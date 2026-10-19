"""Pretrained text diffusion model (Fast-dLLM v2 1.5B) fine-tuned to write Reg M-B teams.

Pre-registration: docs/textdiff.md. Three stages, two interpreters:

    /tmp/vgc-pilot/.venv/bin/python textdiff.py prepare          # project venv: rows -> text examples
    ~/.local/share/vgc-pilot-runtime/venv-dlm/bin/python textdiff.py smoke     # one real update step, throughput
    ~/.local/share/vgc-pilot-runtime/venv-dlm/bin/python textdiff.py train     # LoRA fine-tune
    ~/.local/share/vgc-pilot-runtime/venv-dlm/bin/python textdiff.py gen ARM   # raw team texts per arm
    /tmp/vgc-pilot/.venv/bin/python textdiff_battle.py ...                     # decode, validate, battle, report

The training examples are the rows the scratch model F1_final_medium_noheldout_regmb_v3 trained on
(scripts/retrain_regmb.py train: the test's family + the real corpus, valid under the v3 vocabulary,
plus the 6,000 v3 synthetic teams), serialised as a Showdown paste without Stat Points, so both
models see the same teams and neither models spreads.
"""
import argparse, json, math, os, random, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNTIME = Path.home() / ".local/share/vgc-pilot-runtime"
OUT = RUNTIME / "textdiff"
REG = RUNTIME / "regmb-v3"
FORMAT = "gen9championsvgc2026regmb"
P_UNCOND = 0.15          # the scratch model's label dropout (lossdown DEFAULT p_uncond)
SEED = 20261014

SYSTEM = "You write Pokémon VGC teams as Showdown pastes."
PROMPT = "Format: {fmt}. Six Pokémon. Target win rate vs top meta teams: {ask}."


def prompt_text(ask):
    return PROMPT.format(fmt=FORMAT, ask="unknown" if ask is None else f"{ask:.2f}")


def slot_text(species, item, ability, nature, moves):
    head = species + (f" @ {item}" if item else "")
    lines = [head, f"Ability: {ability}"]
    if nature: lines.append(f"{nature} Nature")
    lines += [f"- {m}" for m in moves]
    return "\n".join(lines)


# ---------------------------------------------------------------- prepare (project venv)
def prepare(args):
    sys.path.insert(0, str(HERE))
    import numpy as np
    import lossdown as L, asked_vs_got as X, heldout as HO, encode
    reg = json.loads((REG / "training-regulation.json").read_text())
    V = encode.Vocab.from_regulation(reg); names = reg["names"]
    corpus, old, *_ = X.setup()
    d = L.load_data(include_heldout=True); mapped = np.load(REG / "remapped.npz"); synthetic = np.load(REG / "synthetic.npy")
    keep = HO.row_mask(d["X"], old)
    tr, val = L.splits(d); tr = tr[keep[tr]]; val = val[keep[val]]; tag = L.tag_of(d["S"])
    idx = tr[(tag[tr] == L.TEST_TAG) | (tag[tr] == 0)]
    idx = idx[mapped["valid"][idx]]; vi = val[mapped["valid"][val]]
    rng = random.Random(SEED)

    def row_text(r):
        slots = []
        for s in range(6):
            g = lambda j: V.decode_field(s * 8 + j, int(r[s * 8 + j]))
            nm = lambda v: names.get(v, v) if v else ""
            moves = [nm(g(3 + j)) for j in range(4) if g(3 + j) not in ("", "[MASK]")]
            slots.append(slot_text(nm(g(0)), nm(g(2)), nm(g(1)), nm(g(7)), moves))
        return "\n\n".join(slots) + "\n"

    def examples(rows, labels, source, dropout):
        out = []
        for r, w in zip(rows, labels):
            w = None if (w is None or not np.isfinite(w)) else float(w)
            ask = None if (w is None or (dropout and rng.random() < P_UNCOND)) else w
            out.append(dict(prompt=prompt_text(ask), answer=row_text(r), label=w, source=source))
        return out
    real = idx[tag[idx] == 0]; fam = idx[tag[idx] != 0]
    train = (examples(mapped["X"][real], d["W"][real], "real", True)
             + examples(mapped["X"][fam], d["W"][fam], "family", True)
             + examples(synthetic, [None] * len(synthetic), "synthetic", False))
    rng.shuffle(train)
    valid = examples(mapped["X"][vi], d["W"][vi], "val", False)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in [("train", train), ("val", valid)]:
        with open(OUT / f"{name}.jsonl", "w") as f:
            for e in rows: f.write(json.dumps(e) + "\n")
    labelled = sum(e["label"] is not None for e in train)
    meta = dict(n_train=len(train), n_real=int(len(real)), n_family=int(len(fam)), n_synthetic=int(len(synthetic)), n_val=len(valid),
                labelled=labelled, prompt_unknown=sum("unknown" in e["prompt"] for e in train),
                p_uncond=P_UNCOND, seed=SEED, format=FORMAT, scratch_counterpart="F1_final_medium_noheldout_regmb_v3",
                regulation=str(REG / "training-regulation.json"))
    (OUT / "data.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2)); print(train[0]["prompt"]); print(train[0]["answer"])


# ---------------------------------------------------------------- tokenisation (dlm venv)
def short_prompt(prompt):
    """'Format: ... Target win rate vs top meta teams: 0.47.' -> 'Win rate 0.47.' (the format is fixed anyway)."""
    return "Win rate " + prompt.rsplit(": ", 1)[1]


def encode_example(tok, prompt, answer, bd=32, style="full"):
    if style == "short":
        msgs = [{"role": "user", "content": short_prompt(prompt)}]
    else:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    a = tok(answer, add_special_tokens=False).input_ids + [tok.eos_token_id]       # <|im_end|> = 151645
    ids = p + a
    labels = [-100] * len(p) + a
    pad = (-len(ids)) % bd
    ids += [tok.pad_token_id] * pad; labels += [-100] * pad
    return ids, labels


def load_jsonl(path):
    return [json.loads(l) for l in open(path)]


KEYS = ("species", "ability", "item", "move", "nature")


def domain_strings():
    """Every display name a paste can hold, as it appears in one: after a space (' @ Choice Scarf',
    'Ability: Intimidate', '- Fake Out', 'Jolly Nature' -> ' Nature' is already one token) and, for species
    and natures, at the start of a line."""
    reg = json.loads((REG / "training-regulation.json").read_text()); names = reg["names"]
    vals = {k: sorted({names.get(v, v) for v in reg["values"][k] if v}) for k in KEYS}
    return sorted({" " + x for k in KEYS for x in vals[k]} | {x for k in ("species", "nature") for x in vals[k]})


def add_domain_tokens(tok, m):
    """One token per domain name (1,588 added): a team is 135 answer tokens instead of 200. New embedding rows
    start at the mean of the old sub-token rows, and only those rows train (PEFT trainable_token_indices; the
    tied output head follows). Returns the new ids."""
    import torch
    new = domain_strings()
    pieces = [tok(x, add_special_tokens=False).input_ids for x in new]      # tokenised before the names are atomic
    n0 = len(tok); added = tok.add_tokens(new); assert added == len(new), (added, len(new))
    ids_new = tok.convert_tokens_to_ids(new)                                # not always contiguous from len(tok)
    m.resize_token_embeddings(max(len(tok), max(ids_new) + 1), mean_resizing=False)
    E = m.get_input_embeddings().weight
    with torch.no_grad():
        for i, ids in zip(ids_new, pieces): E[i] = E[ids].mean(0)
    print(f"domain tokens: {added} added, ids {min(ids_new)}..{max(ids_new)}, embedding rows {n0} -> {E.shape[0]}", flush=True)
    return sorted(ids_new)


def lora_model(m, r=32, alpha=64, dropout=0.0, ckpt=True, ckpt_layers=None, lora_layers=None, token_indices=None):   # no dropout: validation runs the training-mode forward
    """ckpt_layers: how many of the 28 decoder layers recompute their forward in backward (gradient
    checkpointing). Each layer left out keeps its activations instead: about 0.7 GB more at batch 4
    and one 28th of a forward pass saved per step (measured 2026-10-15, docs/textdiff.md)."""
    from peft import LoraConfig, get_peft_model
    if ckpt:
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(use_reentrant=False))
        # Non-reentrant checkpointing needs no input gradient. Without it, autograd never visits the
        # frozen layers below the first adapter (lora_layers), which is where that arm's speed comes from.
        if lora_layers is None: m.enable_input_require_grads()
        if ckpt_layers is not None:
            for layer in m.model.layers[ckpt_layers:]: layer.gradient_checkpointing = False
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                     layers_to_transform=lora_layers, trainable_token_indices=token_indices)
    return get_peft_model(m, cfg)


def span(text):
    """'14-27' -> [14, ..., 27]; None stays None."""
    if not text: return None
    a, b = text.split("-"); return list(range(int(a), int(b) + 1))


def drop_layers(m, text):
    """Remove decoder layers A..B (a shallower model, as in layer-pruning papers); the rest keep their weights."""
    import torch
    if not text: return m
    gone = set(span(text)); keep = [l for i, l in enumerate(m.model.layers) if i not in gone]
    m.model.layers = torch.nn.ModuleList(keep); m.config.num_hidden_layers = len(keep)
    for i, l in enumerate(m.model.layers): l.self_attn.layer_idx = i
    print(f"dropped layers {sorted(gone)}: {len(keep)} remain", flush=True)
    return m


def compile_backbone(m, on):
    """torch.compile of the decoder stack's forward (Metal backend). Compiling the bound method, not the
    module, keeps the parameter names as they are, so adapters save and load unchanged. Dynamic shapes:
    one compile (about 90 s) covers every padded length; static kernels were 5% faster per step but
    recompiled per length and held 2 GB more (2026-10-15)."""
    import torch
    if not on: return
    base = m.get_base_model()
    base.model.forward = torch.compile(base.model.forward, dynamic=True)


def compiled_functional(on):
    """The functional training forward (textdiff_port.functional_forward), compiled with dynamic shapes like the
    released one, or eager."""
    import torch, textdiff_port as P
    return torch.compile(P.functional_forward, dynamic=True) if on else P.functional_forward


def layout_step(m, ids, lab, g, args, fwd):
    """Loss and gradients of one step through a functional layout (--layout full / trim / detach), in micro-batches
    whose gradients add up to the whole batch's: each micro-batch's cross-entropy sum is divided by the whole batch's
    masked-token count. The masks are drawn exactly as diffusion_loss draws them (same calls, same generator), so
    every layout trains on the same noisy copies as the released path."""
    import torch, textdiff_port as P
    base = m.get_base_model(); bd = base.model.bd_size
    p = P.block_mask_rates(ids, bd, generator=g)
    u = torch.rand(ids.shape, device=ids.device, generator=g)
    x1, l1, x2, l2 = P.noisy_pair(ids, lab, p, u)
    o = 1 if P.SHIFT else 0
    total = int((l1[:, o:] != -100).sum() + (l2[:, o:] != -100).sum())
    mb = args.micro_batch or ids.shape[0]
    loss = 0.0
    for s in range(0, ids.shape[0], mb):
        sl = slice(s, s + mb)
        part, _ = P.layout_loss_sum(base, ids[sl], lab[sl], x1[sl], l1[sl], x2[sl], l2[sl], args.layout, args.ckpt_layers, fwd,
                                    args.decoder_layers)
        (part / total).backward()
        loss += float(part.detach()) / total
    return loss


def mezo_step(m, ids, lab, g, args, fwd, params, seed):
    """MeZO (Malladi et al. 2023, arXiv 2305.17333): the loss at theta + eps*z and theta - eps*z on the same masks,
    the projected gradient (L+ - L-) / 2 eps along z, and no backward pass. Leaves p.grad = projected gradient * z for
    a plain SGD step; z is regenerated from the seed instead of stored."""
    import torch, textdiff_port as P
    base = m.get_base_model(); bd = base.model.bd_size
    pm = P.block_mask_rates(ids, bd, generator=g); u = torch.rand(ids.shape, device=ids.device, generator=g)
    x1, l1, x2, l2 = P.noisy_pair(ids, lab, pm, u)
    o = 1 if P.SHIFT else 0
    total = int((l1[:, o:] != -100).sum() + (l2[:, o:] != -100).sum())
    layout = args.layout if args.layout != "released" else "trim"
    def z(q, gen): return torch.randn(q.shape, device=q.device, dtype=q.dtype, generator=gen)
    def shift(scale):
        gen = torch.Generator(device=P.DEV).manual_seed(seed)
        for q in params: q.data.add_(z(q, gen), alpha=scale * args.mezo_eps)
    def loss():
        with torch.no_grad():
            s, _ = P.layout_loss_sum(base, ids, lab, x1, l1, x2, l2, layout, 0, fwd)
        return float(s) / total
    shift(+1); lp = loss(); shift(-2); lm = loss(); shift(+1)
    proj = (lp - lm) / (2 * args.mezo_eps)
    gen = torch.Generator(device=P.DEV).manual_seed(seed)
    for q in params: q.grad = z(q, gen).mul_(proj)
    return (lp + lm) / 2


def lion(groups, lr, betas=(0.9, 0.99)):
    """Lion (Chen et al. 2023, arXiv 2302.06675): the sign of an interpolated momentum; one state per parameter."""
    import torch
    class Lion(torch.optim.Optimizer):
        @torch.no_grad()
        def step(self):
            for grp in self.param_groups:
                b1, b2 = grp["betas"]
                for p in grp["params"]:
                    if p.grad is None: continue
                    st = self.state[p]
                    if not st: st["m"] = torch.zeros_like(p)
                    mom = st["m"]
                    p.add_(torch.sign(mom * b1 + p.grad * (1 - b1)), alpha=-grp["lr"])
                    mom.mul_(b2).add_(p.grad, alpha=1 - b2)
    return Lion(groups, dict(lr=lr, betas=betas))


def make_optimizer(m, args):
    """--optimizer adamw (default) / adamw-fused / sgd (--momentum) / lion / adafactor; --lora-plus R gives the
    LoRA B matrices R times the learning rate (LoRA+, Hayou et al. 2024, arXiv 2402.12354)."""
    import torch
    named = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    if args.lora_plus != 1.0:
        groups = [dict(params=[p for n, p in named if "lora_B" not in n]),
                  dict(params=[p for n, p in named if "lora_B" in n], lr=args.lr * args.lora_plus)]
    else:
        groups = [dict(params=[p for _, p in named])]
    o = args.optimizer
    if o == "adamw": return torch.optim.AdamW(groups, lr=args.lr, weight_decay=0.0)
    if o == "adamw-fused": return torch.optim.AdamW(groups, lr=args.lr, weight_decay=0.0, fused=True)
    if o == "sgd": return torch.optim.SGD(groups, lr=args.lr, momentum=args.momentum)
    if o == "lion": return lion(groups, lr=args.lr)
    if o == "adafactor": return torch.optim.Adafactor(groups, lr=args.lr, weight_decay=0.0)
    if o == "mezo": return torch.optim.SGD(groups, lr=args.lr, momentum=0.0)       # applies mezo_step's p.grad
    raise ValueError(o)


def optimizer_state_gb(opt):
    import torch
    return sum(t.numel() * t.element_size() for st in opt.state.values() for t in st.values() if torch.is_tensor(t)) / 1e9


def prune_mlp(m, tok, rows, keep, n_calib=64, style="full"):
    """Structured width pruning (Minitron-style, arXiv 2407.14679): keep the `keep` fraction of each layer's MLP
    neurons with the largest mean |activation| (the input to down_proj) over n_calib clean training sequences
    (released eval path, block-causal), and slice gate/up/down to them. The dense matrices shrink, so the Metal GPU
    does less work, unlike 2:4 or unstructured sparsity, whose speed-ups need NVIDIA sparse kernels."""
    import torch, textdiff_port as P
    layers = m.model.layers; score = [None] * len(layers); hooks = []
    def hook(i):
        def f(mod, inp):
            a = inp[0].detach().abs().float().sum((0, 1))
            score[i] = a if score[i] is None else score[i] + a
        return f
    for i, l in enumerate(layers): hooks.append(l.mlp.down_proj.register_forward_pre_hook(hook(i)))
    m.eval()
    with torch.no_grad():
        for r in rows[:n_calib]:
            ids, _ = encode_example(tok, r["prompt"], r["answer"], 32, style)
            m(input_ids=torch.tensor([ids], device=P.DEV), block_size=32)
    for h in hooks: h.remove()
    new = int(round(m.config.intermediate_size * keep / 64)) * 64
    for i, l in enumerate(layers):
        idx = score[i].topk(new).indices.sort().values
        for name, dim in (("gate_proj", 0), ("up_proj", 0), ("down_proj", 1)):
            W = getattr(l.mlp, name).weight.data.index_select(dim, idx).clone()
            lin = torch.nn.Linear(W.shape[1], W.shape[0], bias=False, device=W.device, dtype=W.dtype); lin.weight.data = W
            setattr(l.mlp, name, lin)
        l.mlp.intermediate_size = new
    m.config.intermediate_size = new
    print(f"pruned the MLP width to {new} neurons per layer (keep {keep:.2f}, {n_calib} calibration sequences)", flush=True)
    return m


def verify(args):
    """textdiff.py verify: the functional layouts against the released forward on one real batch of four (full prompt):
    loss values and LoRA gradients (adapter B matrices set to small random values so every gradient is non-zero)."""
    import torch, textdiff_port as P
    tok, m = P.load(dtype=torch.float32 if args.arm == "fp32" else torch.bfloat16)     # textdiff.py verify fp32
    rows = load_jsonl(OUT / "train.jsonl")[:4]
    ids, lab = collate([encode_example(tok, r["prompt"], r["answer"], 32, args.prompt_style) for r in rows], tok, 32)
    m = lora_model(m, args.rank, 2 * args.rank, ckpt=True)
    torch.manual_seed(0)
    for n, p in m.named_parameters():
        if "lora_B" in n: p.data.normal_(0, 1e-3)
    print(json.dumps(P.verify_layouts(m, ids.to(P.DEV), lab.to(P.DEV)), indent=1), flush=True)


def batches(rows, tok, bs, bd, rng, max_len, style="full"):
    order = list(range(len(rows))); rng.shuffle(order)
    enc = []
    for i in order:
        ids, lab = encode_example(tok, rows[i]["prompt"], rows[i]["answer"], bd, style)
        if len(ids) > max_len: continue
        enc.append((ids, lab))
        if len(enc) == bs:
            yield collate(enc, tok, bd); enc = []
    if enc: yield collate(enc, tok, bd)


def collate(enc, tok, bd):
    import torch
    L = max(len(i) for i, _ in enc); L = math.ceil(L / bd) * bd
    ids = torch.full((len(enc), L), tok.pad_token_id, dtype=torch.long)
    lab = torch.full((len(enc), L), -100, dtype=torch.long)
    for b, (i, l) in enumerate(enc):
        ids[b, :len(i)] = torch.tensor(i); lab[b, :len(l)] = torch.tensor(l)
    return ids, lab


def smoke(args):
    import torch, textdiff_port as P
    tok, m = P.load()
    rows = load_jsonl(OUT / "train.jsonl")[:64]
    lens = [len(encode_example(tok, r["prompt"], r["answer"])[0]) for r in rows]
    print(f"token length: mean {sum(lens)/len(lens):.0f} max {max(lens)} (prompt+answer, padded to 32)")
    d, scale = P.verify(m, tok); print(f"port check: max |train − eval| logit {d:.4f} at scale {scale:.0f}")
    m = lora_model(m, ckpt=not args.no_ckpt, ckpt_layers=args.ckpt_layers); m.print_trainable_parameters()
    compile_backbone(m, args.compile)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=args.lr)
    before = [p.detach().clone() for p in m.parameters() if p.requires_grad][0]
    rng = random.Random(0); g = torch.Generator(device=P.DEV).manual_seed(0)
    for step, (ids, lab) in enumerate(batches(rows, tok, args.batch, 32, rng, 2048)):
        ids, lab = ids.to(P.DEV), lab.to(P.DEV)
        t0 = time.perf_counter()
        loss = P.diffusion_loss(m, ids, lab, generator=g, shared_clean=not args.legacy_layout)
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.0)
        opt.step(); torch.mps.synchronize()
        dt = time.perf_counter() - t0
        print(f"step {step}: loss {loss.item():.4f} grad-norm {gn.item():.3f} finite {math.isfinite(loss.item())} "
              f"{ids.numel()/dt:.0f} tok/s (×2 complementary) batch {tuple(ids.shape)} {dt:.1f}s "
              f"peak {torch.mps.driver_allocated_memory()/1e9:.1f} GB", flush=True)
        if step == 3: break
    after = [p for p in m.parameters() if p.requires_grad][0]
    print("trainable parameter changed:", bool((after != before).any()))


def train(args):
    import torch, textdiff_port as P
    torch.manual_seed(SEED); random.seed(SEED)
    P.SHIFT = not args.no_shift
    tok, m = P.load(base=args.base or None)
    if args.base and P.qwen3_folder(args.base):
        assert args.layout != "released", "a Qwen3 base trains only through the functional layouts (--layout trim)"
        args.functional_val = True
    rows = load_jsonl(OUT / "train.jsonl"); val = load_jsonl(OUT / "val.jsonl")[:args.nval]
    if args.limit:                                   # every real-corpus row, then the shuffled rest up to the limit
        real = [r for r in rows if r["source"] == "real"]; rest = [r for r in rows if r["source"] != "real"]
        rows = real + rest[:max(0, args.limit - len(real))]; random.Random(SEED).shuffle(rows)
    print(f"training on {len(rows)} examples: {dict(__import__('collections').Counter(r['source'] for r in rows))}", flush=True)
    new_ids = add_domain_tokens(tok, m) if args.domain_tokens else None
    if args.base and not args.base_knows_mask: new_ids = sorted(set(new_ids or []) | {P.MASK_ID})   # a plain Qwen2.5 checkpoint must learn [MASK]
    m = drop_layers(m, args.drop_layers)
    if args.prune_mlp: m = prune_mlp(m, tok, rows, args.prune_mlp, style=args.prompt_style)
    m = lora_model(m, args.rank, 2 * args.rank, ckpt=not args.no_ckpt, ckpt_layers=args.ckpt_layers,
                   lora_layers=span(args.lora_layers), token_indices=new_ids); m.print_trainable_parameters()
    compile_backbone(m, args.compile)
    fwd = compiled_functional(args.compile) if (args.layout != "released" or args.optimizer == "mezo") else None
    params = [p for p in m.parameters() if p.requires_grad]
    opt = make_optimizer(m, args)
    steps_per_epoch = math.ceil(len(rows) / args.batch); total = steps_per_epoch * args.epochs
    warm = max(1, int(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.05, 1 - max(0, s - warm) / max(1, total - warm)))
    g = torch.Generator(device=P.DEV).manual_seed(SEED); rng = random.Random(SEED)
    ck = OUT / args.name; ck.mkdir(parents=True, exist_ok=True)
    if args.domain_tokens: tok.save_pretrained(ck / "tokenizer")       # gen must tokenise with the same vocabulary
    log = []; t0 = time.perf_counter(); step = 0; running = 0.0; tokens = 0; peak = 0.0
    for epoch in range(args.epochs):
        for ids, lab in batches(rows, tok, args.batch, 32, rng, 2048, args.prompt_style):
            if args.max_steps and step >= args.max_steps: break
            ids, lab = ids.to(P.DEV), lab.to(P.DEV)
            if args.optimizer == "mezo":
                opt.zero_grad(set_to_none=True); lv = mezo_step(m, ids, lab, g, args, fwd, params, SEED + step)
            elif fwd is None:
                loss = P.diffusion_loss(m, ids, lab, generator=g, shared_clean=not args.legacy_layout,
                                        complementary=not args.no_complementary)
                if not torch.isfinite(loss): raise RuntimeError("non-finite loss")
                opt.zero_grad(set_to_none=True); loss.backward(); lv = loss.item()
            else:
                opt.zero_grad(set_to_none=True); lv = layout_step(m, ids, lab, g, args, fwd)
                if not math.isfinite(lv): raise RuntimeError("non-finite loss")
            if args.optimizer != "mezo": torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()
            running += lv; tokens += ids.numel(); step += 1
            peak = max(peak, torch.mps.driver_allocated_memory() / 1e9)
            if step % 25 == 0 or step == total:
                entry = dict(epoch=epoch, step=step, total=total, loss=running / 25, lr=sched.get_last_lr()[0],
                             minutes=(time.perf_counter() - t0) / 60, tok_per_s=tokens / (time.perf_counter() - t0),
                             peak_gb=peak, opt_state_gb=optimizer_state_gb(opt))
                log.append(entry); running = 0.0
                print(json.dumps(entry), flush=True)
                torch.mps.empty_cache()          # micro-batches of many lengths fragment the Metal cache (17-18 GB by step 150)
            if step % args.eval_every == 0 or step == total or step == args.max_steps:
                if args.speed_only:
                    (ck / "log.json").write_text(json.dumps(dict(args=vars(args), log=log), indent=1)); continue
                v, na = validate(m, tok, val, g, shared=not args.legacy_layout, style=args.prompt_style, fwd=fwd,
                                 decoder_layers=args.decoder_layers, functional=args.functional_val)
                log.append(dict(step=step, val=v, answer_tokens=na, seconds_per_step=(time.perf_counter() - t0) / step)); print(json.dumps(log[-1]), flush=True)
                m.save_pretrained(ck / f"step{step}")
                (ck / "log.json").write_text(json.dumps(dict(args=vars(args), log=log), indent=1))
    if not args.speed_only: m.save_pretrained(ck / "final")
    (ck / "log.json").write_text(json.dumps(dict(args=vars(args), log=log, minutes=(time.perf_counter() - t0) / 60), indent=1))


def validate(m, tok, val, g, bs=8, shared=True, style="full", fwd=None, decoder_layers=0, functional=False):
    """The published loss (complementary pairs) on the held-out rows: the same function for every arm.
    Returns (mean loss per masked answer token, mean answer tokens per row): with a different vocabulary the
    per-token number changes meaning, their product is the comparable per-team figure. decoder_layers > 0: the
    model the arm trained (noisy tokens through the top N layers only), on the same masks."""
    import torch, textdiff_port as P
    tot, n, na = 0.0, 0, 0
    gv = torch.Generator(device=P.DEV).manual_seed(1)
    with torch.no_grad():
        for ids, lab in batches(val, tok, bs, 32, random.Random(1), 2048, style):
            ids, lab = ids.to(P.DEV), lab.to(P.DEV)
            if decoder_layers or functional:            # functional split path: equals the released forward (verified)
                pm = P.block_mask_rates(ids, 32, generator=gv); u = torch.rand(ids.shape, device=ids.device, generator=gv)
                x1, l1, x2, l2 = P.noisy_pair(ids, lab, pm, u)
                s, k = P.layout_loss_sum(m.get_base_model(), ids, lab, x1, l1, x2, l2, "split", 0, fwd or P.functional_forward, decoder_layers)
                tot += float(s) / k * len(ids)
            else:
                tot += P.diffusion_loss(m, ids, lab, generator=gv, shared_clean=shared).item() * len(ids)
            n += len(ids)
            na += int((lab != -100).sum())
    return tot / max(n, 1), na / max(n, 1)


# ---------------------------------------------------------------- generation (dlm venv)
ARMS = {
    "untouched": dict(adapter=None, ask=None),        # the pretrained checkpoint, same prompt, no fine-tuning
    "base":      dict(ask=None),                       # fine-tuned, win rate unknown (the scratch model's unconditioned arm)
    "ask50":     dict(ask=0.5),                        # fine-tuned, ask 0.5 (the scratch model's cfg arm, guidance 1)
    "ask60":     dict(ask=0.6),
    "ask50_g2":  dict(ask=0.5, guidance=2.0),          # classifier-free guidance 2 in the block sampler
}


def gen(args):
    import torch, textdiff_port as P
    from peft import PeftModel
    cfg = ARMS[args.arm]
    tok, m = P.load()
    adapter = cfg.get("adapter", args.adapter)
    if adapter and (OUT / args.name / "tokenizer").exists():            # trained with domain tokens: same vocabulary
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(OUT / args.name / "tokenizer"), trust_remote_code=True)
        m.resize_token_embeddings(max(len(tok), max(tok.convert_tokens_to_ids(domain_strings())) + 1), mean_resizing=False)
    if adapter: m = PeftModel.from_pretrained(m, str(OUT / args.name / adapter)).merge_and_unload()
    m.eval()
    tag = f"{args.tag}_" if args.tag else ""                    # --tag pool: a separate file and its own seed (stack pool)
    dest = OUT / "gen" / f"{args.name}_{tag}{args.arm}.jsonl"; dest.parent.mkdir(parents=True, exist_ok=True)
    done = sum(1 for _ in open(dest)) if dest.exists() else 0
    torch.manual_seed(SEED + 7 * list(ARMS).index(args.arm) + done + (100000 if args.tag else 0))
    prompt = prompt_text(cfg["ask"])
    chat = lambda t: tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": t}],
                                             add_generation_prompt=True, return_tensors="pt").to(P.DEV)
    ids = chat(prompt)
    unc = chat(prompt_text(None)) if cfg.get("guidance") else None
    with open(dest, "a") as f:
        for i in range(done, args.n):
            t0 = time.perf_counter()
            with torch.no_grad():
                if unc is None:
                    out = m.generate(ids, max_new_tokens=args.max_new, threshold=args.threshold, block_size=32,
                                     small_block_size=8, temperature=args.temperature, top_p=args.top_p)
                else:
                    out = P.generate_cfg(m, ids, unc, cfg["guidance"], max_new_tokens=args.max_new, threshold=args.threshold,
                                         temperature=args.temperature, top_p=args.top_p)
            text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            f.write(json.dumps(dict(i=i, arm=args.arm, text=text, seconds=time.perf_counter() - t0,
                                    tokens=int(out.shape[1] - ids.shape[1]))) + "\n"); f.flush()
            if i % 8 == 0: print(f"{args.arm} {i}/{args.n} {time.perf_counter()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "smoke", "train", "gen", "verify"])
    p.add_argument("arm", nargs="?", default="base")
    p.add_argument("--name", default="lora_r32")
    p.add_argument("--adapter", default="final")
    p.add_argument("--tag", default="", help="pool: write gen/<name>_pool_<arm>.jsonl for textdiff_battle.py stack")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--nval", type=int, default=256)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--no-ckpt", action="store_true", help="no gradient checkpointing at all (30+ GB at batch 4: does not fit)")
    p.add_argument("--ckpt-layers", type=int, default=28, help="decoder layers that recompute in backward; 28 = all (8.4 GB); 20 is ~5%% faster but holds up to 17 GB")
    p.add_argument("--legacy-layout", action="store_true", help="released [noisy | clean] pairs instead of the shared-clean layout")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True, help="torch.compile the decoder stack (--no-compile for eager)")
    p.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (schedule still planned for the full run)")
    p.add_argument("--lora-layers", default="", help="adapters only on decoder layers A-B, e.g. 14-27 (backward stops below A)")
    p.add_argument("--drop-layers", default="", help="remove decoder layers A-B before training, e.g. 10-17")
    p.add_argument("--no-complementary", action="store_true", help="one mask per example per step instead of the complementary pair")
    p.add_argument("--prompt-style", choices=["full", "short"], default="full", help="short: 'Win rate 0.47.' and no system message")
    p.add_argument("--domain-tokens", action="store_true", help="one token per species / ability / item / move / nature")
    p.add_argument("--base", default="", help="a Qwen2 checkpoint to put the block-diffusion recipe on, e.g. Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--layout", choices=["released", "full", "trim", "split", "detach"], default="released",
                   help="training row: released forward; full = the same row through the functional forward; trim = without the "
                        "noisy prompt blocks and the last clean block (same loss and gradient); detach = trim with the clean blocks "
                        "under no_grad (same loss value, no gradient through the clean context)")
    p.add_argument("--decoder-layers", type=int, default=0, help="split / detach: the noisy tokens run only the top N layers (E2D2 shared-KV); 0 = all")
    p.add_argument("--micro-batch", type=int, default=0, help="split each batch into micro-batches of N (functional layouts; same gradient)")
    p.add_argument("--optimizer", choices=["adamw", "adamw-fused", "sgd", "lion", "adafactor", "mezo"], default="adamw")
    p.add_argument("--mezo-eps", type=float, default=1e-3, help="MeZO perturbation size")
    p.add_argument("--momentum", type=float, default=0.9, help="sgd momentum")
    p.add_argument("--lora-plus", type=float, default=1.0, help="learning-rate ratio of the LoRA B matrices to A (LoRA+); 1 = plain LoRA")
    p.add_argument("--prune-mlp", type=float, default=0.0, help="keep this fraction of MLP neurons per layer before training (0 = no pruning)")
    p.add_argument("--speed-only", action="store_true", help="no validation and no adapter saves (speed probes)")
    p.add_argument("--functional-val", action="store_true", help="validate through the functional split path (same masks; equals the released forward)")
    p.add_argument("--base-knows-mask", action="store_true", help="the --base checkpoint is already a diffusion model: do not train its [MASK] row")
    p.add_argument("--no-shift", action="store_true", help="score each masked token at its own position (dllm BD3LM checkpoints), training and validation")
    p.add_argument("--n", type=int, default=96)
    p.add_argument("--max-new", type=int, default=320)
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    a = p.parse_args()
    globals()[a.command](a)
