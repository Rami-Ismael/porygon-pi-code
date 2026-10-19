"""Fast-dLLM v2 (block diffusion, Qwen2.5-1.5B-Instruct adapted) on Apple MPS.

The released modeling.py trains through torch flex_attention compiled for CUDA. This file
swaps that one function for a dense boolean mask through scaled_dot_product_attention and
re-implements the published training loss (per-block mask rate, complementary masks,
[noisy | clean] concatenation, token-shifted cross-entropy on masked positions) so that the
masking is controllable for verification. Nothing else in the released model is changed.

Run with ~/.local/share/vgc-pilot-runtime/venv-dlm (transformers 4.57, peft 0.18).
"""
import sys, time
import torch, torch.nn.functional as F

MODEL_ID = "Efficient-Large-Model/Fast_dLLM_v2_1.5B"
MASK_ID = 151665
SHIFT = True          # token shift: hidden state i predicts token i + 1 (Fast-dLLM v2). False: predicts token i (dllm's BD3LM)
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load(dtype=torch.bfloat16, device=DEV, base=None):
    """base=None: the released Fast-dLLM v2 1.5B. base='Qwen/Qwen2.5-0.5B-Instruct' (or any Qwen2 checkpoint):
    the released block-diffusion architecture and training recipe on that model's weights, with the [MASK] row
    (id 151665, untrained in a plain Qwen2.5 checkpoint) left for the trainer to learn (trainable_token_indices)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if base and qwen3_folder(base):
        return tok, load_qwen3(base, dtype).to(device)
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, dtype=dtype)
    if base:
        from pathlib import Path
        if (Path(base) / "model.safetensors").exists():
            # a local folder of weights and JSON only (e.g. a dllm-hub Qwen2 block-diffusion model downloaded without its
            # .py files): the tensors go into the released architecture, so none of the folder's own code runs
            import json as _json
            from safetensors.torch import load_file
            from transformers import Qwen2Config
            raw = _json.loads((Path(base) / "config.json").read_text())
            raw = {k: v for k, v in raw.items() if k not in ("architectures", "auto_map", "model_type", "transformers_version")}
            src_cfg, state = Qwen2Config.from_dict(raw), {k: v.to(dtype) for k, v in load_file(str(Path(base) / "model.safetensors")).items()}
        else:
            src = AutoModelForCausalLM.from_pretrained(base, dtype=dtype); src_cfg, state = src.config, src.state_dict(); del src
        cfg = type(m.config).from_dict({**src_cfg.to_dict(), "bd_size": m.config.bd_size, "architectures": m.config.architectures,
                                        "auto_map": m.config.auto_map, "model_type": m.config.model_type})
        swapped = type(m)(cfg).to(dtype)
        missing, unexpected = swapped.load_state_dict(state, strict=False); src = None
        assert not unexpected and all(k == "lm_head.weight" for k in missing), (missing, unexpected)
        swapped.tie_weights(); m = swapped; del src
        print(f"block diffusion on {base}: {sum(p.numel() for p in m.parameters())/1e6:.0f}M parameters", flush=True)
    patch(m)
    return tok, m.to(device)


def qwen3_folder(base):
    from pathlib import Path
    import json as _json
    p = Path(base) / "config.json"
    return p.exists() and "qwen3" in _json.loads(p.read_text()).get("model_type", "")


def load_qwen3(base, dtype):
    """A local Qwen3 block-diffusion folder (weights + JSON, e.g. dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1) as a stock
    transformers Qwen3ForCausalLM: none of the folder's own code runs. It trains and validates only through the
    functional layouts (functional_forward handles Qwen3's per-head query/key norms); its [MASK] is <|mask|>."""
    global _MODULE, MASK_ID
    from pathlib import Path
    import json as _json
    from safetensors.torch import load_file
    from transformers import Qwen3Config, Qwen3ForCausalLM
    raw = _json.loads((Path(base) / "config.json").read_text())
    raw = {k: v for k, v in raw.items() if k not in ("architectures", "auto_map", "model_type", "transformers_version")}
    m = Qwen3ForCausalLM(Qwen3Config.from_dict(raw)).to(dtype)
    missing, unexpected = m.load_state_dict({k: v.to(dtype) for k, v in load_file(str(Path(base) / "model.safetensors")).items()}, strict=False)
    assert not unexpected and not missing, (missing, unexpected)
    m.tie_weights(); m.model.bd_size = 32
    _MODULE = type(m.model.layers[0].self_attn).__module__
    added = _json.loads((Path(base) / "added_tokens.json").read_text()); MASK_ID = added["<|mask|>"]
    print(f"Qwen3 block diffusion from {base}: {sum(p.numel() for p in m.parameters())/1e6:.0f}M parameters, [MASK] id {MASK_ID}", flush=True)
    return m


def dense_train_mask(n, block_size, device, nseg=2):
    """The published block_diff_mask over a [noisy | clean] sequence of 2n tokens, as a dense bool.
    nseg=3 is the shared-clean layout [noisy | complementary noisy | clean] (segment_mask)."""
    if nseg != 2: return segment_mask(n, block_size, nseg, device)
    mod = sys.modules[type_module()]
    q = torch.arange(2 * n, device=device)[:, None]
    k = torch.arange(2 * n, device=device)[None, :]
    return mod.block_diff_mask(None, None, q, k, block_size=block_size, n=n)


def segment_mask(n, block_size, nseg, device):
    """block_diff_mask generalised to nseg-1 noisy segments sharing one clean segment (the last).
    A noisy query sees its own block in its own segment and the clean blocks before it; a clean
    query sees clean blocks up to its own. For nseg=2 this is the published mask exactly."""
    idx = torch.arange(nseg * n, device=device)
    seg, blk = idx // n, (idx % n) // block_size
    sq, sk, bq, bk = seg[:, None], seg[None, :], blk[:, None], blk[None, :]
    clean = nseg - 1
    diagonal = (sq == sk) & (sq < clean) & (bq == bk)
    offset_causal = (sq < clean) & (sk == clean) & (bq > bk)
    causal = (sq == clean) & (sk == clean) & (bq >= bk)
    return diagonal | offset_causal | causal


_MODULE = None
def type_module(): return _MODULE


def patch(model):
    """Route training attention through sdpa with a dense mask; keep eval attention as released."""
    global _MODULE
    _MODULE = type(model).__module__
    mod = sys.modules[_MODULE]

    def dense_attention(q, k, v, mask=None):
        # q: (B, Hq, L, d); k, v: (B, Hkv, L, d): the kernel maps the 12 query heads onto the 2 kv heads
        # (enable_gqa), the same grouping flex_attention's enable_gqa=True applied, without copying k and v.
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
    mod.fused_flex_attention = dense_attention

    def gen_mask(self, seqlen, block_size, B, H):
        return dense_train_mask(seqlen, block_size, next(self.parameters()).device, getattr(self, "textdiff_nseg", 2))
    type(model.model).gen_mask = gen_mask

    # The released training attention applies rotary to each half of [noisy | clean] separately,
    # which is the same as tiling the n positions over the segments; written that way, it also
    # covers the shared-clean layout's three segments. Eval attention is untouched.
    attn_cls = type(model.model.layers[0].self_attn)
    released_forward = attn_cls.forward

    def forward(self, hidden_states, position_embeddings, attention_mask, *args, **kwargs):
        if not self.training:
            return released_forward(self, hidden_states, position_embeddings, attention_mask, *args, **kwargs)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        reps = q.shape[2] // cos.shape[1]
        q, k = mod.apply_rotary_pos_emb(q, k, cos.repeat(1, reps, 1), sin.repeat(1, reps, 1))
        out = mod.fused_flex_attention(q, k, v, mask=attention_mask)
        return self.o_proj(out.transpose(1, 2).reshape(*input_shape, -1).contiguous())
    attn_cls.forward = forward
    return model


def block_mask_rates(input_ids, bd, eps=1e-3, generator=None):
    """One mask rate per 32-token block, as in the released forward."""
    B, L = input_ids.shape
    nb = L // bd
    t = torch.rand((B * nb,), device=input_ids.device, generator=generator)
    p = (1 - eps) * t + eps
    return p[:, None].repeat(1, bd).reshape(B, L)


def diffusion_loss(model, input_ids, labels, p_mask=None, complementary=True, generator=None, return_logits=False,
                   shared_clean=True):
    """Exactly the released training loss, with the mask rate exposed.

    input_ids, labels: (B, L) with L % bd == 0; labels are -100 outside the answer.
    Returns the mean token-shifted cross-entropy over masked answer tokens.

    shared_clean: the clean half never attends to noisy tokens, so its hidden states are the same in
    both complementary rows. Run the pair as one row [noisy | complementary noisy | clean] (3L tokens
    instead of 2 x 2L): the same attention pattern and the same loss, a quarter fewer tokens.
    verify_shared() checks the two layouts agree.
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model   # PEFT wrapper -> LoRA-injected LM
    bd = base.model.bd_size
    B, L = input_ids.shape
    assert L % bd == 0, "sequence length must be a multiple of the block size"
    if p_mask is None: p_mask = block_mask_rates(input_ids, bd, generator=generator)
    u = torch.rand((B, L), device=input_ids.device, generator=generator)
    mask_indices = u < p_mask
    answer = labels != -100

    def noisy(mask_indices):
        x_t = torch.where(mask_indices & answer, torch.full_like(input_ids, MASK_ID), input_ids)
        lab = torch.where(x_t == MASK_ID, labels, torch.full_like(labels, -100))
        return x_t, lab
    x1, l1 = noisy(mask_indices)
    base.train()                                   # the released forward branches on self.training; it must still be
    if complementary and shared_clean:             # set when checkpointed layers recompute in backward, so leave it
        x2, l2 = noisy(~mask_indices)
        xs = torch.cat([x1, x2, input_ids], 1); ls = torch.stack([l1, l2], 1)          # (B, 3L), (B, 2, L)
        base.model.textdiff_nseg = 3
        out = base.model(input_ids=xs, labels=xs[:, :L], position_ids=torch.arange(L, device=xs.device)[None])
        h = out.last_hidden_state[:, :2 * L, :].reshape(B, 2, L, -1)
    else:
        xs, ls = torch.cat([x1, input_ids], 1), l1[:, None]                             # (B, 2L), (B, 1, L)
        if complementary:
            x2, l2 = noisy(~mask_indices)
            xs, ls = torch.cat([xs, torch.cat([x2, input_ids], 1)], 0), torch.cat([ls, l2[:, None]], 0)
        base.model.textdiff_nseg = 2
        out = base.model(input_ids=xs, labels=ls[:, 0])
        h = out.last_hidden_state[:, :L, :].unsqueeze(1)
    if return_logits:
        logits = base.lm_head(h).float()
        shift_logits = logits[..., :-1, :].reshape(-1, logits.shape[-1]); shift_labels = ls[..., 1:].reshape(-1)
        return F.cross_entropy(shift_logits, shift_labels, ignore_index=-100), logits.flatten(0, 1)
    # token shift: hidden state i predicts label i+1 (transformers ForCausalLMLoss), within each noisy
    # segment. Only the masked positions carry a label, so run the 152k-way head on those rows alone (memory).
    tgt, hh = (ls[..., 1:], h[..., :-1, :]) if SHIFT else (ls, h)
    sel = tgt != -100
    logits = base.lm_head(hh[sel]).float()
    return F.cross_entropy(logits, tgt[sel])


@torch.no_grad()
def eval_logits(model, input_ids, block_size=32):
    """Released eval path: block-causal mask over a clean sequence (prefix of a multiple of block_size)."""
    model.eval()
    out = model(input_ids=input_ids, block_size=block_size)
    return out.logits.float()


def verify(model, tok, text="Garchomp @ Choice Scarf\nAbility: Rough Skin\nJolly Nature\n- Earthquake\n- Dragon Claw\n- Rock Slide\n- Protect\n"):
    """At mask rate 0 the noisy half is the clean sequence: training-path logits must equal the
    eval-path logits under the block-causal mask. Returns max abs difference."""
    bd = model.model.bd_size
    ids = tok(text, return_tensors="pt").input_ids
    L = (ids.shape[1] // bd) * bd
    ids = ids[:, :L].to(next(model.parameters()).device)
    labels = ids.clone()
    _, lt = diffusion_loss(model, ids, labels, p_mask=torch.zeros_like(ids, dtype=torch.float), complementary=False, return_logits=True)
    le = eval_logits(model, ids, block_size=bd)
    return float((lt - le).abs().max()), float(le.abs().max())


@torch.no_grad()
def verify_shared(model, tok, texts=None, seed=0):
    """The shared-clean layout against the released [noisy | clean] pairs on one batch at a random
    mask: max abs logit difference over the two noisy segments, and the two losses."""
    bd = model.model.bd_size; dev = next(model.parameters()).device
    texts = texts or ["Garchomp @ Choice Scarf\nAbility: Rough Skin\nJolly Nature\n- Earthquake\n- Dragon Claw\n- Rock Slide\n- Protect\n",
                      "Incineroar @ Safety Goggles\nAbility: Intimidate\nCareful Nature\n- Fake Out\n- Flare Blitz\n- Knock Off\n- Parting Shot\n"]
    enc = [tok(t).input_ids for t in texts]; L = max(len(e) for e in enc); L = (L + bd - 1) // bd * bd
    ids = torch.full((len(enc), L), tok.pad_token_id); lab = torch.full((len(enc), L), -100)
    for i, e in enumerate(enc): ids[i, :len(e)] = torch.tensor(e); lab[i, :len(e)] = torch.tensor(e)
    ids, lab = ids.to(dev), lab.to(dev)
    p = block_mask_rates(ids, bd, generator=torch.Generator(device=dev).manual_seed(seed))
    out = {}
    for shared in (False, True):
        g = torch.Generator(device=dev).manual_seed(seed + 1)
        out[shared] = diffusion_loss(model, ids, lab, p_mask=p, generator=g, return_logits=True, shared_clean=shared)
    (loss2, lg2), (loss3, lg3) = out[False], out[True]
    # released layout rows: [row0 mask, row1 mask, row0 complement, row1 complement]; shared: [row0 mask, row0 complement, ...]
    B = len(enc); lg2 = lg2.view(2, B, L, -1).transpose(0, 1).reshape(B * 2, L, -1)
    return float((lg2 - lg3).abs().max()), float(lg2.abs().max()), float(loss2), float(loss3)


if __name__ == "__main__":
    t0 = time.time()
    tok, m = load(dtype=torch.float32 if "fp32" in sys.argv else torch.bfloat16)
    print(f"loaded in {time.time()-t0:.0f}s on {DEV}", flush=True)
    d, scale = verify(m, tok)
    print(f"port check: max |train-path − eval-path| logit = {d:.4f} (logit scale {scale:.1f})", flush=True)
    d, scale, l2, l3 = verify_shared(m, tok)
    print(f"shared-clean check: max |pairs − shared| logit = {d:.4f} (logit scale {scale:.1f}); loss pairs {l2:.5f} shared {l3:.5f}", flush=True)


# ---------------------------------------------------------------- classifier-free guidance in the block sampler
@torch.no_grad()
def generate_cfg(model, cond_ids, uncond_ids, guidance, max_new_tokens=320, threshold=0.9, block_size=32,
                 small_block_size=8, temperature=1.0, top_p=1.0, stop_token=151645, pad_id=151643):
    """The released block sampler, run on a batch of two prompts (conditional, unconditional) whose
    logits are combined as g·cond + (1−g)·uncond at every denoising step and for the autoregressive
    first token of each block. The unconditional prompt is left-padded with <|endoftext|> to the
    conditional prompt's length so both rows share one block structure and one KV cache.
    Sampling, thresholding and stopping follow modeling.py's generate exactly."""
    import torch.nn.functional as F
    bs, small = block_size, small_block_size
    dev = cond_ids.device
    Lc, Lu = cond_ids.shape[1], uncond_ids.shape[1]
    assert Lu <= Lc, "unconditional prompt longer than the conditional one"
    if Lu < Lc: uncond_ids = torch.cat([torch.full((1, Lc - Lu), pad_id, dtype=torch.long, device=dev), uncond_ids], 1)
    x = torch.cat([cond_ids, uncond_ids], 0)
    original = Lc
    combine = lambda lg: guidance * lg[0:1] + (1 - guidance) * lg[1:2]

    def sample(lg):
        if temperature <= 0:
            p = F.softmax(lg, -1); return p.argmax(-1), p
        p = F.softmax(lg / temperature, -1)
        if top_p < 1.0:
            ps, ix = p.sort(-1, descending=True); keep = ps.cumsum(-1) - ps < top_p
            p = torch.zeros_like(p).scatter(-1, ix, ps * keep); p = p / p.sum(-1, keepdim=True)
        return torch.multinomial(p[0], 1).squeeze(1)[None], p

    def stopped(row):                                    # the released check: a stop token with no mask before it
        tail = row[original:]
        if stop_token not in tail: return False
        k = int((tail == stop_token).nonzero()[0, 0])
        return int((tail[:k] == MASK_ID).sum()) == 0

    past = None
    if x.shape[1] > bs:
        out = model.forward(input_ids=x[:, :x.shape[1] // bs * bs], use_cache=True, update_past_key_values=True, block_size=bs)
        past = out.past_key_values
        if x.shape[1] % bs == 0:
            nt = combine(out.logits[:, -1:, :].float()).argmax(-1)
            x = torch.cat([x, nt.expand(2, 1)], 1)
    for _ in range(max_new_tokens // bs):
        if stop_token in x[0, original:]: break
        plen = x.shape[1]
        x = torch.cat([x, torch.full((2, bs - plen % bs), MASK_ID, dtype=torch.long, device=dev)], 1)
        while True:
            if stopped(x[0]): break
            if int((x[0, -bs:] == MASK_ID).sum()) == 0:
                out = model.forward(input_ids=x[:, -bs:], use_cache=True, past_key_values=past, update_past_key_values=True, block_size=bs)
                past = out.past_key_values
                nt = combine(out.logits[:, -1:, :].float()).argmax(-1)
                x = torch.cat([x, nt.expand(2, 1)], 1)
                break
            for sb in range(bs // small):
                start = -bs + sb * small
                end = None if sb == bs // small - 1 else -bs + (sb + 1) * small
                while True:
                    m = (x[0:1, -bs:] == MASK_ID)
                    if int(m[:, start:end].sum()) == 0: break
                    if stopped(x[0]): break
                    out = model.forward(input_ids=x[:, -bs:], use_cache=True, past_key_values=past, update_past_key_values=False)
                    lg = out.logits.float(); lg = torch.cat([lg[:, :1], lg[:, :-1]], 1)
                    lg = combine(lg)[:, start:end]
                    x1, p = sample(lg)
                    conf = p.gather(-1, x1[..., None]).squeeze(-1)
                    conf = torch.where(m[:, start:end], conf, torch.full_like(conf, -float("inf")))
                    un = conf > threshold
                    un[0, conf.argmax(-1)] = True
                    un = (un & m[:, start:end]).expand(2, -1)
                    seg = x[:, start:end]; seg[un] = x1.expand(2, -1)[un]
    if stop_token in x[0, original:]:
        k = int((x[0, original:] == stop_token).nonzero()[0, 0]); x = x[:, :original + k + 1]
    return x[0:1]


# ---------------------------------------------------------------- faster training layouts (2026-10-18)
# A shared-clean training row is [x_t | x_t' | x0], 3L tokens. Two parts of it never reach the loss:
#   - a noisy block whose hidden states predict only prompt labels (token shift: position i predicts i + 1). No other
#     noisy query attends to it, so dropping it from both noisy copies changes nothing (the full prompt fills block 0);
#   - the last clean block: a noisy block attends only to the clean blocks before it.
# layout "trim" removes both: the same loss and the same gradient, ~11% fewer tokens with the full prompt.
# layout "detach" also runs the clean blocks once under no_grad and lets the noisy blocks attend to their keys and
# values: the same loss value, but no gradient through the clean context (E2D2-style split of clean and noisy work).
# layout "full" is the shared-clean row through the same code, the reference for both checks (verify_layouts).
from torch.utils.checkpoint import checkpoint as _checkpoint


def _layer(layer, h, cos, sin, mask, prefix=None, capture=False):
    """One decoder layer exactly as the released training path computes it (patched attention), rotary from explicit
    positions. prefix: keys and values of earlier tokens to prepend (detached clean context)."""
    mod = sys.modules[_MODULE]
    attn = layer.self_attn
    B, T, _ = h.shape
    x = layer.input_layernorm(h)
    q = attn.q_proj(x).view(B, T, -1, attn.head_dim)
    k = attn.k_proj(x).view(B, T, -1, attn.head_dim)
    if hasattr(attn, "q_norm"): q, k = attn.q_norm(q), attn.k_norm(k)          # Qwen3: per-head RMS norm before rotary
    q, k = q.transpose(1, 2), k.transpose(1, 2)
    v = attn.v_proj(x).view(B, T, -1, attn.head_dim).transpose(1, 2)
    q, k = mod.apply_rotary_pos_emb(q, k, cos, sin)
    kv = (k, v) if capture else None
    if prefix is not None:
        k = torch.cat([prefix[0], k], 2); v = torch.cat([prefix[1], v], 2)
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
    h = h + attn.o_proj(o.transpose(1, 2).reshape(B, T, -1))
    h = h + layer.mlp(layer.post_attention_layernorm(h))
    return h, kv


def _layer_kv(layer, h, cos, sin, mask, prefix=None):
    h, kv = _layer(layer, h, cos, sin, mask, prefix, True)
    return h, kv[0], kv[1]


def functional_forward(base, x, pos, mask, ckpt_layers=28, prefix=None, capture=False, first_layer=0):
    """The released training forward written out layer by layer. x (B, T) token ids, pos (T,) absolute positions,
    mask (T, prefix + T) bool. ckpt_layers: the first N layers recompute in backward. capture: also return each
    layer's (k, v), as tensors that keep their gradient when grad is on (layout split). first_layer: start the stack
    there, straight from the embeddings (E2D2's shared-KV decoder: the noisy tokens run only the top layers, each
    reading the clean keys and values of the same layer). prefix: per-layer (k, v) indexed by layer."""
    lm = base.model
    h = lm.embed_tokens(x)
    cos, sin = lm.rotary_emb(h, pos[None])
    kvs = [None] * len(lm.layers)
    for i, layer in enumerate(lm.layers):
        if i < first_layer: continue
        p = None if prefix is None else prefix[i]
        if i < ckpt_layers and torch.is_grad_enabled():
            if capture:
                h, k, v = _checkpoint(lambda h_, l=layer, p_=p: _layer_kv(l, h_, cos, sin, mask, p_), h, use_reentrant=False)
                kvs[i] = (k, v)
            else:
                h = _checkpoint(lambda h_, l=layer, p_=p: _layer(l, h_, cos, sin, mask, p_)[0], h, use_reentrant=False)
        else:
            h, kv = _layer(layer, h, cos, sin, mask, p, capture)
            if capture: kvs[i] = kv
    return lm.norm(h), kvs


def row_mask(seg_q, blk_q, seg_k, blk_k):
    """block_diff_mask over any arrangement of segments. seg 0/1 = the two noisy copies, 2 = clean; blk = absolute
    block index. A noisy query sees its own block in its own copy and the clean blocks before it; a clean query sees
    the clean blocks up to its own."""
    sq, sk, bq, bk = seg_q[:, None], seg_k[None, :], blk_q[:, None], blk_k[None, :]
    noisy_q, clean_k = sq < 2, sk == 2
    same = (sq == sk) & noisy_q & (bq == bk)
    offset = noisy_q & clean_k & (bq > bk)
    causal = (sq == 2) & clean_k & (bq >= bk)
    return same | offset | causal


def _segments(L, bd, start, drop_last_clean, device):
    """(seg, blk, pos) for noisy, noisy', clean with the noisy copies starting at token `start`."""
    Lc = L - bd if drop_last_clean else L
    pn, pc = torch.arange(start, L, device=device), torch.arange(0, Lc, device=device)
    seg = torch.cat([torch.zeros_like(pn), torch.ones_like(pn), torch.full_like(pc, 2)])
    pos = torch.cat([pn, pn, pc])
    return seg, pos // bd, pos, len(pn), Lc


_MASKS = {}
def layout_masks(L, bd, start, layout, device):
    key = (L, bd, start, layout, str(device))
    if key not in _MASKS:
        drop = layout != "full"
        seg, blk, pos, n, Lc = _segments(L, bd, start, drop, device)
        full = row_mask(seg, blk, seg, blk)
        if layout in ("detach", "split"):           # rows: noisy queries; keys: [clean | noisy | noisy']
            order = torch.cat([torch.arange(2 * n, 2 * n + Lc, device=device), torch.arange(0, 2 * n, device=device)])
            _MASKS[key] = dict(clean=full[2 * n:, 2 * n:], noisy=full[:2 * n][:, order], pos=pos, n=n, Lc=Lc)
        else:
            _MASKS[key] = dict(mask=full, pos=pos, n=n, Lc=Lc)
    return _MASKS[key]


def noisy_pair(input_ids, labels, p_mask, u):
    """The two complementary noisy copies and their labels, exactly as diffusion_loss builds them."""
    mask_indices = u < p_mask
    answer = labels != -100
    def noisy(mi):
        x_t = torch.where(mi & answer, torch.full_like(input_ids, MASK_ID), input_ids)
        return x_t, torch.where(x_t == MASK_ID, labels, torch.full_like(labels, -100))
    (x1, l1), (x2, l2) = noisy(mask_indices), noisy(~mask_indices)
    return x1, l1, x2, l2


def trim_start(labels, bd):
    """First token of the noisy copies: whole blocks whose hidden states predict only prompt labels are dropped
    (block j predicts labels j*bd+1 .. (j+1)*bd, so it is droppable when (j+1)*bd < the shortest prompt)."""
    first = int((labels != -100).float().argmax(1).min().item())     # shortest prompt in the (micro-)batch
    if not SHIFT: return first // bd * bd                            # without the shift a block predicts its own labels
    return (first - 1) // bd * bd if first > 0 else 0


def layout_loss_sum(base, ids, labels, x1, l1, x2, l2, layout="trim", ckpt_layers=28, fwd=functional_forward, decoder_layers=0):
    """Sum of token-shifted cross-entropy over the masked answer tokens of both noisy copies, and their count.
    Trailing all-padding blocks are cut first (no earlier token attends to them). Layouts: full (the shared-clean row),
    trim (one row without the dead blocks), split (trim as two passes, clean then noisy, gradient through both: equals
    trim), detach (split with the clean pass under no_grad). decoder_layers N > 0 (split / detach): the noisy tokens run
    only the top N layers (E2D2 shared-KV)."""
    bd = base.model.bd_size
    real = int((ids != PAD_ID).any(0).nonzero().max()) + 1 if (ids == PAD_ID).any() else ids.shape[1]
    L = min(ids.shape[1], (real + bd - 1) // bd * bd)
    ids, labels, x1, l1, x2, l2 = (t[:, :L] for t in (ids, labels, x1, l1, x2, l2))
    start = 0 if layout == "full" else trim_start(labels, bd)
    M = layout_masks(L, bd, start, layout, ids.device)
    n, Lc, pos = M["n"], M["Lc"], M["pos"]
    noisy = torch.cat([x1[:, start:], x2[:, start:]], 1)
    if layout in ("detach", "split"):
        first = len(base.model.layers) - decoder_layers if decoder_layers else 0
        if layout == "detach":
            with torch.no_grad():
                _, kv = fwd(base, ids[:, :Lc], pos[2 * n:], M["clean"], 0, None, True)
        else:
            _, kv = fwd(base, ids[:, :Lc], pos[2 * n:], M["clean"], ckpt_layers, None, True)
        h, _ = fwd(base, noisy, pos[:2 * n], M["noisy"], ckpt_layers, kv, False, first)
    else:
        h, _ = fwd(base, torch.cat([noisy, ids[:, :Lc]], 1), pos, M["mask"], ckpt_layers)
        h = h[:, :2 * n]
    B = ids.shape[0]
    h = h.reshape(B, 2, n, -1)
    tgt = torch.stack([l1[:, start:], l2[:, start:]], 1)
    if SHIFT: tgt, h = tgt[..., 1:], h[..., :-1, :]
    sel = tgt != -100
    logits = base.lm_head(h[sel]).float()
    return F.cross_entropy(logits, tgt[sel], reduction="sum"), int(sel.sum())


PAD_ID = 151643


def verify_layouts(model, ids, lab, seed=0):
    """The functional layouts against the released shared-clean forward on one real batch at one random mask.
    Losses: 'full', 'trim' and 'detach' must all equal the released path. Gradients (LoRA weights): 'trim' must equal
    'full' up to bf16 noise; 'detach' differs by construction (no path through the clean context): reported."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    bd = base.model.bd_size; dev = ids.device; B, L = ids.shape
    g = torch.Generator(device=dev).manual_seed(seed)
    p = block_mask_rates(ids, bd, generator=g); u = torch.rand(ids.shape, device=dev, generator=g)
    x1, l1, x2, l2 = noisy_pair(ids, lab, p, u)
    params = [q for q in model.parameters() if q.requires_grad]
    def grads(fn):
        for q in params: q.grad = None
        loss = fn(); loss.backward()
        return float(loss), [q.grad.detach().float().clone() if q.grad is not None else torch.zeros_like(q, dtype=torch.float) for q in params]
    def released():
        base.train(); base.model.textdiff_nseg = 3
        out = base.model(input_ids=torch.cat([x1, x2, ids], 1), labels=ids, position_ids=torch.arange(L, device=dev)[None])
        h = out.last_hidden_state[:, :2 * L].reshape(B, 2, L, -1)
        tgt = torch.stack([l1, l2], 1)[..., 1:]; sel = tgt != -100
        return F.cross_entropy(base.lm_head(h[..., :-1, :][sel]).float(), tgt[sel])
    def functional(layout):
        def f():
            s, n = layout_loss_sum(base, ids, lab, x1, l1, x2, l2, layout)
            return s / n
        return f
    res = {}; G = {}
    for name, fn in [("released", released), ("full", functional("full")), ("trim", functional("trim")), ("split", functional("split")),
                     ("detach", functional("detach"))]:
        res[name], G[name] = grads(fn)
    norm = lambda gs: float(torch.sqrt(sum((x ** 2).sum() for x in gs)))
    for name in ("full", "trim", "split", "detach"):
        diff = norm([a - b for a, b in zip(G[name], G["released"])])
        res[name + "_grad_rel_diff"] = diff / max(norm(G["released"]), 1e-12)
    for q in params: q.grad = None
    return res
