"""Validation loss of saved adapters on the same 256 held-out rows and the same masks as textdiff.validate, split
into all masked answer tokens and block-interior tokens only (target position not the first of its 32-token block).

Why: with the token shift (Fast-dLLM v2) a block's first token is predicted from the previous block's last position,
which sees neither the clean previous block nor the rest of its own block; without the shift (dllm BD3LM) it is
predicted from its own position and sees both. For every other token the two conventions see exactly the same
context, so the block-interior loss is the same quantity for both kinds of model.

Usage (venv-dlm, from src/): python textdiff_interior.py NAME:ADAPTER_DIR[:BASE[:noshift]] ...
    e.g. control:~/.local/share/vgc-pilot-runtime/textdiff/arm_control/step500
         qwen3:.../r3f_qwen3_06/step500:.../base_models/qwen3_06b_bd3lm:noshift
Writes $EVAL_OUT (default textdiff/interior.json under the runtime folder). docs/textdiff.md, round three.
"""
import json, os, random, sys, time
import torch, torch.nn.functional as F
import textdiff as T, textdiff_port as P


@torch.no_grad()
def per_token(m, tok, val, shift, fwd):
    base = m.get_base_model(); bd = 32
    gv = torch.Generator(device=P.DEV).manual_seed(1)
    tot = dict(all=[0.0, 0], interior=[0.0, 0], first=[0.0, 0]); cnt = dict(all=0, interior=0, first=0)
    for ids, lab in T.batches(val, tok, 8, 32, random.Random(1), 2048, "full"):
        ids, lab = ids.to(P.DEV), lab.to(P.DEV)
        pm = P.block_mask_rates(ids, bd, generator=gv); u = torch.rand(ids.shape, device=ids.device, generator=gv)
        x1, l1, x2, l2 = P.noisy_pair(ids, lab, pm, u)
        B, L = ids.shape; Lc = L - bd
        M = P.layout_masks(L, bd, 0, "split", ids.device); n = M["n"]
        _, kv = fwd(base, ids[:, :Lc], M["pos"][2 * n:], M["clean"], 0, None, True)
        h, _ = fwd(base, torch.cat([x1, x2], 1), M["pos"][:2 * n], M["noisy"], 0, kv)
        h = h.reshape(B, 2, n, -1); tgt = torch.stack([l1, l2], 1)
        pos = torch.arange(n, device=ids.device).expand(B, 2, n)
        if shift: tgt, h, pos = tgt[..., 1:], h[..., :-1, :], pos[..., 1:]
        sel = tgt != -100
        ce = F.cross_entropy(base.lm_head(h[sel]).float(), tgt[sel], reduction="none")
        first = (pos[sel] % bd) == 0
        for key, mask in (("all", torch.ones_like(first)), ("interior", ~first), ("first", first)):
            # validate() averages each batch's mean over rows (weight = batch size); do the same so "all" reproduces it
            tot[key][0] += float(ce[mask].mean()) * B; tot[key][1] += B
            cnt[key] += int(mask.sum())
    return {k: round(s / max(c, 1), 4) for k, (s, c) in tot.items()} | {"n_first": cnt["first"], "n_all": cnt["all"]}


if __name__ == "__main__":
    from peft import PeftModel
    val = T.load_jsonl(T.OUT / "val.jsonl")[:256]
    out = {}
    for spec in sys.argv[1:]:
        parts = spec.split(":")
        name, adapter = parts[0], parts[1]; base = parts[2] if len(parts) > 2 and parts[2] else None
        shift = not (len(parts) > 3 and parts[3] == "noshift")
        P.SHIFT = shift; P.MASK_ID = 151665
        t0 = time.time()
        tok, m = P.load(base=base)
        m = PeftModel.from_pretrained(m, adapter); m.eval()
        out[name] = per_token(m, tok, val, shift, P.functional_forward)
        print(name, json.dumps(out[name]), f"{time.time() - t0:.0f}s", flush=True)
        del m; torch.mps.empty_cache()
    dest = os.environ.get("EVAL_OUT") or str(T.OUT / "interior.json")
    json.dump(out, open(dest, "w"), indent=1)
