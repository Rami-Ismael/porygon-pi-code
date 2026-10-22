"""The scratch team writer: the deployed Regulation M-B diffusion model behind one interface.

    w = TeamWriter.load()                           # the loop-teams model + the battle-records model
    rows = w.sample(64, ask=0.5, guidance=2.0)      # [64, 48] token ids, legal by construction
    text, slots = w.to_paste(rows[0], rng)
    w2 = w.fine_tuned(rows, labels, epochs=40)      # a new writer; w is unchanged

Everything a caller used to rebuild from the loss-down training module (lossdown.py) lives here: which
member checkpoints, the legality tables from the checkpoint's regulation snapshot, the two-member mixture,
the matchup-row padding of the battle-records model, the sampler (fixed unmasking order, ask and guidance,
temperature, top-p, surrogate tilt, species guidance, condition-annealed sampling) and fine-tuning. A fine-tuned writer is cached under a key that
hashes every setting that changes its weights, so changing the budget can never reload a stale copy.

It reproduces the foundation-swap sampler (foundation.Gen("v3")) and the cross-entropy loops' fine-tunes
(cem_v3.members, cem_dbbo.members), except that it never samples a Pokémon with no moves, which Showdown
rejects (scripts/test_team_writer.py). Finished experiments keep calling lossdown directly.
"""
import copy, hashlib, json
from pathlib import Path
import numpy as np
import torch

import lossdown as L
import asked_vs_got as X
import alignment as AL
import diffusion as D
import heldout as HO
import hps_generate
from hps_generate import slot_to_text
from checkpoints import load_checkpoint
from encode import NF, NSLOT, Vocab

MEMBERS = ("F1_final_medium_noheldout_regmb_v3", "G2_family_matrix_noprtrain_noheldout_regmb_v3")
CACHE = L.CK / "team_writer"
V3DIR = Path.home() / ".local/share/vgc-pilot-runtime/regmb-v3"
SPECIES_COLS = [c for c in range(L.COLS) if c % NF == 0]
NAN = float("nan")
FIRST_MOVE = 3                  # field offset of a slot's first move column (species, ability, item, moves, nature)
MATRIX_COLUMNS = 49             # the battle-records model's matchup row: one column per other top-50 meta team


def _digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


class TeamWriter:
    """One scratch team writer: an ensemble of member networks plus what sampling and fine-tuning need.
    Build it with TeamWriter.load() or writer.fine_tuned(); never mutate one in place."""

    @classmethod
    def load(cls, members=MEMBERS, start=None):
        """The members' own checkpoints, or with start=PREFIX the weights saved as PREFIX_<member>.pt
        (start="dial_precise" is the precise-label writer of ask_dial.py)."""
        ck = load_checkpoint(L.CK / (members[0] + ".pt"), map_location="cpu")
        if not ck.get("regulation"):
            raise ValueError("the team writer needs a versioned checkpoint with a regulation snapshot")
        V = Vocab.from_regulation(ck["regulation"])
        corpus, corpus_vocab, _, look, spreads = X.setup()
        look.update(V.regulation["names"])
        tables = L.Tables(V, corpus)
        nets = []
        for name in members:
            net, cfg, _ = L.load_net(name, V)
            if start:
                net.load_state_dict(load_checkpoint(L.CK / f"{start}_{name}.pt", map_location=L.DEV)["sd"]); net.eval()
            nets.append((name, net, cfg))
        _, _, _, modes, natures = AL.build_tables(corpus)
        shared = dict(V=V, tables=tables, corpus=corpus, corpus_vocab=corpus_vocab, look=look, spreads=spreads,
                      modes=modes, natures=natures, original_rows={})
        return cls(shared, nets, dict(members=list(members), start=start))

    def __init__(self, shared, nets, provenance):
        self._shared, self._nets, self.provenance = shared, nets, provenance
        self.key = _digest(provenance)
        self._f = L.ensemble_fn([L.logprobs_fn(net, shared["tables"], tuple(cfg["rules"].split(","))) for _, net, cfg in nets])

    V = property(lambda self: self._shared["V"])
    tables = property(lambda self: self._shared["tables"])
    corpus = property(lambda self: self._shared["corpus"])

    # ---------------------------------------------------------------- scoring
    def logprobs(self, xin, t, w):
        """{field: log-probs [B, columns, values]} of the member mixture, legality-masked."""
        return self._f(xin, t, w)

    def column_logprobs(self, x, t, w, c):
        k = L.KEY_OF_COL[c]; return self._f(x, t, w)[k][:, L.COLS_BY_KEY[k].index(c)]

    def legal_values(self, x, c):
        """Boolean [B, values] for column c given the visible fields; never empty, never [MASK]. Moves are
        stored sorted with the empty move last, so a slot's first move column may not be empty: Showdown
        rejects a Pokémon with no moves, which the shared tables allow."""
        k = L.KEY_OF_COL[c]; o = self.tables.allowed(x)[k][:, L.COLS_BY_KEY[k].index(c)].clone()
        E = self.tables.EMV
        if c % NF == FIRST_MOVE and E is not None:                    # only where a real move is still allowed
            real = o.clone(); real[:, [0, E]] = False; o[real.any(1), E] = False
        o[~o.any(1)] = True; o[:, 0] = False
        return o

    # ---------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample(self, n, ask=None, guidance=1.0, temperature=1.0, top_p=1.0, surrogate=None, lam=0.0,
               species_guidance=None, cads=None):
        """n teams in diffusion.ORDER. ask: the win-rate condition (None = unconditioned); guidance: the
        classifier-free weight against the unconditioned writer; species_guidance: that weight on the six
        species columns only; surrogate/lam: add lam * surrogate.tilt to every column's logits; cads: the noise
        scale of condition-annealed sampling (CADS), noise on the condition embedding annealed away over the
        steps (docs/guidance-timing-cads.md)."""
        w = torch.full((n,), NAN if ask is None else ask, device=L.DEV); wn = torch.full((n,), NAN, device=L.DEV)
        x = torch.zeros(n, L.COLS, dtype=torch.long, device=L.DEV)
        try:
            for step, c in enumerate(D.ORDER):
                t = torch.full((n,), 1.0 - step / L.COLS, device=L.DEV)
                L.COND_NOISE = None if cads is None else (cads, step / (L.COLS - 1))
                gc = species_guidance if (species_guidance is not None and c in SPECIES_COLS) else guidance
                lc = self.column_logprobs(x, t, w, c)
                if ask is not None and gc != 1.0: lc = gc * lc + (1 - gc) * self.column_logprobs(x, t, wn, c)
                if surrogate is not None and lam > 0: lc = lc + lam * surrogate.tilt(self.V, x, c)
                lc = lc.masked_fill(~self.legal_values(x, c), -float("inf")) / temperature
                p = torch.softmax(lc, -1)
                if top_p < 1.0:
                    ps, ix = p.sort(-1, descending=True); keep = ps.cumsum(-1) - ps < top_p
                    p = torch.zeros_like(p).scatter(1, ix, ps * keep); p = p / p.sum(-1, keepdim=True)
                x[:, c] = torch.multinomial(p, 1).squeeze(1)
        finally:
            L.COND_NOISE = None
        return x

    def to_paste(self, row, rng, repair=False):
        """(Showdown paste, slot dicts) for one sampled row. Stat Points come from a same-species corpus
        team, or a random legal spread for a species the corpus lacks; repair=True aligns the nature."""
        sh = self._shared; V, look = sh["V"], sh["look"]
        slots = []
        for s in range(NSLOT):
            g = lambda j: V.decode_field(s * NF + j, int(row[s * NF + j]))
            sp = g(0); pl = sh["spreads"].get(sp) or [hps_generate.sample_spread(rng)]
            slot = dict(species=look.get(sp, sp), item=look.get(g(2), g(2)), ability=look.get(g(1), g(1)),
                        nature=look.get(g(7), g(7)), moves=[look.get(g(3 + j), g(3 + j)) for j in range(4)
                                                            if g(3 + j) not in ("", "[MASK]")],
                        evs=pl[int(rng.integers(0, len(pl)))])
            if repair: slot["nature"] = AL.repair_nature(slot, sh["modes"], sh["natures"])
            slots.append(slot)
        return "\n\n".join(slot_to_text(s) for s in slots) + "\n", slots

    # ---------------------------------------------------------------- fine-tuning
    def fine_tuned(self, rows, labels, *, epochs=None, steps=None, batch=64, lr=1e-4, mix_original=False, seed=0,
                   cache_dir=CACHE):
        """A new writer: every member trained further from this writer's weights on (rows, labels).
        Training starts from this writer's weights: fine-tune TeamWriter.load() for the original members, or
        TeamWriter.load(start="dial_precise") for the precise-label ones.
        epochs=E: E shuffled passes over the rows (the cross-entropy loop's re-steer, cem_v3). steps=S: S batches
        drawn with replacement; with mix_original=True half of each batch comes from the member's own training
        rows (the diffusion black-box-optimization arm, cem_dbbo). AdamW, weight decay 0.01, gradients clipped
        at 1. Cached under a hash of this writer, every argument above except cache_dir, and, with
        mix_original, the files the members' own rows are read from."""
        if (epochs is None) == (steps is None):
            raise ValueError("give exactly one of epochs or steps")
        if mix_original and steps is None:
            raise ValueError("mix_original needs steps: an epoch over the member's own rows is not defined")
        rows = np.asarray(rows, dtype=np.int64); labels = np.asarray(labels, dtype=np.float32)
        recipe = dict(parent=self.key, rows=hashlib.sha256(rows.tobytes()).hexdigest(),
                      labels=hashlib.sha256(labels.tobytes()).hexdigest(), n=len(rows), epochs=epochs, steps=steps,
                      batch=batch, lr=lr, mix_original=bool(mix_original), seed=seed)
        if mix_original: recipe["original_data"] = self._original_data_digest()
        key = _digest(recipe); cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
        tables = self.tables; nets = []
        for name, parent, cfg in self._nets:
            net = copy.deepcopy(parent); ck = cache_dir / f"{key[:20]}_{name}.pt"
            if ck.exists():
                net.load_state_dict(load_checkpoint(ck, map_location=L.DEV)["sd"]); net.eval()
                nets.append((name, net, cfg)); continue
            torch.manual_seed(seed)
            xr = torch.as_tensor(rows, device=L.DEV); wr = torch.as_tensor(labels, device=L.DEV)
            if cfg.get("mrow"): wr = torch.cat([wr[:, None], torch.full((len(wr), MATRIX_COLUMNS), NAN, device=L.DEV)], 1)
            sched = L.sched_tensor(cfg["sched"]).to(L.DEV); rules = tuple(cfg["rules"].split(","))
            opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
            net.train()
            def update(bx, bw):
                loss = L.batch_loss(net, tables, rules, bx, bw, None, sched, cfg)
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            if epochs is not None:
                for _ in range(epochs):
                    perm = torch.randperm(len(xr), device=L.DEV)
                    for i in range(0, len(xr), batch):
                        b = perm[i:i + batch]; update(xr[b], wr[b])
            elif mix_original:
                xo, wo = self._original_rows(name, cfg); half = batch // 2
                for _ in range(steps):
                    bo = torch.randint(len(xo), (half,), device=L.DEV); bl = torch.randint(len(xr), (half,), device=L.DEV)
                    update(torch.cat([xo[bo], xr[bl]]), torch.cat([wo[bo], wr[bl]]))
            else:
                for _ in range(steps):
                    b = torch.randint(len(xr), (batch,), device=L.DEV); update(xr[b], wr[b])
            net.eval()
            torch.save({"sd": net.state_dict(), "recipe": {**recipe, "member": name}}, ck)
            nets.append((name, net, cfg))
        return TeamWriter(self._shared, nets, dict(self.provenance, fine_tune=recipe))

    def _original_data_digest(self):
        """sha256 of the three files the members' own training rows come from."""
        if "original_digest" not in self._shared:
            h = hashlib.sha256()
            for f in (L.DATA, V3DIR / "remapped.npz", V3DIR / "synthetic.npy"): h.update(Path(f).read_bytes())
            self._shared["original_digest"] = h.hexdigest()
        return self._shared["original_digest"]

    def _original_rows(self, name, cfg):
        """The member's own training rows, rebuilt exactly as scripts/retrain_regmb.py chose them."""
        cache = self._shared["original_rows"]
        if name in cache: return cache[name]
        d = L.load_data(include_heldout=True); mapped = np.load(V3DIR / "remapped.npz"); syn = np.load(V3DIR / "synthetic.npy")
        keep = HO.row_mask(d["X"], self._shared["corpus_vocab"]); tr, _ = L.splits(d); tr = tr[keep[tr]]; tag = L.tag_of(d["S"])
        idx = tr[(tag[tr] == L.TEST_TAG) | (tag[tr] == 0) | ((d["M"][tr, :, 1].sum(1) > 0) if cfg.get("mrow") else False)]
        if cfg.get("mrow"): idx = np.setdiff1d(idx, L.matrix_holdout(d))
        idx = idx[mapped["valid"][idx]]
        w = L.cond_matrix(d, idx) if cfg.get("mrow") else d["W"][idx]
        sw = np.full((len(syn), 1 + MATRIX_COLUMNS) if cfg.get("mrow") else (len(syn),), np.nan, np.float32)
        xx = np.concatenate([mapped["X"][idx], syn]); ww = np.concatenate([w, sw]).astype(np.float32)
        cache[name] = (torch.as_tensor(xx, device=L.DEV), torch.as_tensor(ww, device=L.DEV))
        return cache[name]
