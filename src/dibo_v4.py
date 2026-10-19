"""Online discrete DiBO replacement for CEM on the regulation-v3 foundation.

python src/dibo_v4.py run --smoke --output results/dibo_v4_smoke.json
python src/dibo_v4.py run
python src/dibo_v4.py final
python src/dibo_v4.py report
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/tmp/vgc-pilot/src")
sys.path.insert(0, str(HERE))
import dibo_core_v4 as DC
import foundation as FD
import lossdown as L
import lossdown_battle as LB
import matchup_db as MDB
import heldout as HO
import diffusion as D
from corpus import parse_team_text
from encode import NF
from hps_generate import Validator
from checkpoints import load_checkpoint


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def atomic_checkpoint(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def parse_slots(paste):
    # The legacy parser merges item-less adjacent slots; parsing blocks avoids that.
    return [s for block in paste.strip().split("\n\n") for s in parse_team_text(block)]


class Denoiser(nn.Module):
    """Trainable mixture of foundation members, always unconditioned on win-rate labels."""
    def __init__(self, nets, configs, tables):
        super().__init__()
        self.nets, self.configs, self.tables = nn.ModuleList(nets), configs, tables

    def distributions(self, x, t):
        w = torch.full((len(x),), float("nan"), device=x.device)
        outputs = []
        for net, cfg in zip(self.nets, self.configs):
            if cfg.get("prime", 1) != 1:
                raise ValueError("DiBO path adapter requires the full-token foundation")
            logits = net.logits(net(x, t, w))
            outputs.append({k: F.log_softmax(v.float(), -1) for k, v in logits.items()})
        return {k: torch.logsumexp(torch.stack([o[k] for o in outputs]), 0) - math.log(len(outputs))
                for k in outputs[0]}

    def forward(self, x, t, c):
        k = L.KEY_OF_COL[c]
        j = L.COLS_BY_KEY[k].index(c)
        lp = self.distributions(x, t)[k][:, j]
        allowed = self.tables.allowed(x)[k][:, j].clone()
        # Common full-support fallback for a dead prefix; validator rejects bad endpoints.
        allowed[~allowed.any(1)] = True
        allowed[:, 0] = False
        return F.log_softmax(lp.masked_fill(~allowed, -float("inf")), -1)

    def denoising_loss(self, x, mask, t):
        xin = x.masked_fill(mask, 0)
        distributions = self.distributions(xin, t)
        ce = torch.zeros(x.shape, device=x.device)
        for k, cols in L.COLS_BY_KEY.items():
            ce[:, cols] = -distributions[k].gather(2, x[:, cols, None]).squeeze(-1)
        # Per-row uniform absorbing-mask diffusion objective. CE is a training loss,
        # not the cross-entropy SEARCH method that this implementation replaces.
        return (ce * mask / t[:, None]).sum(1) / x.shape[1]


class Runtime:
    def __init__(self, cfg, outpath):
        self.cfg, self.outpath = cfg, Path(outpath)
        torch.set_num_threads(cfg["threads"])
        torch.manual_seed(cfg["seed"])
        LB._stack()
        self.generator = FD.Gen("v3")
        self.V = self.generator.V
        nets, configs = [], []
        for name in FD.V3_MEMBERS:
            net, c, _ = L.load_net(name, self.V)
            nets.append(net)
            configs.append(c)
        self.base = Denoiser(nets, configs, self.generator.tables).to(L.DEV).eval()
        self.con = MDB.connect()
        self.validator = Validator()
        self.render_cache = {}
        self.checkdir = L.CK / ("dibo_v4_" + hashlib.sha256(str(self.outpath.resolve()).encode()).hexdigest()[:12])
        self.signature = dict(format="discrete-dibo-v4", config=cfg, vocabulary=self.V.itos,
                              regulation=self.V.regulation,
                              meta=self.con.execute("SELECT team_id,weight FROM team_set WHERE set_name='top50' ORDER BY team_id").fetchall(),
                              policy=self.con.execute("SELECT format_id,row_policy,col_policy,simulator FROM policy WHERE policy_id=?", (LB.POLICY,)).fetchone(),
                              implementation={p.name: self.file_hash(p) for p in (Path(__file__), HERE / "dibo_core_v4.py")},
                              foundation={n: self.file_hash(L.CK / (n + ".pt")) for n in FD.V3_MEMBERS})
        if not self.signature["meta"]:
            raise ValueError("the top50 matchup set must be initialized")

    @staticmethod
    def file_hash(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def close(self):
        self.validator.close()
        self.con.close()

    def render(self, row):
        key = tuple(int(v) for v in row)
        if key not in self.render_cache:
            seed = int.from_bytes(hashlib.sha256(np.asarray(key, np.int64).tobytes()).digest()[:8], "little")
            # Keep this mapping fixed for the run: no nature repair after probability scoring.
            paste, _ = self.generator.to_paste(np.asarray(key), np.random.default_rng(seed), repair=False)
            legal = self.validator(paste) is None and not HO.is_heldout_text(paste)
            self.render_cache[key] = (paste, legal, HO.grid_key_text(paste))
        return self.render_cache[key]

    def initial(self):
        rows = []
        for score in MDB.labels(self.con, LB.POLICY, "top50"):
            if score["coverage"] != score["columns"]:
                continue
            tid = score["team_id"]
            paste, legal = self.con.execute("SELECT paste,legal FROM team WHERE team_id=?", (tid,)).fetchone()
            if not legal or HO.is_heldout_text(paste):
                continue
            slots = parse_slots(paste)
            if len(slots) != 6:
                continue
            try:
                x = self.V.encode(slots)
            except ValueError:
                continue
            if (x == 0).any():
                continue
            rows.append(dict(paste=paste, x=x.tolist(), y=float(score["score"]), team_id=tid,
                             key=HO.grid_key_text(paste), src="initial", battles=score["battles"]))
        # One label per semantic grid; freeze this initial snapshot in the run manifest.
        unique = {}
        for row in sorted(rows, key=lambda r: -r["battles"]):
            unique.setdefault(row["key"], row)
        rows = list(unique.values())
        if len(rows) < 2:
            raise ValueError("DiBO needs at least two fully evaluated, non-heldout legal teams")
        rng = np.random.default_rng(self.cfg["seed"])
        if len(rows) > self.cfg["initial_limit"]:
            rows = [rows[i] for i in rng.choice(len(rows), self.cfg["initial_limit"], replace=False)]
        return rows

    def ensemble(self, seed):
        members = []
        keys = [L.KEY_OF_COL[c] for c in range(NF)]
        for i in range(self.cfg["ensemble"]):
            torch.manual_seed(seed + i)
            members.append(DC.Predictor(self.V.sizes, keys, width=self.cfg["width"], hidden=self.cfg["hidden"]))
        return DC.PredictorEnsemble(members).to(L.DEV)

    def models(self, rows, generation):
        seed = self.cfg["seed"] + generation * 1000
        torch.manual_seed(seed)
        ck = self.checkdir / f"round{generation}.pt"
        prior = DC.posterior_copy(DC.PathSampler(self.base, D.ORDER, L.DEV))
        ensemble = self.ensemble(seed)
        if ck.exists():
            data = load_checkpoint(ck, map_location=L.DEV)
            if data["signature"] != self.signature or data["rows"] != rows:
                raise ValueError("model cache does not match this round's observations")
            prior.model.load_state_dict(data["prior"])
            posterior = DC.posterior_copy(prior)
            posterior.model.load_state_dict(data["posterior"])
            ensemble.load_state_dict(data["ensemble"])
            return prior, posterior, ensemble.eval(), data["training"]
        x = torch.tensor([r["x"] for r in rows], dtype=torch.long, device=L.DEV)
        y = torch.tensor([r["y"] for r in rows], dtype=torch.float32, device=L.DEV)
        progress = lambda message: print(f"round {generation}: {message}", flush=True)
        loss_prior = DC.train_prior(prior.model, x, y, self.cfg["prior_epochs"], self.cfg["batch"], self.cfg["lr"], progress)
        loss_predictors = ensemble.fit(x, y, self.cfg["predictor_epochs"], self.cfg["batch"], self.cfg["predictor_lr"], seed, progress)
        posterior = DC.posterior_copy(prior)
        def tilt(terminal):
            mu, sigma = ensemble.standardized(terminal)
            result = self.cfg["beta"] * (mu + self.cfg["gamma"] * sigma)
            legal = torch.tensor([self.render(r)[1] for r in terminal.cpu().numpy()], device=L.DEV)
            # Positive finite barrier permits RTB training on invalid paths. Proposals
            # subsequently condition on validity by rejection, which preserves the
            # desired target restricted to legal, non-heldout endpoints.
            return torch.where(legal, result, torch.full_like(result, -40.0*self.cfg['beta']))
        training_rtb, log_z = DC.train_rtb(prior, posterior, tilt, self.cfg["rtb_steps"],
                                         self.cfg["rtb_batch"], self.cfg["rtb_lr"], replay=x, progress=progress,
                                         temperature=self.cfg['beta'], microbatch=self.cfg['rtb_microbatch'])
        training = dict(prior_loss=loss_prior, predictor_mse=loss_predictors, rtb=training_rtb, log_z=log_z,
                        y_mean=float(ensemble.y_mean), y_scale=float(ensemble.y_scale))
        atomic_checkpoint(ck, dict(signature=self.signature, rows=rows, prior=prior.model.state_dict(),
                                  posterior=posterior.model.state_dict(), ensemble=ensemble.state_dict(), training=training))
        print(f"round {generation}: weighted prior + {len(ensemble.members)} predictors + {len(training_rtb)} RTB updates", flush=True)
        return prior, posterior, ensemble.eval(), training

    @torch.no_grad()
    def records(self, terminal, prior, ensemble, terms=None):
        mu, sigma = ensemble(terminal)
        zmu, zsigma = ensemble.standardized(terminal)
        terms = prior.log_prob_terms(terminal) if terms is None else terms
        log_prior = terms.sum(1)
        records = []
        for i, r in enumerate(terminal.cpu().numpy()):
            paste, legal, key = self.render(r)
            if not legal or not torch.isfinite(log_prior[i]):
                continue
            ucb = float(mu[i] + self.cfg["gamma"] * sigma[i])
            zucb = float(zmu[i] + self.cfg['gamma'] * zsigma[i])
            records.append(dict(x=r.tolist(), paste=paste, key=key, mu=float(mu[i]), sigma=float(sigma[i]),
                                ucb=ucb, log_prior=float(log_prior[i]),
                                standardized_ucb=zucb, _terms=terms[i].cpu().tolist(),
                                log_target=float(log_prior[i]) + self.cfg["beta"] * zucb))
        return records

    @torch.no_grad()
    def refine(self, records, prior, ensemble, seen=()):
        """Categorical ascent: cheap neighbor proposal, exact full-density acceptance.

        A surrogate/conditional score proposes one of four alternatives. The
        unchanged density prefix is cached; the changed suffix is rescored exactly.
        This is a discrete adaptation, not the paper's continuous gradient step.
        """
        current = list(records); excluded = set(seen)
        for iteration in range(self.cfg["local_steps"]):
            columns = np.random.randint(L.COLS, size=len(current))
            improved = 0
            for c in sorted(set(columns)):
                indices = np.flatnonzero(columns == c)
                pos = prior.order.index(int(c))
                for offset in range(0,len(indices),self.cfg["sample_batch"]):
                    owners = indices[offset:offset+self.cfg["sample_batch"]]
                    row = torch.tensor([current[i]["x"] for i in owners], dtype=torch.long, device=L.DEV)
                    prefix = torch.zeros_like(row)
                    prefix[:,list(prior.order[:pos])] = row[:,list(prior.order[:pos])]
                    t = torch.full((len(row),),1-pos/L.COLS,device=L.DEV)
                    lp = prior.model(prefix,t,int(c))
                    # Include the incumbent, then sample four legal-prefix alternatives.
                    alternatives = torch.multinomial(lp.exp(),self.cfg["local_neighbors"],replacement=True)
                    alternatives = torch.cat([row[:,int(c):int(c)+1],alternatives],1)
                    k = alternatives.shape[1]
                    variants = row[:,None,:].repeat(1,k,1)
                    variants[:,:,int(c)] = alternatives
                    flat = variants.flatten(0,1)
                    mu, sigma = ensemble.standardized(flat)
                    cheap = lp.gather(1,alternatives) + self.cfg["beta"]*(mu+self.cfg["gamma"]*sigma).reshape(len(row),k)
                    best = cheap.argmax(1)
                    candidate = variants[torch.arange(len(row),device=L.DEV),best]
                    changed = (candidate!=row).any(1)
                    good = [i for i in changed.nonzero().flatten().cpu().tolist()
                            if self.render(candidate[i].cpu().numpy())[1]
                            and self.render(candidate[i].cpu().numpy())[2] not in excluded]
                    if not good:
                        continue
                    chosen = candidate[good]
                    old_terms = torch.tensor([current[owners[i]]["_terms"] for i in good],device=L.DEV)
                    terms = prior.log_prob_terms(chosen,start=pos,previous=old_terms)
                    candidates = self.records(chosen,prior,ensemble,terms=terms)
                    by_x = {tuple(r["x"]):r for r in candidates}
                    for j,i in enumerate(good):
                        r = by_x.get(tuple(chosen[j].cpu().tolist()))
                        owner = owners[i]
                        if r is not None and r["log_target"]>current[owner]["log_target"]:
                            current[owner] = r; improved += 1
            print(f"local refinement {iteration+1}/{self.cfg['local_steps']}: {improved}/{len(current)} improved",flush=True)
        return current

    def propose(self, prior, posterior, ensemble, generation, seen):
        torch.manual_seed(self.cfg["seed"] + generation * 1000 + 500)
        np.random.seed(self.cfg["seed"] + generation * 1000 + 500)
        records, keys, tried = [], set(seen), 0
        raw_accepted = []
        limit = self.cfg["propose"] * self.cfg["attempt_factor"]
        while len(records) < self.cfg["propose"] and tried < limit:
            n = min(self.cfg["sample_batch"], limit - tried)
            x = posterior.sample(n)
            tried += n
            for record in self.records(x, prior, ensemble):
                if len(raw_accepted) < 512:
                    raw_accepted.append({k:v for k,v in record.items() if k!='_terms'})
                if record["key"] not in keys:
                    keys.add(record["key"])
                    records.append(record)
                    if len(records) == self.cfg["propose"]:
                        break
            if len(records) and (tried % 512 == 0 or len(records) == self.cfg['propose']):
                print(f'round {generation}: {len(records)}/{self.cfg["propose"]} distinct proposals from {tried} draws',flush=True)
        if len(records) < self.cfg["battle"]:
            raise RuntimeError(f"only {len(records)} new legal proposals after {tried} draws; need {self.cfg['battle']}")
        records = self.refine(records, prior, ensemble, seen)
        records = [{k:v for k,v in r.items() if k!='_terms'} for r in records]
        selected = DC.unique_top(records, self.cfg["battle"], seen)
        if len(selected) != self.cfg["battle"]:
            raise RuntimeError("local refinement collapsed below the distinct evaluation batch size")
        return dict(proposals=records, selected=selected, tried=tried, raw_accepted_sample=raw_accepted)


def observations(out, generation, buffer):
    rows = list(out["initial"])
    for g, result in sorted(out["rounds"].items(), key=lambda kv: int(kv[0])):
        if int(g) >= generation:
            continue
        rows.extend(result.get("evaluated", []))
    # Moving top-L buffer is DiBO's bounded dataset, not an elite fraction.
    return sorted(rows, key=lambda r: -r["y"])[:buffer]


def load_out(path, signature=None):
    out = json.loads(Path(path).read_text())
    if signature is not None and out["signature"] != json.loads(json.dumps(signature)):
        raise ValueError("resume configuration, foundation, vocabulary or meta schedule changed")
    return out


def run(args):
    cfg = configuration(args)
    runtime = Runtime(cfg, args.output)
    try:
        if args.output.exists():
            out = load_out(args.output, runtime.signature)
        else:
            out = dict(signature=runtime.signature, initial=runtime.initial(), rounds={})
            atomic_json(args.output, out)
        for g in range(cfg["rounds"]):
            result = out["rounds"].setdefault(str(g), {})
            if "evaluated" in result:
                continue
            prior, posterior, ensemble, training = runtime.models(observations(out, g, cfg["buffer"]), g)
            result["training"] = training
            if "selected" not in result:
                seen = {r["key"] for r in out["initial"]}
                seen.update(r["key"] for v in out["rounds"].values() for r in v.get("evaluated", []))
                result.update(runtime.propose(prior, posterior, ensemble, g, seen))
                atomic_json(args.output, out)
            scores = MDB.score_pastes(runtime.con, [r["paste"] for r in result["selected"]],
                                      per_cell=cfg["per_cell"], set_name="top50", seed=cfg["seed"] + g,
                                      origin=f"dibo:{args.output.stem}:round{g}", policy_id=LB.POLICY)
            if any(s.get("score") is None or s.get("coverage") != s.get("columns") for s in scores):
                raise RuntimeError("objective evaluation incomplete; saved proposals can be resumed")
            result["evaluated"] = [dict(paste=r["paste"], x=r["x"], key=r["key"], y=float(s["score"]),
                                        team_id=s["team_id"], src=f"round{g}", battles=s["battles"])
                                   for r, s in zip(result["selected"], scores)]
            result["mean"] = float(np.mean([r["y"] for r in result["evaluated"]]))
            result["best"] = max(r["y"] for r in result["evaluated"])
            atomic_json(args.output, out)
            print(f"round {g}: {len(result['evaluated'])} legal distinct teams evaluated; mean={result['mean']:.3f} best={result['best']:.3f}", flush=True)
        out["search_complete"] = True
        atomic_json(args.output, out)
    finally:
        runtime.close()


def final(args):
    """Select by search labels once, then fresh top50 confirmation and heldout evaluation."""
    import pool
    out = load_out(args.output)
    if not out.get("search_complete"):
        raise ValueError("finish the search before final selection")
    cfg = out["signature"]["config"]
    if "final_selected" not in out:
        rows = [dict(r, log_target=r["y"]) for v in out["rounds"].values() for r in v["evaluated"]]
        out["final_selected"] = DC.unique_top(rows, cfg["final_n"])
        atomic_json(args.output, out)
    LB._stack()
    directory = Path("/tmp/vgc-pilot") / (args.output.stem + "_final")
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    for i, row in enumerate(out["final_selected"]):
        path = directory / f"{i:02d}.txt"
        path.write_text(MDB.canonical_paste(row["paste"]))
        files.append(str(path))
    if not files:
        raise ValueError("no evaluated candidates for confirmation")
    if "confirmation" not in out:
        result = pool.score([(f, FD.A.opponents()) for f in files], battles=cfg["final_battles"], conc=50)
        if any(result.get(f, {}).get("win_rate") is None or
               result.get(f, {}).get("battles") != cfg["final_battles"] for f in files):
            raise RuntimeError("fresh confirmation is incomplete")
        out["confirmation"] = [result[f] for f in files]
        atomic_json(args.output, out)
    if "heldout" not in out:
        result = HO.evaluate(files, battles=cfg["final_battles"], conc=50)
        if any(result.get(f, {}).get("win_rate") is None or
               result.get(f, {}).get("battles") != cfg["final_battles"] for f in files):
            raise RuntimeError("heldout evaluation is incomplete")
        out["heldout"] = [result[f] for f in files]
        atomic_json(args.output, out)
    report(args)


def report(args):
    out = load_out(args.output)
    for g, row in sorted(out["rounds"].items(), key=lambda kv: int(kv[0])):
        print(f"round {g}: mean={row.get('mean')} best={row.get('best')} evaluated={len(row.get('evaluated', []))}")
    print(f"search complete: {out.get('search_complete', False)}; confirmation: {len(out.get('confirmation', []))}; heldout: {len(out.get('heldout', []))}")


def configuration(args):
    cfg = {k: getattr(args, k) for k in ("rounds", "propose", "battle", "buffer", "initial_limit", "ensemble",
           "width", "hidden", "prior_epochs", "predictor_epochs", "rtb_steps", "batch", "rtb_batch", "rtb_microbatch",
           "sample_batch", "local_steps", "local_neighbors", "attempt_factor", "per_cell", "seed", "threads",
           "final_n", "final_battles", "lr", "predictor_lr", "rtb_lr", "beta", "gamma")}
    if args.smoke:
        cfg.update(rounds=2, propose=8, battle=2, initial_limit=16, prior_epochs=1, predictor_epochs=2,
                   rtb_steps=2, batch=8, rtb_batch=2, rtb_microbatch=2, sample_batch=4, local_steps=1, local_neighbors=2,
                   final_n=2, final_battles=8)
    if cfg["propose"] < cfg["battle"] or cfg["ensemble"] < 2 or cfg["buffer"] < 2 or cfg["initial_limit"] < 2:
        raise ValueError("pool must cover the battle batch; buffer, initial labels and ensemble must be >=2")
    for key, value in cfg.items():
        if not math.isfinite(value) or value < 0 or (value == 0 and key not in ("beta", "gamma", "local_steps", "seed")):
            raise ValueError(f"invalid {key}: {value}")
    if cfg["rtb_batch"] < 2:
        raise ValueError("RTB batch must have space for on-policy and replay samples")
    return cfg


def main(argv=None):
    parser = argparse.ArgumentParser(description="Discrete DiBO posterior inference for Pokémon team search")
    parser.add_argument("stage", choices=("run", "final", "report"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    integers = dict(rounds=10, propose=12800, battle=128, buffer=1000, initial_limit=5000, ensemble=5,
                    width=32, hidden=256, prior_epochs=50, predictor_epochs=50, rtb_steps=50,
                    batch=64, rtb_batch=256, rtb_microbatch=128, sample_batch=128, local_steps=10, local_neighbors=4,
                    attempt_factor=12, per_cell=1, seed=5050, threads=4, final_n=16, final_battles=192)
    floats = dict(lr=1e-4, predictor_lr=1e-3, rtb_lr=1e-4, beta=30.0, gamma=1.0)
    for key, default in integers.items():
        parser.add_argument("--" + key.replace("_", "-"), type=int, default=default)
    for key, default in floats.items():
        parser.add_argument("--" + key.replace("_", "-"), type=float, default=default)
    args = parser.parse_args(argv)
    args.output = args.output or HERE.parent / "results" / ("dibo_v4_smoke.json" if args.smoke else "dibo_v4.json")
    {"run": run, "final": final, "report": report}[args.stage](args)


if __name__ == "__main__":
    main()
