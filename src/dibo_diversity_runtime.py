"""Opt-in DiBO v4 runtime; frozen v4 sources and current runs remain untouched."""
from pathlib import Path
import math
import numpy as np
import torch
from torch import nn
import dibo_v4 as DB
import dibo_core_v4 as DC
import dibo_diversity as DV
from boltzmann_selection import linear_temperature
from checkpoints import load_checkpoint


class ProposalMask(nn.Module):
    """Proposal-only restriction. Density is always scored by the original prior."""
    def __init__(self, model, species_columns, banned=(), pins=None):
        super().__init__()
        self.model = model
        self.species_columns = set(species_columns)
        self.banned = tuple(banned)
        self.pins = pins or {}

    def forward(self, x, t, c):
        lp = self.model(x, t, c).clone()
        if c in self.species_columns and self.banned:
            lp[:, list(self.banned)] = -torch.inf
        if c in self.pins:
            allowed = lp[:, self.pins[c]].clone()
            lp[:] = -torch.inf
            lp[:, self.pins[c]] = allowed
        if not torch.isfinite(lp).any(1).all():
            raise ValueError('diversity constraint produced a dead prefix; constraints were not relaxed')
        return torch.log_softmax(lp, -1)


class RewardView(nn.Module):
    def __init__(self, ensemble, arm):
        super().__init__()
        self.ensemble, self.arm = ensemble, arm

    def forward(self, x):
        return self.ensemble(x)

    def standardized(self, x):
        mu, sigma = self.ensemble.standardized(x)
        if self.arm in ('no_guidance', 'random'):
            return torch.zeros_like(mu), torch.zeros_like(sigma)
        return (-mu if self.arm == 'negative' else mu), sigma


def train_prior(prior, x, weights, cfg, entropy_bonus=0):
    weights = torch.as_tensor(weights, dtype=torch.float32, device=x.device)
    opt = torch.optim.AdamW(prior.parameters(), lr=cfg['lr'])
    losses = []
    prior.train()
    for epoch in range(cfg['prior_epochs']):
        for ix in torch.randperm(len(x), device=x.device).split(cfg['batch']):
            xb = x[ix]
            t = torch.rand(len(ix), device=x.device).clamp(min=1e-3)
            mask = torch.rand(xb.shape, device=x.device) < t[:, None]
            empty = ~mask.any(1)
            mask[empty, torch.randint(xb.shape[1], (int(empty.sum()),), device=x.device)] = True
            if entropy_bonus:
                distributions = prior.distributions(xb.masked_fill(mask, 0), t)
                per_field = torch.zeros(xb.shape, device=x.device)
                for key, cols in DB.L.COLS_BY_KEY.items():
                    lp = distributions[key]
                    ce = -lp.gather(2, xb[:, cols, None]).squeeze(-1)
                    entropy = -(lp.exp() * lp).sum(-1)
                    per_field[:, cols] = ce - entropy_bonus * entropy
                per_row = (per_field * mask / t[:, None]).sum(1) / xb.shape[1]
            else:
                per_row = prior.denoising_loss(xb, mask, t)
            loss = (per_row * weights[ix] * len(x)).mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(prior.parameters(), 1); opt.step()
            losses.append(float(loss.detach()))
        if (epoch + 1) % 10 == 0 or epoch + 1 == cfg['prior_epochs']:
            print(f'diversity prior epoch {epoch + 1}: {losses[-1]:.4f}', flush=True)
    prior.eval()
    return losses


class Runtime(DB.Runtime):
    def __init__(self, cfg, outpath):
        super().__init__(cfg, outpath)
        self.arm = cfg['arm']
        self.signature['format'] = 'discrete-dibo-v4-diversity-v1'
        paths = [Path(__file__), Path(DV.__file__), DB.HERE / 'dibo_diversity_experiment.py',
                 DB.HERE / 'boltzmann_selection.py', DB.HERE / 'temperature_experiment.py',
                 DB.HERE / 'dibo_comparison_v2.py', Path(DB.MDB.__file__)]
        self.signature['implementation'].update({p.name: self.file_hash(p) for p in paths})
        self.banned, self.pins, self.previous = [], {}, set()

    def models(self, rows, generation):
        # Same initialization, losses and RNG sequence as v4 for the control.
        seed = self.cfg['seed'] + generation * 1000
        torch.manual_seed(seed)
        prior = DC.posterior_copy(DC.PathSampler(self.base, DB.D.ORDER, DB.L.DEV))
        ensemble = self.ensemble(seed)
        ck = self.checkdir / f'round{generation}.pt'
        if ck.exists():
            saved = load_checkpoint(ck, map_location=DB.L.DEV)
            if saved['signature'] != self.signature or saved['rows'] != rows:
                raise ValueError('diversity checkpoint inputs changed')
            prior.model.load_state_dict(saved['prior'])
            posterior = DC.posterior_copy(prior)
            posterior.model.load_state_dict(saved['posterior'])
            ensemble.load_state_dict(saved['ensemble'])
            return prior, posterior, RewardView(ensemble.eval(), self.arm), saved['training']
        if len(rows) < 2:
            raise ValueError('diversity constraint left fewer than two training observations')
        x = torch.tensor([r['x'] for r in rows], dtype=torch.long, device=DB.L.DEV)
        y = torch.tensor([r['y'] for r in rows], dtype=torch.float32, device=DB.L.DEV)
        weights, info = DV.prior_weights(rows, self.arm, generation, self.cfg['rounds'])
        bonus = linear_temperature(generation + 1, self.cfg['rounds']) if self.arm == 'refit' else 0
        weighted = self.arm in ('boltz', 'fixed_010', 'fixed_020', 'fixed_030', 'adaptive', 'niche', 'refit', 'negative', 'random')
        loss = (train_prior(prior.model, x, weights, self.cfg, bonus) if weighted else
                DC.train_prior(prior.model, x, y, self.cfg['prior_epochs'], self.cfg['batch'], self.cfg['lr']))
        pred = ensemble.fit(x, y, self.cfg['predictor_epochs'], self.cfg['batch'], self.cfg['predictor_lr'], seed)
        view = RewardView(ensemble.eval(), self.arm)
        posterior = DC.posterior_copy(prior)
        def tilt(terminal):
            mu, sigma = view.standardized(terminal)
            value = self.cfg['beta'] * (mu + self.cfg['gamma'] * sigma)
            legal = torch.tensor([self.render(r)[1] for r in terminal.cpu().numpy()], device=DB.L.DEV)
            return torch.where(legal, value, torch.full_like(value, -40 * self.cfg['beta']))
        history, log_z = DC.train_rtb(prior, posterior, tilt, self.cfg['rtb_steps'], self.cfg['rtb_batch'],
                                     self.cfg['rtb_lr'], replay=x, temperature=self.cfg['beta'],
                                     microbatch=self.cfg['rtb_microbatch'])
        training = dict(prior_loss=loss, predictor_mse=pred, rtb=history, log_z=log_z,
                        weighting=info, entropy_bonus=bonus, observation_count=len(rows))
        DB.atomic_checkpoint(ck, dict(signature=self.signature, rows=rows, prior=prior.model.state_dict(),
                                     posterior=posterior.model.state_dict(), ensemble=ensemble.state_dict(), training=training))
        return prior, posterior, view, training

    def allowed(self, row):
        if set(DV.composition(row)) & set(self.banned):
            return False
        return all(row['x'][c] == token for c, token in self.pins.items())

    def records(self, terminal, prior, ensemble, terms=None):
        # Also protects local refinement: it cannot reintroduce peeled species
        # or change an island's pinned columns.
        return [r for r in super().records(terminal, prior, ensemble, terms) if self.allowed(r)]

    def propose(self, prior, posterior, ensemble, generation, seen):
        seed = self.cfg['seed'] + generation * 1000 + 500
        torch.manual_seed(seed); np.random.seed(seed)
        count = self.cfg['propose'] * (self.cfg['pool_multiplier'] if self.arm in ('big', 'dpp') else 1)
        banned_tokens = [self.V.stoi['species'][s] for s in self.banned if s in self.V.stoi['species']]
        sampler = DC.PathSampler(ProposalMask(posterior.model, DB.FD.SPECIES_COLS, banned_tokens, self.pins),
                                 posterior.order, posterior.device) if self.banned or self.pins else posterior
        records, raw, keys, tried, dead = [], [], set(seen), 0, 0
        taboo = DV.Taboo(self.previous) if self.arm == 'taboo' else None
        limit = count * self.cfg['attempt_factor']
        while len(records) < count and tried < limit:
            n = min(self.cfg['sample_batch'], limit - tried); tried += n
            try:
                x = sampler.sample(n)
            except ValueError as e:
                if 'dead prefix' not in str(e):
                    raise
                dead += n
                continue
            for row in self.records(x, prior, ensemble):
                if len(raw) < self.cfg['metric_sample']:
                    raw.append({k: v for k, v in row.items() if k != '_terms'})
                if row['key'] in keys or (taboo and not taboo.accept(row)):
                    continue
                keys.add(row['key']); records.append(row)
                if len(records) == count:
                    break
        if len(records) != count:
            raise RuntimeError(f'{self.arm}: filled {len(records)}/{count} proposals in {tried} draws; no budget silently reduced')
        pre_refine = list(records)
        records = self.refine(records, prior, ensemble, seen)
        # Refinement can merge grids/compositions. Fall back to original legal
        # proposals to fill quotas; never relax taboo, pinning or peel constraints.
        fallback = pre_refine if self.arm in ('taboo', 'peel', 'island') else []
        candidates = DV.distinct(sorted(records + fallback, key=lambda r: -r['log_target']), seen)
        if taboo:
            gate = DV.Taboo(self.previous)
            candidates = [r for r in candidates if gate.accept(r)]
        candidates = [{k: v for k, v in r.items() if k != '_terms'} for r in candidates]
        rank = None
        if self.arm == 'dpp':
            selected, rank = DV.select_dpp(candidates, self.cfg['battle'])
        elif self.arm == 'random':
            ix = np.random.default_rng(seed).permutation(len(candidates))[:self.cfg['battle']]
            selected = [candidates[i] for i in ix]
        else:
            selected = DC.unique_top(candidates, self.cfg['battle'], seen)
        if len(selected) != self.cfg['battle']:
            raise RuntimeError('diversity/refinement could not fill the distinct battle batch')
        # History includes all accepted pre-refinement proposals as in CEM taboo.
        history = sorted({DV.composition(r) for r in pre_refine + candidates})
        return dict(proposals=candidates, selected=selected, tried=tried, dead_prefix_draws=dead,
                    raw_accepted_sample=raw, proposal_compositions=history,
                    diversity=dict(raw=DV.metrics(raw), output=DV.metrics(candidates[:self.cfg['metric_sample']]),
                                   selected=DV.metrics(selected)), dpp_rank=rank,
                    banned=self.banned, pins=self.pins)
