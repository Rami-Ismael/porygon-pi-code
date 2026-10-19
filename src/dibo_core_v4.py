"""Discrete DiBO mechanics, independent of the Pokémon runtime.

A fixed unmasking order gives each ordered terminal grid exactly one trajectory.
Consequently the product of normalized transition probabilities is its exact
density (not an ELBO or a sum of bidirectional masked-token scores).
"""
import copy
import math
import torch
from torch import nn


def score_weights(y):
    if y.ndim != 1 or not len(y) or not torch.isfinite(y).all():
        raise ValueError("scores must be a nonempty finite vector")
    return torch.softmax(standardize(y)[0], 0)


def standardize(y):
    mean = y.mean()
    scale = y.std(unbiased=True).clamp(min=1e-7) if len(y) > 1 else y.new_tensor(1e-7)
    return (y-mean)/scale, mean, scale


class PathSampler:
    """conditional(prefix, time, column) must return normalized log probabilities."""
    def __init__(self, model, order, device="cpu"):
        self.model, self.order, self.device = model, tuple(order), torch.device(device)
        if sorted(self.order) != list(range(len(self.order))):
            raise ValueError("order must reveal every column exactly once")

    def transitions(self, terminal):
        x = torch.zeros_like(terminal)
        for step, c in enumerate(self.order):
            t = torch.full((len(x),), 1 - step / len(self.order), device=x.device)
            lp = self.model(x, t, c)
            yield lp.gather(1, terminal[:, c:c + 1]).squeeze(1)
            # Fresh storage: autograd may retain the prefix for embedding gradients.
            x = x.clone()
            x[:, c] = terminal[:, c]

    def log_prob(self, terminal):
        if terminal.ndim != 2 or terminal.shape[1] != len(self.order) or (terminal <= 0).any():
            raise ValueError("terminal grids must contain a non-mask token in every column")
        return torch.stack(list(self.transitions(terminal))).sum(0)

    def log_prob_terms(self, terminal, start=0, previous=None):
        """Exact suffix rescoring when only a coordinate at/after start changed."""
        if start and previous is None:
            raise ValueError('suffix rescoring needs unchanged prefix terms')
        x = torch.zeros_like(terminal)
        x[:,list(self.order[:start])] = terminal[:,list(self.order[:start])]
        terms = [] if not start else list(previous[:,:start].unbind(1))
        for step in range(start,len(self.order)):
            c = self.order[step]
            t = torch.full((len(x),),1-step/len(self.order),device=x.device)
            lp = self.model(x,t,c)
            terms.append(lp.gather(1,terminal[:,c:c+1]).squeeze(1))
            x = x.clone(); x[:,c] = terminal[:,c]
        return torch.stack(terms,1)

    @torch.no_grad()
    def sample(self, n):
        if n <= 0:
            raise ValueError("sample count must be positive")
        self.model.eval()
        x = torch.zeros(n, len(self.order), dtype=torch.long, device=self.device)
        for step, c in enumerate(self.order):
            t = torch.full((n,), 1 - step / len(self.order), device=self.device)
            p = self.model(x, t, c).exp()
            if not torch.isfinite(p).all() or (p.sum(1) <= 0).any():
                raise ValueError("invalid transition distribution")
            x[:, c] = torch.multinomial(p, 1).squeeze(1)
        return x


class Predictor(nn.Module):
    """Permutation-invariant slot embeddings include every categorical field."""
    def __init__(self, sizes, field_keys, slots=6, width=32, hidden=128):
        super().__init__()
        self.field_keys, self.slots = tuple(field_keys), slots
        self.emb = nn.ModuleDict({k: nn.Embedding(v, width) for k, v in sizes.items()})
        self.slot = nn.Sequential(nn.Linear(width * 5, hidden), nn.GELU(),
                                  nn.Linear(hidden, hidden), nn.GELU())
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x):
        x = x.reshape(len(x), self.slots, len(self.field_keys))
        if self.field_keys != ('species','ability','item','move','move','move','move','nature'):
            raise ValueError('predictor expects the six-Pokemon categorical representation')
        features = torch.cat([self.emb['species'](x[:,:,0]), self.emb['ability'](x[:,:,1]),
                              self.emb['item'](x[:,:,2]), self.emb['move'](x[:,:,3:7]).mean(2),
                              self.emb['nature'](x[:,:,7])], -1)
        # Fuse fields within each Pokemon BEFORE pooling the team.
        return self.mlp(self.slot(features).mean(1)).squeeze(-1)


class PredictorEnsemble(nn.Module):
    def __init__(self, members):
        super().__init__()
        if len(members) < 2:
            raise ValueError("uncertainty requires at least two independently trained predictors")
        self.members = nn.ModuleList(members)
        self.register_buffer('y_mean', torch.tensor(0.))
        self.register_buffer('y_scale', torch.tensor(1.))

    def forward(self, x):
        mu, sigma = self.standardized(x)
        return self.y_mean + self.y_scale * mu, self.y_scale * sigma

    def standardized(self, x):
        predictions = torch.stack([net(x) for net in self.members])
        return predictions.mean(0), predictions.std(0, unbiased=False)

    def fit(self, x, y, epochs, batch, lr, seed, progress=None):
        weights = score_weights(y)
        target, mean, scale = standardize(y)
        self.y_mean.copy_(mean); self.y_scale.copy_(scale)
        logs = []
        for i, net in enumerate(self.members):
            torch.manual_seed(seed + i)
            # Independent bootstrap produces disagreement even with small labelled pools.
            indices = torch.randint(len(x), (len(x),), device=x.device)
            opt = torch.optim.AdamW(net.parameters(), lr=lr)
            net.train()
            for _ in range(epochs):
                for ix in indices[torch.randperm(len(indices), device=x.device)].split(batch):
                    loss = ((net(x[ix]) - target[ix]).square() * weights[ix] * len(x)).mean()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
            net.eval()
            with torch.no_grad():
                logs.append(float((self.y_mean+self.y_scale*net(x) - y).square().mean()))
            if progress:
                progress(f"predictor {i + 1}/{len(self.members)} finished: MSE={logs[-1]:.5f}")
        return logs


def train_prior(prior, x, y, epochs, batch, lr, progress=None):
    """Score-weighted masked denoising. All observed rows participate, no elite cutoff."""
    weights = score_weights(y)
    opt = torch.optim.AdamW(prior.parameters(), lr=lr)
    losses = []
    prior.train()
    for epoch in range(epochs):
        for ix in torch.randperm(len(x), device=x.device).split(batch):
            xb = x[ix]
            t = torch.rand(len(ix), device=x.device).clamp(min=1e-3)
            mask = torch.rand(xb.shape, device=x.device) < t[:, None]
            empty = ~mask.any(1)
            mask[empty, torch.randint(xb.shape[1], (int(empty.sum()),), device=x.device)] = True
            per_row = prior.denoising_loss(xb, mask, t)
            loss = (per_row * weights[ix] * len(x)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(prior.parameters(), 1)
            opt.step()
            losses.append(float(loss.detach()))
        if progress and ((epoch + 1) % 10 == 0 or epoch + 1 == epochs):
            progress(f"weighted prior epoch {epoch + 1}/{epochs}: loss={losses[-1]:.4f}")
    prior.eval()
    return losses


def replay_probabilities(log_tilts, temperature=1.0):
    """Prioritize standardized UCB, rather than beta-scaled UCB."""
    return torch.softmax(log_tilts / temperature, 0)


def train_rtb(prior, posterior, log_tilt, steps, batch, lr, replay=None, progress=None,
              temperature=1.0, microbatch=32):
    """Scaled RTB with UCB-prioritized growing replay and detached trajectories."""
    if prior.order != posterior.order:
        raise ValueError("prior and posterior must use the same trajectory law")
    prior.model.eval(); posterior.model.eval()
    alpha = 1.0 / temperature
    with torch.no_grad():
        initial_tilt = log_tilt(prior.sample(batch))
        initial_z = torch.logsumexp(initial_tilt, 0) - math.log(batch)
        replay_x = replay.clone() if replay is not None else None
        replay_rewards = torch.cat([log_tilt(b) for b in replay_x.split(microbatch)]) if replay_x is not None else None
        validation = replay_x[:min(32, len(replay_x))].clone() if replay_x is not None else prior.sample(32)
        validation_target = prior.log_prob(validation) + log_tilt(validation)
    z = nn.Parameter((alpha * initial_z).detach().clone())
    opt = torch.optim.Adam([{"params": posterior.model.parameters(), "lr": lr},
                            {"params": [z], "lr": max(lr, .01)}])
    history = []
    for step in range(steps):
        off_policy = replay_x is not None and bool(torch.rand(()) < .5)
        with torch.no_grad():
            if off_policy:
                ix = torch.multinomial(replay_probabilities(replay_rewards, temperature), batch, replacement=True)
                terminal = replay_x[ix]
            else:
                terminal = posterior.sample(batch)
            rewards = torch.cat([log_tilt(b) for b in terminal.split(microbatch)])
            residuals = []
            for offset in range(0, batch, microbatch):
                xb = terminal[offset:offset+microbatch]
                target = prior.log_prob(xb) + rewards[offset:offset+microbatch]
                residuals.append(z.detach() + alpha*(posterior.log_prob(xb)-target))
            residual = torch.cat(residuals)
            if not torch.isfinite(residual).all():
                raise ValueError("nonfinite RTB residual")
            replay_x = terminal.clone() if replay_x is None else torch.cat([replay_x, terminal])
            replay_rewards = rewards.clone() if replay_rewards is None else torch.cat([replay_rewards, rewards])
        opt.zero_grad(set_to_none=True)
        for offset in range(0, batch, microbatch):
            xb = terminal[offset:offset+microbatch]
            coefficient = 2*alpha*residual[offset:offset+microbatch]/batch
            for lp in posterior.transitions(xb):
                (coefficient*lp).sum().backward()
        (2*residual.mean()*z).backward()
        nn.utils.clip_grad_norm_(posterior.model.parameters(), 1)
        opt.step()
        row = dict(step=step, loss=float(residual.square().mean()), log_z=float(z.detach()/alpha),
                   off_policy=off_policy, replay_size=len(replay_x))
        if (step+1)%10==0 or step+1==steps:
            with torch.no_grad():
                vr = z.detach()+alpha*(posterior.log_prob(validation)-validation_target)
                row["fixed_validation_loss"] = float(vr.square().mean())
            if progress:
                progress(f"RTB update {step+1}/{steps}: scaled_loss={row['loss']:.4f} fixed_validation={row['fixed_validation_loss']:.4f}")
        history.append(row)
    return history, float(z.detach()/alpha)


def posterior_copy(prior):
    return PathSampler(copy.deepcopy(prior.model), prior.order, prior.device)


def unique_top(records, budget, seen=()):
    """Rank by full log prior + β UCB, never just μ or a top-quantile label cutoff."""
    if budget < 1:
        raise ValueError("selection budget must be positive")
    selected, used = [], set(seen)
    for row in sorted(records, key=lambda r: -r["log_target"]):
        if row["key"] not in used:
            used.add(row["key"])
            selected.append(row)
            if len(selected) == budget:
                break
    return selected
