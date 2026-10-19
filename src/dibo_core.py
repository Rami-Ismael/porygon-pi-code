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
    return torch.softmax(y, 0)


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
        self.mlp = nn.Sequential(nn.Linear(width * len(field_keys), hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        x = x.reshape(len(x), self.slots, len(self.field_keys))
        features = torch.cat([self.emb[k](x[:, :, j]) for j, k in enumerate(self.field_keys)], -1)
        return self.mlp(features.mean(1)).squeeze(-1)


class PredictorEnsemble(nn.Module):
    def __init__(self, members):
        super().__init__()
        if len(members) < 2:
            raise ValueError("uncertainty requires at least two independently trained predictors")
        self.members = nn.ModuleList(members)

    def forward(self, x):
        predictions = torch.stack([net(x) for net in self.members])
        return predictions.mean(0), predictions.std(0, unbiased=False)

    def fit(self, x, y, epochs, batch, lr, seed, progress=None):
        weights = score_weights(y)
        logs = []
        for i, net in enumerate(self.members):
            torch.manual_seed(seed + i)
            # Independent bootstrap produces disagreement even with small labelled pools.
            indices = torch.randint(len(x), (len(x),), device=x.device)
            opt = torch.optim.AdamW(net.parameters(), lr=lr)
            net.train()
            for _ in range(epochs):
                for ix in indices[torch.randperm(len(indices), device=x.device)].split(batch):
                    loss = ((net(x[ix]) - y[ix]).square() * weights[ix] * len(x)).mean()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
            net.eval()
            with torch.no_grad():
                logs.append(float((net(x) - y).square().mean()))
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


def train_rtb(prior, posterior, log_tilt, steps, batch, lr, replay=None, progress=None):
    """Equation 14. Draws are detached; gradient is 2 residual ∇ log pψ(trajectory).

    Recompute and backprop each transition separately to bound memory. The frozen
    prior and posterior MUST use identical support, normalization and path order.
    """
    if prior.order != posterior.order:
        raise ValueError("prior and posterior must use the same trajectory law")
    prior.model.eval()
    posterior.model.eval()  # eval disables dropout, but retains parameter gradients
    with torch.no_grad():
        initial_tilt = log_tilt(prior.sample(batch))
        if not torch.isfinite(initial_tilt).all():
            raise ValueError("nonfinite reward while initializing the partition estimate")
        # Prior expectation of exp(log_tilt) estimates the partition function.
        # It is only an initialization; RTB continues to learn the normalizer.
        initial_z = torch.logsumexp(initial_tilt, 0) - math.log(batch)
    log_z = nn.Parameter(initial_z.detach().clone())
    opt = torch.optim.AdamW([{"params": posterior.model.parameters(), "lr": lr},
                            {"params": [log_z], "lr": max(lr, .01)}], weight_decay=0)
    history = []
    for step in range(steps):
        with torch.no_grad():
            terminal = posterior.sample(batch)
            if replay is not None and len(replay):
                n = batch // 2
                terminal[:n] = replay[torch.randint(len(replay), (n,), device=replay.device)]
            target = prior.log_prob(terminal) + log_tilt(terminal)
            residual = log_z.detach() + posterior.log_prob(terminal) - target
            if not torch.isfinite(residual).all():
                raise ValueError("nonfinite RTB residual; incompatible support or invalid reward")
            coefficient = 2 * residual / len(terminal)
            loss = residual.square().mean()
        opt.zero_grad(set_to_none=True)
        for lp in posterior.transitions(terminal):
            (coefficient * lp).sum().backward()
        (coefficient.sum() * log_z).backward()
        nn.utils.clip_grad_norm_(posterior.model.parameters(), 1)
        opt.step()
        history.append(dict(step=step, loss=float(loss), log_z=float(log_z.detach())))
        if progress and ((step + 1) % 10 == 0 or step + 1 == steps):
            progress(f"RTB update {step + 1}/{steps}: loss={float(loss):.4f} logZ={float(log_z.detach()):.4f}")
    return history, float(log_z.detach())


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
