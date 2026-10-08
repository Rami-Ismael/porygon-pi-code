"""Boltzmann *pool selection*, independent of denoising and decoding temperature.

The adaptive arm solves for an ESS fraction, not a learned network parameter.
All temperatures stay positive: measured battle scores have sampling noise.
"""
import numpy as np

ARMS = ("annealed", "fixed_010", "fixed_020", "fixed_030", "adaptive")


def boltz_weights(y, T):
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all():
        raise ValueError("scores must be a nonempty finite one-dimensional array")
    if not np.isfinite(T) or T <= 0:
        raise ValueError("selection temperature must be finite and positive")
    with np.errstate(over="ignore", under="ignore"):
        w = np.exp((y - y.max()) / T)
    return w / w.sum()


def concentration(w):
    w = np.asarray(w, dtype=float)
    p = w[w > 0]
    ess = float(1 / np.dot(w, w))
    return dict(ess=ess, ess_fraction=ess / len(w), max_weight=float(w.max()),
                entropy=float(-np.dot(p, np.log(p))))


def linear_temperature(generation, generations=11, start=.30, end=.10):
    if generation < 1 or generations < 1:
        raise ValueError("generation counts must be positive")
    if not np.isfinite([start, end]).all() or min(start, end) <= 0:
        raise ValueError("schedule endpoints must be finite and positive")
    # One-generation smoke runs use the starting temperature.
    fraction = min(generation - 1, generations - 1) / max(generations - 1, 1)
    return float(start + (end - start) * fraction)


def adaptive_temperature(y, target=.50, lower=.10, upper=.30):
    """Choose T in [lower, upper] for ESS/n=target via monotone bisection.

    ESS increases with T. Flat/tied scores may make the target unreachable;
    return the closest boundary and explicitly report that condition.
    """
    if not 0 < target <= 1 or not 0 < lower <= upper:
        raise ValueError("require 0 < target <= 1 and 0 < lower <= upper")
    def stats(t):
        return concentration(boltz_weights(y, t))
    lo_stats, hi_stats = stats(lower), stats(upper)
    if lo_stats["ess_fraction"] >= target:
        temp, status = lower, "lower_bound"
    elif hi_stats["ess_fraction"] <= target:
        temp, status = upper, "upper_bound"
    else:
        lo, hi = lower, upper
        for _ in range(60):
            mid = (lo + hi) / 2
            if stats(mid)["ess_fraction"] < target:
                lo = mid
            else:
                hi = mid
        temp, status = (lo + hi) / 2, "target"
    return float(temp), dict(target_ess_fraction=target, status=status, **stats(temp))


def selection_temperature(arm, generation, y, generations=11, target=.50):
    if arm not in ARMS:
        raise ValueError(f"unknown temperature arm: {arm}")
    if arm == "adaptive":
        return adaptive_temperature(y, target)
    temp = (linear_temperature(generation, generations) if arm == "annealed"
            else {"fixed_010": .10, "fixed_020": .20, "fixed_030": .30}[arm])
    return temp, dict(status="scheduled" if arm == "annealed" else "fixed",
                      **concentration(boltz_weights(y, temp)))


def select_boltz(lab_y, k, T, rng):
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("selection draw count must be a positive integer")
    w = boltz_weights(lab_y, T)
    return rng.choice(len(w), size=k, replace=True, p=w), w
