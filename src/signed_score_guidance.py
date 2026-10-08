"""Reverse the frozen gloss guidance contribution without changing its gain."""

import math


class SignedGuide:
    def __init__(self, guide, direction):
        if direction not in (-1, 1):
            raise ValueError("direction must be -1 or +1")
        self.guide = guide
        self.direction = direction

    def tilt(self, *args, **kwargs):
        return self.direction * self.guide.tilt(*args, **kwargs)

    def gap(self, *args, **kwargs):
        # Keep the original quality-dependent gain, including its zero clamp.
        return self.guide.gap(*args, **kwargs)


def sample_direction(sampler, model, constraints, guide, n, strength):
    if not math.isfinite(strength):
        raise ValueError("strength must be finite")
    mode = "none" if strength == 0 else "gloss"
    signed = SignedGuide(guide, -1 if strength < 0 else 1)
    # The frozen sampler's lam > 0 gate requires a positive magnitude.
    return sampler(model, constraints, signed, n, mode, abs(strength))
