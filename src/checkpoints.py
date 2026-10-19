"""One place to load PyTorch checkpoints.

torch.load is banned elsewhere by the Ruff rule TID251 (see pyproject.toml at
the repository root). weights_only=True refuses arbitrary pickled objects; it is
already torch's default on 2.12 and 2.14, so this makes it explicit.
Pass trusted_objects=True only for a file you wrote that stores more than
tensors and plain Python containers (numpy arrays, custom classes).
"""
import torch


def load_checkpoint(path, *, trusted_objects=False, **kwargs):
    return torch.load(path, weights_only=not trusted_objects, **kwargs)  # noqa: TID251
