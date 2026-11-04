"""Same finite predicate for every tensor, one host decision per boundary."""
import torch

def finite_all(tensors):
    tensors=tuple(tensors)
    assert tensors and all(x is not None for x in tensors), 'Missing gradient/tensor'
    return torch.stack([torch.isfinite(x).all() for x in tensors]).all()

def require_finite(tensors):
    assert bool(finite_all(tensors)), 'Non-finite tensor in batched guard'
