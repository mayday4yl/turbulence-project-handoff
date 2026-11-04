"""Immutable physical-state caches for the authorized V5 A6k/B4k experiment."""


def __getattr__(name):
    if name in {"AlignmentDataset", "AlignmentBatchSampler"}:
        from .dataset import AlignmentDataset, AlignmentBatchSampler
        return {"AlignmentDataset": AlignmentDataset, "AlignmentBatchSampler": AlignmentBatchSampler}[name]
    raise AttributeError(name)
