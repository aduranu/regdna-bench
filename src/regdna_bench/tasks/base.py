"""task interface + stratification helpers shared by tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Task(ABC):
    """a benchmark task: prepare inputs, score a model, reduce to metrics.

    ``requires`` lists the model capability interfaces (from regdna_bench.base)
    the runner checks before scoring, so a missing capability fails fast.
    """

    name: str = ""
    requires: tuple = ()

    @abstractmethod
    def prepare(self, dataset):
        # pull a loaded dataset into the shape run() consumes.
        ...

    @abstractmethod
    def run(self, model, prepared):
        # score the model; return raw per-item outputs.
        ...

    @abstractmethod
    def metrics(self, prepared, output):
        # reduce raw outputs to a json-able summary (overall + by-class).
        ...


def group_indices(groups):
    # {group label: [row indices]}, preserving first-seen order so per-group
    # slices stay aligned with the input arrays.
    out = {}
    for i, g in enumerate(groups):
        out.setdefault(g, []).append(i)

    return out


def subset(values, idx):
    # index any per-item container: lists/tuples by comprehension, arrays/tensors
    # by fancy index, None passes through.
    if values is None:
        return None

    if isinstance(values, (list, tuple)):
        return [values[i] for i in idx]

    return values[idx]
