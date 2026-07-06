"""dataset interface + canonical containers consumed by tasks.

a Dataset assembles model-agnostic inputs (CANON token windows + metadata). the
VEP tasks consume a VariantSet; other task families add their own container
behind the same ABC.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class VariantSet:
    """paired ref/alt CANON windows + labels for variant-effect tasks.

    all arrays are row-aligned over N variants. ref_ids/alt_ids are (N, L) CANON
    windows; ref_base/alt_base are the CANON base id (1-4) at the variant offset,
    used by the likelihood task to index per-position logits.
    """

    ref_ids: np.ndarray
    alt_ids: np.ndarray
    offsets: np.ndarray
    ref_base: np.ndarray
    alt_base: np.ndarray
    groups: np.ndarray
    labels: np.ndarray | None = None
    effect_sizes: np.ndarray | None = None

    def __len__(self):
        return len(self.ref_ids)


class Dataset(ABC):
    """base class for benchmark datasets; load() returns a task-ready container."""

    @property
    @abstractmethod
    def name(self):
        ...

    @abstractmethod
    def load(self):
        # assemble and return the canonical container (e.g. VariantSet).
        ...
