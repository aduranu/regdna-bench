"""Model interfaces for the regdna-bench benchmark.

A benchmark "model" is the only thing that should be model-specific. Tasks are
written against these interfaces, not against any concrete model, so a new model
plugs in by implementing the minimal capability it supports.

The capabilities are split so a model only implements what it can do:

* ``BenchModel.forward`` is the base contract (run on canonical DNA, return a
  generated sequence).
* ``EmbeddingModel.embed`` powers embedding / probing tasks.
* ``LikelihoodModel.predict_logits`` + ``vocab_index`` power likelihood tasks.

All tasks pass and receive sequence as CANON ids (see ``CANON``). Each model
owns the mapping from CANON to its own input/output vocab, so datasets stay
model-agnostic.
"""

from abc import ABC, abstractmethod

# benchmark-wide canonical DNA alphabet. N is carried in-band so a window can
# hold chromosome-edge padding without a separate mask. datasets emit ids in
# this scheme; models translate to their own vocab inside the adapter.
CANON = {"N": 0, "A": 1, "C": 2, "G": 3, "T": 4}


class CapabilityError(TypeError):
    """raised when a task is run against a model lacking a required capability."""


class BenchModel(ABC):
    """base class every model wrapper must implement."""

    @abstractmethod
    def forward(self, windows):
        """run the model and return a generated sequence.

        Args:
            windows: (B, L) CANON token ids to condition on.

        Returns:
            the model's generated sequence output (e.g. predicted tokens).
        """
        ...


class EmbeddingModel(BenchModel):
    """capability for embedding / probing tasks."""

    @abstractmethod
    def embed(self, windows):
        """return last-layer hidden representations.

        Args:
            windows: (B, L) CANON token ids.

        Returns:
            (B, L, H) last-layer hidden states in fp32. tasks pool over L.
        """
        ...


class LikelihoodModel(BenchModel):
    """capability for likelihood-based zero-shot tasks."""

    @abstractmethod
    def predict_logits(self, windows):
        """return per-position scores.

        Args:
            windows: (B, L) CANON token ids.

        Returns:
            (B, L, V) per-position scores over the model's output vocab.
        """
        ...

    @abstractmethod
    def vocab_index(self, canon_base):
        """map a CANON base id (1-4) to its column in the predict_logits vocab."""
        ...
