"""Abstract model interface for the regdna-bench benchmark.

A benchmark "model" is anything that can (a) generate DNA sequence and
(b) expose its last-layer embeddings. Prediction tasks (zero-shot,
probing) are written against this interface instead of taking a concrete
model parameter, so any model implementing it can be plugged in.
"""

from abc import ABC, abstractmethod


class BenchModel(ABC):
    """Base class every model wrapper in regdna-bench must implement.

    Two capabilities cover the task families we care about first:

    * ``forward`` powers *zero-shot* tasks: run the model on input
      sequence(s) and return generated sequence output (e.g. an
      autoregressively sampled continuation / predicted tokens).
      Zero-shot scores are derived from this output.

    * ``add_last_layer_embedding_extraction`` powers *probing* tasks:
      arrange for the model's final-layer hidden representations to be
      captured so a lightweight probe can be trained on frozen
      embeddings.
    """

    @abstractmethod
    def forward(self, sequence):
        """Run the model and return a generated sequence.

        Args:
            sequence: input DNA sequence(s) / context to condition on.

        Returns:
            The model's generated sequence output. Consumed by
            zero-shot tasks.
        """
        ...

    @abstractmethod
    def add_last_layer_embedding_extraction(self):
        """Enable extraction of the model's last-layer embeddings.

        Concrete implementations register whatever is needed (e.g. a
        forward hook, or ``output_hidden_states=True``) so that the
        final hidden-layer representation is available after
        ``forward``. Consumed by probing tasks.
        """
        ...
