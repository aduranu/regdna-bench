"""D3 (DNA discrete diffusion) wrapper implementing the regdna-bench BenchModel.

example adapter showing how a diffusion model plugs into the benchmark. D3 is
not autoregressive, so two D3-specific choices are made here and documented:

- forward() runs a single denoising pass at a fixed noise level and returns the
  model's predicted (argmax) tokens, to satisfy the "return a generated
  sequence" contract. zero-shot variant scoring should read the denoising
  logits instead (predict_logits), since a diffusion model has no autoregressive
  continuation.
- last-layer embeddings are captured with a forward hook on the final
  transformer block. add_last_layer_embedding_extraction() installs the hook;
  the result is read from .last_layer_embedding after each forward().

the D3 backbone (d3_dna) is imported lazily inside from_pretrained so the rest
of the benchmark does not depend on the D3 environment.
"""

from __future__ import annotations

import torch

from regdna_bench.base import BenchModel

# precision: D3 samples under fp16 autocast, but embeddings and any probe-side
# reductions are kept in fp32 to avoid instability when fitting linear probes.
INFERENCE_DTYPE = torch.float16
EMBED_DTYPE = torch.float32

_DNA = ("A", "C", "G", "T")
_CHAR_TO_TOK = {c: i for i, c in enumerate(_DNA)}


class D3Model(BenchModel):
    """benchmark wrapper around a loaded D3 TransformerModel (DDiT backbone)."""

    def __init__(self, model, config, device="cuda", default_sigma=0.02):
        self._model = model.to(device).eval()
        self._config = config
        self._device = device

        # diffusion forward needs a noise level; autoregressive models would not.
        # a single fixed sigma matches the D3 lentiMPRA probing setup.
        self._default_sigma = default_sigma

        self._last_layer_embedding = None
        self._hook_handle = None

    @classmethod
    def from_pretrained(cls, checkpoint, config, device="cuda", default_sigma=0.02):
        from omegaconf import OmegaConf
        from d3_dna.models import TransformerModel
        from d3_dna.modules.checkpoint import load_checkpoint

        cfg = OmegaConf.load(config) if isinstance(config, str) else config

        model = TransformerModel(cfg)
        # load_checkpoint applies EMA weights and preserves checkpoint precision
        model, _graph, _noise = load_checkpoint(checkpoint, model=model, config=cfg, device=device)
        model.eval()

        return cls(model, cfg, device=device, default_sigma=default_sigma)

    # accepts ACGT strings, (B, L) token ids, or (B, L, 4) one-hot
    def _to_tokens(self, sequence):
        if isinstance(sequence, str):
            sequence = [sequence]

        if isinstance(sequence, (list, tuple)):
            return torch.tensor(
                [[_CHAR_TO_TOK[c] for c in s] for s in sequence],
                dtype=torch.long, device=self._device,
            )

        tokens = sequence
        if tokens.dim() == 3:
            tokens = tokens.argmax(dim=-1)

        return tokens.to(self._device).long()

    def _sigma_for(self, batch_size):
        return torch.full(
            (batch_size,), float(self._default_sigma),
            device=self._device, dtype=EMBED_DTYPE,
        )

    # forward hook is a bound method (not a nested def) so it captures self
    # without violating the no-nested-functions rule. block output is
    # (B, L, hidden); probing pools over the length dimension.
    def _capture_hook(self, _module, _inputs, output):
        self._last_layer_embedding = output.detach().to(EMBED_DTYPE)

    def forward(self, sequence):
        tokens = self._to_tokens(sequence)
        sigma = self._sigma_for(tokens.shape[0])

        with torch.no_grad():
            logits, _rep = self._model(tokens, None, train=False, sigma=sigma)

        # contract: return a generated sequence. for diffusion this is the
        # single-step denoised prediction; use predict_logits for a continuous
        # variant-effect score.
        return logits.argmax(dim=-1)

    def add_last_layer_embedding_extraction(self):
        if self._hook_handle is not None:
            return

        self._hook_handle = self._model.blocks[-1].register_forward_hook(self._capture_hook)

    # ---- extras consumed by tasks (beyond the minimal BenchModel ABC) ----

    @property
    def last_layer_embedding(self):
        # populated by the capture hook after forward(); (B, L, hidden)
        return self._last_layer_embedding

    # denoising logits (B, L, vocab); zero-shot variant scoring reads these
    # rather than forward()'s argmax tokens, since the diffusion model has no
    # autoregressive likelihood to compare ref vs alt alleles directly.
    def predict_logits(self, sequence):
        tokens = self._to_tokens(sequence)
        sigma = self._sigma_for(tokens.shape[0])

        with torch.no_grad():
            logits, _rep = self._model(tokens, None, train=False, sigma=sigma)

        return logits
