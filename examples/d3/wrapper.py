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

the only thing a caller must provide is a checkpoint: from_pretrained reads the
config embedded in the checkpoint and derives the probing noise level from the
model's own schedule. the D3 backbone (d3_dna) is imported lazily so the rest of
the benchmark does not depend on the D3 environment.
"""

from __future__ import annotations

import contextlib

import torch

from regdna_bench.base import BenchModel

# the D3 backbone uses FlashAttention, which only supports fp16/bf16, so the
# model forward must run under autocast (transformer -> bf16, per d3_dna's
# precision policy). captured embeddings/logits are cast back to fp32 here.
EMBED_DTYPE = torch.float32

DNA = ("A", "C", "G", "T")
CHAR_TO_TOK = {c: i for i, c in enumerate(DNA)}


# config lives inside PL checkpoints under hyper_parameters.cfg, so callers can
# pass just a checkpoint. a path or an already-loaded OmegaConf still override it.
def _resolve_config(checkpoint, config, device):
    from omegaconf import OmegaConf

    if config is not None:
        return OmegaConf.load(config) if isinstance(config, str) else config

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    hyper = ckpt.get("hyper_parameters", {}) if isinstance(ckpt, dict) else {}

    if "cfg" not in hyper:
        raise ValueError("no config given and checkpoint has no embedded cfg; pass config explicitly")

    return OmegaConf.create(dict(hyper["cfg"]))


class D3Model(BenchModel):
    """benchmark wrapper around a loaded D3 TransformerModel (DDiT backbone)."""

    def __init__(self, model, config, noise=None, device="cuda", default_sigma=None,
                 autocast_dtype=None):
        self._model = model.to(device).eval()
        self._config = config
        self._noise = noise
        self._device = device

        # autocast dtype for the FlashAttention forward; None disables autocast
        # (e.g. cpu debugging). from_pretrained sets it from the model config.
        self._autocast_dtype = autocast_dtype

        # diffusion forward needs a noise level; autoregressive models would not.
        # derive it from the model's own schedule unless caller pins one.
        if default_sigma is None:
            if noise is None:
                raise ValueError("provide default_sigma, or a noise schedule to derive it from")
            default_sigma = self._canonical_sigma()
        self._default_sigma = default_sigma

        self._last_layer_embedding = None
        self._hook_handle = None

    @classmethod
    def from_pretrained(cls, checkpoint, config=None, device="cuda", default_sigma=None):
        from d3_dna.models import TransformerModel
        from d3_dna.modules.checkpoint import load_checkpoint
        from d3_dna.modules.precision import autocast_dtype_for_cfg

        cfg = _resolve_config(checkpoint, config, device)

        model = TransformerModel(cfg)
        # load_checkpoint applies EMA weights and returns the noise schedule we
        # need to pick the probing sigma
        model, _graph, noise = load_checkpoint(checkpoint, model=model, config=cfg, device=device)
        model.eval()

        # reuse d3's own precision policy (transformer -> bf16) so the forward
        # autocast matches how the model was trained / is sampled.
        autocast_dtype = autocast_dtype_for_cfg(cfg)

        return cls(model, cfg, noise=noise, device=device, default_sigma=default_sigma,
                   autocast_dtype=autocast_dtype)

    # matches the lentiMPRA VEP default: read the 5th-from-last (low-noise) sigma
    # of the geometric schedule, so embeddings come from near the clean end of
    # the diffusion trajectory.
    def _canonical_sigma(self):
        steps = int(self._config.sampling.steps) if hasattr(self._config, "sampling") else 128
        eps = 1e-5

        timesteps = torch.linspace(1.0, eps, steps + 1, device=self._device)
        sigmas = self._noise.total_noise(timesteps)

        idx = max(0, len(sigmas) - 5)

        return float(sigmas[idx].item())

    # accepts ACGT strings, numpy/torch (B, L) token ids, or (B, L, 4) one-hot
    def _to_tokens(self, sequence):
        if isinstance(sequence, str):
            sequence = [sequence]

        if isinstance(sequence, (list, tuple)):
            return torch.tensor(
                [[CHAR_TO_TOK[c] for c in s] for s in sequence],
                dtype=torch.long, device=self._device,
            )

        tokens = torch.as_tensor(sequence)
        if tokens.dim() == 3:
            tokens = tokens.argmax(dim=-1)

        return tokens.to(self._device).long()

    def _sigma_for(self, batch_size):
        return torch.full(
            (batch_size,), float(self._default_sigma),
            device=self._device, dtype=EMBED_DTYPE,
        )

    # autocast context for the model forward; nullcontext when disabled (cpu).
    def _autocast(self):
        if self._autocast_dtype is None:
            return contextlib.nullcontext()

        device_type = "cuda" if "cuda" in str(self._device) else "cpu"

        return torch.amp.autocast(device_type, dtype=self._autocast_dtype,
                                  enabled=(device_type == "cuda"))

    # forward hook is a bound method (not a nested def) so it captures self
    # without violating the no-nested-functions rule. block output is
    # (B, L, hidden); probing pools over the length dimension.
    def _capture_hook(self, _module, _inputs, output):
        self._last_layer_embedding = output.detach().to(EMBED_DTYPE)

    def forward(self, sequence):
        tokens = self._to_tokens(sequence)
        sigma = self._sigma_for(tokens.shape[0])

        with torch.no_grad(), self._autocast():
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

        with torch.no_grad(), self._autocast():
            logits, _rep = self._model(tokens, None, train=False, sigma=sigma)

        return logits
