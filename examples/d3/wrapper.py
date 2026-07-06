"""D3 (DNA discrete diffusion) adapter implementing the regdna-bench model interface.

example of how a diffusion model plugs into the benchmark. D3 is not
autoregressive, so two D3-specific choices are made here and documented:

- forward() runs a single denoising pass at a fixed noise level and returns the
  argmax tokens, to satisfy the "return a generated sequence" contract. zero-shot
  variant scoring reads the denoising logits (predict_logits) instead.
- embed() captures the final transformer block output with a forward hook.

D3's input/output vocab is exactly the benchmark CANON alphabet {N:0,A:1,C:2,G:3,T:4}
(it trained on the same zoonomia tokenization), so the CANON-to-vocab mapping is
identity: windows pass straight through and vocab_index returns the base id.

the only thing a caller must provide is a checkpoint: from_pretrained reads the
config embedded in the checkpoint and derives the probing noise level from the
model's own schedule. the D3 backbone (d3_dna) is imported lazily so the rest of
the benchmark does not depend on the D3 environment.
"""

from __future__ import annotations

import contextlib

import torch

from regdna_bench.base import EmbeddingModel, LikelihoodModel

# the D3 backbone uses FlashAttention (fp16/bf16 only), so the forward runs under
# autocast (transformer -> bf16, per d3_dna's precision policy). captured
# embeddings/logits are cast back to fp32 here.
EMBED_DTYPE = torch.float32


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


class D3Model(EmbeddingModel, LikelihoodModel):
    """benchmark adapter around a loaded D3 TransformerModel (DDiT backbone)."""

    def __init__(self, model, config, noise=None, device="cuda", default_sigma=None,
                 autocast_dtype=None):
        self._model = model.to(device).eval()
        self._config = config
        self._noise = noise
        self._device = device

        # autocast dtype for the FlashAttention forward; None disables autocast
        # (e.g. cpu debugging). from_pretrained sets it from the model config.
        self._autocast_dtype = autocast_dtype

        # diffusion forward needs a noise level; derive it from the model's own
        # schedule unless caller pins one.
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

    # CANON ids in, D3 token tensor out. identity vocab, so this only moves the
    # window onto the device; (B, L, 4) one-hot is accepted defensively.
    def _to_tokens(self, windows):
        tokens = torch.as_tensor(windows)
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

    def _denoise(self, windows):
        # shared single-step denoising pass; returns (logits, last-block output).
        tokens = self._to_tokens(windows)
        sigma = self._sigma_for(tokens.shape[0])

        with torch.no_grad(), self._autocast():
            logits, _rep = self._model(tokens, None, train=False, sigma=sigma)

        return logits

    def forward(self, windows):
        # contract: return a generated sequence. for diffusion this is the
        # single-step denoised prediction.
        return self._denoise(windows).argmax(dim=-1)

    # ---- EmbeddingModel ----

    def _capture_hook(self, _module, _inputs, output):
        self._last_layer_embedding = output.detach().to(EMBED_DTYPE)

    def embed(self, windows):
        if self._hook_handle is None:
            self._hook_handle = self._model.blocks[-1].register_forward_hook(self._capture_hook)

        self._denoise(windows)

        return self._last_layer_embedding

    # ---- LikelihoodModel ----

    # denoising logits (B, L, vocab); zero-shot variant scoring reads these rather
    # than forward()'s argmax tokens, since the diffusion model has no
    # autoregressive likelihood to compare ref vs alt alleles directly.
    def predict_logits(self, windows):
        return self._denoise(windows)

    # identity: D3's logit vocab is the CANON alphabet.
    def vocab_index(self, canon_base):
        return canon_base
