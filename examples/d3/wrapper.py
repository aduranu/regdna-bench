"""D3 (DNA discrete diffusion) adapter implementing the regdna-bench model interface.

example of how a diffusion model plugs into the benchmark. D3 is not
autoregressive, so two D3-specific choices are made here and documented:

- forward() runs a single denoising pass at a fixed noise level and returns the
  argmax tokens, to satisfy the "return a generated sequence" contract. zero-shot
  variant scoring reads the denoising logits (predict_logits) instead, since a
  diffusion model has no autoregressive continuation.
- last-layer embeddings are captured with a forward hook on the final
  transformer block. the vep embedding task calls embed(); probing installs the
  hook via add_last_layer_embedding_extraction() and reads .last_layer_embedding
  after each forward().

D3's zoonomia vocab is the benchmark CANON alphabet {N:0,A:1,C:2,G:3,T:4}, so
CANON id windows (from the vep datasets) pass straight through. acgt one-hot /
string inputs (from the probing h5 splits) are shifted onto the model vocab by
token_offset (1 for zoonomia, 0 for the acgt-only k562/lentimpra models).

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
# embeddings/logits are cast back to fp32 here; the cnn probe also keeps fp32.
EMBED_DTYPE = torch.float32

_DNA = ("A", "C", "G", "T")
_CHAR_TO_TOK = {c: i for i, c in enumerate(_DNA)}


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
                 token_offset=0, autocast_dtype=None):
        self._model = model.to(device).eval()
        self._config = config
        self._noise = noise
        self._device = device

        # token_offset maps acgt one-hot channels (0..3) onto the model's own
        # vocab: 0 for the lentimpra/k562 models (0=A..3=T), 1 for zoonomia
        # (0=N, 1=A..4=T). CANON id windows are already in vocab and pass as-is.
        self._token_offset = token_offset

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
    def from_pretrained(cls, checkpoint, config=None, device="cuda", default_sigma=None,
                        token_offset=0):
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
                   token_offset=token_offset, autocast_dtype=autocast_dtype)

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

    # accepts CANON/model-vocab (B, L) token ids, acgt (B, L, 4) one-hot, or ACGT
    # strings. token_offset shifts acgt onto the model's vocab; pre-tokenized
    # (B, L) ids are assumed already in the model's convention and left as-is.
    def _to_tokens(self, sequence):
        if isinstance(sequence, str):
            sequence = [sequence]

        if isinstance(sequence, (list, tuple)):
            return torch.tensor(
                [[_CHAR_TO_TOK[c] + self._token_offset for c in s] for s in sequence],
                dtype=torch.long, device=self._device,
            )

        tokens = torch.as_tensor(sequence)
        if tokens.dim() == 3:
            tokens = tokens.argmax(dim=-1) + self._token_offset

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

    def _denoise(self, sequence):
        # shared single-step denoising pass used by forward()/embed()/predict_logits().
        tokens = self._to_tokens(sequence)
        sigma = self._sigma_for(tokens.shape[0])

        with torch.no_grad(), self._autocast():
            logits, _rep = self._model(tokens, None, train=False, sigma=sigma)

        return logits

    def forward(self, sequence):
        # contract: return a generated sequence. for diffusion this is the
        # single-step denoised prediction; use predict_logits for a continuous
        # variant-effect score.
        return self._denoise(sequence).argmax(dim=-1)

    # ---- EmbeddingModel ----

    def add_last_layer_embedding_extraction(self):
        if self._hook_handle is not None:
            return

        self._hook_handle = self._model.blocks[-1].register_forward_hook(self._capture_hook)

    @property
    def last_layer_embedding(self):
        # populated by the capture hook after forward(); (B, L, hidden)
        return self._last_layer_embedding

    def embed(self, windows):
        # vep embedding path: install the hook, run the pass, return (B, L, hidden).
        # probing drives the hook directly via add_last_layer_embedding_extraction().
        self.add_last_layer_embedding_extraction()

        self._denoise(windows)

        return self._last_layer_embedding

    # ---- LikelihoodModel ----

    # denoising logits (B, L, vocab); zero-shot variant scoring reads these rather
    # than forward()'s argmax tokens, since the diffusion model has no
    # autoregressive likelihood to compare ref vs alt alleles directly.
    def predict_logits(self, sequence):
        return self._denoise(sequence)

    # identity: D3's logit vocab is the CANON alphabet.
    def vocab_index(self, canon_base):
        return canon_base
