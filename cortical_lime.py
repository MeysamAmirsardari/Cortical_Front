"""
CorticalLIME: Local Interpretable Model-agnostic Explanations for the
biomimetic auditory frontend (diffAudNeuro, Famularo et al. 2024).

WHY perturb (Rate, Scale) instead of (Time, Frequency)?
------------------------------------------------------
Standard audio LIME (e.g. SoundLIME, audioLIME) masks rectangular patches of
the time-frequency plane. Those patches have no biological referent: a "block
of pixels" in a spectrogram does not correspond to any neural population in
the auditory cortex, so the resulting explanations are difficult to map back
to the perceptual mechanisms the model is supposed to emulate.

The diffAudNeuro frontend instead expands each auditory spectrogram into a
bank of N STRF channels indexed by a (spectral scale Omega [cyc/oct],
temporal rate omega [Hz]) pair. Each channel is a direct analogue of a
modulation-tuned neuron in primary auditory cortex (A1) (Chi, Ru & Shamma
2005). Masking an entire (Omega, omega) channel therefore "lesions" a
biologically meaningful population across the whole utterance and tells us
which modulation channels the downstream classifier is actually relying on.

This module provides a small, framework-agnostic CorticalLIME class. The
forward pass uses the JAX/Flax `vSupervisedSTRF` model (this codebase is
JAX-only); the LIME logic itself is plain NumPy + scikit-learn so it is easy
to read, port, and unit-test. PyTorch is not used because the trained
checkpoints are JAX pickles -- there is no PyTorch backend to call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.linear_model import Ridge


# ---------------------------------------------------------------------------
# Perturbation generation
# ---------------------------------------------------------------------------

def perturb_cortical_features(
    n_strfs: int,
    n_samples: int,
    keep_prob: float = 0.5,
    rng: Optional[np.random.Generator] = None,
    include_reference: bool = True,
) -> np.ndarray:
    """Generate binary masks over the STRF channel axis.

    Each row of the returned matrix is a binary vector of length `n_strfs`
    where 1 means "keep this (Omega, omega) channel" and 0 means "zero it
    out across the entire time-frequency plane".

    Parameters
    ----------
    n_strfs : int
        Number of STRF (Omega, omega) channels in the cortical representation.
    n_samples : int
        Number of perturbed samples to draw.
    keep_prob : float
        Bernoulli probability of keeping each channel. 0.5 is a good default
        (maximum-entropy masks); lower values stress-test the model harder.
    rng : np.random.Generator, optional
        Source of randomness; if None a fresh default_rng() is used.
    include_reference : bool
        If True, the first row is the all-ones mask (the unperturbed input).
        Including it stabilises the Ridge fit and gives a clean reference
        prediction.
    """
    if rng is None:
        rng = np.random.default_rng()
    masks = (rng.random((n_samples, n_strfs)) < keep_prob).astype(np.float32)
    if include_reference:
        masks[0] = 1.0
    # Avoid degenerate all-zero masks: force at least one active channel.
    empty = masks.sum(axis=1) == 0
    if empty.any():
        idx = rng.integers(0, n_strfs, size=empty.sum())
        masks[empty, idx] = 1.0
    return masks


# ---------------------------------------------------------------------------
# Distance / kernel
# ---------------------------------------------------------------------------

def mask_distance(masks: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """Distance from each mask to the all-ones reference.

    "cosine" follows the original LIME paper (Ribeiro et al. 2016); "l2"
    is provided as a simple alternative.
    """
    ref = np.ones((1, masks.shape[1]), dtype=masks.dtype)
    if metric == "cosine":
        num = (masks * ref).sum(axis=1)
        den = np.linalg.norm(masks, axis=1) * np.linalg.norm(ref) + 1e-12
        return 1.0 - num / den
    if metric == "l2":
        return np.linalg.norm(masks - ref, axis=1) / np.sqrt(masks.shape[1])
    raise ValueError(f"Unknown metric: {metric}")


def kernel_weights(distances: np.ndarray, kernel_width: float = 0.25) -> np.ndarray:
    """Exponential kernel mapping distance -> sample weight (LIME-style)."""
    return np.exp(-(distances ** 2) / (kernel_width ** 2))


# ---------------------------------------------------------------------------
# Surrogate fitting
# ---------------------------------------------------------------------------

def fit_surrogate(
    masks: np.ndarray,
    target_probs: np.ndarray,
    sample_weights: np.ndarray,
    alpha: float = 1.0,
) -> tuple[Ridge, float]:
    """Fit a weighted Ridge regression: mask -> P(class)."""
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(masks, target_probs, sample_weight=sample_weights)
    score = float(model.score(masks, target_probs, sample_weight=sample_weights))
    return model, score


# ---------------------------------------------------------------------------
# Main explainer
# ---------------------------------------------------------------------------

@dataclass
class CorticalLIMEResult:
    """Container for a single CorticalLIME explanation."""
    target_class: int
    target_prob: float
    importances: np.ndarray         # (n_strfs,) Ridge coefficients
    sr: np.ndarray                  # (n_strfs, 2) (scale, rate) pairs
    masks: np.ndarray               # (n_samples, n_strfs)
    target_probs: np.ndarray        # (n_samples,)
    distances: np.ndarray           # (n_samples,)
    weights: np.ndarray             # (n_samples,)
    surrogate_r2: float
    intercept: float

    def top_k(self, k: int = 5) -> list[tuple[int, float, float, float]]:
        """Return [(channel, scale, rate, importance), ...] sorted by |importance|."""
        order = np.argsort(-np.abs(self.importances))[:k]
        return [
            (int(i), float(self.sr[i, 0]), float(self.sr[i, 1]), float(self.importances[i]))
            for i in order
        ]

    def rate_scale_grid(self, n_rate_bins: int = 12, n_scale_bins: int = 8):
        """Bin importances onto a (Rate x Scale) grid for visualisation.

        Returns (grid, rate_edges, scale_edges). The grid is a mean over
        STRF channels falling into each cell, with NaN for empty cells.
        """
        rates = self.sr[:, 1]
        scales = self.sr[:, 0]
        rate_edges = np.linspace(rates.min(), rates.max(), n_rate_bins + 1)
        scale_edges = np.linspace(scales.min(), scales.max(), n_scale_bins + 1)
        grid = np.full((n_scale_bins, n_rate_bins), np.nan)
        counts = np.zeros_like(grid)
        for imp, s, r in zip(self.importances, scales, rates):
            ri = min(np.searchsorted(rate_edges, r, side="right") - 1, n_rate_bins - 1)
            si = min(np.searchsorted(scale_edges, s, side="right") - 1, n_scale_bins - 1)
            ri = max(ri, 0); si = max(si, 0)
            if np.isnan(grid[si, ri]):
                grid[si, ri] = 0.0
            grid[si, ri] += imp
            counts[si, ri] += 1
        with np.errstate(invalid="ignore"):
            grid = grid / np.where(counts == 0, np.nan, counts)
        return grid, rate_edges, scale_edges


class CorticalLIME:
    """LIME-style explainer that perturbs STRF (Omega, omega) channels.

    The explainer is model-agnostic in the LIME sense: it only needs two
    callables that wrap the trained network.

    Parameters
    ----------
    encode_fn : Callable[[np.ndarray], np.ndarray]
        Maps a batch of waveforms (B, T_audio) -> cortical features
        (B, freq, time, n_strfs). For diffAudNeuro this is
        `vSupervisedSTRF.apply(..., method=encode)`.
    decode_fn : Callable[[np.ndarray], np.ndarray]
        Maps cortical features (B, freq, time, n_strfs) -> per-frame logits
        (B, time_pooled, n_classes). For diffAudNeuro this is
        `vSupervisedSTRF.apply(..., method=conv)`.
    sr : np.ndarray
        (n_strfs, 2) array of (scale, rate) pairs that parameterise the STRFs;
        used purely for plotting / book-keeping.
    n_samples : int
        Number of perturbed inputs to draw.
    keep_prob : float
        Bernoulli keep probability for each channel.
    distance_metric : {"cosine", "l2"}
    kernel_width : float
    ridge_alpha : float
    batch_size : int
        How many perturbed inputs to push through `decode_fn` at once.
    seed : int
    """

    def __init__(
        self,
        encode_fn: Callable[[np.ndarray], np.ndarray],
        decode_fn: Callable[[np.ndarray], np.ndarray],
        sr: np.ndarray,
        n_samples: int = 1000,
        keep_prob: float = 0.5,
        distance_metric: str = "cosine",
        kernel_width: float = 0.25,
        ridge_alpha: float = 1.0,
        batch_size: int = 32,
        seed: int = 0,
    ):
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn
        self.sr = np.asarray(sr)
        self.n_strfs = int(self.sr.shape[0])
        self.n_samples = int(n_samples)
        self.keep_prob = float(keep_prob)
        self.distance_metric = distance_metric
        self.kernel_width = float(kernel_width)
        self.ridge_alpha = float(ridge_alpha)
        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(seed)

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
        z = logits - logits.max(axis=axis, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=axis, keepdims=True)

    def _utterance_class_probs(self, logits: np.ndarray) -> np.ndarray:
        """Average softmax over time -> single (B, n_classes) vector per utt."""
        if logits.ndim == 4:                # (B, time, freq_pooled, n_classes)
            logits = logits.mean(axis=2)
        # logits now (B, time, n_classes)
        probs = self._softmax(logits, axis=-1)
        return probs.mean(axis=1)            # (B, n_classes)

    # -- public API -------------------------------------------------------

    def explain(
        self,
        waveform: np.ndarray,
        target_class: Optional[int] = None,
    ) -> CorticalLIMEResult:
        """Explain a single waveform.

        `waveform` is a 1-D float array of audio samples; `target_class`
        defaults to the model's argmax prediction.
        """
        wav = np.asarray(waveform, dtype=np.float32)[None, :]   # (1, T)

        # 1) One forward pass to get reference cortical features.
        cortical = np.asarray(self.encode_fn(wav))               # (1, F, T, S)
        ref_logits = np.asarray(self.decode_fn(cortical))
        ref_probs = self._utterance_class_probs(ref_logits)[0]   # (n_classes,)
        if target_class is None:
            target_class = int(np.argmax(ref_probs))

        # 2) Draw binary masks over STRF channels.
        masks = perturb_cortical_features(
            self.n_strfs, self.n_samples,
            keep_prob=self.keep_prob, rng=self.rng,
        )

        # 3) Apply masks and run the decoder in mini-batches.
        target_probs = np.empty(self.n_samples, dtype=np.float32)
        cortical0 = cortical[0]                                  # (F, T, S)
        for start in range(0, self.n_samples, self.batch_size):
            end = min(start + self.batch_size, self.n_samples)
            mb = masks[start:end]                                 # (b, S)
            # Broadcast mask along (F, T) axes -> (b, F, T, S).
            perturbed = cortical0[None, :, :, :] * mb[:, None, None, :]
            logits = np.asarray(self.decode_fn(perturbed))
            probs = self._utterance_class_probs(logits)
            target_probs[start:end] = probs[:, target_class]

        # 4) Distances, kernel weights, Ridge fit.
        distances = mask_distance(masks, metric=self.distance_metric)
        weights = kernel_weights(distances, kernel_width=self.kernel_width)
        ridge, r2 = fit_surrogate(
            masks, target_probs, weights, alpha=self.ridge_alpha,
        )

        return CorticalLIMEResult(
            target_class=int(target_class),
            target_prob=float(ref_probs[target_class]),
            importances=ridge.coef_.astype(np.float32),
            sr=self.sr,
            masks=masks,
            target_probs=target_probs,
            distances=distances,
            weights=weights,
            surrogate_r2=r2,
            intercept=float(ridge.intercept_),
        )


# ---------------------------------------------------------------------------
# JAX glue: build encode/decode callables for vSupervisedSTRF
# ---------------------------------------------------------------------------

def make_jax_callables(model, nn_params, aud_params):
    """Return (encode_fn, decode_fn) bound to a trained vSupervisedSTRF.

    `model` must be a `vSupervisedSTRF` instance, `nn_params` the FrozenDict
    returned by `model.init` (or loaded from a checkpoint), and `aud_params`
    the dict containing 'sr', 'compression_params', optionally 'alpha'.

    Both callables accept and return NumPy arrays so they can be used from
    pure-NumPy code (and from notebooks) without leaking JAX types.
    """
    @jax.jit
    def _encode(x):
        return model.apply(nn_params, x, aud_params, method=model.encode)

    @jax.jit
    def _decode(feats):
        return model.apply(nn_params, feats, method=model.conv)

    def encode_fn(wav_np):
        return np.asarray(_encode(jnp.asarray(wav_np)))

    def decode_fn(feats_np):
        return np.asarray(_decode(jnp.asarray(feats_np)))

    return encode_fn, decode_fn
