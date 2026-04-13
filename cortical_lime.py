"""
CorticalLIME — Perturbation-based interpretability for the diffAudNeuro
biomimetic auditory frontend (Famularo et al., arXiv:2409.08997).

==========================================================================
WHY PERTURB (RATE ω, SCALE Ω) INSTEAD OF (TIME, FREQUENCY)?
==========================================================================
Standard audio LIME methods (SoundLIME, audioLIME) perturb rectangular
patches of the time-frequency plane.  Those patches have *no biological
referent*: a block of spectrogram pixels does not map to any identifiable
neural population in primary auditory cortex (A1).

The diffAudNeuro frontend decomposes the auditory spectrogram into a bank
of S STRF channels indexed by (Ω, ω) — spectral scale [cyc/oct] and
temporal rate [Hz].  Each channel models a modulation-tuned cortical
neuron (Chi, Ru & Shamma, JASA 2005).  Masking an entire (Ω, ω) channel
lesions a *biologically meaningful* population across the whole utterance,
revealing which modulation channels drive the downstream classifier's
phonetic discrimination.

This framing connects directly to classical electrophysiology of A1
(Depireux et al. 2001; Atencio & Schreiner 2012) and gives explanations
that can be cross-validated against known phonetic–acoustic correlates
(e.g., temporal modulations ≤ 16 Hz for syllabic rate, spectral scales
> 2 cyc/oct for formant structure).

==========================================================================
ARCHITECTURE
==========================================================================
cortical_lime.py        Core explainer + alternative baselines
cortical_lime_metrics.py  Faithfulness, stability, population analysis

The forward pass is JAX/Flax (`vSupervisedSTRF`).  All LIME / analysis
logic is pure NumPy + scikit-learn for portability and testability.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.linear_model import BayesianRidge, Lasso, Ridge


# ═══════════════════════════════════════════════════════════════════════════
# 1.  PERTURBATION STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════

class PerturbStrategy(str, Enum):
    """Supported perturbation strategies for the interpretable domain.

    BERNOULLI   Binary on/off per channel, i.i.d. Bernoulli(keep_prob).
                Maximises information per sample when keep_prob=0.5.
    GAUSSIAN    Continuous scaling: each channel is multiplied by a sample
                from N(1, σ²), clipped to [0, ∞).  Richer signal per
                perturbation at the cost of a non-binary interpretable
                domain (requires continuous surrogate).
    STRUCTURED  Binary, but channels with similar (Ω, ω) are grouped into
                K super-channels via k-means on the (scale, rate) plane.
                Masking is per-group, reducing the effective dimensionality
                and improving surrogate fit when S is large.
    """
    BERNOULLI = "bernoulli"
    GAUSSIAN = "gaussian"
    STRUCTURED = "structured"


def _bernoulli_masks(
    n_strfs: int,
    n_samples: int,
    keep_prob: float,
    rng: np.random.Generator,
) -> np.ndarray:
    masks = (rng.random((n_samples, n_strfs)) < keep_prob).astype(np.float32)
    # Fix degenerate all-zero rows.
    empty = masks.sum(axis=1) == 0
    if empty.any():
        masks[empty, rng.integers(0, n_strfs, size=int(empty.sum()))] = 1.0
    return masks


def _gaussian_masks(
    n_strfs: int,
    n_samples: int,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Continuous multiplicative perturbation, clipped to [0, +∞)."""
    masks = rng.normal(loc=1.0, scale=sigma, size=(n_samples, n_strfs))
    masks = np.clip(masks, 0.0, None).astype(np.float32)
    return masks


def _structured_masks(
    n_strfs: int,
    n_samples: int,
    keep_prob: float,
    sr: np.ndarray,
    n_groups: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Group (Ω, ω) channels by k-means, then Bernoulli-mask per group.

    Returns (masks, group_labels) where masks is (n_samples, n_strfs)
    and group_labels is (n_strfs,).
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    sr_norm = StandardScaler().fit_transform(sr)
    km = KMeans(n_clusters=n_groups, n_init=10, random_state=int(rng.integers(1 << 31)))
    labels = km.fit_predict(sr_norm)

    group_masks = (rng.random((n_samples, n_groups)) < keep_prob).astype(np.float32)
    empty = group_masks.sum(axis=1) == 0
    if empty.any():
        group_masks[empty, rng.integers(0, n_groups, size=int(empty.sum()))] = 1.0
    masks = group_masks[:, labels]
    return masks, labels


def generate_perturbation_masks(
    n_strfs: int,
    n_samples: int,
    strategy: PerturbStrategy | str = PerturbStrategy.BERNOULLI,
    keep_prob: float = 0.5,
    sigma: float = 0.5,
    sr: Optional[np.ndarray] = None,
    n_groups: int = 8,
    include_reference: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Generate perturbation masks for STRF channels.

    Returns
    -------
    masks : (n_samples, n_strfs) float32
    group_labels : (n_strfs,) int or None  (only for STRUCTURED)
    """
    strategy = PerturbStrategy(strategy)
    if rng is None:
        rng = np.random.default_rng()

    group_labels = None
    if strategy == PerturbStrategy.BERNOULLI:
        masks = _bernoulli_masks(n_strfs, n_samples, keep_prob, rng)
    elif strategy == PerturbStrategy.GAUSSIAN:
        masks = _gaussian_masks(n_strfs, n_samples, sigma, rng)
    elif strategy == PerturbStrategy.STRUCTURED:
        if sr is None:
            raise ValueError("STRUCTURED strategy requires `sr` (scale-rate pairs).")
        masks, group_labels = _structured_masks(
            n_strfs, n_samples, keep_prob, sr, n_groups, rng,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    if include_reference:
        masks[0] = 1.0
    return masks, group_labels


# ═══════════════════════════════════════════════════════════════════════════
# 2.  DISTANCE & KERNEL FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

class DistanceMetric(str, Enum):
    COSINE = "cosine"
    L2 = "l2"
    HAMMING = "hamming"


def compute_distances(
    masks: np.ndarray,
    reference: Optional[np.ndarray] = None,
    metric: DistanceMetric | str = DistanceMetric.COSINE,
) -> np.ndarray:
    """Distance from each mask to the reference (default: all-ones).

    Uses scipy.spatial.distance.cdist for correctness and speed.
    """
    metric = DistanceMetric(metric)
    if reference is None:
        reference = np.ones((1, masks.shape[1]), dtype=masks.dtype)
    elif reference.ndim == 1:
        reference = reference[None, :]

    if metric == DistanceMetric.COSINE:
        d = cdist(masks, reference, metric="cosine").ravel()
        # cdist returns NaN for zero-norm rows; map to 1.0 (maximum).
        d = np.nan_to_num(d, nan=1.0)
        return d
    if metric == DistanceMetric.L2:
        return cdist(masks, reference, metric="euclidean").ravel() / np.sqrt(masks.shape[1])
    if metric == DistanceMetric.HAMMING:
        binary = (masks > 0.5).astype(np.float32)
        ref_bin = (reference > 0.5).astype(np.float32)
        return cdist(binary, ref_bin, metric="hamming").ravel()
    raise ValueError(f"Unknown metric: {metric}")


def exponential_kernel(distances: np.ndarray, width: float = 0.25) -> np.ndarray:
    """LIME exponential kernel: w = exp(-d² / σ²)."""
    return np.exp(-(distances ** 2) / (width ** 2))


# ═══════════════════════════════════════════════════════════════════════════
# 3.  SURROGATE MODELS
# ═══════════════════════════════════════════════════════════════════════════

class SurrogateType(str, Enum):
    """Surrogate model families.

    RIDGE          L2-penalised linear regression (Ribeiro et al., 2016).
    LASSO          L1-penalised; yields genuinely sparse explanations.
    BAYESIAN_RIDGE Bayesian linear regression; provides per-coefficient
                   posterior variance for principled confidence intervals
                   without bootstrap.
    """
    RIDGE = "ridge"
    LASSO = "lasso"
    BAYESIAN_RIDGE = "bayesian_ridge"


@dataclass
class SurrogateResult:
    """Output of a surrogate fit."""
    coef: np.ndarray            # (n_features,) importance weights
    intercept: float
    r2: float                   # weighted R²
    coef_std: Optional[np.ndarray] = None  # per-coef std (BayesianRidge only)


def fit_surrogate(
    masks: np.ndarray,
    target: np.ndarray,
    sample_weights: np.ndarray,
    model_type: SurrogateType | str = SurrogateType.RIDGE,
    alpha: float = 1.0,
) -> SurrogateResult:
    """Fit a weighted linear surrogate: mask → P(class).

    For BayesianRidge, `alpha` is ignored (the model learns its own).
    """
    model_type = SurrogateType(model_type)
    coef_std = None

    if model_type == SurrogateType.RIDGE:
        m = Ridge(alpha=alpha, fit_intercept=True)
        m.fit(masks, target, sample_weight=sample_weights)
    elif model_type == SurrogateType.LASSO:
        m = Lasso(alpha=alpha, fit_intercept=True, max_iter=5000)
        m.fit(masks, target, sample_weight=sample_weights)
    elif model_type == SurrogateType.BAYESIAN_RIDGE:
        m = BayesianRidge(fit_intercept=True, max_iter=500)
        # BayesianRidge does not natively support sample_weight;
        # emulate by repeating rows proportional to weight.
        w_norm = sample_weights / sample_weights.sum() * len(sample_weights)
        # Weighted least squares via diagonal sqrt(W) transform.
        sw = np.sqrt(np.maximum(w_norm, 0.0))
        m.fit(masks * sw[:, None], target * sw)
        coef_std = np.sqrt(np.diag(m.sigma_)).astype(np.float32)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    r2 = float(m.score(masks, target, sample_weight=sample_weights))
    return SurrogateResult(
        coef=m.coef_.astype(np.float32),
        intercept=float(m.intercept_),
        r2=r2,
        coef_std=coef_std,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 4.  CorticalLIME RESULT
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CorticalLIMEResult:
    """Complete output of a single CorticalLIME explanation.

    Carries all intermediate data so downstream analyses (faithfulness
    metrics, bootstrap, population studies) can be computed without
    re-running the expensive forward passes.
    """
    # -- identity ---
    target_class: int
    target_prob: float               # P(target_class) on unperturbed input
    full_probs: np.ndarray           # (n_classes,) full prob vector, reference

    # -- explanation ---
    importances: np.ndarray          # (n_strfs,) surrogate coefficients
    importances_std: Optional[np.ndarray]  # (n_strfs,) coef std (BayesianRidge)
    intercept: float
    surrogate_r2: float
    surrogate_type: str

    # -- STRF metadata ---
    sr: np.ndarray                   # (n_strfs, 2)  columns = [scale, rate]
    n_strfs: int

    # -- perturbation data (for post-hoc analysis) ---
    masks: np.ndarray                # (n_samples, n_strfs)
    target_probs: np.ndarray         # (n_samples,) P(target_class) per mask
    all_probs: np.ndarray            # (n_samples, n_classes) full prob vectors
    distances: np.ndarray            # (n_samples,)
    weights: np.ndarray              # (n_samples,)
    group_labels: Optional[np.ndarray]  # (n_strfs,) for STRUCTURED

    # -- config ---
    strategy: str = ""
    kernel_width: float = 0.25
    seed: int = 0

    # -- convenience methods ---

    def top_k(self, k: int = 5) -> list[dict]:
        """Top-k channels by |importance|.  Returns list of dicts."""
        order = np.argsort(-np.abs(self.importances))[:k]
        return [
            dict(
                channel=int(i),
                scale=float(self.sr[i, 0]),
                rate=float(self.sr[i, 1]),
                importance=float(self.importances[i]),
                importance_std=float(self.importances_std[i]) if self.importances_std is not None else None,
            )
            for i in order
        ]

    def signed_rank(self) -> np.ndarray:
        """Rank channels by signed importance (most positive first)."""
        return np.argsort(-self.importances)

    def confidence_intervals(self, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) bounds on importances.

        If BayesianRidge was used, these are analytic Gaussian CIs.
        Otherwise returns (importances, importances) — run bootstrap
        for non-Bayesian surrogates.
        """
        if self.importances_std is not None:
            from scipy.stats import norm
            z = norm.ppf(1 - alpha / 2)
            lo = self.importances - z * self.importances_std
            hi = self.importances + z * self.importances_std
            return lo, hi
        return self.importances.copy(), self.importances.copy()

    def rate_scale_grid(
        self,
        n_rate_bins: int = 10,
        n_scale_bins: int = 6,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bin importances onto a regular (Rate × Scale) grid.

        Returns (grid, rate_edges, scale_edges).
        Grid cells with no data are NaN.
        """
        rates, scales = self.sr[:, 1], self.sr[:, 0]
        re = np.linspace(rates.min(), rates.max(), n_rate_bins + 1)
        se = np.linspace(scales.min(), scales.max(), n_scale_bins + 1)
        grid = np.full((n_scale_bins, n_rate_bins), np.nan)
        cnt = np.zeros_like(grid)
        for imp, s, r in zip(self.importances, scales, rates):
            ri = np.clip(np.searchsorted(re, r, side="right") - 1, 0, n_rate_bins - 1)
            si = np.clip(np.searchsorted(se, s, side="right") - 1, 0, n_scale_bins - 1)
            grid[si, ri] = np.nansum([grid[si, ri], imp])
            cnt[si, ri] += 1
        with np.errstate(invalid="ignore"):
            grid /= np.where(cnt == 0, np.nan, cnt)
        return grid, re, se


# ═══════════════════════════════════════════════════════════════════════════
# 5.  BOOTSTRAP CONFIDENCE INTERVALS
# ═══════════════════════════════════════════════════════════════════════════

def bootstrap_importances(
    result: CorticalLIMEResult,
    n_bootstrap: int = 500,
    alpha_ci: float = 0.05,
    surrogate_type: SurrogateType | str = SurrogateType.RIDGE,
    surrogate_alpha: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Non-parametric bootstrap on surrogate coefficients.

    Re-samples (mask, target_prob, weight) tuples and re-fits the
    surrogate, yielding a distribution over importance vectors.

    Returns
    -------
    boot_mean : (n_strfs,)
    boot_lo   : (n_strfs,) lower percentile
    boot_hi   : (n_strfs,) upper percentile
    """
    rng = np.random.default_rng(seed)
    n = len(result.masks)
    coefs = np.empty((n_bootstrap, result.n_strfs), dtype=np.float32)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sr = fit_surrogate(
            result.masks[idx],
            result.target_probs[idx],
            result.weights[idx],
            model_type=surrogate_type,
            alpha=surrogate_alpha,
        )
        coefs[b] = sr.coef

    plo = 100 * alpha_ci / 2
    phi = 100 * (1 - alpha_ci / 2)
    return (
        coefs.mean(axis=0),
        np.percentile(coefs, plo, axis=0),
        np.percentile(coefs, phi, axis=0),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 6.  MAIN EXPLAINER CLASS
# ═══════════════════════════════════════════════════════════════════════════

class CorticalLIME:
    """LIME-style explainer operating on the STRF modulation channel axis
    of the diffAudNeuro biomimetic auditory frontend.

    The explainer is *model-agnostic* in the LIME sense: it only requires
    two callables (encode, decode) that wrap the frozen network.  All
    expensive forward passes are batched for GPU/TPU throughput.

    Parameters
    ----------
    encode_fn : (B, T_audio) -> (B, F, T_cortical, S)
    decode_fn : (B, F, T_cortical, S) -> (B, T_pooled, n_classes)
    sr : (S, 2)  — columns [scale, rate]
    strategy : perturbation strategy
    n_samples : number of perturbed inputs
    keep_prob : Bernoulli keep probability (BERNOULLI / STRUCTURED)
    sigma : std for Gaussian perturbation
    n_groups : number of super-channels (STRUCTURED)
    distance_metric : cosine | l2 | hamming
    kernel_width : exponential kernel σ
    surrogate_type : ridge | lasso | bayesian_ridge
    surrogate_alpha : regularisation strength (Ridge / Lasso)
    batch_size : mini-batch size for decode_fn
    seed : PRNG seed
    """

    def __init__(
        self,
        encode_fn: Callable,
        decode_fn: Callable,
        sr: np.ndarray,
        *,
        strategy: PerturbStrategy | str = PerturbStrategy.BERNOULLI,
        n_samples: int = 2000,
        keep_prob: float = 0.5,
        sigma: float = 0.5,
        n_groups: int = 8,
        distance_metric: DistanceMetric | str = DistanceMetric.COSINE,
        kernel_width: float = 0.25,
        surrogate_type: SurrogateType | str = SurrogateType.RIDGE,
        surrogate_alpha: float = 1.0,
        batch_size: int = 64,
        seed: int = 0,
    ):
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn
        self.sr = np.asarray(sr, dtype=np.float64)
        self.n_strfs = self.sr.shape[0]

        self.strategy = PerturbStrategy(strategy)
        self.n_samples = n_samples
        self.keep_prob = keep_prob
        self.sigma = sigma
        self.n_groups = n_groups

        self.distance_metric = DistanceMetric(distance_metric)
        self.kernel_width = kernel_width

        self.surrogate_type = SurrogateType(surrogate_type)
        self.surrogate_alpha = surrogate_alpha
        self.batch_size = batch_size
        self.seed = seed

    # -- internal helpers ---------------------------------------------------

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        e = np.exp(x - x.max(axis=axis, keepdims=True))
        return e / e.sum(axis=axis, keepdims=True)

    def _logits_to_probs(self, logits: np.ndarray) -> np.ndarray:
        """(B, T, [F,] C) → (B, C) by averaging softmax across time.

        The CNN decoder may output (B, T, C) or (B, T, F_pool, C) depending
        on pooling_stride; we handle both.
        """
        if logits.ndim == 4:
            logits = logits.mean(axis=2)
        return self._softmax(logits, axis=-1).mean(axis=1)

    def _run_decoder_batched(
        self,
        cortical_ref: np.ndarray,
        masks: np.ndarray,
    ) -> np.ndarray:
        """Apply masks and run decoder in mini-batches.

        cortical_ref : (F, T, S) — single-utterance cortical features
        masks        : (N, S)
        Returns      : (N, C) probability vectors
        """
        n = masks.shape[0]
        all_probs = []
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            mb = masks[start:end]
            perturbed = cortical_ref[None, :, :, :] * mb[:, None, None, :]
            logits = np.asarray(self.decode_fn(perturbed))
            all_probs.append(self._logits_to_probs(logits))
        return np.concatenate(all_probs, axis=0)

    # -- public API ---------------------------------------------------------

    def explain(
        self,
        waveform: np.ndarray,
        target_class: Optional[int] = None,
    ) -> CorticalLIMEResult:
        """Produce a CorticalLIME explanation for a single waveform.

        Parameters
        ----------
        waveform : (T_audio,) float32, mono, 16 kHz, RMS-normalised.
        target_class : class index to explain; defaults to argmax.
        """
        wav = np.asarray(waveform, dtype=np.float32)[None, :]   # (1, T)

        # ---- 1. Reference forward pass -----------------------------------
        cortical = np.asarray(self.encode_fn(wav))               # (1, F, T, S)
        ref_logits = np.asarray(self.decode_fn(cortical))
        ref_probs = self._logits_to_probs(ref_logits)[0]
        if target_class is None:
            target_class = int(np.argmax(ref_probs))

        cortical0 = cortical[0]  # (F, T, S)

        # ---- 2. Generate perturbation masks ------------------------------
        rng = np.random.default_rng(self.seed)
        masks, group_labels = generate_perturbation_masks(
            n_strfs=self.n_strfs,
            n_samples=self.n_samples,
            strategy=self.strategy,
            keep_prob=self.keep_prob,
            sigma=self.sigma,
            sr=self.sr,
            n_groups=self.n_groups,
            include_reference=True,
            rng=rng,
        )

        # ---- 3. Batched decoder probing ----------------------------------
        all_probs = self._run_decoder_batched(cortical0, masks)
        target_probs = all_probs[:, target_class]

        # ---- 4. Distances & kernel weights -------------------------------
        distances = compute_distances(masks, metric=self.distance_metric)
        weights = exponential_kernel(distances, width=self.kernel_width)

        # ---- 5. Fit surrogate --------------------------------------------
        sr_result = fit_surrogate(
            masks, target_probs, weights,
            model_type=self.surrogate_type,
            alpha=self.surrogate_alpha,
        )

        return CorticalLIMEResult(
            target_class=int(target_class),
            target_prob=float(ref_probs[target_class]),
            full_probs=ref_probs.astype(np.float32),
            importances=sr_result.coef,
            importances_std=sr_result.coef_std,
            intercept=sr_result.intercept,
            surrogate_r2=sr_result.r2,
            surrogate_type=self.surrogate_type.value,
            sr=self.sr.astype(np.float32),
            n_strfs=self.n_strfs,
            masks=masks,
            target_probs=target_probs,
            all_probs=all_probs.astype(np.float32),
            distances=distances,
            weights=weights,
            group_labels=group_labels,
            strategy=self.strategy.value,
            kernel_width=self.kernel_width,
            seed=self.seed,
        )

    def explain_batch(
        self,
        waveforms: Sequence[np.ndarray],
        target_classes: Optional[Sequence[int]] = None,
        verbose: bool = True,
    ) -> list[CorticalLIMEResult]:
        """Explain multiple waveforms (convenience wrapper).

        Useful for building per-phoneme importance profiles over a dataset.
        """
        results = []
        targets = target_classes or [None] * len(waveforms)
        for i, (wav, tc) in enumerate(zip(waveforms, targets)):
            if verbose and (i + 1) % 10 == 0:
                print(f"  CorticalLIME: {i+1}/{len(waveforms)}")
            results.append(self.explain(wav, target_class=tc))
        return results


# ═══════════════════════════════════════════════════════════════════════════
# 7.  BASELINE EXPLANATION METHODS (for comparison in the paper)
# ═══════════════════════════════════════════════════════════════════════════

class OcclusionSensitivity:
    """Leave-one-out ablation baseline.

    Zeroes each STRF channel in turn and records the change in P(class).
    Simple, deterministic, no surrogate model — serves as ground truth
    for faithfulness evaluation of CorticalLIME.
    """

    def __init__(self, encode_fn: Callable, decode_fn: Callable, sr: np.ndarray):
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn
        self.sr = np.asarray(sr, dtype=np.float32)
        self.n_strfs = self.sr.shape[0]

    @staticmethod
    def _softmax(x, axis=-1):
        e = np.exp(x - x.max(axis=axis, keepdims=True))
        return e / e.sum(axis=axis, keepdims=True)

    def _logits_to_probs(self, logits):
        if logits.ndim == 4:
            logits = logits.mean(axis=2)
        return self._softmax(logits, axis=-1).mean(axis=1)

    def explain(self, waveform: np.ndarray, target_class: Optional[int] = None) -> dict:
        wav = np.asarray(waveform, dtype=np.float32)[None, :]
        cortical = np.asarray(self.encode_fn(wav))
        ref_logits = np.asarray(self.decode_fn(cortical))
        ref_probs = self._logits_to_probs(ref_logits)[0]
        if target_class is None:
            target_class = int(np.argmax(ref_probs))
        ref_p = ref_probs[target_class]

        cortical0 = cortical[0]
        deltas = np.empty(self.n_strfs, dtype=np.float32)
        for ch in range(self.n_strfs):
            mask = np.ones(self.n_strfs, dtype=np.float32)
            mask[ch] = 0.0
            perturbed = cortical0[None, :, :, :] * mask[None, None, None, :]
            logits = np.asarray(self.decode_fn(perturbed))
            p = self._logits_to_probs(logits)[0, target_class]
            deltas[ch] = ref_p - p  # positive = channel was helping

        return dict(
            target_class=target_class,
            target_prob=float(ref_p),
            importances=deltas,
            sr=self.sr,
        )


class CorticalIntegratedGradients:
    """Integrated Gradients along the STRF channel axis.

    Computes the path integral from a zero-cortical baseline to the actual
    cortical representation, accumulating gradients w.r.t. a per-channel
    scaling vector α ∈ [0,1]^S.

    This is a gradient-based (white-box) method that leverages the full
    differentiability of the JAX model.  It serves as a second comparison
    baseline alongside OcclusionSensitivity.

    Reference: Sundararajan, Taly & Yan (ICML 2017).
    """

    def __init__(
        self,
        model,
        nn_params,
        aud_params,
        n_steps: int = 50,
    ):
        self.model = model
        self.nn_params = nn_params
        self.aud_params = aud_params
        self.n_steps = n_steps
        self.sr = np.asarray(aud_params["sr"], dtype=np.float32)
        self.n_strfs = self.sr.shape[0]

        # Build a non-vmapped base model for conv calls (see make_jax_callables
        # docstring for why the vmapped model's conv cannot be called directly).
        from supervisedSTRF import supervisedSTRF
        base_model = supervisedSTRF(
            n_phones=model.n_phones,
            input_type=model.input_type,
            update_lin=model.update_lin,
            use_class=model.use_class,
            encoder_type=model.encoder_type,
            decoder_type=model.decoder_type,
            compression_method=model.compression_method,
            conv_feats=model.conv_feats,
            pooling_stride=model.pooling_stride,
        )

        # Build the JIT-compiled gradient function.
        # We parameterise the channel mask as a continuous vector α,
        # compute logits(cortical * α), and take ∂logits/∂α.
        @jax.jit
        def _encode(x):
            return model.apply(nn_params, x, aud_params, method=model.encode)

        def _conv_single(x_single):
            return base_model.apply(nn_params, x_single, method=base_model.conv)

        def _class_score(alpha_vec, cortical_single, target_cls):
            """Scalar: mean-over-time softmax probability of target_cls."""
            masked = cortical_single * alpha_vec[None, None, :]  # (F, T, S)
            logits = _conv_single(masked)
            if logits.ndim == 3:
                logits = logits.mean(axis=1)
            # logits: (T, C) or (C,).  Softmax + mean over time.
            probs = jax.nn.softmax(logits, axis=-1)
            if probs.ndim == 2:
                probs = probs.mean(axis=0)
            return probs[target_cls]

        self._encode = _encode
        self._conv_batch = jax.jit(jax.vmap(_conv_single))
        self._grad_fn = jax.jit(jax.grad(_class_score, argnums=0))

    def explain(self, waveform: np.ndarray, target_class: Optional[int] = None) -> dict:
        wav = jnp.asarray(waveform, dtype=jnp.float32)[None, :]
        cortical = self._encode(wav)
        cortical0 = cortical[0]

        if target_class is None:
            logits = self._conv_batch(cortical)
            if logits.ndim == 4:
                logits = logits.mean(axis=2)
            probs = jax.nn.softmax(logits[0], axis=-1).mean(axis=0)
            target_class = int(jnp.argmax(probs))

        # Riemann sum along the straight-line path from 0 to 1 in α-space.
        alphas = jnp.linspace(0.0, 1.0, self.n_steps + 1)[1:]  # exclude 0
        grads_sum = jnp.zeros(self.n_strfs)
        for alpha_val in alphas:
            alpha_vec = jnp.full(self.n_strfs, float(alpha_val))
            g = self._grad_fn(alpha_vec, cortical0, target_class)
            grads_sum = grads_sum + g
        ig = np.asarray(grads_sum / self.n_steps)

        return dict(
            target_class=int(target_class),
            importances=ig.astype(np.float32),
            sr=self.sr,
        )


# ═══════════════════════════════════════════════════════════════════════════
# 8.  JAX MODEL BINDING
# ═══════════════════════════════════════════════════════════════════════════

def detect_n_phones(nn_params) -> int:
    """Auto-detect n_phones from checkpoint weights.

    The CNN decoder's final Dense layer has kernel shape (in_features,
    n_phones + 1).  We find it by scanning the parameter tree for Dense
    kernels and picking the one whose output dimension is the class count.
    """
    from flax.traverse_util import flatten_dict
    flat = flatten_dict(nn_params, sep="/")
    # Look for the last Dense kernel (Dense_0/kernel in the conv sub-tree).
    dense_shapes = {k: v.shape for k, v in flat.items()
                    if "Dense" in k and "kernel" in k}
    if not dense_shapes:
        raise RuntimeError("Cannot detect n_phones: no Dense kernel in checkpoint.")
    # The class-output Dense is the one with the largest output dim.
    key, shape = max(dense_shapes.items(), key=lambda kv: kv[1][-1])
    n_classes = int(shape[-1])   # = n_phones + 1 (includes blank)
    return n_classes - 1


def build_model(nn_params, aud_params):
    """Convenience: build a vSupervisedSTRF with n_phones auto-detected.

    Returns (model, n_phones).
    """
    from supervisedSTRF import vSupervisedSTRF

    n_phones = detect_n_phones(nn_params)

    # Detect update_lin: if 'alpha' key exists in aud_params, LIN was updated.
    update_lin = "alpha" in aud_params

    model = vSupervisedSTRF(
        n_phones=n_phones,
        input_type="audio",
        update_lin=update_lin,
        use_class=False,
        encoder_type="strf",
        decoder_type="cnn",
        compression_method="power",
        conv_feats=[10, 20, 40],
        pooling_stride=2,
    )
    return model, n_phones


def make_jax_callables(model, nn_params, aud_params):
    """Build numpy-in / numpy-out (encode_fn, decode_fn) for a trained
    vSupervisedSTRF.

    The resulting functions are JIT-compiled and gradient-free — the JAX
    equivalent of PyTorch's `torch.no_grad()`.  They are safe to call from
    pure-NumPy LIME code.

    Implementation note
    -------------------
    ``vSupervisedSTRF`` is produced by ``nn.vmap(..., in_axes=(0, None))``,
    which expects **two** positional arguments for every vmapped method.
    ``encode(self, x, params)`` has two args so it works natively, but
    ``conv(self, x)`` has only one — calling it through the vmapped wrapper
    raises ``ValueError: Tuple arity mismatch: 1 != 2``.

    We work around this by importing the non-vmapped base class
    ``supervisedSTRF`` and applying ``jax.vmap`` ourselves for the decoder.
    Since ``variable_axes={'params': None}`` was used in the original vmap,
    the parameter tree structure is identical between the base class and the
    vmapped wrapper.
    """
    from supervisedSTRF import supervisedSTRF

    # Build a non-vmapped twin sharing the same hyper-parameters.
    base_model = supervisedSTRF(
        n_phones=model.n_phones,
        input_type=model.input_type,
        update_lin=model.update_lin,
        use_class=model.use_class,
        encoder_type=model.encoder_type,
        decoder_type=model.decoder_type,
        compression_method=model.compression_method,
        conv_feats=model.conv_feats,
        pooling_stride=model.pooling_stride,
    )

    @jax.jit
    def _encode(x):
        return model.apply(nn_params, x, aud_params, method=model.encode)

    @jax.jit
    def _decode(feats):
        # Manually vmap over the batch dimension because the vmapped model's
        # in_axes=(0, None) arity does not match conv's single argument.
        return jax.vmap(
            lambda x: base_model.apply(nn_params, x, method=base_model.conv)
        )(feats)

    def encode_fn(wav_np):
        return np.asarray(_encode(jnp.asarray(wav_np)))

    def decode_fn(feats_np):
        return np.asarray(_decode(jnp.asarray(feats_np)))

    return encode_fn, decode_fn
