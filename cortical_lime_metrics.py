"""
Faithfulness, stability, and population-level analysis for CorticalLIME.

This module provides the quantitative evaluation toolbox needed for a
NeurIPS-grade analysis of CorticalLIME explanations:

  §1  Deletion / Insertion curves  (Petsiuk et al., BMVC 2018)
  §2  AOPC  (Samek et al., 2017)
  §3  Infidelity  (Yeh et al., NeurIPS 2019)
  §4  Explanation stability / Lipschitz continuity
  §5  Per-phoneme population analysis with statistical tests
  §6  Cross-method agreement (rank correlation between explainers)

All functions operate on plain NumPy arrays and the CorticalLIMEResult
dataclass from cortical_lime.py, making them usable from any notebook or
evaluation script.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
from scipy import stats

# _trapz was removed in NumPy 2.0; use np.trapezoid when available.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


# ═══════════════════════════════════════════════════════════════════════════
# §1  DELETION / INSERTION CURVES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DeletionInsertionResult:
    """Output of a deletion or insertion curve computation."""
    fractions: np.ndarray     # (K+1,) fraction of channels removed/inserted
    probs: np.ndarray         # (K+1,) P(target_class) at each step
    auc: float                # area under the curve (trapezoidal)
    channel_order: np.ndarray # (S,) order in which channels are ablated/added


def deletion_curve(
    cortical_features: np.ndarray,
    importances: np.ndarray,
    decode_fn: Callable,
    target_class: int,
    steps: Optional[int] = None,
) -> DeletionInsertionResult:
    """Progressively zero the most-important STRF channels and track P(class).

    A faithful explanation should cause a *steep* drop — i.e. low AUC.

    Parameters
    ----------
    cortical_features : (F, T, S) single-utterance cortical repr.
    importances : (S,) importance weights (higher = more important).
    decode_fn : callable mapping (B, F, T, S) -> logits.
    target_class : index of the class to track.
    steps : number of deletion steps; default S (one channel at a time).
    """
    S = cortical_features.shape[-1]
    if steps is None:
        steps = S

    order = np.argsort(-importances)            # most important first
    indices = np.round(np.linspace(0, S, steps + 1)).astype(int)

    probs = np.empty(len(indices), dtype=np.float32)
    mask = np.ones(S, dtype=np.float32)

    for i, n_removed in enumerate(indices):
        # Zero out the first n_removed channels in `order`.
        mask[:] = 1.0
        mask[order[:n_removed]] = 0.0
        perturbed = cortical_features[None] * mask[None, None, None, :]
        logits = np.asarray(decode_fn(perturbed))
        p = _logits_to_single_prob(logits, target_class)
        probs[i] = p

    fractions = indices / S
    auc = float(_trapz(probs, fractions))
    return DeletionInsertionResult(fractions, probs, auc, order)


def insertion_curve(
    cortical_features: np.ndarray,
    importances: np.ndarray,
    decode_fn: Callable,
    target_class: int,
    steps: Optional[int] = None,
) -> DeletionInsertionResult:
    """Start from a zero baseline and progressively *insert* the most-important
    channels.  A faithful explanation should cause a *steep* rise — i.e. high AUC.
    """
    S = cortical_features.shape[-1]
    if steps is None:
        steps = S

    order = np.argsort(-importances)
    indices = np.round(np.linspace(0, S, steps + 1)).astype(int)

    probs = np.empty(len(indices), dtype=np.float32)
    mask = np.zeros(S, dtype=np.float32)

    for i, n_inserted in enumerate(indices):
        mask[:] = 0.0
        mask[order[:n_inserted]] = 1.0
        perturbed = cortical_features[None] * mask[None, None, None, :]
        logits = np.asarray(decode_fn(perturbed))
        probs[i] = _logits_to_single_prob(logits, target_class)

    fractions = indices / S
    auc = float(_trapz(probs, fractions))
    return DeletionInsertionResult(fractions, probs, auc, order)


def random_baseline_curves(
    cortical_features: np.ndarray,
    decode_fn: Callable,
    target_class: int,
    n_repeats: int = 20,
    steps: Optional[int] = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean±std deletion and insertion AUCs with random channel ordering.

    Returns (del_aucs, ins_aucs, del_mean_curve, ins_mean_curve).
    """
    S = cortical_features.shape[-1]
    rng = np.random.default_rng(seed)
    del_aucs, ins_aucs = [], []
    del_curves, ins_curves = [], []

    for _ in range(n_repeats):
        rand_imp = rng.random(S).astype(np.float32)
        d = deletion_curve(cortical_features, rand_imp, decode_fn, target_class, steps)
        i_ = insertion_curve(cortical_features, rand_imp, decode_fn, target_class, steps)
        del_aucs.append(d.auc)
        ins_aucs.append(i_.auc)
        del_curves.append(d.probs)
        ins_curves.append(i_.probs)

    return (
        np.array(del_aucs),
        np.array(ins_aucs),
        np.mean(del_curves, axis=0),
        np.mean(ins_curves, axis=0),
    )


# ═══════════════════════════════════════════════════════════════════════════
# §2  AOPC  (Average Over Perturbation Curve)
# ═══════════════════════════════════════════════════════════════════════════

def aopc(
    cortical_features: np.ndarray,
    importances: np.ndarray,
    decode_fn: Callable,
    target_class: int,
    K: int = 10,
) -> float:
    """AOPC_{MoRF} metric (Samek et al., 2017).

    Measures mean probability drop when the top-K most relevant channels
    are progressively removed.

    AOPC = (1 / (K+1)) Σ_{k=0}^{K} [P(class|x) − P(class|x_{MoRF,k})]

    Higher AOPC = more faithful explanation.
    """
    S = cortical_features.shape[-1]
    order = np.argsort(-importances)
    K = min(K, S)

    mask = np.ones(S, dtype=np.float32)
    ref_logits = np.asarray(decode_fn(cortical_features[None]))
    ref_p = _logits_to_single_prob(ref_logits, target_class)

    total = 0.0
    for k in range(K + 1):
        mask[:] = 1.0
        mask[order[:k]] = 0.0
        perturbed = cortical_features[None] * mask[None, None, None, :]
        logits = np.asarray(decode_fn(perturbed))
        p = _logits_to_single_prob(logits, target_class)
        total += (ref_p - p)

    return float(total / (K + 1))


# ═══════════════════════════════════════════════════════════════════════════
# §3  INFIDELITY  (Yeh et al., NeurIPS 2019)
# ═══════════════════════════════════════════════════════════════════════════

def infidelity(
    cortical_features: np.ndarray,
    importances: np.ndarray,
    decode_fn: Callable,
    target_class: int,
    n_samples: int = 200,
    sigma: float = 0.5,
    seed: int = 0,
) -> float:
    """Infidelity metric: E[(φᵀΔx − Δf)²], where Δx is a random
    perturbation direction and Δf is the resulting output change.

    Lower infidelity = more faithful explanation.

    We sample isotropic Gaussian perturbations of the per-channel
    scaling, matching the spirit of the original definition.
    """
    rng = np.random.default_rng(seed)
    S = cortical_features.shape[-1]

    ref_logits = np.asarray(decode_fn(cortical_features[None]))
    ref_p = _logits_to_single_prob(ref_logits, target_class)

    total_sq = 0.0
    for _ in range(n_samples):
        delta = rng.normal(0, sigma, size=S).astype(np.float32)
        mask = np.clip(1.0 - delta, 0.0, None)
        perturbed = cortical_features[None] * mask[None, None, None, :]
        logits = np.asarray(decode_fn(perturbed))
        p = _logits_to_single_prob(logits, target_class)
        delta_f = ref_p - p
        phi_delta = float(np.dot(importances, delta))
        total_sq += (phi_delta - delta_f) ** 2

    return float(total_sq / n_samples)


# ═══════════════════════════════════════════════════════════════════════════
# §4  EXPLANATION STABILITY
# ═══════════════════════════════════════════════════════════════════════════

def explanation_stability(
    explainer,
    waveform: np.ndarray,
    n_runs: int = 10,
    seeds: Optional[Sequence[int]] = None,
) -> dict:
    """Measure how stable CorticalLIME explanations are across random seeds.

    Computes pairwise Spearman rank correlation and L2 distance between
    importance vectors from different PRNG seeds on the *same* input.
    High mean-rho / low mean-L2 = stable.
    """
    if seeds is None:
        seeds = list(range(n_runs))

    original_seed = explainer.seed
    results = []
    for s in seeds:
        explainer.seed = s
        results.append(explainer.explain(waveform))
    explainer.seed = original_seed

    imps = np.stack([r.importances for r in results])  # (n_runs, S)

    # Pairwise Spearman
    rhos = []
    l2s = []
    for i in range(len(imps)):
        for j in range(i + 1, len(imps)):
            rho, _ = stats.spearmanr(imps[i], imps[j])
            rhos.append(rho)
            l2s.append(float(np.linalg.norm(imps[i] - imps[j])))

    return dict(
        mean_spearman=float(np.mean(rhos)),
        std_spearman=float(np.std(rhos)),
        mean_l2=float(np.mean(l2s)),
        std_l2=float(np.std(l2s)),
        all_importances=imps,
        surrogate_r2s=[r.surrogate_r2 for r in results],
    )


def input_sensitivity(
    explainer,
    waveform: np.ndarray,
    noise_levels: Sequence[float] = (1e-4, 1e-3, 1e-2, 5e-2),
    n_repeats: int = 5,
    seed: int = 0,
) -> dict:
    """Measure explanation sensitivity to small input perturbations.

    For each noise level, adds Gaussian noise to the waveform and computes
    Spearman rho with the clean explanation.  An explanation method with
    low sensitivity (high rho even under noise) is more trustworthy.
    """
    rng = np.random.default_rng(seed)
    clean = explainer.explain(waveform)
    records = []

    for eps in noise_levels:
        rhos = []
        for _ in range(n_repeats):
            noisy = waveform + rng.normal(0, eps, size=waveform.shape).astype(np.float32)
            noisy_result = explainer.explain(noisy)
            rho, _ = stats.spearmanr(clean.importances, noisy_result.importances)
            rhos.append(rho)
        records.append(dict(
            noise_level=eps,
            mean_rho=float(np.mean(rhos)),
            std_rho=float(np.std(rhos)),
        ))

    return dict(clean_result=clean, sensitivity=records)


# ═══════════════════════════════════════════════════════════════════════════
# §5  PER-PHONEME POPULATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PhonemeProfile:
    """Aggregated importance profile for a single phoneme class."""
    phoneme: str
    class_idx: int
    n_utterances: int
    mean_importances: np.ndarray    # (S,)
    std_importances: np.ndarray     # (S,)
    se_importances: np.ndarray      # (S,) standard error
    mean_r2: float
    all_importances: np.ndarray     # (n_utterances, S)


def build_phoneme_profiles(
    results: Sequence,
    phoneme_labels: dict[int, str],
) -> dict[str, PhonemeProfile]:
    """Aggregate CorticalLIME results by target phoneme class.

    Parameters
    ----------
    results : list of CorticalLIMEResult
    phoneme_labels : {class_idx: "aa", ...}

    Returns dict keyed by phoneme string.
    """
    from collections import defaultdict
    buckets: dict[int, list] = defaultdict(list)
    for r in results:
        buckets[r.target_class].append(r)

    profiles = {}
    for cls, group in buckets.items():
        phn = phoneme_labels.get(cls, f"cls_{cls}")
        imps = np.stack([r.importances for r in group])
        n = len(group)
        profiles[phn] = PhonemeProfile(
            phoneme=phn,
            class_idx=cls,
            n_utterances=n,
            mean_importances=imps.mean(axis=0),
            std_importances=imps.std(axis=0),
            se_importances=imps.std(axis=0) / np.sqrt(max(n, 1)),
            mean_r2=float(np.mean([r.surrogate_r2 for r in group])),
            all_importances=imps,
        )
    return profiles


def phoneme_family_comparison(
    profiles: dict[str, PhonemeProfile],
    families: dict[str, list[str]],
) -> dict[str, dict]:
    """Compare importance profiles across phoneme families.

    `families` maps family name -> list of phoneme strings, e.g.
    {"stops": ["b", "d", "g", "p", "t", "k"],
     "vowels": ["aa", "ae", "ah", "iy", "uw", ...]}

    For each pair of families, runs a two-sample t-test on the mean
    importance of each STRF channel, with Bonferroni correction.
    """
    family_imps = {}
    for fname, members in families.items():
        all_rows = []
        for phn in members:
            if phn in profiles:
                all_rows.append(profiles[phn].all_importances)
        if all_rows:
            family_imps[fname] = np.concatenate(all_rows, axis=0)

    comparisons = {}
    fnames = list(family_imps.keys())
    for i in range(len(fnames)):
        for j in range(i + 1, len(fnames)):
            fa, fb = fnames[i], fnames[j]
            S = family_imps[fa].shape[1]
            t_vals = np.empty(S)
            p_vals = np.empty(S)
            for ch in range(S):
                t, p = stats.ttest_ind(
                    family_imps[fa][:, ch],
                    family_imps[fb][:, ch],
                    equal_var=False,
                )
                t_vals[ch] = t
                p_vals[ch] = p
            # Bonferroni correction.
            p_corrected = np.minimum(p_vals * S, 1.0)
            significant = p_corrected < 0.05
            comparisons[f"{fa}_vs_{fb}"] = dict(
                t_values=t_vals,
                p_values=p_vals,
                p_corrected=p_corrected,
                n_significant=int(significant.sum()),
                significant_channels=np.where(significant)[0].tolist(),
                effect_sizes=(
                    family_imps[fa].mean(0) - family_imps[fb].mean(0)
                ) / np.sqrt(
                    (family_imps[fa].var(0) + family_imps[fb].var(0)) / 2 + 1e-12
                ),  # Cohen's d
            )
    return comparisons


# ═══════════════════════════════════════════════════════════════════════════
# §6  CROSS-METHOD AGREEMENT
# ═══════════════════════════════════════════════════════════════════════════

def cross_method_agreement(
    importances_a: np.ndarray,
    importances_b: np.ndarray,
) -> dict:
    """Compare two explanation vectors (e.g. CorticalLIME vs Occlusion).

    Returns Spearman rho, Kendall tau, top-k overlap, and sign agreement.
    """
    rho_sp, p_sp = stats.spearmanr(importances_a, importances_b)
    tau, p_tau = stats.kendalltau(importances_a, importances_b)

    top5_a = set(np.argsort(-np.abs(importances_a))[:5])
    top5_b = set(np.argsort(-np.abs(importances_b))[:5])
    top10_a = set(np.argsort(-np.abs(importances_a))[:10])
    top10_b = set(np.argsort(-np.abs(importances_b))[:10])

    sign_agree = float(np.mean(np.sign(importances_a) == np.sign(importances_b)))

    return dict(
        spearman_rho=float(rho_sp),
        spearman_p=float(p_sp),
        kendall_tau=float(tau),
        kendall_p=float(p_tau),
        top5_overlap=len(top5_a & top5_b),
        top10_overlap=len(top10_a & top10_b),
        sign_agreement=sign_agree,
    )


def rank_correlation_matrix(
    method_importances: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    """Spearman rank correlation matrix across multiple explanation methods.

    method_importances: {"CorticalLIME": array(S,), "Occlusion": ..., ...}
    Returns (S×S correlation matrix, ordered method names).
    """
    names = list(method_importances.keys())
    n = len(names)
    mat = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            rho, _ = stats.spearmanr(
                method_importances[names[i]],
                method_importances[names[j]],
            )
            mat[i, j] = mat[j, i] = rho
    return mat, names


# ═══════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _logits_to_single_prob(logits: np.ndarray, target_class: int) -> float:
    """(B, T, [F,] C) -> scalar P(target_class), averaged over time."""
    if logits.ndim == 4:
        logits = logits.mean(axis=2)
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = e / e.sum(axis=-1, keepdims=True)
    return float(probs.mean(axis=1)[0, target_class])
