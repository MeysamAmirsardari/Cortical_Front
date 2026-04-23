"""Mechanistic interpretability figures for the biomimetic auditory frontend.

Given a CorticalLIME run's raw output (``results_raw.npz``) plus several
optional companion artifacts written by ``run.py`` (``hero_cochleagrams.npz``,
``sanity_arrays.npz``, ``manifold.npz``), this module produces five
publication-quality figures:

    1.  ``plot_acoustic_vs_cortical_hero``   – Auditory cochleagram (top)
                                                + 5×n_strfs cortical
                                                saliency heatmap (bottom).
    2.  ``plot_sanity_dashboard``             – Skeptic-proof 2×2 grid:
                                                faithfulness curves, stability
                                                heatmap, fidelity vs confidence,
                                                cross-method ρ histogram.
    3.  ``plot_phonetic_dendrogram``          – Emergent phonetic taxonomy
                                                (L2 + PCA-suppressed Ward
                                                hierarchical clustering on
                                                39 × 330 LIME matrix).
    4.  ``plot_estrf_reconstruction``         – Effective STRF (eSTRF):
                                                analytic Gabor reconstruction
                                                of the 2-D time-frequency
                                                receptive field implied by
                                                the 330 LIME coefficients.
    5.  ``plot_phonetic_manifold_trajectory`` – UMAP / t-SNE projection of
                                                sliding-window LIME vectors
                                                with a directed phoneme path
                                                through a single word.

All figures use serif fonts, removed top/right spines, 400 DPI, and a
colour-blind-friendly (ColorBrewer-inspired) palette, and are saved as PNG.

The pipeline is strictly data-driven — no dummy data is generated. Call
``render_all`` or use the CLI::

    python lingo_analysis.py --results results_raw.npz --outdir figures_lingo/

``render_all`` is also imported by ``run.py`` and invoked automatically at
the end of the full analysis pipeline.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

# =============================================================================
#                               GLOBAL AESTHETICS
# =============================================================================

# ColorBrewer-inspired, colour-blind-friendly qualitative palette.
FAMILY_COLORS: Dict[str, str] = {
    "Vowels":     "#D7263D",   # red
    "Stops":      "#1B4F8B",   # blue
    "Fricatives": "#2A9D3F",   # green
    "Nasals":     "#6A359C",   # purple
}

SEQ_CMAP = "viridis"
DIV_CMAP = "RdBu_r"

# TIMIT-39 phones grouped by manner of articulation (canonical order).
PHONEMES_BY_FAMILY: Dict[str, List[str]] = {
    "Vowels": [
        "iy", "ih", "eh", "ae", "aa", "ah", "uw", "uh", "er",
        "ey", "ay", "oy", "aw", "ow",
    ],
    "Stops":      ["b", "d", "g", "p", "t", "k"],
    "Fricatives": ["s", "sh", "z", "f", "th", "v", "dh", "hh"],
    "Nasals":     ["m", "n", "ng"],
}

# Flattened phone → family lookup.
FAMILY_OF: Dict[str, str] = {
    p: fam for fam, ps in PHONEMES_BY_FAMILY.items() for p in ps
}

# TIMIT 61 → 39 folding (Lee & Hon, 1989), restricted to phones used here.
PHONE_61_TO_39: Dict[str, str] = {
    "aa": "aa", "ae": "ae", "ah": "ah", "ao": "aa", "aw": "aw",
    "ax": "ah", "ax-h": "ah", "axr": "er", "ay": "ay",
    "b": "b", "bcl": "sil", "ch": "ch",
    "d": "d", "dcl": "sil", "dh": "dh", "dx": "dx",
    "eh": "eh", "el": "l", "em": "m", "en": "n", "eng": "ng",
    "epi": "sil", "er": "er", "ey": "ey",
    "f": "f", "g": "g", "gcl": "sil",
    "h#": "sil", "hh": "hh", "hv": "hh",
    "ih": "ih", "ix": "ih", "iy": "iy",
    "jh": "jh", "k": "k", "kcl": "sil",
    "l": "l", "m": "m", "n": "n", "ng": "ng", "nx": "n",
    "ow": "ow", "oy": "oy", "p": "p", "pau": "sil", "pcl": "sil",
    "q": "", "r": "r", "s": "s", "sh": "sh",
    "t": "t", "tcl": "sil", "th": "th",
    "uh": "uh", "uw": "uw", "ux": "uw",
    "v": "v", "w": "w", "y": "y", "z": "z", "zh": "sh",
}

_TIMIT_61 = [
    "aa", "ae", "ah", "ao", "aw", "ax", "ax-h", "axr", "ay",
    "b", "bcl", "ch", "d", "dcl", "dh", "dx",
    "eh", "el", "em", "en", "eng", "epi", "er", "ey",
    "f", "g", "gcl", "h#", "hh", "hv",
    "ih", "ix", "iy",
    "jh", "k", "kcl", "l", "m", "n", "ng", "nx",
    "ow", "oy", "p", "pau", "pcl", "q", "r",
    "s", "sh", "t", "tcl", "th", "uh", "uw", "ux",
    "v", "w", "y", "z", "zh",
]
IDX_TO_PHONE61 = {i + 1: p for i, p in enumerate(_TIMIT_61)}

LINGUISTIC_BAND_NAMES: List[str] = [
    "Base / Voicing", "F1 (Height)", "F2 (Frontness)",
    "F3 (Rhoticity)", "High Freq (Friction)",
]

DEFAULT_BAND_EDGES_HZ: np.ndarray = np.array([
    [0.0, 400.0], [400.0, 1000.0], [1000.0, 2500.0],
    [2500.0, 3500.0], [3500.0, 8000.0],
], dtype=np.float64)


def set_academic_style() -> None:
    """Configure matplotlib with a restrained, publication-grade style."""
    sns.set_theme(context="paper", style="ticks", font="serif")
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.format": "png",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _despine(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# =============================================================================
#                           DATA CONTAINERS / LOADING
# =============================================================================

@dataclass
class LimeResults:
    """Contents of ``results_raw.npz`` from ``run.py``.

    Supports both legacy single-mask mode (S = n_strfs) and the
    high-resolution 2-D band mode (S_total = n_bands × n_strfs).
    """
    importances: np.ndarray            # (N, S_total)
    target_classes: np.ndarray         # (N,) int, 1-indexed TIMIT-61
    target_probs: Optional[np.ndarray] # (N,) target-class probability
    surrogate_r2s: Optional[np.ndarray]
    sr_pairs: np.ndarray               # (n_strfs, 2)  cols: [scale, rate]
    n_bands: int = 1
    band_edges_hz: Optional[np.ndarray] = None   # (n_bands, 2)
    band_names: Optional[List[str]] = None

    @property
    def n_strfs(self) -> int:
        return int(self.sr_pairs.shape[0])

    @property
    def rates(self) -> np.ndarray:
        return self.sr_pairs[:, 1]

    @property
    def scales(self) -> np.ndarray:
        return self.sr_pairs[:, 0]

    @property
    def is_band_mode(self) -> bool:
        return self.n_bands > 1

    def importance_2d(self, i: int) -> np.ndarray:
        """Reshape one row to (n_bands, n_strfs)."""
        if not self.is_band_mode:
            raise ValueError("Not in band mode.")
        return self.importances[i].reshape(self.n_bands, self.n_strfs)

    def collapse_bands(self) -> np.ndarray:
        """Return (N, n_strfs): sum |.| over bands. Pass-through if not band-mode."""
        if not self.is_band_mode:
            return self.importances
        N = self.importances.shape[0]
        out = np.abs(self.importances).reshape(N, self.n_bands, self.n_strfs)
        return out.sum(axis=1)


def load_lime_results(path: str) -> LimeResults:
    with np.load(path, allow_pickle=True) as d:
        importances = np.asarray(d["importances"], dtype=np.float64)
        target_classes = np.asarray(d["target_classes"], dtype=np.int64)
        sr_pairs = np.asarray(d["sr_pairs"], dtype=np.float64)
        n_bands = int(d["n_bands"]) if "n_bands" in d.files else 1
        band_edges = (
            np.asarray(d["band_edges_hz"], dtype=np.float64)
            if "band_edges_hz" in d.files and n_bands > 1 else None
        )
        target_probs = (
            np.asarray(d["target_probs"], dtype=np.float64)
            if "target_probs" in d.files else None
        )
        r2s = (
            np.asarray(d["surrogate_r2s"], dtype=np.float64)
            if "surrogate_r2s" in d.files else None
        )
    band_names = LINGUISTIC_BAND_NAMES[:n_bands] if n_bands > 1 else None
    return LimeResults(
        importances=importances, target_classes=target_classes,
        target_probs=target_probs, surrogate_r2s=r2s,
        sr_pairs=sr_pairs, n_bands=n_bands,
        band_edges_hz=band_edges, band_names=band_names,
    )


def _phone61_idx_to_39(idx: int) -> str:
    p61 = IDX_TO_PHONE61.get(int(idx), "")
    return PHONE_61_TO_39.get(p61, "")


def aggregate_per_phoneme(
    res: LimeResults,
    keep_band_axis: bool = False,
) -> Dict[str, np.ndarray]:
    """Mean absolute importance per TIMIT-39 phoneme."""
    band = keep_band_axis and res.is_band_mode
    bucket: Dict[str, List[np.ndarray]] = {}
    for imp, tc in zip(res.importances, res.target_classes):
        p39 = _phone61_idx_to_39(tc)
        if not p39 or p39 == "sil":
            continue
        a = np.abs(imp)
        if res.is_band_mode:
            a = a.reshape(res.n_bands, res.n_strfs)
            if not band:
                a = a.sum(axis=0)
        bucket.setdefault(p39, []).append(a)
    return {p: np.mean(np.stack(v, 0), 0) for p, v in bucket.items()}


def aggregate_signed_per_phoneme(
    res: LimeResults,
) -> Dict[str, np.ndarray]:
    """Mean *signed* coefficient per phoneme (for eSTRF reconstruction)."""
    bucket: Dict[str, List[np.ndarray]] = {}
    for imp, tc in zip(res.importances, res.target_classes):
        p39 = _phone61_idx_to_39(tc)
        if not p39 or p39 == "sil":
            continue
        bucket.setdefault(p39, []).append(np.asarray(imp, dtype=np.float64))
    return {p: np.mean(np.stack(v, 0), 0) for p, v in bucket.items()}


def _strf_sort_order(sr_pairs: np.ndarray) -> np.ndarray:
    """Group columns visually by (rate, scale)."""
    return np.lexsort((sr_pairs[:, 0], sr_pairs[:, 1]))


# =============================================================================
#               FIGURE 1 — ACOUSTIC vs CORTICAL SALIENCY (HERO)
# =============================================================================

def plot_acoustic_vs_cortical_hero(
    res: LimeResults,
    cochleagrams: Dict[str, Dict],
    phones: Sequence[str] = ("s", "aa", "iy", "t"),
    band_edges_hz: Optional[np.ndarray] = None,
    band_names: Optional[Sequence[str]] = None,
) -> plt.Figure:
    """Figure 1 — Acoustic cochleagram (top) vs 5×n_strfs LIME heatmap (bottom).

    Parameters
    ----------
    cochleagrams : ``{phone: {"cochleagram": (F, T), "freqs_hz": (F,),
                                "duration_s": float, ...}}``  written by
        ``generate_hero_cochleagrams.py``. May optionally include a
        ``"lime_importance"`` (n_bands, n_strfs) field — if present, the
        heatmap is drawn from this *crop-matched* vector rather than from
        the population-mean ``aggregate_per_phoneme`` fallback. May also
        include ``"target_prob"``, ``"surrogate_r2"``, ``"phone_dur_s"``
        and ``"pad_ms"`` for per-panel header annotation.
    phones : ordered list of phones to render (one column each).
    """
    if not res.is_band_mode:
        raise ValueError("Hero figure requires band-mode results.")

    band_edges = (
        np.asarray(band_edges_hz)
        if band_edges_hz is not None
        else (res.band_edges_hz if res.band_edges_hz is not None
              else DEFAULT_BAND_EDGES_HZ[: res.n_bands])
    )
    band_names = list(
        band_names or res.band_names or LINGUISTIC_BAND_NAMES[: res.n_bands]
    )

    phone_means = aggregate_per_phoneme(res, keep_band_axis=True)
    available = [
        p for p in phones
        if p in cochleagrams and (
            "lime_importance" in cochleagrams[p] or p in phone_means
        )
    ]
    if not available:
        raise ValueError(
            f"None of {list(phones)} have both LIME and cochleagram data. "
            f"LIME: {sorted(phone_means)}, "
            f"Cochleagrams: {sorted(cochleagrams)}"
        )

    n = len(available)
    fig, axes = plt.subplots(
        2, n,
        figsize=(4.4 * n, 7.6),
        gridspec_kw={
            "hspace": 0.46, "wspace": 0.18,
            "height_ratios": [1.0, 0.95],
        },
        squeeze=False,
    )

    order = _strf_sort_order(res.sr_pairs)
    rates_sorted = res.rates[order]

    # ── Build per-panel LIME matrices (prefer crop-matched). ──
    L_mats: Dict[str, np.ndarray] = {}
    for p in available:
        coc = cochleagrams[p]
        if "lime_importance" in coc:
            imp = np.abs(np.asarray(coc["lime_importance"], dtype=np.float64))
            if imp.ndim == 1:
                imp = imp.reshape(res.n_bands, res.n_strfs)
        else:
            imp = phone_means[p]
        L_mats[p] = imp[:, order]
    vmax_l = max(M.max() for M in L_mats.values()) + 1e-12

    im_cochlea = None
    im_lime = None
    for c, p in enumerate(available):
        coc = cochleagrams[p]
        S_db = np.asarray(coc["cochleagram"], dtype=np.float64)
        freqs = np.asarray(coc["freqs_hz"], dtype=np.float64)
        dur = float(coc.get("duration_s", S_db.shape[1] / 100.0))
        phone_dur = coc.get("phone_dur_s", None)
        pad_ms = coc.get("pad_ms", None)
        prob = coc.get("target_prob", None)
        r2 = coc.get("surrogate_r2", None)

        fam = FAMILY_OF.get(p, None)
        col = FAMILY_COLORS.get(fam, "black")

        # ── TOP: Auditory cochleagram with band guides ──
        ax_top = axes[0][c]
        im_cochlea = ax_top.imshow(
            S_db, origin="lower", aspect="auto", cmap="magma",
            extent=[0.0, dur * 1000.0,
                    float(freqs.min()), float(freqs.max())],
        )
        ax_top.set_yscale("log")
        ax_top.set_yticks([125, 250, 500, 1000, 2000, 4000, 8000])
        ax_top.get_yaxis().set_major_formatter(
            mpl.ticker.FuncFormatter(lambda x, _: f"{int(x)}")
        )
        ax_top.minorticks_off()
        # Frequency band guides + (only on the first column) right-edge labels.
        f_min_show = float(freqs.min())
        f_max_show = float(freqs.max())
        for bi, ((lo, hi), name) in enumerate(zip(band_edges, band_names)):
            if 0.0 < lo <= f_max_show:
                ax_top.axhline(lo, color="white", lw=0.7, ls="--", alpha=0.55)
            if 0.0 < hi <= f_max_show:
                ax_top.axhline(hi, color="white", lw=0.7, ls="--", alpha=0.55)
            if c == n - 1:
                lo_eff = max(lo, f_min_show * 1.02)
                hi_eff = min(hi, f_max_show / 1.02)
                if hi_eff > lo_eff:
                    mid_y = np.sqrt(lo_eff * hi_eff)
                    ax_top.text(
                        dur * 1000.0 * 1.015, mid_y, name,
                        ha="left", va="center",
                        color="#222222", fontsize=7.5,
                        bbox=dict(boxstyle="round,pad=0.18",
                                  fc="white", ec="#cccccc", lw=0.5,
                                  alpha=0.95),
                    )
        ax_top.set_xlabel("Time (ms)", labelpad=2)
        if c == 0:
            ax_top.set_ylabel("Frequency (Hz)")
        # Title with phone in family colour, duration in subtitle.
        sub = []
        if phone_dur is not None:
            sub.append(f"{float(phone_dur) * 1000.0:.0f} ms")
        if pad_ms is not None:
            sub.append(f"\u00B1{float(pad_ms):.0f} ms pad")
        subtitle = "   ".join(sub)
        ax_top.set_title(
            f"/{p}/", color=col, fontsize=15, fontweight="bold", pad=6,
        )
        if subtitle:
            ax_top.text(
                0.5, 1.012, subtitle, transform=ax_top.transAxes,
                ha="center", va="bottom", fontsize=8.5, color="#444444",
                style="italic",
            )
        _despine(ax_top)

        # ── BOTTOM: 5×n_strfs cortical-saliency heatmap ──
        ax_bot = axes[1][c]
        M = L_mats[p] / vmax_l
        im_lime = ax_bot.imshow(
            M, origin="upper", aspect="auto", cmap=SEQ_CMAP,
            vmin=0.0, vmax=1.0,
        )
        ax_bot.set_yticks(np.arange(res.n_bands))
        ax_bot.set_yticklabels(
            band_names if c == 0 else [""] * res.n_bands, fontsize=8.5,
        )
        # Light vertical guides at rate transitions.
        last = None
        for j, rt in enumerate(rates_sorted):
            if last is not None and not np.isclose(rt, last):
                ax_bot.axvline(j - 0.5, color="white", lw=0.35, alpha=0.35)
            last = rt
        ax_bot.set_xticks([])
        ax_bot.set_xlabel(
            r"STRF channels  (rate $\omega \uparrow$, scale $\Omega$ within)",
            labelpad=2,
        )
        # Per-panel header: P(class) and surrogate R² when available.
        head_bits = []
        if prob is not None:
            head_bits.append(f"P={float(prob):.2f}")
        if r2 is not None:
            head_bits.append(f"R\u00B2={float(r2):.2f}")
        head = "   ".join(head_bits) if head_bits else ""
        ax_bot.set_title(
            "cortical saliency" + (f"   ({head})" if head else ""),
            color="#333333", fontsize=10, pad=4,
        )
        _despine(ax_bot)

    # ── Colour bars (compact, on the right) ──
    cbar_top = fig.colorbar(
        im_cochlea, ax=axes[0, :].tolist(),
        orientation="vertical", shrink=0.78, pad=0.025, aspect=18,
    )
    cbar_top.set_label("Power (dB)", fontsize=9)
    cbar_top.ax.tick_params(labelsize=8)
    cbar_top.outline.set_visible(False)

    cbar_bot = fig.colorbar(
        im_lime, ax=axes[1, :].tolist(),
        orientation="vertical", shrink=0.78, pad=0.025, aspect=18,
    )
    cbar_bot.set_label(r"Normalised $|$importance$|$", fontsize=9)
    cbar_bot.ax.tick_params(labelsize=8)
    cbar_bot.outline.set_visible(False)

    fig.suptitle(
        "Acoustic vs. Cortical Saliency",
        fontsize=15, fontweight="bold", y=0.995,
    )
    fig.text(
        0.5, 0.955,
        "Per-phoneme cochleagram (top) and crop-matched CorticalLIME "
        "weights across linguistic bands × STRF channels (bottom)",
        ha="center", va="top", fontsize=10, style="italic", color="#444444",
    )
    return fig


# =============================================================================
#                FIGURE 2 — SKEPTIC-PROOF SANITY DASHBOARD (2×2)
# =============================================================================

def plot_sanity_dashboard(arrs: Dict[str, np.ndarray]) -> plt.Figure:
    """Figure 2 — 2×2 sanity dashboard, driven by ``sanity_arrays.npz``.

    Expected keys:
      faith_steps           : (K,)            deletion fraction (0…1)
      faith_lime_curves     : (n_utt, K)      P(class) under LIME deletion
      faith_random_curves   : (n_utt, K)      P(class) under random deletion
      stability_matrix      : (n_seeds, n_seeds) Spearman ρ across seeds
      fidelity_r2           : (n_utt,)        surrogate R²
      fidelity_prob         : (n_utt,)        target-class P(class)
      crossmethod_rhos      : (M,)            Spearman ρ LIME vs Occlusion
    """
    fig, ax = plt.subplots(2, 2, figsize=(11.4, 8.4),
                           gridspec_kw={"hspace": 0.36, "wspace": 0.28})

    # ── Panel A: Faithfulness curves ──
    a = ax[0, 0]
    if {"faith_steps", "faith_lime_curves", "faith_random_curves"} <= arrs.keys():
        steps = np.asarray(arrs["faith_steps"], dtype=np.float64)
        L = np.asarray(arrs["faith_lime_curves"], dtype=np.float64)
        R = np.asarray(arrs["faith_random_curves"], dtype=np.float64)
        m_l, s_l = L.mean(0), L.std(0)
        m_r, s_r = R.mean(0), R.std(0)
        a.plot(steps, m_l, color="#C0392B", lw=2.0, marker="o", ms=4,
               label="CorticalLIME (top-k)")
        a.fill_between(steps, m_l - s_l, m_l + s_l, color="#C0392B", alpha=0.18)
        a.plot(steps, m_r, color="#7F8C8D", lw=2.0, marker="s", ms=4,
               ls="--", label="Random baseline")
        a.fill_between(steps, m_r - s_r, m_r + s_r, color="#7F8C8D", alpha=0.18)
        a.set(xlabel="Fraction of channels deleted", ylabel=r"P(target class)",
              title="(A) Faithfulness — Deletion curve")
        a.legend(loc="upper right")
        a.grid(True, axis="y", ls=":", lw=0.6, alpha=0.5)
    else:
        a.text(0.5, 0.5, "Faithfulness arrays missing",
               ha="center", va="center", transform=a.transAxes)
        a.set_axis_off()
    _despine(a)

    # ── Panel B: Stability heatmap ──
    b = ax[0, 1]
    if "stability_matrix" in arrs:
        M = np.asarray(arrs["stability_matrix"], dtype=np.float64)
        im = b.imshow(M, cmap=DIV_CMAP, vmin=-1.0, vmax=1.0)
        b.set(xlabel="Seed", ylabel="Seed",
              title=f"(B) Stability — Spearman ρ across {M.shape[0]} seeds  "
                    f"(mean={M[np.triu_indices_from(M, 1)].mean():.3f})")
        cb = fig.colorbar(im, ax=b, shrink=0.85, pad=0.02, aspect=18)
        cb.set_label(r"$\rho$")
        cb.outline.set_visible(False)
    else:
        b.text(0.5, 0.5, "Stability matrix missing",
               ha="center", va="center", transform=b.transAxes)
        b.set_axis_off()

    # ── Panel C: Fidelity vs Confidence ──
    c = ax[1, 0]
    if "fidelity_r2" in arrs and "fidelity_prob" in arrs:
        r2 = np.asarray(arrs["fidelity_r2"], dtype=np.float64)
        pr = np.asarray(arrs["fidelity_prob"], dtype=np.float64)
        c.scatter(pr, r2, s=22, alpha=0.55, color="#2980B9",
                  edgecolors="white", lw=0.4)
        # OLS line for visual guidance.
        if len(r2) >= 3:
            slope, intercept = np.polyfit(pr, r2, 1)
            xs = np.linspace(pr.min(), pr.max(), 50)
            c.plot(xs, slope * xs + intercept, color="#1B4F8B", lw=1.2, ls="--")
        c.set(xlabel="P(target class)", ylabel=r"Surrogate $R^{2}$",
              title="(C) Fidelity vs. Confidence",
              ylim=(min(0.0, r2.min() - 0.05), 1.02))
        c.grid(True, ls=":", lw=0.6, alpha=0.5)
    else:
        c.text(0.5, 0.5, "Fidelity arrays missing",
               ha="center", va="center", transform=c.transAxes)
        c.set_axis_off()
    _despine(c)

    # ── Panel D: Cross-method agreement ──
    d = ax[1, 1]
    if "crossmethod_rhos" in arrs:
        rhos = np.asarray(arrs["crossmethod_rhos"], dtype=np.float64)
        d.hist(rhos, bins=22, color="#6A359C", edgecolor="white", lw=0.4)
        m = float(np.nanmean(rhos))
        d.axvline(m, color="black", lw=1.4, ls="--",
                  label=f"mean={m:.3f}")
        d.axvline(0, color="grey", lw=0.8, ls=":")
        d.set(xlabel=r"Spearman $\rho$  (LIME vs. Occlusion)", ylabel="Count",
              title=f"(D) Cross-Method Agreement  (n={len(rhos)})")
        d.legend(loc="upper left")
        d.grid(True, axis="y", ls=":", lw=0.6, alpha=0.5)
    else:
        d.text(0.5, 0.5, "Cross-method ρ missing",
               ha="center", va="center", transform=d.transAxes)
        d.set_axis_off()
    _despine(d)

    fig.suptitle(
        "Skeptic-Proof Sanity Dashboard:\n"
        "Faithfulness · Stability · Fidelity · Cross-Method Agreement",
        fontsize=14, fontweight="bold", y=1.005,
    )
    return fig


# =============================================================================
#               FIGURE 3 — EMERGENT PHONETIC TAXONOMY (DENDROGRAM)
# =============================================================================

def _l2_normalise_rows(M: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return M / norms


def _pca_project(M: np.ndarray, k: int) -> np.ndarray:
    """Top-k PCA on the row-mean-centred matrix (no sklearn dep)."""
    Mc = M - M.mean(axis=0, keepdims=True)
    # Use SVD on the centred matrix; keep top-k right-singular components.
    U, S, Vt = np.linalg.svd(Mc, full_matrices=False)
    k = min(k, S.size)
    # rows projected onto top-k principal axes
    return U[:, :k] * S[:k]


def _link_color_map(
    Z: np.ndarray, families: Sequence[str], family_colors: Dict[str, str],
    default: str = "#8A8A8A",
) -> Dict[int, str]:
    n = len(families)
    leaves_under: Dict[int, set] = {i: {i} for i in range(n)}
    for i, row in enumerate(Z):
        a, b = int(row[0]), int(row[1])
        leaves_under[n + i] = leaves_under[a] | leaves_under[b]
    colors: Dict[int, str] = {}
    for node_id in range(n, 2 * n - 1):
        fams = {families[idx] for idx in leaves_under[node_id]}
        if len(fams) == 1:
            colors[node_id] = family_colors[next(iter(fams))]
        else:
            colors[node_id] = default
    return colors


def _ordered_phone_weight_matrix(
    phone_means: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, List[str], List[str]]:
    phones, families, rows = [], [], []
    for fam in ["Vowels", "Stops", "Fricatives", "Nasals"]:
        for p in PHONEMES_BY_FAMILY[fam]:
            if p in phone_means:
                phones.append(p)
                families.append(fam)
                rows.append(phone_means[p].ravel())
    if not rows:
        raise ValueError("No TIMIT-39 phones found in results.")
    return np.vstack(rows), phones, families


def plot_phonetic_dendrogram(
    res: LimeResults,
    phone_means: Optional[Dict[str, np.ndarray]] = None,
    n_pca: int = 20,
    method: str = "ward",
    metric: str = "euclidean",
) -> plt.Figure:
    """Figure 3 — Emergent phonetic taxonomy.

    Critical preprocessing (per the user's spec):
      1. Use the full (n_phones × 330) band-mode LIME matrix.
      2. L2-normalise rows to remove energy bias.
      3. PCA-reduce to ``n_pca=20`` components to suppress noise.
      4. Ward linkage on the reduced matrix.

    Branch colours and leaf labels are painted with the manner-class
    palette (strict per-subtree purity rule).
    """
    if phone_means is None:
        # Use band-aware aggregation: keep_band_axis=True if available so the
        # full 330-dim representation is used; else fall back to (n_strfs,).
        phone_means = aggregate_per_phoneme(res, keep_band_axis=res.is_band_mode)

    W, phones, families = _ordered_phone_weight_matrix(phone_means)
    W = _l2_normalise_rows(W)
    if n_pca and n_pca < min(W.shape):
        W_red = _pca_project(W, k=n_pca)
    else:
        W_red = W
    Z = linkage(W_red, method=method, metric=metric)
    link_colors = _link_color_map(Z, families, FAMILY_COLORS)

    fig, ax = plt.subplots(figsize=(7.8, 9.4))
    dendrogram(
        Z,
        orientation="right",
        labels=phones,
        leaf_font_size=10,
        link_color_func=lambda k: link_colors.get(k, "#8A8A8A"),
        above_threshold_color="#8A8A8A",
        ax=ax,
    )
    fam_of = dict(zip(phones, families))
    for lbl in ax.get_ymajorticklabels():
        p = lbl.get_text()
        lbl.set_color(FAMILY_COLORS[fam_of[p]])
        lbl.set_fontweight("bold")

    ax.set_xlabel("Ward linkage distance  (L2 + PCA-{})".format(n_pca))
    ax.set_title(
        "Emergent Phonetic Taxonomy:\n"
        "Hierarchical Clustering of L2 + PCA Cortical-Saliency Vectors",
        pad=14,
    )
    _despine(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(True, axis="x", ls=":", lw=0.6, alpha=0.55)

    # Sonorant / Obstruent annotations.
    ordered_phones = [t.get_text() for t in ax.get_ymajorticklabels()]
    ordered_fams = [fam_of[p] for p in ordered_phones]
    sonorant = {"Vowels", "Nasals"}
    obstruent = {"Stops", "Fricatives"}

    def _yrange(target: set) -> Optional[Tuple[float, float]]:
        idxs = [i for i, f in enumerate(ordered_fams) if f in target]
        if not idxs:
            return None
        return (10 * (min(idxs) + 0.5) - 4, 10 * (max(idxs) + 0.5) + 4)

    x_max = float(np.max(Z[:, 2]))
    x_anno = x_max * 1.03
    ax.set_xlim(right=x_max * 1.22)
    for label, target in [
        ("Sonorants\n(Vowels + Nasals)", sonorant),
        ("Obstruents\n(Stops + Fricatives)", obstruent),
    ]:
        yr = _yrange(target)
        if yr is None:
            continue
        y0, y1 = yr
        ax.annotate(
            "",
            xy=(x_anno, y0), xytext=(x_anno, y1),
            arrowprops=dict(arrowstyle="-", lw=1.6, color="#3A3A3A"),
            annotation_clip=False,
        )
        ax.text(x_anno + (x_max * 0.03), (y0 + y1) / 2, label,
                va="center", ha="left", fontsize=10, fontweight="bold",
                color="#3A3A3A")

    handles = [
        plt.Line2D([0], [0], color=FAMILY_COLORS[f], lw=3, label=f)
        for f in ["Vowels", "Stops", "Fricatives", "Nasals"]
    ]
    ax.legend(handles=handles, loc="lower right", title="Manner class",
              title_fontsize=9)
    fig.tight_layout()
    return fig


# =============================================================================
#                FIGURE 4 — EFFECTIVE STRF (eSTRF RECONSTRUCTION)
# =============================================================================

def _gabor_kernel(
    omega: float, Omega: float,
    t: np.ndarray, f_oct: np.ndarray,
    sigma_t_s: float = 0.15, sigma_f_oct: float = 0.6,
) -> np.ndarray:
    """A 2-D Gabor STRF parameterised by temporal rate ω (Hz) and spectral
    scale Ω (cyc/oct). Time t in seconds; f_oct in octaves relative to base.

    Returns a real-valued (n_t, n_f) kernel.
    """
    env_t = np.exp(-(t / sigma_t_s) ** 2 / 2.0)
    env_f = np.exp(-(f_oct / sigma_f_oct) ** 2 / 2.0)
    carrier_t = np.cos(2.0 * np.pi * omega * t)
    carrier_f = np.cos(2.0 * np.pi * Omega * f_oct)
    return np.outer(env_t * carrier_t, env_f * carrier_f)


def _band_freq_centers_oct(
    band_edges_hz: np.ndarray, base_hz: float = 125.0,
) -> np.ndarray:
    """Geometric-mean centre of each band, in octaves above ``base_hz``."""
    centres = np.sqrt(np.maximum(band_edges_hz[:, 0], 1.0) * band_edges_hz[:, 1])
    return np.log2(centres / base_hz)


def reconstruct_estrf(
    coefs: np.ndarray,
    sr_pairs: np.ndarray,
    band_edges_hz: np.ndarray,
    n_t: int = 192,
    n_f: int = 128,
    t_window_s: float = 0.30,
    f_range_oct: Tuple[float, float] = (-1.0, 6.0),
    sigma_t_s: float = 0.10,
    sigma_f_oct: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct a time-frequency receptive field from band-mode coeffs.

    The 330-dim coefficient vector is reshaped to (n_bands, n_strfs).
    For each (band b, strf k) we synthesise a 2-D Gabor STRF centred at
    the band's geometric-mean frequency (in octaves) and at t = 0, with
    rate / scale (ω_k, Ω_k). The eSTRF is then the linear combination of
    these Gabor kernels weighted by the LIME coefficients.

    Returns
    -------
    estrf  : (n_t, n_f) time × frequency  (real-valued)
    t      : (n_t,) seconds
    f_oct  : (n_f,) octaves above the base of the lowest band
    """
    n_bands, n_strfs = band_edges_hz.shape[0], sr_pairs.shape[0]
    coefs_2d = np.asarray(coefs, dtype=np.float64).reshape(n_bands, n_strfs)
    band_centres = _band_freq_centers_oct(band_edges_hz)

    t = np.linspace(-t_window_s / 2.0, t_window_s / 2.0, n_t)
    f_oct = np.linspace(f_range_oct[0], f_range_oct[1], n_f)

    estrf = np.zeros((n_t, n_f), dtype=np.float64)
    for b in range(n_bands):
        for k in range(n_strfs):
            w = coefs_2d[b, k]
            if w == 0.0:
                continue
            Omega = float(sr_pairs[k, 0])
            omega = float(sr_pairs[k, 1])
            # Place Gabor centred at the band's frequency:
            kernel = _gabor_kernel(
                omega, Omega,
                t, f_oct - band_centres[b],
                sigma_t_s=sigma_t_s, sigma_f_oct=sigma_f_oct,
            )
            estrf += w * kernel
    return estrf, t, f_oct


def plot_estrf_reconstruction(
    res: LimeResults,
    phones: Sequence[str] = ("s", "aa", "iy", "t", "f", "m"),
    cols: int = 3,
) -> plt.Figure:
    """Figure 4 — Effective STRF (eSTRF) reconstructions.

    For each phoneme, sums the analytic Gabor basis functions of the
    frontend's STRF channels weighted by the *signed* mean LIME
    coefficients. The result is a 2-D time × frequency receptive field
    showing exactly what acoustic pattern the model is "looking for".
    """
    if not res.is_band_mode:
        raise ValueError("eSTRF reconstruction requires band-mode results.")

    band_edges = (
        res.band_edges_hz if res.band_edges_hz is not None
        else DEFAULT_BAND_EDGES_HZ[: res.n_bands]
    )

    signed = aggregate_signed_per_phoneme(res)
    available = [p for p in phones if p in signed]
    if not available:
        raise ValueError(f"None of {phones} present.")

    # Reconstruct all eSTRFs and find a joint colour scale.
    panels: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    vmax = 0.0
    for p in available:
        estrf, t, f_oct = reconstruct_estrf(signed[p], res.sr_pairs, band_edges)
        panels[p] = (estrf, t, f_oct)
        vmax = max(vmax, float(np.max(np.abs(estrf))))
    vmax = vmax + 1e-12

    n = len(available)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(4.4 * cols, 3.4 * rows),
        squeeze=False,
    )
    base_hz = max(1.0, float(band_edges[0, 0])) if band_edges[0, 0] > 0 else 125.0

    im = None
    for k, p in enumerate(available):
        r, c = divmod(k, cols)
        ax = axes[r][c]
        estrf, t, f_oct = panels[p]
        im = ax.imshow(
            estrf.T, origin="lower", aspect="auto", cmap=DIV_CMAP,
            vmin=-vmax, vmax=vmax,
            extent=[t[0] * 1000.0, t[-1] * 1000.0,
                    float(f_oct[0]), float(f_oct[-1])],
        )
        ax.axvline(0, color="black", lw=0.6, ls=":")
        # Mark band edges on the spectral axis.
        for (lo, hi) in band_edges:
            for edge in (lo, hi):
                if edge <= 0:
                    continue
                y = float(np.log2(edge / base_hz))
                if f_oct[0] <= y <= f_oct[-1]:
                    ax.axhline(y, color="black", lw=0.4, ls="--", alpha=0.4)
        ax.set_xlabel("Time relative to centre (ms)")
        if c == 0:
            # Map octave ticks back to Hz for readability.
            yt_oct = ax.get_yticks()
            yt_hz = base_hz * (2.0 ** yt_oct)
            ax.set_yticks(yt_oct)
            ax.set_yticklabels([f"{int(round(h))}" for h in yt_hz])
            ax.set_ylabel("Frequency (Hz)")
        fam = FAMILY_OF.get(p, None)
        col = FAMILY_COLORS.get(fam, "black")
        ax.set_title(f"/{p}/  eSTRF", color=col, fontsize=11)
        _despine(ax)

    for k in range(n, rows * cols):
        r, c = divmod(k, cols)
        axes[r][c].axis("off")

    fig.suptitle(
        "Effective STRF (eSTRF) Reconstruction:\n"
        "Acoustic patterns the model is tuned to, by phoneme",
        fontsize=14, fontweight="bold", y=1.005,
    )
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(),
                        shrink=0.85, pad=0.02, aspect=30)
    cbar.set_label(r"Signed weight  ($w_{b,k}$ · Gabor)")
    cbar.outline.set_visible(False)
    return fig


# =============================================================================
#         FIGURE 5 — PHONETIC MANIFOLD & WORD TRAJECTORY (UMAP / t-SNE)
# =============================================================================

def _embed_2d(X: np.ndarray, method: str = "umap",
              random_state: int = 0) -> np.ndarray:
    """2-D embedding of (N, D). Tries UMAP first; falls back to t-SNE
    (sklearn) and finally to truncated SVD."""
    if method == "umap":
        try:
            import umap  # type: ignore
            return umap.UMAP(
                n_components=2, random_state=random_state,
                n_neighbors=min(15, max(5, X.shape[0] // 5)),
                min_dist=0.15, metric="cosine",
            ).fit_transform(X)
        except Exception:
            method = "tsne"
    if method == "tsne":
        try:
            from sklearn.manifold import TSNE
            perp = max(5, min(30, X.shape[0] // 5))
            return TSNE(
                n_components=2, perplexity=perp,
                init="pca", random_state=random_state, metric="cosine",
            ).fit_transform(X)
        except Exception:
            pass
    # SVD fallback.
    Mc = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Mc, full_matrices=False)
    return Mc @ Vt[:2].T


def plot_phonetic_manifold_trajectory(
    manifold: Dict,
    method: str = "umap",
    random_state: int = 0,
) -> plt.Figure:
    """Figure 5 — UMAP/t-SNE manifold of sliding-window LIME vectors.

    ``manifold`` is loaded from ``manifold.npz`` and contains:
      background_imps   : (M, S_total) LIME vectors of full utterances
      background_phones : (M,) TIMIT-39 phone strings
      window_imps       : (W, S_total) sliding-window LIME on a single word
      window_phones     : (W,) per-window dominant phone strings
      word              : str
    """
    bg_imp = np.asarray(manifold["background_imps"], dtype=np.float64)
    bg_phn = [str(p) for p in manifold["background_phones"]]
    win_imp = np.asarray(manifold["window_imps"], dtype=np.float64)
    win_phn = [str(p) for p in manifold["window_phones"]]
    word = str(manifold.get("word", "?"))

    X = np.vstack([bg_imp, win_imp])
    # L2-normalise to focus on direction (phonetic identity), not magnitude.
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.where(norms < 1e-12, 1.0, norms)
    Y = _embed_2d(X, method=method, random_state=random_state)
    Y_bg, Y_w = Y[: len(bg_imp)], Y[len(bg_imp):]

    fig, ax = plt.subplots(figsize=(8.4, 7.0))
    # ── Background scatter coloured by manner family. ──
    fam_bg = [FAMILY_OF.get(p, None) for p in bg_phn]
    for fam, colour in FAMILY_COLORS.items():
        m = np.array([f == fam for f in fam_bg], dtype=bool)
        if not m.any():
            continue
        ax.scatter(
            Y_bg[m, 0], Y_bg[m, 1],
            s=22, alpha=0.55, color=colour, label=fam,
            edgecolors="white", linewidths=0.3, zorder=2,
        )
    others = np.array([f is None for f in fam_bg], dtype=bool)
    if others.any():
        ax.scatter(
            Y_bg[others, 0], Y_bg[others, 1],
            s=14, alpha=0.25, color="#999999", label="Other / silence",
            edgecolors="white", linewidths=0.25, zorder=1,
        )

    # ── Word trajectory (arrows). ──
    if len(Y_w) >= 2:
        for i in range(len(Y_w) - 1):
            ax.annotate(
                "",
                xy=(Y_w[i + 1, 0], Y_w[i + 1, 1]),
                xytext=(Y_w[i, 0], Y_w[i, 1]),
                arrowprops=dict(
                    arrowstyle="->,head_width=0.36,head_length=0.6",
                    color="black", lw=1.4, alpha=0.95,
                ),
                zorder=4,
            )
        ax.scatter(
            Y_w[:, 0], Y_w[:, 1],
            s=70, facecolors="white",
            edgecolors=[FAMILY_COLORS.get(FAMILY_OF.get(p, ""), "black")
                        for p in win_phn],
            linewidths=1.6, zorder=5,
        )
        # Phoneme labels on transitions.
        seen_idx: set = set()
        last = None
        for i, p in enumerate(win_phn):
            if p == last:
                continue
            seen_idx.add(i); last = p
        for i in sorted(seen_idx):
            ax.annotate(
                f"/{win_phn[i]}/", xy=(Y_w[i, 0], Y_w[i, 1]),
                xytext=(6, 6), textcoords="offset points",
                fontsize=10, fontweight="bold", color="black",
                bbox=dict(boxstyle="round,pad=0.18",
                          fc="white", ec="black", alpha=0.85),
                zorder=6,
            )

    ax.set_xlabel(f"{method.upper()}-1")
    ax.set_ylabel(f"{method.upper()}-2")
    ax.set_title(
        f'Phonetic Manifold & Word Trajectory: "{word}"\n'
        f"Sliding-window CorticalLIME → 2-D embedding",
        pad=12,
    )
    ax.legend(loc="best", title="Manner class")
    _despine(ax)
    fig.tight_layout()
    return fig


# =============================================================================
#                  ARTIFACT LOADERS (cochleagrams / sanity / manifold)
# =============================================================================

def load_cochleagrams(path: str) -> Dict[str, Dict]:
    """Load ``hero_cochleagrams.npz``.

    Required per-phone keys: ``__cochleagram``, ``__freqs_hz``, ``__duration_s``.
    Optional per-phone keys: ``__lime_importance`` (n_bands, n_strfs),
    ``__phone_dur_s``, ``__pad_ms``, ``__target_prob``, ``__surrogate_r2``,
    ``__utterance_id``, ``__phone_label``.
    """
    optional = (
        "lime_importance", "phone_dur_s", "pad_ms",
        "target_prob", "surrogate_r2", "utterance_id", "phone_label",
    )
    with np.load(path, allow_pickle=True) as d:
        files = set(d.files)
        keys = [k for k in d.files if k.endswith("__cochleagram")]
        out: Dict[str, Dict] = {}
        for k in keys:
            phn = k.replace("__cochleagram", "")
            entry: Dict = {
                "cochleagram": np.asarray(d[k]),
                "freqs_hz": np.asarray(d[f"{phn}__freqs_hz"]),
                "duration_s": float(d[f"{phn}__duration_s"]),
            }
            for opt in optional:
                fk = f"{phn}__{opt}"
                if fk in files:
                    val = d[fk]
                    if opt == "lime_importance":
                        entry[opt] = np.asarray(val, dtype=np.float64)
                    elif opt in ("phone_dur_s", "pad_ms",
                                 "target_prob", "surrogate_r2"):
                        entry[opt] = float(np.asarray(val).item())
                    else:
                        entry[opt] = str(np.asarray(val).item())
            out[phn] = entry
    return out


def load_sanity_arrays(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as d:
        return {k: np.asarray(d[k]) for k in d.files}


def load_manifold(path: str) -> Dict:
    with np.load(path, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


# =============================================================================
#                             TOP-LEVEL RENDERING
# =============================================================================

def render_all(
    results_path: str,
    cochleagrams_path: Optional[str] = None,
    sanity_path: Optional[str] = None,
    manifold_path: Optional[str] = None,
    outdir: str = "figures_lingo",
    *,
    hero_phones: Sequence[str] = ("s", "aa", "iy", "t"),
    estrf_phones: Sequence[str] = ("s", "aa", "iy", "t", "f", "m"),
    embed_method: str = "umap",
) -> Dict[str, str]:
    """Produce the 5 publication figures.

    All companion artifact paths are optional: if a file is missing, the
    corresponding figure is skipped (with a printed warning) rather than
    raising. ``results_raw.npz`` is the only required input.
    """
    set_academic_style()
    os.makedirs(outdir, exist_ok=True)
    written: Dict[str, str] = {}

    res = load_lime_results(results_path)

    # ── Figure 1: Hero (cochleagram + 5×n_strfs heatmap) ──
    if cochleagrams_path and os.path.exists(cochleagrams_path) and res.is_band_mode:
        try:
            cochs = load_cochleagrams(cochleagrams_path)
            fig1 = plot_acoustic_vs_cortical_hero(
                res, cochs, phones=hero_phones,
            )
            p1 = os.path.join(outdir, "fig1_acoustic_vs_cortical_hero.png")
            fig1.savefig(p1); plt.close(fig1); written["fig1"] = p1
        except Exception as e:
            print(f"[lingo] fig1 skipped: {e}")
    else:
        print("[lingo] fig1 skipped (cochleagrams unavailable or not band-mode)")

    # ── Figure 2: Sanity dashboard ──
    if sanity_path and os.path.exists(sanity_path):
        try:
            arrs = load_sanity_arrays(sanity_path)
            fig2 = plot_sanity_dashboard(arrs)
            p2 = os.path.join(outdir, "fig2_sanity_dashboard.png")
            fig2.savefig(p2); plt.close(fig2); written["fig2"] = p2
        except Exception as e:
            print(f"[lingo] fig2 skipped: {e}")
    else:
        print("[lingo] fig2 skipped (sanity_arrays.npz unavailable)")

    # ── Figure 3: Phonetic dendrogram ──
    try:
        fig3 = plot_phonetic_dendrogram(res)
        p3 = os.path.join(outdir, "fig3_phonetic_taxonomy.png")
        fig3.savefig(p3); plt.close(fig3); written["fig3"] = p3
    except Exception as e:
        print(f"[lingo] fig3 skipped: {e}")

    # ── Figure 4: eSTRF reconstruction ──
    if res.is_band_mode:
        try:
            fig4 = plot_estrf_reconstruction(res, phones=estrf_phones)
            p4 = os.path.join(outdir, "fig4_estrf_reconstruction.png")
            fig4.savefig(p4); plt.close(fig4); written["fig4"] = p4
        except Exception as e:
            print(f"[lingo] fig4 skipped: {e}")
    else:
        print("[lingo] fig4 skipped (requires band-mode results)")

    # ── Figure 5: Manifold + word trajectory ──
    if manifold_path and os.path.exists(manifold_path):
        try:
            mani = load_manifold(manifold_path)
            fig5 = plot_phonetic_manifold_trajectory(mani, method=embed_method)
            p5 = os.path.join(outdir, "fig5_phonetic_manifold.png")
            fig5.savefig(p5); plt.close(fig5); written["fig5"] = p5
        except Exception as e:
            print(f"[lingo] fig5 skipped: {e}")
    else:
        print("[lingo] fig5 skipped (manifold.npz unavailable)")

    return written


# =============================================================================
#                                 CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True, help="path to results_raw.npz")
    ap.add_argument("--cochleagrams", default=None,
                    help="path to hero_cochleagrams.npz (Figure 1)")
    ap.add_argument("--sanity", default=None,
                    help="path to sanity_arrays.npz (Figure 2)")
    ap.add_argument("--manifold", default=None,
                    help="path to manifold.npz (Figure 5)")
    ap.add_argument("--outdir", default="figures_lingo")
    ap.add_argument("--embed_method", choices=["umap", "tsne"], default="umap")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    written = render_all(
        results_path=args.results,
        cochleagrams_path=args.cochleagrams,
        sanity_path=args.sanity,
        manifold_path=args.manifold,
        outdir=args.outdir,
        embed_method=args.embed_method,
    )
    print("Saved figures:")
    for k, v in written.items():
        print(f"  {k}: {v}")
