"""Publication-quality figures for CorticalLIME analyses.

This module is fully self-contained: every plot has a matching dummy-data
generator that produces structured, neuro-linguistically motivated data so the
figures convey the intended message out of the box. Replace the dummy
generators with real outputs from the CorticalLIME pipeline to produce the
final camera-ready versions.

Four figures are produced:

    1.  ``plot_temporal_rate_marginal``   – "The Rhythmic Signature"
    2.  ``plot_manner_heatmaps``          – "Phonetic Universals"
    3.  ``plot_phonetic_dendrogram``      – "The Aha! Moment"
    4.  ``plot_cortical_trajectory``      – "Time-Series Shift"

All figures share the same academic aesthetic: serif fonts, top/right spines
removed, high DPI (400), colour-blind-friendly (ColorBrewer) palettes, and PNG
output.

Usage
-----
>>> python lingo_analysis.py                # writes PNGs to ./figures/
>>> python lingo_analysis.py --outdir out/  # custom output directory
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy.cluster.hierarchy import dendrogram, linkage, set_link_color_palette

# =============================================================================
#                               GLOBAL AESTHETICS
# =============================================================================

# Colour-blind-friendly qualitative palette (ColorBrewer "Set2"/"Dark2" hybrids)
FAMILY_COLORS: Dict[str, str] = {
    "Vowels":     "#D7263D",   # red
    "Stops":      "#1B4F8B",   # blue
    "Fricatives": "#2A9D3F",   # green
    "Nasals":     "#6A359C",   # purple
}

# Sequential colour-blind-friendly map for heatmaps
SEQ_CMAP = "viridis"
DIVERGING_CMAP = "RdBu_r"


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
            "axes.labelweight": "regular",
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
    """Hard-remove the top/right spines on a given Axes object."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# =============================================================================
#                              SHARED CONSTANTS
# =============================================================================

# Biomimetic STRF axes (Chi, Ru & Shamma 2005 conventions).
TEMPORAL_RATES_HZ = np.array(
    [1.0, 1.4, 2.0, 2.8, 4.0, 5.7, 8.0, 11.3, 16.0, 22.6, 32.0], dtype=np.float64
)
SPECTRAL_SCALES_CPO = np.array(
    [0.25, 0.5, 1.0, 2.0, 4.0, 8.0], dtype=np.float64
)

# Canonical TIMIT-39 inventory (Lee & Hon 1989), grouped by manner.
PHONEMES_BY_FAMILY: Dict[str, List[str]] = {
    "Vowels": [
        "iy", "ih", "eh", "ae", "aa", "ah", "uw", "uh", "er", "ey", "ay",
        "oy", "aw", "ow",
    ],
    "Stops":      ["b", "d", "g", "p", "t", "k"],
    "Fricatives": ["s", "sh", "z", "f", "th", "v", "dh", "hh"],
    "Nasals":     ["m", "n", "ng"],
}


# =============================================================================
#                       DUMMY-DATA GENERATORS (HYPOTHESIS-DRIVEN)
# =============================================================================

def _gauss2d(
    rates: np.ndarray,
    scales: np.ndarray,
    mu_rate: float,
    mu_scale: float,
    sigma_rate: float,
    sigma_scale: float,
    amp: float = 1.0,
) -> np.ndarray:
    """2-D Gaussian bump on a (scale, rate) grid, in log space."""
    R, S = np.meshgrid(np.log2(rates), np.log2(scales))
    Z = amp * np.exp(
        -((R - np.log2(mu_rate)) ** 2) / (2 * sigma_rate ** 2)
        - ((S - np.log2(mu_scale)) ** 2) / (2 * sigma_scale ** 2)
    )
    return Z


def dummy_temporal_rate_importance(
    rates: np.ndarray = TEMPORAL_RATES_HZ,
    n_utterances: int = 240,
    seed: int = 0,
) -> np.ndarray:
    """Dummy |importance| bootstrap sample per rate channel.

    Encodes the hypothesis that English (stress-timed) peaks near ~4 Hz with a
    broad shoulder across 2–8 Hz (syllabic rate), decaying toward both ends.

    Returns
    -------
    (n_utterances, len(rates)) float array.
    """
    rng = np.random.default_rng(seed)
    log_r = np.log2(rates)
    peak = np.log2(4.0)
    # Skewed log-normal-ish envelope + second micro-peak at ~16 Hz (phonemic).
    envelope = 1.0 * np.exp(-((log_r - peak) ** 2) / (2 * 0.9 ** 2)) \
        + 0.28 * np.exp(-((log_r - np.log2(16.0)) ** 2) / (2 * 0.55 ** 2))
    envelope = 0.05 + envelope  # baseline floor
    # Multiplicative log-normal noise across utterances
    noise = rng.lognormal(mean=0.0, sigma=0.18, size=(n_utterances, rates.size))
    samples = envelope[None, :] * noise
    return samples.astype(np.float64)


def dummy_manner_heatmaps(
    rates: np.ndarray = TEMPORAL_RATES_HZ,
    scales: np.ndarray = SPECTRAL_SCALES_CPO,
    seed: int = 1,
) -> Dict[str, np.ndarray]:
    """Synthetic |importance| heatmap per manner class (scale × rate)."""
    rng = np.random.default_rng(seed)

    # Vowels ─ high scale (>2 cyc/oct), low rate (<8 Hz): slow harmonic stack.
    vowels = _gauss2d(rates, scales, mu_rate=3.0, mu_scale=4.0,
                      sigma_rate=0.6, sigma_scale=0.6, amp=1.0)
    # Stops ─ low scale (<1 cyc/oct), high rate (>16 Hz): fast broadband burst.
    stops = _gauss2d(rates, scales, mu_rate=20.0, mu_scale=0.5,
                     sigma_rate=0.55, sigma_scale=0.55, amp=1.0)
    # Fricatives ─ mid/high scale, mid/high rate: sustained noise w/ texture.
    fricatives = _gauss2d(rates, scales, mu_rate=10.0, mu_scale=3.0,
                          sigma_rate=0.7, sigma_scale=0.7, amp=1.0)

    out = {
        "Vowels": vowels,
        "Stops": stops,
        "Fricatives": fricatives,
    }
    for k, v in out.items():
        v = v + 0.04 * rng.standard_normal(v.shape)
        v = np.clip(v, 0.0, None)
        out[k] = v / v.max()
    return out


def dummy_phonetic_weight_matrix(
    n_filters: int = 66,  # 11 rates × 6 scales
    seed: int = 2,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Dummy (39 × N_filters) CorticalLIME weight matrix for TIMIT-39.

    Constructed so that hierarchical clustering naturally recovers the four
    manner families — the "aha" moment of the paper.
    """
    rng = np.random.default_rng(seed)

    rates = TEMPORAL_RATES_HZ
    scales = SPECTRAL_SCALES_CPO

    family_templates = {
        "Vowels":     _gauss2d(rates, scales, 3.0, 4.0, 0.55, 0.55, 1.0),
        "Stops":      _gauss2d(rates, scales, 20.0, 0.5, 0.55, 0.55, 1.0),
        "Fricatives": _gauss2d(rates, scales, 10.0, 3.0, 0.65, 0.65, 1.0),
        "Nasals":     _gauss2d(rates, scales, 2.0, 1.5, 0.55, 0.65, 1.0),
    }
    # Flatten templates to filter-vectors of length n_filters.
    template_vecs = {
        k: v.flatten()[:n_filters] for k, v in family_templates.items()
    }

    phones: List[str] = []
    families: List[str] = []
    rows: List[np.ndarray] = []
    for fam, plist in PHONEMES_BY_FAMILY.items():
        base = template_vecs[fam]
        base = base / (np.linalg.norm(base) + 1e-8)
        for p in plist:
            jitter = 0.18 * rng.standard_normal(n_filters)
            row = base + jitter
            rows.append(row)
            phones.append(p)
            families.append(fam)

    W = np.vstack(rows)  # (39, n_filters)
    return W, phones, families


def dummy_cortical_trajectory(
    n_frames: int = 240,
    seed: int = 3,
) -> Tuple[np.ndarray, List[str], List[Tuple[int, str]]]:
    """Dummy time-resolved cortical importance trajectory for the word 'stop'.

    Returns
    -------
    trajectory : (n_filters_subset, n_frames) float array in [0, 1]
    filter_labels : names of the cortical filter bands
    boundaries : list of (frame_idx, phone_symbol) tuples
    """
    rng = np.random.default_rng(seed)
    filter_labels = [
        "High Rate / Low Scale\n(Stops)",
        "Low Rate / High Scale\n(Vowels)",
        "Mid Rate / High Scale\n(Fricatives)",
    ]
    n_filters = len(filter_labels)
    T = n_frames

    # Phoneme boundaries for /s/ /t/ /aa/ /p/ with relative durations
    durations = np.array([0.28, 0.14, 0.38, 0.20])  # sum = 1
    durations = durations / durations.sum()
    edges = np.concatenate([[0], np.cumsum(durations) * T]).astype(int)
    phones = ["/s/", "/t/", "/aa/", "/p/"]
    boundaries = [(int(edges[i]), phones[i]) for i in range(len(phones))]

    traj = np.zeros((n_filters, T), dtype=np.float64)
    t = np.arange(T)

    def bump(center: float, width: float, amp: float = 1.0) -> np.ndarray:
        return amp * np.exp(-((t - center) ** 2) / (2 * width ** 2))

    # /s/ → Fricatives filter active
    s_center = (edges[0] + edges[1]) / 2
    traj[2] += bump(s_center, (edges[1] - edges[0]) / 2.5, amp=0.95)
    # /t/ → Stops filter spikes sharply at burst onset
    t_center = edges[1] + 2
    traj[0] += bump(t_center, (edges[2] - edges[1]) / 3.5, amp=1.0)
    # /aa/ → Vowels filter sustained, strong
    a_center = (edges[2] + edges[3]) / 2
    traj[1] += bump(a_center, (edges[3] - edges[2]) / 2.2, amp=1.0)
    # /p/ → Stops burst, slightly weaker than /t/
    p_center = (edges[3] + edges[4]) / 2
    traj[0] += bump(p_center, (edges[4] - edges[3]) / 2.8, amp=0.85)

    traj += 0.04 * rng.standard_normal(traj.shape)
    traj = np.clip(traj, 0.0, None)
    traj /= traj.max()
    return traj, filter_labels, boundaries


# =============================================================================
#                                PLOT 1 – RHYTHM
# =============================================================================

def plot_temporal_rate_marginal(
    importances: np.ndarray,
    rates: np.ndarray = TEMPORAL_RATES_HZ,
    ci: float = 95.0,
    ax: plt.Axes | None = None,
    highlight: Tuple[float, float] = (2.0, 8.0),
) -> plt.Figure:
    """Plot 1 – Temporal Rate Marginal Importance ("The Rhythmic Signature")."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
    else:
        fig = ax.figure

    mean = importances.mean(axis=0)
    lo = np.percentile(importances, (100 - ci) / 2, axis=0)
    hi = np.percentile(importances, 100 - (100 - ci) / 2, axis=0)

    colour = FAMILY_COLORS["Vowels"]
    ax.fill_between(rates, lo, hi, color=colour, alpha=0.22,
                    label=f"{ci:.0f}% CI")
    ax.plot(rates, mean, color=colour, lw=2.2, marker="o", ms=5,
            mfc="white", mec=colour, mew=1.3, label="Mean |importance|")

    # Shade the English stress-timing band.
    ax.axvspan(*highlight, color="#F4A261", alpha=0.18,
               label="Stress-timed band (2–8 Hz)")

    ax.set_xscale("log", base=2)
    ax.set_xticks(rates)
    ax.set_xticklabels([f"{r:g}" for r in rates])
    ax.set_xlabel(r"Temporal rate $\omega$ (Hz)")
    ax.set_ylabel(r"Mean $|$importance$|$")
    ax.set_title("The Rhythmic Signature:\nTemporal Rate Marginal Importance")
    ax.grid(True, axis="y", ls=":", lw=0.6, alpha=0.55)
    ax.legend(loc="upper right")
    _despine(ax)
    fig.tight_layout()
    return fig


# =============================================================================
#                           PLOT 2 – MANNER HEATMAPS
# =============================================================================

def plot_manner_heatmaps(
    manner_maps: Dict[str, np.ndarray],
    rates: np.ndarray = TEMPORAL_RATES_HZ,
    scales: np.ndarray = SPECTRAL_SCALES_CPO,
) -> plt.Figure:
    """Plot 2 – Manner-of-articulation heatmaps ("Phonetic Universals")."""
    order = ["Vowels", "Stops", "Fricatives"]
    fig, axes = plt.subplots(
        1, 3, figsize=(12.2, 3.9), sharey=True,
        gridspec_kw={"wspace": 0.18},
    )

    vmax = max(m.max() for m in manner_maps.values())
    im = None
    for ax, name in zip(axes, order):
        M = manner_maps[name]  # (n_scales, n_rates)
        im = ax.imshow(
            M, origin="lower", aspect="auto", cmap=SEQ_CMAP,
            vmin=0.0, vmax=vmax,
            extent=[-0.5, len(rates) - 0.5, -0.5, len(scales) - 0.5],
        )
        ax.set_xticks(np.arange(len(rates)))
        ax.set_xticklabels([f"{r:g}" for r in rates], rotation=45, ha="right")
        ax.set_yticks(np.arange(len(scales)))
        ax.set_yticklabels([f"{s:g}" for s in scales])
        ax.set_xlabel(r"Temporal rate $\omega$ (Hz)")
        ax.set_title(f"({'abc'[order.index(name)]})  {name}",
                     color=FAMILY_COLORS[name])
        _despine(ax)

    axes[0].set_ylabel(r"Spectral scale $\Omega$ (cyc / oct)")
    fig.suptitle("Phonetic Universals:\nCortical Importance by Manner of Articulation",
                 fontsize=14, fontweight="bold", y=1.04)

    cbar = fig.colorbar(im, ax=axes, shrink=0.88, pad=0.02, aspect=22)
    cbar.set_label(r"Normalised $|$importance$|$")
    cbar.outline.set_visible(False)
    return fig


# =============================================================================
#                         PLOT 3 – PHONETIC DENDROGRAM
# =============================================================================

def plot_phonetic_dendrogram(
    W: np.ndarray,
    phones: Sequence[str],
    families: Sequence[str],
    method: str = "ward",
    metric: str = "euclidean",
) -> plt.Figure:
    """Plot 3 – Unsupervised phonetic dendrogram ("The Aha! Moment").

    Hierarchical clustering of the (phonemes × filters) CorticalLIME weight
    matrix, with branches & leaf labels colour-coded by manner family.
    """
    assert W.shape[0] == len(phones) == len(families)

    Z = linkage(W, method=method, metric=metric)

    # Supply a per-family colour palette for the dendrogram link colouring.
    palette = [FAMILY_COLORS[f] for f in ["Vowels", "Stops", "Fricatives",
                                          "Nasals"]]
    set_link_color_palette(palette)

    fig, ax = plt.subplots(figsize=(7.2, 8.8))
    ddata = dendrogram(
        Z,
        orientation="right",
        labels=list(phones),
        leaf_font_size=10,
        color_threshold=0.7 * np.max(Z[:, 2]),
        above_threshold_color="#8A8A8A",
        ax=ax,
    )

    # Colour the leaf labels by family.
    fam_of = dict(zip(phones, families))
    for lbl in ax.get_ymajorticklabels():
        p = lbl.get_text()
        lbl.set_color(FAMILY_COLORS[fam_of[p]])
        lbl.set_fontweight("bold")

    ax.set_xlabel("Ward linkage distance")
    ax.set_title("The Aha! Moment:\nUnsupervised Phonetic Taxonomy from CorticalLIME Weights",
                 pad=14)
    _despine(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(True, axis="x", ls=":", lw=0.6, alpha=0.55)

    # Manual legend for the four families.
    handles = [
        plt.Line2D([0], [0], color=FAMILY_COLORS[f], lw=3, label=f)
        for f in ["Vowels", "Stops", "Fricatives", "Nasals"]
    ]
    ax.legend(handles=handles, loc="lower right", title="Manner family",
              title_fontsize=9)
    fig.tight_layout()
    return fig


# =============================================================================
#                       PLOT 4 – CORTICAL TRAJECTORY
# =============================================================================

def plot_cortical_trajectory(
    trajectory: np.ndarray,
    filter_labels: Sequence[str],
    boundaries: Sequence[Tuple[int, str]],
    word: str = "stop",
) -> plt.Figure:
    """Plot 4 – Dynamic cortical trajectory ("Time-Series Shift")."""
    n_filt, T = trajectory.shape

    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    im = ax.imshow(
        trajectory, aspect="auto", origin="lower", cmap=SEQ_CMAP,
        extent=[0, T, -0.5, n_filt - 0.5],
    )

    ax.set_yticks(np.arange(n_filt))
    ax.set_yticklabels(filter_labels)
    ax.set_xlabel("Time frame")
    ax.set_title(f"Time-Series Shift:\nCortical Importance Trajectory for the word “{word}”",
                 pad=12)

    # Dashed phoneme boundaries + labels.
    edges = [b[0] for b in boundaries] + [T]
    for b_idx, (frame, phn) in enumerate(boundaries):
        ax.axvline(frame, color="white", ls="--", lw=1.1, alpha=0.85)
        center = (edges[b_idx] + edges[b_idx + 1]) / 2
        ax.text(
            center, n_filt - 0.35, phn,
            ha="center", va="top", color="white",
            fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none",
                      alpha=0.55),
        )
    # Final trailing edge marker
    ax.axvline(T, color="white", ls="--", lw=1.1, alpha=0.85)

    _despine(ax)
    cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.9, aspect=18)
    cbar.set_label(r"Normalised $|$importance$|$")
    cbar.outline.set_visible(False)
    fig.tight_layout()
    return fig


# =============================================================================
#                                   MAIN
# =============================================================================

@dataclass
class FigurePaths:
    outdir: str

    def path(self, name: str) -> str:
        os.makedirs(self.outdir, exist_ok=True)
        return os.path.join(self.outdir, name)


def render_all(outdir: str = "figures", seed: int = 0) -> Dict[str, str]:
    """Generate all four figures from dummy data and save as PNGs."""
    set_academic_style()
    paths = FigurePaths(outdir)
    written: Dict[str, str] = {}

    # ── Plot 1 ────────────────────────────────────────────────────────────
    samples = dummy_temporal_rate_importance(seed=seed)
    fig1 = plot_temporal_rate_marginal(samples)
    p1 = paths.path("fig1_temporal_rate_marginal.png")
    fig1.savefig(p1)
    plt.close(fig1)
    written["fig1"] = p1

    # ── Plot 2 ────────────────────────────────────────────────────────────
    manner = dummy_manner_heatmaps(seed=seed + 1)
    fig2 = plot_manner_heatmaps(manner)
    p2 = paths.path("fig2_manner_heatmaps.png")
    fig2.savefig(p2)
    plt.close(fig2)
    written["fig2"] = p2

    # ── Plot 3 ────────────────────────────────────────────────────────────
    W, phones, families = dummy_phonetic_weight_matrix(seed=seed + 2)
    fig3 = plot_phonetic_dendrogram(W, phones, families)
    p3 = paths.path("fig3_phonetic_dendrogram.png")
    fig3.savefig(p3)
    plt.close(fig3)
    written["fig3"] = p3

    # ── Plot 4 ────────────────────────────────────────────────────────────
    traj, flabels, bounds = dummy_cortical_trajectory(seed=seed + 3)
    fig4 = plot_cortical_trajectory(traj, flabels, bounds, word="stop")
    p4 = paths.path("fig4_cortical_trajectory.png")
    fig4.savefig(p4)
    plt.close(fig4)
    written["fig4"] = p4

    return written


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="figures",
                    help="directory to write PNGs to (default: ./figures)")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for dummy data")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    written = render_all(outdir=args.outdir, seed=args.seed)
    print("Saved figures:")
    for k, v in written.items():
        print(f"  {k}: {v}")
