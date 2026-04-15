"""Mechanistic interpretability figures for the biomimetic auditory frontend.

Given a CorticalLIME run's raw output (``results_raw.npz``) and an optional
per-word trajectory file (``trajectories.npz``), this module produces five
publication-quality figures:

    1.  ``plot_temporal_rate_marginal``     – Temporal-rate marginal importance
                                              (the "rhythmic signature").
    2.  ``plot_manner_heatmaps``            – Rate-×-scale importance heatmaps
                                              grouped by manner of articulation.
    3.  ``plot_phonetic_dendrogram``        – Unsupervised hierarchical taxonomy
                                              of the TIMIT-39 inventory.
    4a. ``plot_cortical_trajectory_panel``  – Dynamic cortical trajectories,
    4b.                                      split across two figures of five
                                              target words each.

All figures use serif fonts, removed top/right spines, 400 DPI, and a
colour-blind-friendly (ColorBrewer-inspired) palette, and are saved as PNG.

The pipeline is strictly data-driven — no dummy data is generated. Call
``render_all`` or use the CLI::

    python lingo_analysis.py --results results_raw.npz \
                             --trajectories trajectories.npz \
                             --outdir figures_lingo/

``render_all`` is also imported by ``run.py`` and invoked automatically at the
end of the full analysis pipeline.
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

# Canonical 1-indexed TIMIT-61 inventory.
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
#                         LOADING & AGGREGATION
# =============================================================================

@dataclass
class LimeResults:
    """Container for the contents of ``results_raw.npz``."""
    importances: np.ndarray   # (N, S)
    target_classes: np.ndarray  # (N,) int, 1-indexed TIMIT-61
    sr_pairs: np.ndarray      # (S, 2)  columns: [scale (cyc/oct), rate (Hz)]

    @property
    def rates(self) -> np.ndarray:
        return self.sr_pairs[:, 1]

    @property
    def scales(self) -> np.ndarray:
        return self.sr_pairs[:, 0]


def load_lime_results(path: str) -> LimeResults:
    """Load ``results_raw.npz`` produced by ``run.py``."""
    with np.load(path, allow_pickle=True) as d:
        importances = np.asarray(d["importances"], dtype=np.float64)
        target_classes = np.asarray(d["target_classes"], dtype=np.int64)
        sr_pairs = np.asarray(d["sr_pairs"], dtype=np.float64)
    return LimeResults(importances, target_classes, sr_pairs)


def _phone61_idx_to_39(idx: int) -> str:
    p61 = IDX_TO_PHONE61.get(int(idx), "")
    return PHONE_61_TO_39.get(p61, "")


def aggregate_per_phoneme(res: LimeResults) -> Dict[str, np.ndarray]:
    """Mean absolute importance per TIMIT-39 phoneme.

    Returns
    -------
    dict mapping phone_39 → (S,) mean |importance| vector.
    """
    bucket: Dict[str, List[np.ndarray]] = {}
    for imp, tc in zip(res.importances, res.target_classes):
        p39 = _phone61_idx_to_39(tc)
        if not p39 or p39 == "sil":
            continue
        bucket.setdefault(p39, []).append(np.abs(imp))
    return {p: np.mean(np.stack(v, 0), 0) for p, v in bucket.items()}


def _unique_axis_values(values: np.ndarray) -> np.ndarray:
    return np.array(sorted(set(np.round(values, 6).tolist())))


def marginalise_by_rate(res: LimeResults) -> Tuple[np.ndarray, np.ndarray]:
    """Per-utterance marginal |importance| over unique temporal-rate bins.

    Returns
    -------
    rates_unique : (R,) sorted unique rate values
    samples      : (N, R)  mean |importance| per utterance per rate bin
    """
    abs_imp = np.abs(res.importances)
    rates = res.rates
    rates_unique = _unique_axis_values(rates)
    out = np.zeros((abs_imp.shape[0], rates_unique.size), dtype=np.float64)
    for j, r in enumerate(rates_unique):
        mask = np.isclose(rates, r)
        out[:, j] = abs_imp[:, mask].mean(axis=1)
    return rates_unique, out


def grid_by_scale_rate(
    mean_vec: np.ndarray,
    sr_pairs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reshape a (S,) vector into a (n_scales, n_rates) grid.

    The model's STRF bank may store duplicates (e.g. upward / downward FM);
    we average within each (scale, rate) cell.
    """
    scales = _unique_axis_values(sr_pairs[:, 0])
    rates = _unique_axis_values(sr_pairs[:, 1])
    grid = np.full((scales.size, rates.size), np.nan, dtype=np.float64)
    count = np.zeros_like(grid)
    for v, (sc, rt) in zip(mean_vec, sr_pairs):
        i = int(np.argmin(np.abs(scales - sc)))
        j = int(np.argmin(np.abs(rates - rt)))
        if np.isnan(grid[i, j]):
            grid[i, j] = 0.0
        grid[i, j] += v
        count[i, j] += 1
    with np.errstate(invalid="ignore"):
        grid = grid / np.where(count == 0, 1, count)
    grid[count == 0] = 0.0
    return grid, scales, rates


# =============================================================================
#                     PLOT 1 — TEMPORAL RATE MARGINAL IMPORTANCE
# =============================================================================

def plot_temporal_rate_marginal(
    res: LimeResults,
    ci: float = 95.0,
    highlight: Tuple[float, float] = (2.0, 8.0),
) -> plt.Figure:
    """Plot 1 – Temporal-rate marginal importance (the Rhythmic Signature).

    For each unique temporal-rate value in the STRF bank, computes the mean
    absolute importance across all explained utterances and plots the mean
    curve with a 95 % shaded confidence interval. Highlights the 2-8 Hz
    stress-timed band and annotates the rate at which the peak occurs.
    """
    rates_unique, samples = marginalise_by_rate(res)
    mean = samples.mean(axis=0)
    lo = np.percentile(samples, (100 - ci) / 2, axis=0)
    hi = np.percentile(samples, 100 - (100 - ci) / 2, axis=0)

    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    colour = FAMILY_COLORS["Vowels"]

    ax.axvspan(*highlight, color="#F4A261", alpha=0.18,
               label="Stress-timed band (2–8 Hz)")
    ax.fill_between(rates_unique, lo, hi, color=colour, alpha=0.22,
                    label=f"{ci:.0f} % CI")
    ax.plot(rates_unique, mean, color=colour, lw=2.2, marker="o", ms=5,
            mfc="white", mec=colour, mew=1.3, label="Mean |importance|")

    # Annotate the peak.
    j_peak = int(np.argmax(mean))
    r_peak = rates_unique[j_peak]
    ax.annotate(
        f"Peak at {r_peak:.1f} Hz",
        xy=(r_peak, mean[j_peak]),
        xytext=(r_peak * 1.6, mean[j_peak] * 1.05),
        arrowprops=dict(arrowstyle="->", lw=0.9, color="black"),
        fontsize=10, ha="left",
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks(rates_unique)
    ax.set_xticklabels([f"{r:g}" for r in rates_unique])
    ax.set_xlabel(r"Temporal rate $\omega$ (Hz)")
    ax.set_ylabel(r"Mean $|$importance$|$")
    ax.set_title("The Rhythmic Signature:\nTemporal-Rate Marginal Importance")
    ax.grid(True, axis="y", ls=":", lw=0.6, alpha=0.55)
    ax.legend(loc="upper right")
    _despine(ax)
    fig.tight_layout()
    return fig


# =============================================================================
#                     PLOT 2 — MANNER-OF-ARTICULATION HEATMAPS
# =============================================================================

def _family_mean_grid(
    phone_means: Dict[str, np.ndarray],
    family: str,
    sr_pairs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    members = [p for p in PHONEMES_BY_FAMILY[family] if p in phone_means]
    if not members:
        return (
            np.zeros((1, 1)),
            _unique_axis_values(sr_pairs[:, 0]),
            _unique_axis_values(sr_pairs[:, 1]),
        )
    stacked = np.stack([phone_means[p] for p in members], 0).mean(0)
    return grid_by_scale_rate(stacked, sr_pairs)


def plot_manner_heatmaps(
    res: LimeResults,
    phone_means: Optional[Dict[str, np.ndarray]] = None,
) -> plt.Figure:
    """Plot 2 – Manner-of-articulation heatmaps (Phonetic Universals).

    For each manner class (vowels, stops, fricatives), averages CorticalLIME
    importance across member phones, folds the (S,) vector onto the
    (scale × rate) grid of the STRF bank, and renders a 2-D heatmap. Axes
    are shared and a common colour bar conveys the normalised importance.
    """
    if phone_means is None:
        phone_means = aggregate_per_phoneme(res)

    order = ["Vowels", "Stops", "Fricatives"]
    grids: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        k: _family_mean_grid(phone_means, k, res.sr_pairs) for k in order
    }
    # Normalise jointly so panels are comparable.
    gmax = max(g[0].max() for g in grids.values()) + 1e-12
    for k in order:
        g, s, r = grids[k]
        grids[k] = (g / gmax, s, r)

    fig, axes = plt.subplots(
        1, 3, figsize=(12.4, 4.0), sharey=True,
        gridspec_kw={"wspace": 0.18},
    )
    im = None
    for ax, name in zip(axes, order):
        M, scales, rates = grids[name]
        im = ax.imshow(
            M, origin="lower", aspect="auto", cmap=SEQ_CMAP,
            vmin=0.0, vmax=1.0,
            extent=[-0.5, len(rates) - 0.5, -0.5, len(scales) - 0.5],
        )
        ax.set_xticks(np.arange(len(rates)))
        ax.set_xticklabels([f"{r:g}" for r in rates], rotation=45, ha="right")
        ax.set_yticks(np.arange(len(scales)))
        ax.set_yticklabels([f"{s:g}" for s in scales])
        ax.set_xlabel(r"Temporal rate $\omega$ (Hz)")
        ax.set_title(f"({'abc'[order.index(name)]}) {name}",
                     color=FAMILY_COLORS[name])
        _despine(ax)

    axes[0].set_ylabel(r"Spectral scale $\Omega$ (cyc / oct)")
    fig.suptitle(
        "Phonetic Universals:\nCortical Importance by Manner of Articulation",
        fontsize=14, fontweight="bold", y=1.04,
    )
    cbar = fig.colorbar(im, ax=axes, shrink=0.88, pad=0.02, aspect=22)
    cbar.set_label(r"Normalised $|$importance$|$")
    cbar.outline.set_visible(False)
    return fig


# =============================================================================
#                 PLOT 3 — UNSUPERVISED PHONETIC TAXONOMY
# =============================================================================

def _link_color_map(
    Z: np.ndarray, families: Sequence[str], family_colors: Dict[str, str],
    default: str = "#8A8A8A",
) -> Dict[int, str]:
    """Map every internal node of ``Z`` to a colour.

    A link is coloured with a family's colour iff *all* leaves under it
    belong to that family; otherwise it is neutral grey. This guarantees
    the dendrogram branch colours are consistent with the leaf-label
    colours — a link inherits a family colour only when its subtree is
    pure. The top of the tree (the sonorant/obstruent split) is grey.
    """
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
    phones: List[str] = []
    families: List[str] = []
    rows: List[np.ndarray] = []
    for fam in ["Vowels", "Stops", "Fricatives", "Nasals"]:
        for p in PHONEMES_BY_FAMILY[fam]:
            if p in phone_means:
                phones.append(p)
                families.append(fam)
                rows.append(phone_means[p])
    if not rows:
        raise ValueError("No TIMIT-39 phones found in results.")
    return np.vstack(rows), phones, families


def plot_phonetic_dendrogram(
    res: LimeResults,
    phone_means: Optional[Dict[str, np.ndarray]] = None,
    method: str = "ward",
    metric: str = "euclidean",
) -> plt.Figure:
    """Plot 3 – Unsupervised phonetic taxonomy.

    Hierarchical clustering (Ward linkage) of the (N_phones × S_filters)
    mean CorticalLIME weight matrix. Branch colours and leaf labels are
    painted with the same manner-class palette (strict match: a branch
    inherits a family colour only if its entire subtree is pure in that
    family). The two highest-level clusters are annotated as "Sonorants"
    and "Obstruents" to guide the reader.
    """
    if phone_means is None:
        phone_means = aggregate_per_phoneme(res)

    W, phones, families = _ordered_phone_weight_matrix(phone_means)
    Z = linkage(W, method=method, metric=metric)
    link_colors = _link_color_map(Z, families, FAMILY_COLORS)

    fig, ax = plt.subplots(figsize=(7.8, 9.4))
    ddata = dendrogram(
        Z,
        orientation="right",
        labels=phones,
        leaf_font_size=10,
        link_color_func=lambda k: link_colors.get(k, "#8A8A8A"),
        above_threshold_color="#8A8A8A",
        ax=ax,
    )
    # Colour leaf labels by family.
    fam_of = dict(zip(phones, families))
    for lbl in ax.get_ymajorticklabels():
        p = lbl.get_text()
        lbl.set_color(FAMILY_COLORS[fam_of[p]])
        lbl.set_fontweight("bold")

    ax.set_xlabel("Ward linkage distance")
    ax.set_title(
        "Emergent Phonetic Taxonomy:\n"
        "Unsupervised Hierarchical Clustering of CorticalLIME Weights",
        pad=14,
    )
    _despine(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(True, axis="x", ls=":", lw=0.6, alpha=0.55)

    # ── Sonorant / Obstruent annotations ─────────────────────────────────
    ordered_phones = [t.get_text() for t in ax.get_ymajorticklabels()]
    ordered_fams = [fam_of[p] for p in ordered_phones]
    sonorant_set = {"Vowels", "Nasals"}
    obstruent_set = {"Stops", "Fricatives"}

    def _yrange(target: set) -> Optional[Tuple[float, float]]:
        idxs = [i for i, f in enumerate(ordered_fams) if f in target]
        if not idxs:
            return None
        return (10 * (min(idxs) + 0.5) - 4, 10 * (max(idxs) + 0.5) + 4)

    x_max = float(np.max(Z[:, 2]))
    x_anno = x_max * 1.03
    ax.set_xlim(right=x_max * 1.22)

    for label, target, color in [
        ("Sonorants\n(Vowels + Nasals)", sonorant_set, "#3A3A3A"),
        ("Obstruents\n(Stops + Fricatives)", obstruent_set, "#3A3A3A"),
    ]:
        yr = _yrange(target)
        if yr is None:
            continue
        y0, y1 = yr
        ax.annotate(
            "",
            xy=(x_anno, y0), xytext=(x_anno, y1),
            arrowprops=dict(arrowstyle="-", lw=1.6, color=color),
            annotation_clip=False,
        )
        ax.text(
            x_anno + (x_max * 0.03), (y0 + y1) / 2, label,
            va="center", ha="left", fontsize=10, fontweight="bold",
            color=color,
        )

    handles = [
        plt.Line2D([0], [0], color=FAMILY_COLORS[f], lw=3, label=f)
        for f in ["Vowels", "Stops", "Fricatives", "Nasals"]
    ]
    ax.legend(handles=handles, loc="lower right", title="Manner class",
              title_fontsize=9)
    fig.tight_layout()
    return fig


# =============================================================================
#                   PLOT 4 — DYNAMIC CORTICAL TRAJECTORIES
# =============================================================================

# Filter categorisation used to build per-frame trajectory rows.
FILTER_GROUPS: List[Tuple[str, str]] = [
    ("High Rate / Low Scale\n(Stops)",      "Stops"),
    ("Low Rate / High Scale\n(Vowels)",     "Vowels"),
    ("Mid Rate / High Scale\n(Fricatives)", "Fricatives"),
]


def categorise_filters(sr_pairs: np.ndarray) -> Dict[str, np.ndarray]:
    """Partition STRF channels into three linguistically-motivated groups.

    The (scale, rate) thresholds follow Chi, Ru & Shamma (2005):
        Vowels     →  rate  ≤ 8 Hz  and  scale ≥ 2 cyc/oct
        Stops      →  rate  ≥ 11 Hz and  scale ≤ 1 cyc/oct
        Fricatives →  rate  ≥ 5 Hz  and  scale ≥ 1 cyc/oct (excl. Vowels/Stops)
    """
    scale = sr_pairs[:, 0]
    rate = sr_pairs[:, 1]

    vowels = (rate <= 8.0) & (scale >= 2.0)
    stops = (rate >= 11.0) & (scale <= 1.0)
    fricatives = (rate >= 5.0) & (scale >= 1.0) & ~vowels & ~stops

    return {
        "Vowels": np.where(vowels)[0],
        "Stops": np.where(stops)[0],
        "Fricatives": np.where(fricatives)[0],
    }


def _plot_single_trajectory(
    ax: plt.Axes,
    traj: np.ndarray,
    boundaries: Sequence[Tuple[int, str]],
    word: str,
    filter_labels: Sequence[str],
    show_ylabels: bool = True,
) -> "mpl.image.AxesImage":
    n_g, T = traj.shape
    im = ax.imshow(
        traj, aspect="auto", origin="lower", cmap=SEQ_CMAP,
        vmin=0.0, vmax=1.0,
        extent=[0, T, -0.5, n_g - 0.5],
    )
    ax.set_yticks(np.arange(n_g))
    if show_ylabels:
        ax.set_yticklabels(filter_labels)
    else:
        ax.set_yticklabels([])
    ax.set_xlabel("Time frame")
    ax.set_title(f"“{word}”", pad=6)

    edges = [int(b[0]) for b in boundaries] + [T]
    for b_idx, (frame, phn) in enumerate(boundaries):
        ax.axvline(frame, color="white", ls="--", lw=1.0, alpha=0.9)
        center = (edges[b_idx] + edges[b_idx + 1]) / 2
        ax.text(
            center, n_g - 0.35, phn,
            ha="center", va="top", color="white",
            fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.18", fc="black", ec="none",
                      alpha=0.55),
        )
    ax.axvline(T, color="white", ls="--", lw=1.0, alpha=0.9)
    _despine(ax)
    return im


def plot_cortical_trajectory_panel(
    trajectories: List[Dict],
    filter_labels: Sequence[str] = (fl for fl, _ in FILTER_GROUPS),
    title: str = "Dynamic Cortical Trajectories",
) -> plt.Figure:
    """Plot 4 – One figure containing 5 stacked word trajectories.

    Each row shows a single word's time-resolved cortical saliency
    (``|LIME importance| × per-frame cortical energy``, aggregated by
    filter category). Dashed white lines mark phoneme boundaries.
    """
    filter_labels = list(filter_labels)
    n = len(trajectories)
    fig, axes = plt.subplots(
        n, 1, figsize=(9.0, 2.1 * n + 0.6), sharex=False,
    )
    if n == 1:
        axes = [axes]
    im = None
    for i, (ax, td) in enumerate(zip(axes, trajectories)):
        im = _plot_single_trajectory(
            ax, td["trajectory"], td["boundaries"], td["word"],
            filter_labels, show_ylabels=True,
        )
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0.0, 0.0, 0.95, 0.98])
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02, aspect=30)
    cbar.set_label(r"Normalised saliency")
    cbar.outline.set_visible(False)
    return fig


def load_trajectories(path: str) -> List[Dict]:
    """Load ``trajectories.npz`` written by ``run.py``.

    The file stores an object array of per-word dicts with keys
    ``word``, ``trajectory`` (n_groups × T), and ``boundaries``
    (list of (frame_idx, phone) tuples).
    """
    with np.load(path, allow_pickle=True) as d:
        arr = d["trajectories"]
    return [dict(x) for x in arr.tolist()]


# =============================================================================
#                             TOP-LEVEL RENDERING
# =============================================================================

def render_all(
    results_path: str,
    trajectories_path: Optional[str] = None,
    outdir: str = "figures_lingo",
) -> Dict[str, str]:
    """Produce all five figures.

    Parameters
    ----------
    results_path : path to ``results_raw.npz``
    trajectories_path : path to ``trajectories.npz`` (optional; if absent,
        Plot 4 panels are skipped).
    outdir : directory to write PNGs into.
    """
    set_academic_style()
    os.makedirs(outdir, exist_ok=True)
    written: Dict[str, str] = {}

    res = load_lime_results(results_path)
    phone_means = aggregate_per_phoneme(res)

    fig1 = plot_temporal_rate_marginal(res)
    p1 = os.path.join(outdir, "fig1_temporal_rate_marginal.png")
    fig1.savefig(p1); plt.close(fig1); written["fig1"] = p1

    fig2 = plot_manner_heatmaps(res, phone_means)
    p2 = os.path.join(outdir, "fig2_manner_heatmaps.png")
    fig2.savefig(p2); plt.close(fig2); written["fig2"] = p2

    fig3 = plot_phonetic_dendrogram(res, phone_means)
    p3 = os.path.join(outdir, "fig3_phonetic_taxonomy.png")
    fig3.savefig(p3); plt.close(fig3); written["fig3"] = p3

    if trajectories_path is not None and os.path.exists(trajectories_path):
        trajs = load_trajectories(trajectories_path)
        # Split into two figures of 5 words each.
        first = trajs[:5]
        second = trajs[5:10]
        if first:
            f4a = plot_cortical_trajectory_panel(
                first,
                title="Dynamic Cortical Trajectories (I/II):\n"
                      "Temporal Saliency Across Five Representative Words",
            )
            p4a = os.path.join(outdir, "fig4a_cortical_trajectory.png")
            f4a.savefig(p4a); plt.close(f4a); written["fig4a"] = p4a
        if second:
            f4b = plot_cortical_trajectory_panel(
                second,
                title="Dynamic Cortical Trajectories (II/II):\n"
                      "Temporal Saliency Across Five Representative Words",
            )
            p4b = os.path.join(outdir, "fig4b_cortical_trajectory.png")
            f4b.savefig(p4b); plt.close(f4b); written["fig4b"] = p4b

    return written


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True,
                    help="path to results_raw.npz")
    ap.add_argument("--trajectories", default=None,
                    help="path to trajectories.npz (optional)")
    ap.add_argument("--outdir", default="figures_lingo",
                    help="directory to write PNGs to")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    written = render_all(args.results, args.trajectories, args.outdir)
    print("Saved figures:")
    for k, v in written.items():
        print(f"  {k}: {v}")
