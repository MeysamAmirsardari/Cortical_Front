#!/usr/bin/env python3
"""Heavy-duty phonetic taxonomy from CorticalLIME results.

The lightweight dendrogram in ``lingo_analysis`` clusters the 39
TIMIT-folded phonemes directly in the raw 330-dim band-mode LIME
feature space.  At that scale, Euclidean distance is dominated by
LIME perturbation noise and by per-phoneme magnitude differences
(stops are intrinsically smaller-energy than vowels), so the tree
collapses into a one-cluster blob.

This script does the rigorous preprocessing the analysis actually
needs before any linkage is computed:

  A.  Per-phoneme mean of the **absolute** LIME importances
      → (39, 330) matrix X.
  B.  L2 row-normalisation of X — forces the clustering to focus on
      the *pattern* of cortical tuning rather than absolute weight
      magnitude, so an obstruent burst pattern is comparable to a
      vowel formant pattern.
  C.  PCA denoising with `n_components=0.95, svd_solver='full'`
      — automatically picks the number of components needed to
      explain 95 % of the variance and discards the noisy tail.

Linkage is then Ward on Euclidean distances in the PCA-denoised
space.  The figure paints leaves by manner of articulation and
recolours each subtree edge if (and only if) every leaf below the
edge belongs to the same manner family — so a pure-vowel subtree
turns red, a pure-stop subtree turns blue, and any mixed subtree
falls back to neutral grey.

Usage::

    ./generate_heavy_taxonomy.py
    ./generate_heavy_taxonomy.py --variance 0.97
    ./generate_heavy_taxonomy.py --results /path/to/results_raw.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

# ────────────────────────────────────────────────────────────────────────
# Project imports — mirror run.py path resolution so this works wherever
# the user invokes it from.
# ────────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_r_candidate = _SCRIPT_DIR.parent / "r_code"
if not _r_candidate.is_dir():
    _r_candidate = _SCRIPT_DIR / "r_code"
if not _r_candidate.is_dir():
    _r_candidate = _SCRIPT_DIR
_R_CODE = _r_candidate.resolve()
sys.path.insert(0, str(_R_CODE))
sys.path.insert(0, str(_SCRIPT_DIR))

from lingo_analysis import (  # noqa: E402
    FAMILY_COLORS, FAMILY_OF, IDX_TO_PHONE61, PHONE_61_TO_39,
    PHONEMES_BY_FAMILY,
)


# ────────────────────────────────────────────────────────────────────────
# Project-canonical default paths.
# ────────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path("/Users/eminent/Projects/Cortical_Front")
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "r_code" / "analysis_outputs2"
_DEFAULT_RESULTS = _DEFAULT_OUTPUT_DIR / "results_raw.npz"


# Canonical leaf order: walk the four manner classes in a fixed order so
# the dendrogram leaves are at least *initially* arranged sensibly.  Ward
# linkage will reorder them anyway.
_FAMILY_ORDER: List[str] = ["Vowels", "Stops", "Fricatives", "Nasals"]


# ────────────────────────────────────────────────────────────────────────
# Pipeline
# ────────────────────────────────────────────────────────────────────────

def _phone61_idx_to_39(idx: int) -> str:
    p61 = IDX_TO_PHONE61.get(int(idx), "")
    return PHONE_61_TO_39.get(p61, "")


def build_phoneme_matrix(
    importances: np.ndarray,        # (N, 330)
    target_classes: np.ndarray,     # (N,)  1-indexed TIMIT-61
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Step A — mean of |importance| per TIMIT-39 phoneme.

    Returns
    -------
    X        : (P, F) matrix, F = 330 (n_bands × n_strfs).
    phones   : (P,) list of TIMIT-39 phone strings (manner-ordered).
    families : (P,) list of manner-class labels matching ``phones``.
    """
    abs_imp = np.abs(np.asarray(importances, dtype=np.float64))
    bucket: Dict[str, List[np.ndarray]] = {}
    for vec, tc in zip(abs_imp, target_classes):
        p39 = _phone61_idx_to_39(int(tc))
        if not p39 or p39 == "sil":
            continue
        bucket.setdefault(p39, []).append(vec)

    means = {p: np.mean(np.stack(v, 0), axis=0) for p, v in bucket.items()}

    phones, families, rows = [], [], []
    for fam in _FAMILY_ORDER:
        for p in PHONEMES_BY_FAMILY[fam]:
            if p in means:
                phones.append(p)
                families.append(fam)
                rows.append(means[p])
    if not rows:
        raise ValueError("No TIMIT-39 phones survived the silence filter.")
    X = np.vstack(rows)
    return X, phones, families


def heavy_preprocess(
    X: np.ndarray, variance_target: float = 0.95, random_state: int = 0,
) -> Tuple[np.ndarray, int, np.ndarray]:
    """Steps B + C — L2 row-normalise, then PCA-denoise to ``variance_target``.

    Returns
    -------
    X_pca           : (P, k) projected data
    n_components    : k, chosen automatically by sklearn
    explained_var   : (k,) explained-variance ratios
    """
    # Step B: L2 normalisation — pattern over magnitude.
    X_l2 = normalize(X, norm="l2", axis=1)

    # Step C: PCA-denoise.  svd_solver='full' is mandatory when
    # n_components is a float (variance fraction).
    pca = PCA(
        n_components=variance_target,
        svd_solver="full",
        random_state=random_state,
        whiten=False,
    )
    X_pca = pca.fit_transform(X_l2)
    return X_pca, int(pca.n_components_), pca.explained_variance_ratio_


def cluster_ward(X_pca: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """pdist + Ward linkage."""
    D = pdist(X_pca, metric="euclidean")
    Z = linkage(D, method="ward", optimal_ordering=False)
    return Z, D


# ────────────────────────────────────────────────────────────────────────
# Subtree-purity branch colouring
# ────────────────────────────────────────────────────────────────────────

def _build_link_color_func(
    Z: np.ndarray,
    families: Sequence[str],
    family_colors: Dict[str, str],
    mixed_color: str = "#8A8A8A",
):
    """Return a function ``id -> hex`` for ``dendrogram(link_color_func=…)``.

    A branch is painted with its family's colour iff every leaf descended
    from that branch belongs to the same manner class; otherwise it gets
    ``mixed_color``.
    """
    n = len(families)
    leaves_under: Dict[int, set] = {i: {i} for i in range(n)}
    for i, row in enumerate(Z):
        a, b = int(row[0]), int(row[1])
        leaves_under[n + i] = leaves_under[a] | leaves_under[b]

    colors: Dict[int, str] = {}
    for node_id in range(n, 2 * n - 1):
        fams = {families[idx] for idx in leaves_under[node_id]}
        colors[node_id] = (
            family_colors[next(iter(fams))] if len(fams) == 1 else mixed_color
        )

    def _color_func(k: int) -> str:
        return colors.get(k, mixed_color)

    return _color_func, leaves_under


# ────────────────────────────────────────────────────────────────────────
# Plotting
# ────────────────────────────────────────────────────────────────────────

def _set_academic_style() -> None:
    sns.set_theme(context="paper", style="ticks", font="serif")
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.frameon": False,
        "figure.dpi": 150,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.format": "png",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _despine(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_heavy_dendrogram(
    Z: np.ndarray,
    phones: Sequence[str],
    families: Sequence[str],
    n_components: int,
    variance_target: float,
) -> plt.Figure:
    """Render the publication-grade dendrogram."""
    fam_of = dict(zip(phones, families))
    color_func, leaves_under = _build_link_color_func(
        Z, families, FAMILY_COLORS, mixed_color="#8A8A8A",
    )

    fig, ax = plt.subplots(figsize=(8.6, 10.4))
    dn = dendrogram(
        Z,
        orientation="right",
        labels=list(phones),
        leaf_font_size=11,
        link_color_func=color_func,
        above_threshold_color="#8A8A8A",
        ax=ax,
    )

    # Colour + bold leaf labels by family.
    for lbl in ax.get_ymajorticklabels():
        p = lbl.get_text()
        lbl.set_color(FAMILY_COLORS[fam_of[p]])
        lbl.set_fontweight("bold")

    ax.set_xlabel(
        f"Ward linkage distance  "
        f"(L2 + PCA-{n_components}, {variance_target:.0%} variance)",
        labelpad=8,
    )
    ax.set_title(
        "Emergent Phonetic Taxonomy\n"
        "Hierarchical clustering of CorticalLIME tuning patterns",
        pad=14,
    )
    _despine(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(True, axis="x", ls=":", lw=0.6, alpha=0.55)

    # ── Sonorant / Obstruent macro-split annotation. ──
    # Re-derive the on-screen y-coords scipy gave us and decide whether
    # the tree actually separates the two macro-classes (annotate only if
    # so — never lie about a structure that isn't there).
    ordered_phones = [t.get_text() for t in ax.get_ymajorticklabels()]
    ordered_fams = [fam_of[p] for p in ordered_phones]
    sonorant = {"Vowels", "Nasals"}
    obstruent = {"Stops", "Fricatives"}

    def _is_clean_block(labels: Sequence[str], target: set) -> bool:
        """True iff all positions whose family is in `target` form a
        contiguous, undivided run in `labels`."""
        idxs = [i for i, f in enumerate(labels) if f in target]
        if not idxs:
            return False
        return idxs == list(range(min(idxs), max(idxs) + 1))

    son_clean = _is_clean_block(ordered_fams, sonorant)
    obs_clean = _is_clean_block(ordered_fams, obstruent)
    if son_clean and obs_clean:
        # scipy assigns each leaf a y-coordinate of 5 + 10*i (i=row index).
        x_max = float(np.max(Z[:, 2]))
        x_anno = x_max * 1.06
        ax.set_xlim(right=x_max * 1.30)

        for label, target in [
            ("Sonorants\n(Vowels + Nasals)", sonorant),
            ("Obstruents\n(Stops + Fricatives)", obstruent),
        ]:
            idxs = [i for i, f in enumerate(ordered_fams) if f in target]
            y0 = 10 * (min(idxs) + 0.5) - 4
            y1 = 10 * (max(idxs) + 0.5) + 4
            ax.annotate(
                "",
                xy=(x_anno, y0), xytext=(x_anno, y1),
                arrowprops=dict(
                    arrowstyle="-", lw=1.8, color="#3A3A3A",
                    shrinkA=0, shrinkB=0,
                ),
                annotation_clip=False,
            )
            ax.text(
                x_anno + (x_max * 0.04), (y0 + y1) / 2, label,
                va="center", ha="left", fontsize=11, fontweight="bold",
                color="#3A3A3A",
            )
    else:
        # Honest reporting: the tree did not separate the macro-classes
        # into contiguous blocks.  We print a note rather than annotate
        # something misleading on the figure.
        print(
            "[plot] Sonorant <-> Obstruent macro-split not cleanly "
            "recovered by the tree — annotation suppressed."
        )

    # Legend.
    handles = [
        plt.Line2D([0], [0], color=FAMILY_COLORS[f], lw=3, label=f)
        for f in _FAMILY_ORDER
    ]
    ax.legend(
        handles=handles, loc="lower right",
        title="Manner of articulation", title_fontsize=10,
    )
    fig.tight_layout()
    return fig


# ────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default=str(_DEFAULT_RESULTS),
                   help="Path to results_raw.npz.")
    p.add_argument("--output_dir", default=str(_DEFAULT_OUTPUT_DIR),
                   help="Where to write the rendered PNG.")
    p.add_argument("--variance", type=float, default=0.95,
                   help="PCA variance fraction to retain (e.g. 0.95).")
    p.add_argument(
        "--output_name", default="fig3_heavy_phonetic_taxonomy.png",
        help="Output PNG filename (saved into --output_dir).",
    )
    args = p.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        raise SystemExit(f"results_raw.npz not found at {results_path}.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / args.output_name

    print(f"Loading {results_path} ...")
    with np.load(results_path, allow_pickle=True) as d:
        importances = np.asarray(d["importances"], dtype=np.float64)
        target_classes = np.asarray(d["target_classes"], dtype=np.int64)
        n_bands = int(d["n_bands"]) if "n_bands" in d.files else 1
    print(f"  N={importances.shape[0]}  F={importances.shape[1]}  "
          f"n_bands={n_bands}")

    # ── A. Build (39, F) phoneme matrix. ──
    X, phones, families = build_phoneme_matrix(importances, target_classes)
    print(f"\n[A] Phoneme matrix: {X.shape[0]} phones × "
          f"{X.shape[1]} features.")
    print(f"    Phones retained: {', '.join(phones)}")

    # ── B + C. L2 normalise + PCA-denoise. ──
    X_pca, k, evr = heavy_preprocess(X, variance_target=args.variance)
    print(f"\n[B] L2-normalised rows.")
    print(f"[C] PCA(svd_solver='full', n_components={args.variance:.2f}) "
          f"⇒ retained k = {k} components")
    print(f"    Cumulative variance explained: {evr.sum():.4f}")
    print(f"    First 10 component ratios: "
          f"{', '.join(f'{r:.3f}' for r in evr[:10])}")

    # ── Cluster. ──
    Z, _ = cluster_ward(X_pca)
    print(f"\n[D] Ward linkage on Euclidean distances in {k}-D PCA space.")

    # ── Render. ──
    _set_academic_style()
    fig = plot_heavy_dendrogram(
        Z, phones, families,
        n_components=k, variance_target=args.variance,
    )
    fig.savefig(png_path)
    plt.close(fig)
    print(f"\nSaved → {png_path}")


if __name__ == "__main__":
    main()
