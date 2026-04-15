#!/usr/bin/env python3
"""
CorticalLIME population-level analysis over TIMIT.

This script runs CorticalLIME (and comparison baselines) across a sample
of TIMIT utterances and produces the full set of figures and tables for
the "Interpreting Biomimetic Auditory Neural Networks" paper.

Analyses
--------
  A.  Dataset & model overview
  B.  Single-utterance deep dive (3 methods side-by-side)
  C.  Faithfulness across the dataset (deletion/insertion AUC distributions)
  D.  Stability analysis (seed + noise)
  E.  Per-phoneme STRF importance profiles
  F.  Manner-of-articulation comparison (stops / fricatives / vowels / nasals)
  G.  Voicing contrasts (b/p, d/t, g/k, z/s, ...)
  H.  Place-of-articulation within stops & fricatives
  I.  STRF filter utilisation — universal vs. class-specific channels
  J.  Model confidence vs. explanation quality
  K.  Temporal-rate vs. spectral-scale decomposition
  L.  Cross-method agreement at population level

Each section writes one or more publication-quality PDF/PNG figures into
an output directory.

Usage
-----
    # From the r_code/ directory (so cochlear_filter_params.npz is found):
    cd /path/to/Cortical_Front/r_code
    python ../cortical_lime_analysis.py \\
        --n_utterances 200 \\
        --n_lime_samples 2000 \\
        --output_dir ../analysis_outputs

    # Or use defaults (100 utterances, 1500 perturbations):
    python ../cortical_lime_analysis.py
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
import glob
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy import stats

# ── Repo imports ──────────────────────────────────────────────────────────
# Expect to be run from r_code/ or with r_code on sys.path.
SCRIPT_DIR = Path(__file__).resolve().parent
R_CODE = SCRIPT_DIR.parent / "r_code" if (SCRIPT_DIR.parent / "r_code").is_dir() else SCRIPT_DIR
sys.path.insert(0, str(R_CODE))
sys.path.insert(0, str(SCRIPT_DIR))

from cortical_lime import (
    CorticalLIME, OcclusionSensitivity, CorticalIntegratedGradients,
    make_jax_callables, bootstrap_importances, build_model,
)
from cortical_lime_metrics import (
    deletion_curve, insertion_curve, random_baseline_curves,
    aopc, infidelity,
    explanation_stability,
    build_phoneme_profiles, phoneme_family_comparison,
    cross_method_agreement, rank_correlation_matrix,
)
from timit_dataset import (
    TIMITDataset, TIMITUtterance,
    TIMIT_PHONEMES, IDX_TO_PHONEME, PHONEME_TO_IDX,
    PHONEME_FAMILIES, VOICING_PAIRS, PLACE_GROUPS,
)

warnings.filterwarnings("ignore", category=FutureWarning)


# ═══════════════════════════════════════════════════════════════════════════
# Plotting defaults
# ═══════════════════════════════════════════════════════════════════════════

def set_pub_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "font.size": 10,
        "font.family": "serif",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "figure.constrained_layout.use": True,
    })

FAMILY_COLORS = {
    "vowels": "#e74c3c",
    "stops": "#2980b9",
    "fricatives": "#27ae60",
    "nasals": "#8e44ad",
    "affricates": "#e67e22",
    "closures": "#95a5a6",
    "semivowels_glides": "#1abc9c",
    "silence": "#bdc3c7",
}


# ═══════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════

def load_model(model_dir: str):
    """Load the latest TIMIT phoneme checkpoint and build callables."""
    mdir = Path(model_dir)
    ckpts = sorted(
        glob.glob(str(mdir / "chkStep_*.p")),
        key=lambda p: int(re.search(r"chkStep_(\d+)\.p$", p).group(1)),
    )
    if not ckpts:
        raise FileNotFoundError(f"No chkStep_*.p in {mdir}")
    ckpt = ckpts[-1]
    print(f"Checkpoint: {ckpt}")

    with open(ckpt, "rb") as f:
        obj = pickle.load(f)
    nn_params, aud_params = (
        obj if isinstance(obj, (list, tuple))
        else (obj["nn_params"], obj["params"])
    )

    model, n_phones = build_model(nn_params, aud_params)
    encode_fn, decode_fn = make_jax_callables(model, nn_params, aud_params)

    # Warm-up JIT.
    _d = np.zeros((1, 16000), dtype=np.float32)
    encode_fn(_d); decode_fn(encode_fn(_d))

    sr_pairs = np.asarray(aud_params["sr"])
    return model, nn_params, aud_params, encode_fn, decode_fn, sr_pairs


# ═══════════════════════════════════════════════════════════════════════════
# Helper: get model prediction for an utterance
# ═══════════════════════════════════════════════════════════════════════════

def model_predict(encode_fn, decode_fn, wav):
    """Return (predicted_class, prob_vector, entropy)."""
    feats = encode_fn(wav[None, :])
    logits = np.asarray(decode_fn(feats))
    if logits.ndim == 4:
        logits = logits.mean(axis=2)
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = (e / e.sum(axis=-1, keepdims=True)).mean(axis=1)[0]
    pred = int(np.argmax(probs))
    entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    return pred, probs, entropy


# ═══════════════════════════════════════════════════════════════════════════
# A.  Dataset & model overview
# ═══════════════════════════════════════════════════════════════════════════

def analysis_A_dataset_overview(ds: TIMITDataset, out: Path):
    print("\n=== A. Dataset overview ===")
    s = ds.summary()
    for k, v in s.items():
        print(f"  {k}: {v}")

    # Figure A1: Phone distribution.
    dist = ds.phone_distribution(fold_to_39=False)
    phones = list(dist.keys())[:30]
    counts = [dist[p] for p in phones]

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.barh(range(len(phones)), counts, color="#2a7f62", edgecolor="white", lw=0.3)
    ax.set_yticks(range(len(phones)))
    ax.set_yticklabels(phones, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Total frames")
    ax.set_title(f"Phone distribution (top 30, {s['n_utterances']} utterances)")
    fig.savefig(out / "A1_phone_distribution.png")
    plt.close(fig)

    # Figure A2: Duration histogram.
    durs = [u.duration_sec for u in ds]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(durs, bins=30, color="#555", edgecolor="white", lw=0.3)
    ax.set_xlabel("Utterance duration (s)")
    ax.set_ylabel("Count")
    ax.set_title("Utterance duration distribution")
    fig.savefig(out / "A2_duration_hist.png")
    plt.close(fig)

    # Figure A3: Dialect region counts.
    regions = defaultdict(int)
    for u in ds:
        regions[u.dialect_region] += 1
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(sorted(regions.keys()), [regions[r] for r in sorted(regions.keys())],
           color="#2980b9", edgecolor="white", lw=0.3)
    ax.set_xlabel("Dialect region")
    ax.set_ylabel("Utterances")
    ax.set_title("Dialect region coverage")
    fig.savefig(out / "A3_dialect_regions.png")
    plt.close(fig)
    print("  Figures saved: A1, A2, A3")


# ═══════════════════════════════════════════════════════════════════════════
# B.  Single-utterance deep dive
# ═══════════════════════════════════════════════════════════════════════════

def analysis_B_single_utterance(
    utt: TIMITUtterance,
    explainer: CorticalLIME,
    occ: OcclusionSensitivity,
    ig_explainer: CorticalIntegratedGradients,
    sr_pairs: np.ndarray,
    out: Path,
):
    print("\n=== B. Single-utterance deep dive ===")
    wav = utt.segment_audio(1.0, mode="center")
    result = explainer.explain(wav)
    occ_r = occ.explain(wav, target_class=result.target_class)
    ig_r = ig_explainer.explain(wav, target_class=result.target_class)

    phn = IDX_TO_PHONEME.get(result.target_class, "?")
    print(f"  Utterance: {utt.utterance_id}")
    print(f"  Predicted: /{phn}/  P={result.target_prob:.4f}  R²={result.surrogate_r2:.4f}")

    methods = {
        "CorticalLIME": result.importances,
        "Occlusion": occ_r["importances"],
        "Integrated Gradients": ig_r["importances"],
    }

    # Figure B1: Three methods side-by-side scatter.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, (name, imp) in zip(axes, methods.items()):
        vmax = np.max(np.abs(imp)) + 1e-12
        sc = ax.scatter(
            sr_pairs[:, 1], sr_pairs[:, 0],
            c=imp, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
            s=60 + 400 * (np.abs(imp) / vmax),
            edgecolors="k", linewidths=0.4, zorder=3,
        )
        ax.axvline(0, color="grey", lw=0.5, ls=":")
        ax.set_xlabel(r"$\omega$ (Hz)")
        ax.set_title(name, fontsize=11)
    axes[0].set_ylabel(r"$\Omega$ (cyc/oct)")
    fig.suptitle(
        f'XAI comparison — /{phn}/  (P={result.target_prob:.3f})',
        y=1.02, fontsize=13,
    )
    fig.savefig(out / "B1_three_methods.png")
    plt.close(fig)

    # Figure B2: Surrogate diagnostics.
    y_true = result.target_probs
    y_pred = result.masks @ result.importances + result.intercept

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].scatter(y_pred, y_true, c=result.weights, cmap="viridis", s=6, alpha=0.6)
    lims = [min(y_pred.min(), y_true.min()), max(y_pred.max(), y_true.max())]
    axes[0].plot(lims, lims, "k--", lw=0.7)
    axes[0].set(xlabel="Ridge prediction", ylabel="P(class)",
                title=f"Fidelity (R²={result.surrogate_r2:.3f})")

    axes[1].hist(result.distances, bins=30, color="#555", edgecolor="white", lw=0.3)
    axes[1].set(xlabel="Cosine distance", ylabel="Count", title="Distance distribution")

    axes[2].hist(result.weights, bins=30, color="#2a7f62", edgecolor="white", lw=0.3)
    axes[2].set(xlabel="Kernel weight", ylabel="Count", title="Kernel weights")
    fig.savefig(out / "B2_surrogate_diagnostics.png")
    plt.close(fig)

    # Figure B3: Bootstrap CIs.
    boot_mean, boot_lo, boot_hi = bootstrap_importances(
        result, n_bootstrap=500, alpha_ci=0.05, seed=42,
    )
    order = np.argsort(-np.abs(boot_mean))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(boot_mean))
    ax.bar(x, boot_mean[order], color="#2a7f62", edgecolor="white", lw=0.3)
    ax.vlines(x, boot_lo[order], boot_hi[order], color="k", lw=0.8)
    ax.axhline(0, color="k", lw=0.5)
    sig = (boot_lo[order] > 0) | (boot_hi[order] < 0)
    for i, s in enumerate(sig):
        if s:
            ax.scatter(i, boot_hi[order][i] + 0.002, marker="*", color="goldenrod", s=25, zorder=5)
    ax.set(xlabel="STRF channel (sorted)", ylabel="Coefficient",
           title=f"95% bootstrap CIs — {int(sig.sum())}/{len(sig)} significant")
    fig.savefig(out / "B3_bootstrap_ci.png")
    plt.close(fig)

    # Cross-method agreement.
    print("  Cross-method agreement:")
    for (a, ia), (b, ib) in [
        (("CorticalLIME", result.importances), ("Occlusion", occ_r["importances"])),
        (("CorticalLIME", result.importances), ("IntGrad", ig_r["importances"])),
    ]:
        ag = cross_method_agreement(ia, ib)
        print(f"    {a} vs {b}: ρ={ag['spearman_rho']:.3f}  "
              f"top5={ag['top5_overlap']}/5")

    print("  Figures saved: B1, B2, B3")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# C.  Faithfulness across dataset
# ═══════════════════════════════════════════════════════════════════════════

def analysis_C_faithfulness(
    utterances: list[TIMITUtterance],
    results: list,
    encode_fn, decode_fn,
    sr_pairs: np.ndarray,
    out: Path,
):
    print("\n=== C. Faithfulness across dataset ===")
    del_aucs_lime, ins_aucs_lime = [], []
    del_aucs_rand, ins_aucs_rand = [], []
    aopc_vals, infid_vals = [], []

    n = min(len(utterances), len(results))
    for i in range(n):
        wav = utterances[i].segment_audio(1.0)
        feats = encode_fn(wav[None, :])[0]
        tc = results[i].target_class
        imp = results[i].importances

        dc = deletion_curve(feats, imp, decode_fn, tc)
        ic = insertion_curve(feats, imp, decode_fn, tc)
        del_aucs_lime.append(dc.auc)
        ins_aucs_lime.append(ic.auc)

        a = aopc(feats, imp, decode_fn, tc, K=10)
        aopc_vals.append(a)

        inf = infidelity(feats, imp, decode_fn, tc, n_samples=100)
        infid_vals.append(inf)

        # One random baseline per utterance.
        rand_imp = np.random.default_rng(i).random(sr_pairs.shape[0]).astype(np.float32)
        rd = deletion_curve(feats, rand_imp, decode_fn, tc)
        ri = insertion_curve(feats, rand_imp, decode_fn, tc)
        del_aucs_rand.append(rd.auc)
        ins_aucs_rand.append(ri.auc)

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n}...")

    del_aucs_lime = np.array(del_aucs_lime)
    ins_aucs_lime = np.array(ins_aucs_lime)
    del_aucs_rand = np.array(del_aucs_rand)
    ins_aucs_rand = np.array(ins_aucs_rand)

    print(f"  Deletion AUC:  LIME {del_aucs_lime.mean():.4f}±{del_aucs_lime.std():.4f}  "
          f"Random {del_aucs_rand.mean():.4f}±{del_aucs_rand.std():.4f}")
    print(f"  Insertion AUC: LIME {ins_aucs_lime.mean():.4f}±{ins_aucs_lime.std():.4f}  "
          f"Random {ins_aucs_rand.mean():.4f}±{ins_aucs_rand.std():.4f}")
    print(f"  AOPC: {np.mean(aopc_vals):.4f}±{np.std(aopc_vals):.4f}")
    print(f"  Infidelity: {np.mean(infid_vals):.6f}±{np.std(infid_vals):.6f}")

    # Statistical test: LIME vs Random.
    t_del, p_del = stats.ttest_rel(del_aucs_lime, del_aucs_rand)
    t_ins, p_ins = stats.ttest_rel(ins_aucs_lime, ins_aucs_rand)
    print(f"  Paired t-test (del): t={t_del:.3f}  p={p_del:.2e}")
    print(f"  Paired t-test (ins): t={t_ins:.3f}  p={p_ins:.2e}")

    # Figure C1: Deletion / Insertion AUC distributions.
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (cl, cr, title, direction) in zip(axes, [
        (del_aucs_lime, del_aucs_rand, "Deletion AUC", "↓ better"),
        (ins_aucs_lime, ins_aucs_rand, "Insertion AUC", "↑ better"),
    ]):
        bins = np.linspace(
            min(cl.min(), cr.min()) - 0.02,
            max(cl.max(), cr.max()) + 0.02, 25,
        )
        ax.hist(cr, bins=bins, alpha=0.5, color="grey", label="Random", edgecolor="white", lw=0.3)
        ax.hist(cl, bins=bins, alpha=0.7, color="#c0392b", label="CorticalLIME", edgecolor="white", lw=0.3)
        ax.set(xlabel=f"{title}  ({direction})", ylabel="Count")
        ax.set_title(title)
        ax.legend()
    fig.savefig(out / "C1_faithfulness_distributions.png")
    plt.close(fig)

    # Figure C2: AOPC vs Infidelity scatter.
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(aopc_vals, infid_vals, s=12, alpha=0.6, color="#2980b9")
    ax.set_xlabel("AOPC (↑ better)")
    ax.set_ylabel("Infidelity (↓ better)")
    ax.set_title("Faithfulness: AOPC vs Infidelity per utterance")
    fig.savefig(out / "C2_aopc_vs_infidelity.png")
    plt.close(fig)
    print("  Figures saved: C1, C2")

    return dict(
        del_aucs_lime=del_aucs_lime, ins_aucs_lime=ins_aucs_lime,
        del_aucs_rand=del_aucs_rand, ins_aucs_rand=ins_aucs_rand,
        aopc_vals=np.array(aopc_vals), infid_vals=np.array(infid_vals),
    )


# ═══════════════════════════════════════════════════════════════════════════
# D.  Stability
# ═══════════════════════════════════════════════════════════════════════════

def analysis_D_stability(
    utt: TIMITUtterance,
    explainer: CorticalLIME,
    sr_pairs: np.ndarray,
    out: Path,
):
    print("\n=== D. Stability analysis ===")
    wav = utt.segment_audio(1.0)
    stab = explanation_stability(explainer, wav, n_runs=10)
    print(f"  Seed stability: ρ={stab['mean_spearman']:.4f}±{stab['std_spearman']:.4f}")

    all_imps = stab["all_importances"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    im = axes[0].imshow(all_imps, aspect="auto", cmap="RdBu_r",
                        vmin=-np.max(np.abs(all_imps)), vmax=np.max(np.abs(all_imps)))
    axes[0].set(xlabel="STRF channel", ylabel="Seed", title="Importances across seeds")
    fig.colorbar(im, ax=axes[0], shrink=0.85)

    mean_imp = all_imps.mean(0)
    std_imp = all_imps.std(0)
    order = np.argsort(-np.abs(mean_imp))
    axes[1].bar(range(len(mean_imp)), mean_imp[order], yerr=std_imp[order],
                capsize=1.5, color="#2a7f62", edgecolor="white", lw=0.3)
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set(xlabel="STRF (sorted)", ylabel="Coef", title="Mean ± std across seeds")
    fig.savefig(out / "D1_stability.png")
    plt.close(fig)
    print("  Figure saved: D1")


# ═══════════════════════════════════════════════════════════════════════════
# E.  Per-phoneme STRF importance profiles
# ═══════════════════════════════════════════════════════════════════════════

def analysis_E_phoneme_profiles(
    results: list,
    sr_pairs: np.ndarray,
    out: Path,
):
    print("\n=== E. Per-phoneme profiles ===")
    profiles = build_phoneme_profiles(results, IDX_TO_PHONEME)
    top_phns = sorted(profiles.items(), key=lambda x: -x[1].n_utterances)[:9]

    # Figure E1: Top-9 phoneme scatter grids.
    n_show = min(9, len(top_phns))
    rows = (n_show + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(13, 3.5 * rows), sharey=True)
    axes = axes.ravel() if n_show > 3 else [axes] if n_show == 1 else axes
    for ax, (phn, prof) in zip(axes, top_phns[:n_show]):
        vmax = np.max(np.abs(prof.mean_importances)) + 1e-12
        ax.scatter(
            sr_pairs[:, 1], sr_pairs[:, 0],
            c=prof.mean_importances, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
            s=30 + 250 * (np.abs(prof.mean_importances) / vmax),
            edgecolors="k", linewidths=0.3,
        )
        ax.axvline(0, color="grey", lw=0.5, ls=":")
        ax.set_title(f"/{phn}/  (n={prof.n_utterances}, R²={prof.mean_r2:.2f})", fontsize=10)
    for ax in axes[:n_show]:
        ax.set_xlabel(r"$\omega$", fontsize=9)
    for i in range(0, n_show, 3):
        axes[i].set_ylabel(r"$\Omega$")
    # Hide unused axes.
    for ax in axes[n_show:]:
        ax.set_visible(False)
    fig.suptitle("Mean CorticalLIME importance per predicted phoneme", y=1.01, fontsize=13)
    fig.savefig(out / "E1_phoneme_profiles.png")
    plt.close(fig)

    # Figure E2: Mean absolute importance per phoneme (bar chart).
    mean_abs = {phn: float(np.mean(np.abs(p.mean_importances)))
                for phn, p in profiles.items() if p.n_utterances >= 3}
    sorted_phns = sorted(mean_abs, key=lambda p: -mean_abs[p])[:20]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(range(len(sorted_phns)), [mean_abs[p] for p in sorted_phns],
           color="#2980b9", edgecolor="white", lw=0.3)
    ax.set_xticks(range(len(sorted_phns)))
    ax.set_xticklabels([f"/{p}/" for p in sorted_phns], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean |importance|")
    ax.set_title("Phonemes ranked by mean absolute STRF importance")
    fig.savefig(out / "E2_importance_magnitude.png")
    plt.close(fig)
    print(f"  {len(profiles)} phoneme profiles built. Figures saved: E1, E2")
    return profiles


# ═══════════════════════════════════════════════════════════════════════════
# F.  Manner-of-articulation comparison
# ═══════════════════════════════════════════════════════════════════════════

def analysis_F_manner(profiles: dict, sr_pairs: np.ndarray, out: Path):
    print("\n=== F. Manner-of-articulation comparison ===")
    comparisons = phoneme_family_comparison(profiles, PHONEME_FAMILIES)

    for pair, res in comparisons.items():
        print(f"  {pair}: {res['n_significant']}/{sr_pairs.shape[0]} significant channels")

    # Figure F1: Family-averaged importance curves.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for fname, members in PHONEME_FAMILIES.items():
        rows = [profiles[p].mean_importances for p in members if p in profiles]
        if len(rows) < 2:
            continue
        fam_mean = np.mean(rows, axis=0)
        fam_std = np.std(rows, axis=0)
        order = np.argsort(-np.abs(fam_mean))
        c = FAMILY_COLORS.get(fname, "grey")
        ax.plot(range(sr_pairs.shape[0]), fam_mean[order], "-", lw=1.5,
                color=c, label=fname)
        ax.fill_between(range(sr_pairs.shape[0]),
                        fam_mean[order] - fam_std[order],
                        fam_mean[order] + fam_std[order],
                        alpha=0.12, color=c)
    ax.axhline(0, color="k", lw=0.5)
    ax.set(xlabel="STRF channel (sorted)", ylabel="Mean coefficient",
           title="Manner-of-articulation importance profiles")
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(out / "F1_manner_profiles.png")
    plt.close(fig)

    # Figure F2: Effect sizes for key comparisons.
    key_pairs = ["stops_vs_vowels", "stops_vs_fricatives", "fricatives_vs_vowels",
                 "vowels_vs_nasals"]
    available = [p for p in key_pairs if p in comparisons]
    if available:
        fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4), sharey=True)
        if len(available) == 1:
            axes = [axes]
        for ax, pair in zip(axes, available):
            cd = comparisons[pair]["effect_sizes"]
            cdmax = np.max(np.abs(cd)) + 0.1
            ax.scatter(
                sr_pairs[:, 1], sr_pairs[:, 0],
                c=cd, cmap="PiYG", vmin=-cdmax, vmax=cdmax,
                s=40 + 300 * (np.abs(cd) / cdmax),
                edgecolors="k", linewidths=0.3,
            )
            ax.axvline(0, color="grey", lw=0.5, ls=":")
            ax.set_xlabel(r"$\omega$ (Hz)")
            fa, fb = pair.split("_vs_")
            ax.set_title(f"Cohen's d: {fa} − {fb}", fontsize=10)
        axes[0].set_ylabel(r"$\Omega$ (cyc/oct)")
        fig.savefig(out / "F2_manner_effect_sizes.png")
        plt.close(fig)

    print("  Figures saved: F1, F2")
    return comparisons


# ═══════════════════════════════════════════════════════════════════════════
# G.  Voicing contrasts
# ═══════════════════════════════════════════════════════════════════════════

def analysis_G_voicing(profiles: dict, sr_pairs: np.ndarray, out: Path):
    print("\n=== G. Voicing contrasts ===")
    pairs_with_data = []
    for v, uv in VOICING_PAIRS:
        if v in profiles and uv in profiles:
            pairs_with_data.append((v, uv))

    if not pairs_with_data:
        print("  Insufficient data for voicing analysis.")
        return

    n = len(pairs_with_data)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (v, uv) in zip(axes, pairs_with_data):
        diff = profiles[v].mean_importances - profiles[uv].mean_importances
        vmax = np.max(np.abs(diff)) + 1e-12
        ax.scatter(
            sr_pairs[:, 1], sr_pairs[:, 0],
            c=diff, cmap="PiYG", vmin=-vmax, vmax=vmax,
            s=40 + 300 * (np.abs(diff) / vmax),
            edgecolors="k", linewidths=0.3,
        )
        ax.axvline(0, color="grey", lw=0.5, ls=":")
        ax.set_xlabel(r"$\omega$ (Hz)")
        ax.set_title(f"/{v}/ − /{uv}/", fontsize=10)
    axes[0].set_ylabel(r"$\Omega$ (cyc/oct)")
    fig.suptitle("Voicing contrasts (voiced − voiceless)", y=1.02, fontsize=13)
    fig.savefig(out / "G1_voicing_contrasts.png")
    plt.close(fig)
    print(f"  {len(pairs_with_data)} pairs plotted. Figure saved: G1")


# ═══════════════════════════════════════════════════════════════════════════
# H.  Place-of-articulation
# ═══════════════════════════════════════════════════════════════════════════

def analysis_H_place(profiles: dict, sr_pairs: np.ndarray, out: Path):
    print("\n=== H. Place-of-articulation ===")
    comps = phoneme_family_comparison(profiles, PLACE_GROUPS)
    for pair, res in comps.items():
        print(f"  {pair}: {res['n_significant']} significant channels")

    # Figure H1: Mean importance per place group.
    fig, ax = plt.subplots(figsize=(7, 4))
    place_colors = {"labial": "#e74c3c", "alveolar": "#2980b9",
                    "velar": "#27ae60", "palatal": "#8e44ad", "dental": "#e67e22"}
    for pname, members in PLACE_GROUPS.items():
        rows = [profiles[p].mean_importances for p in members if p in profiles]
        if not rows:
            continue
        fam_mean = np.mean(rows, axis=0)
        order = np.argsort(-np.abs(fam_mean))
        ax.plot(range(sr_pairs.shape[0]), fam_mean[order], "-o", ms=2, lw=1.2,
                color=place_colors.get(pname, "grey"), label=pname)
    ax.axhline(0, color="k", lw=0.5)
    ax.set(xlabel="STRF channel (sorted)", ylabel="Mean coefficient",
           title="Place-of-articulation importance profiles")
    ax.legend()
    fig.savefig(out / "H1_place_profiles.png")
    plt.close(fig)
    print("  Figure saved: H1")


# ═══════════════════════════════════════════════════════════════════════════
# I.  STRF filter utilisation
# ═══════════════════════════════════════════════════════════════════════════

def analysis_I_utilisation(results: list, sr_pairs: np.ndarray, out: Path):
    print("\n=== I. STRF filter utilisation ===")
    all_imps = np.stack([r.importances for r in results])  # (N, S)
    S = all_imps.shape[1]

    # Per-channel statistics across all utterances.
    mean_abs = np.mean(np.abs(all_imps), axis=0)
    std_across = np.std(all_imps, axis=0)
    # Coefficient of variation of |importance| — high CV = class-specific.
    cv = std_across / (mean_abs + 1e-12)

    # Figure I1: Universal vs. class-specific channels.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    order = np.argsort(-mean_abs)
    axes[0].bar(range(S), mean_abs[order], color="#2a7f62", edgecolor="white", lw=0.3)
    axes[0].set(xlabel="STRF channel (sorted by mean |coef|)", ylabel="Mean |coef|",
                title="Per-channel mean absolute importance")

    vmax = np.max(np.abs(mean_abs)) + 1e-12
    sc = axes[1].scatter(
        sr_pairs[:, 1], sr_pairs[:, 0],
        c=cv, cmap="inferno", s=60 + 400 * (mean_abs / vmax),
        edgecolors="k", linewidths=0.4,
    )
    axes[1].axvline(0, color="grey", lw=0.5, ls=":")
    axes[1].set_xlabel(r"$\omega$ (Hz)")
    axes[1].set_ylabel(r"$\Omega$ (cyc/oct)")
    axes[1].set_title("CV of importance (high = class-specific)")
    fig.colorbar(sc, ax=axes[1], label="CV", shrink=0.85)
    fig.savefig(out / "I1_filter_utilisation.png")
    plt.close(fig)

    # Figure I2: Importance correlation matrix between channels.
    corr = np.corrcoef(all_imps.T)  # (S, S)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set(xlabel="STRF channel", ylabel="STRF channel",
           title="Inter-channel importance correlation")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.savefig(out / "I2_channel_correlation.png")
    plt.close(fig)
    print("  Figures saved: I1, I2")


# ═══════════════════════════════════════════════════════════════════════════
# J.  Model confidence vs. explanation quality
# ═══════════════════════════════════════════════════════════════════════════

def analysis_J_confidence(
    utterances: list,
    results: list,
    encode_fn, decode_fn,
    out: Path,
):
    print("\n=== J. Confidence vs. explanation quality ===")
    entropies, r2s, probs = [], [], []
    for utt, res in zip(utterances, results):
        wav = utt.segment_audio(1.0)
        _, _, ent = model_predict(encode_fn, decode_fn, wav)
        entropies.append(ent)
        r2s.append(res.surrogate_r2)
        probs.append(res.target_prob)

    entropies = np.array(entropies)
    r2s = np.array(r2s)
    probs = np.array(probs)

    rho_r2, p_r2 = stats.spearmanr(entropies, r2s)
    rho_prob, p_prob = stats.spearmanr(probs, r2s)
    print(f"  Entropy vs R²: ρ={rho_r2:.3f}  p={p_r2:.2e}")
    print(f"  P(class) vs R²: ρ={rho_prob:.3f}  p={p_prob:.2e}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].scatter(entropies, r2s, s=15, alpha=0.6, color="#2980b9")
    axes[0].set(xlabel="Prediction entropy", ylabel="Surrogate R²",
                title=f"Entropy vs R²  (ρ={rho_r2:.3f})")

    axes[1].scatter(probs, r2s, s=15, alpha=0.6, color="#27ae60")
    axes[1].set(xlabel="P(predicted class)", ylabel="Surrogate R²",
                title=f"Confidence vs R²  (ρ={rho_prob:.3f})")
    fig.savefig(out / "J1_confidence_vs_r2.png")
    plt.close(fig)
    print("  Figure saved: J1")


# ═══════════════════════════════════════════════════════════════════════════
# K.  Rate vs. Scale decomposition
# ═══════════════════════════════════════════════════════════════════════════

def analysis_K_rate_vs_scale(results: list, sr_pairs: np.ndarray, out: Path):
    print("\n=== K. Rate vs. Scale decomposition ===")
    all_imps = np.stack([r.importances for r in results])
    rates = sr_pairs[:, 1]
    scales = sr_pairs[:, 0]

    # For each utterance, compute fraction of total |importance| attributable
    # to high-rate (|ω| > median) vs. low-rate channels.
    rate_thresh = np.median(np.abs(rates))
    high_rate_mask = np.abs(rates) > rate_thresh
    low_rate_mask = ~high_rate_mask

    scale_thresh = np.median(scales)
    high_scale_mask = scales > scale_thresh
    low_scale_mask = ~high_scale_mask

    frac_high_rate = np.abs(all_imps[:, high_rate_mask]).sum(1) / (np.abs(all_imps).sum(1) + 1e-12)
    frac_high_scale = np.abs(all_imps[:, high_scale_mask]).sum(1) / (np.abs(all_imps).sum(1) + 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].hist(frac_high_rate, bins=25, color="#c0392b", edgecolor="white", lw=0.3, alpha=0.8)
    axes[0].axvline(0.5, color="k", lw=0.8, ls="--")
    axes[0].set(xlabel=f"Fraction |imp| in high-rate (|ω|>{rate_thresh:.1f} Hz)",
                ylabel="Count", title="Temporal rate contribution")

    axes[1].hist(frac_high_scale, bins=25, color="#2980b9", edgecolor="white", lw=0.3, alpha=0.8)
    axes[1].axvline(0.5, color="k", lw=0.8, ls="--")
    axes[1].set(xlabel=f"Fraction |imp| in high-scale (Ω>{scale_thresh:.1f} cyc/oct)",
                ylabel="Count", title="Spectral scale contribution")
    fig.savefig(out / "K1_rate_vs_scale.png")
    plt.close(fig)

    # Figure K2: Rate-marginal and scale-marginal importance.
    abs_imp_mean = np.mean(np.abs(all_imps), axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    axes[0].scatter(np.abs(rates), abs_imp_mean, s=40, c=scales, cmap="viridis",
                    edgecolors="k", linewidths=0.3)
    axes[0].set_xlabel("|Temporal rate| (Hz)")
    axes[0].set_ylabel("Mean |importance|")
    axes[0].set_title("Importance vs. temporal rate")
    cb = fig.colorbar(axes[0].collections[0], ax=axes[0])
    cb.set_label("Scale (cyc/oct)")

    axes[1].scatter(scales, abs_imp_mean, s=40, c=np.abs(rates), cmap="magma",
                    edgecolors="k", linewidths=0.3)
    axes[1].set_xlabel("Spectral scale (cyc/oct)")
    axes[1].set_ylabel("Mean |importance|")
    axes[1].set_title("Importance vs. spectral scale")
    cb = fig.colorbar(axes[1].collections[0], ax=axes[1])
    cb.set_label("|Rate| (Hz)")
    fig.savefig(out / "K2_marginal_importance.png")
    plt.close(fig)

    print(f"  Mean frac_high_rate:  {frac_high_rate.mean():.3f}")
    print(f"  Mean frac_high_scale: {frac_high_scale.mean():.3f}")
    print("  Figures saved: K1, K2")


# ═══════════════════════════════════════════════════════════════════════════
# L.  Cross-method agreement at population level
# ═══════════════════════════════════════════════════════════════════════════

def analysis_L_cross_method(
    utterances: list,
    lime_results: list,
    occ_explainer: OcclusionSensitivity,
    out: Path,
):
    print("\n=== L. Cross-method agreement (population) ===")
    rhos = []
    n = min(30, len(utterances))  # Occlusion is slow; cap at 30.
    for i in range(n):
        wav = utterances[i].segment_audio(1.0)
        tc = lime_results[i].target_class
        occ_r = occ_explainer.explain(wav, target_class=tc)
        rho, _ = stats.spearmanr(lime_results[i].importances, occ_r["importances"])
        rhos.append(rho)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n}...")

    rhos = np.array(rhos)
    print(f"  CorticalLIME vs Occlusion:  ρ = {rhos.mean():.3f} ± {rhos.std():.3f}")

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(rhos, bins=20, color="#8e44ad", edgecolor="white", lw=0.3)
    ax.axvline(rhos.mean(), color="k", lw=1.2, ls="--",
               label=f"mean={rhos.mean():.3f}")
    ax.set(xlabel="Spearman ρ (CorticalLIME vs Occlusion)", ylabel="Count",
           title=f"Cross-method agreement (n={n})")
    ax.legend()
    fig.savefig(out / "L1_cross_method_agreement.png")
    plt.close(fig)
    print("  Figure saved: L1")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="CorticalLIME population analysis over TIMIT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_dir", type=str,
                        default="nishit_trained_models/main_jax_phoneme_rec_timit",
                        help="Checkpoint directory (relative to r_code/ or absolute).")
    parser.add_argument("--timit_local", type=str, default=None,
                        help="Local TIMIT root (skip Kaggle download).")
    parser.add_argument("--n_utterances", type=int, default=100,
                        help="Number of test utterances to analyse.")
    parser.add_argument("--n_lime_samples", type=int, default=1500,
                        help="Perturbation samples per CorticalLIME run.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="analysis_outputs",
                        help="Directory for output figures.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip", type=str, nargs="*", default=[],
                        help="Skip sections (e.g. --skip C L).")
    args = parser.parse_args()

    set_pub_style()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out.resolve()}")

    # ── Load model ────────────────────────────────────────────────────
    model, nn_params, aud_params, encode_fn, decode_fn, sr_pairs = load_model(args.model_dir)
    S = sr_pairs.shape[0]
    print(f"STRF bank: {S} channels")

    # ── Load dataset ──────────────────────────────────────────────────
    ds = TIMITDataset(split="TEST", local_path=args.timit_local)
    print(f"TIMIT TEST: {len(ds)} utterances")

    # ── A. Dataset overview ───────────────────────────────────────────
    if "A" not in args.skip:
        analysis_A_dataset_overview(ds, out)

    # ── Sample utterances ─────────────────────────────────────────────
    utts = ds.sample(args.n_utterances, seed=args.seed, exclude_sa=True)
    print(f"\nSampled {len(utts)} utterances for analysis.")

    # ── Build explainers ──────────────────────────────────────────────
    explainer = CorticalLIME(
        encode_fn=encode_fn, decode_fn=decode_fn, sr=sr_pairs,
        strategy="bernoulli", n_samples=args.n_lime_samples, keep_prob=0.85,
        kernel_width=0.25, surrogate_type="ridge", surrogate_alpha=1.0,
        batch_size=args.batch_size, seed=args.seed,
    )
    occ_explainer = OcclusionSensitivity(encode_fn, decode_fn, sr_pairs)
    ig_explainer = CorticalIntegratedGradients(model, nn_params, aud_params, n_steps=50)

    # ── B. Single-utterance deep dive ─────────────────────────────────
    if "B" not in args.skip:
        analysis_B_single_utterance(utts[0], explainer, occ_explainer, ig_explainer, sr_pairs, out)

    # ── Run CorticalLIME on all utterances ────────────────────────────
    print(f"\nRunning CorticalLIME on {len(utts)} utterances...")
    all_results = []
    for i, utt in enumerate(utts):
        wav = utt.segment_audio(1.0)
        all_results.append(explainer.explain(wav))
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(utts)}  "
                  f"(mean R²={np.mean([r.surrogate_r2 for r in all_results]):.4f})")
    print(f"Done. Mean R²={np.mean([r.surrogate_r2 for r in all_results]):.4f}")

    # ── C. Faithfulness ───────────────────────────────────────────────
    if "C" not in args.skip:
        analysis_C_faithfulness(utts, all_results, encode_fn, decode_fn, sr_pairs, out)

    # ── D. Stability ──────────────────────────────────────────────────
    if "D" not in args.skip:
        analysis_D_stability(utts[0], explainer, sr_pairs, out)

    # ── E. Phoneme profiles ───────────────────────────────────────────
    if "E" not in args.skip:
        profiles = analysis_E_phoneme_profiles(all_results, sr_pairs, out)
    else:
        profiles = build_phoneme_profiles(all_results, IDX_TO_PHONEME)

    # ── F. Manner of articulation ─────────────────────────────────────
    if "F" not in args.skip:
        analysis_F_manner(profiles, sr_pairs, out)

    # ── G. Voicing contrasts ──────────────────────────────────────────
    if "G" not in args.skip:
        analysis_G_voicing(profiles, sr_pairs, out)

    # ── H. Place of articulation ──────────────────────────────────────
    if "H" not in args.skip:
        analysis_H_place(profiles, sr_pairs, out)

    # ── I. Filter utilisation ─────────────────────────────────────────
    if "I" not in args.skip:
        analysis_I_utilisation(all_results, sr_pairs, out)

    # ── J. Confidence vs. explanation ─────────────────────────────────
    if "J" not in args.skip:
        analysis_J_confidence(utts, all_results, encode_fn, decode_fn, out)

    # ── K. Rate vs Scale ──────────────────────────────────────────────
    if "K" not in args.skip:
        analysis_K_rate_vs_scale(all_results, sr_pairs, out)

    # ── L. Cross-method agreement ─────────────────────────────────────
    if "L" not in args.skip:
        analysis_L_cross_method(utts, all_results, occ_explainer, out)

    # ── Save raw results ──────────────────────────────────────────────
    np.savez(
        out / "results_raw.npz",
        importances=np.stack([r.importances for r in all_results]),
        target_classes=np.array([r.target_class for r in all_results]),
        target_probs=np.array([r.target_prob for r in all_results]),
        surrogate_r2s=np.array([r.surrogate_r2 for r in all_results]),
        sr_pairs=sr_pairs,
        utterance_ids=np.array([u.utterance_id for u in utts]),
    )
    print(f"\nRaw results saved to {out / 'results_raw.npz'}")
    print(f"All figures in {out.resolve()}/")
    print("Done.")


if __name__ == "__main__":
    main()
