#!/usr/bin/env python3
"""
run.py — One-command launcher for the full CorticalLIME analysis pipeline.

Handles everything: dependency checks, cochlear filter cache, TIMIT download,
model loading, CorticalLIME + baselines across the dataset, and all 12
analysis sections with publication-ready PDF figures.

Usage
-----
    python run.py                         # defaults (200 utterances, full analysis)
    python run.py --profile quick         # 50 utterances, 500 perturbations
    python run.py --profile full          # 300 utterances, 3000 perturbations
    python run.py --skip C L              # skip slow sections (faithfulness, cross-method)
    python run.py --timit_local /my/timit # skip Kaggle download

Output
------
    analysis_outputs/
    ├── A1_phone_distribution.pdf
    ├── A2_duration_hist.pdf
    ├── ...
    ├── L1_cross_method_agreement.pdf
    ├── results_raw.npz
    └── run_summary.txt
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import re
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
# 0.  Resolve paths
# ═══════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent
# r_code/ lives one level above the worktree, or beside this script.
_candidate = SCRIPT_DIR.parent / "r_code"
if not _candidate.is_dir():
    _candidate = SCRIPT_DIR / "r_code"
if not _candidate.is_dir():
    # Maybe we ARE in r_code/.
    _candidate = SCRIPT_DIR
R_CODE = _candidate.resolve()

sys.path.insert(0, str(R_CODE))
sys.path.insert(0, str(SCRIPT_DIR))


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Pre-set analysis profiles
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AnalysisConfig:
    """All tuneable knobs for a CorticalLIME analysis run."""

    # -- Dataset --
    timit_split: str = "TEST"
    timit_local: str | None = None           # None → auto-download from Kaggle
    n_utterances: int = 200                   # utterances to explain
    exclude_sa: bool = True                   # SA sentences repeated → inflate coverage

    # -- Model --
    model_dir: str = "nishit_trained_models/main_jax_phoneme_rec_timit"
    n_phones: int = 61

    # -- CorticalLIME --
    n_lime_samples: int = 4000               # perturbation masks per utterance
    keep_prob: float = 0.85                   # Bernoulli keep probability (perturb ~15% of filters / sample)
    kernel_width: float = 0.25               # LIME exponential kernel σ
    surrogate_type: str = "ridge"            # ridge | lasso | bayesian_ridge
    surrogate_alpha: float = 1.0             # regularisation strength
    batch_size: int = 64                     # decode_fn mini-batch

    # -- High-resolution band-mode --
    band_mode: bool = True                   # 2-D (band × STRF) interpretable map
    save_masks: bool = True                  # persist perturbation masks (uint8)

    # -- Stability --
    stability_seeds: int = 10                # seeds for seed-consistency test
    noise_levels: list = None                # input-noise σ values
    noise_repeats: int = 5

    # -- Faithfulness --
    faithfulness_n_random: int = 1           # random baselines per utterance
    infidelity_samples: int = 100            # perturbation samples for infidelity
    aopc_K: int = 10                         # top-K for AOPC

    # -- Cross-method --
    cross_method_n: int = 30                 # utterances for occlusion comparison
    ig_steps: int = 50                       # steps for integrated gradients

    # -- Bootstrap --
    n_bootstrap: int = 500
    bootstrap_alpha: float = 0.05

    # -- Output --
    output_dir: str = "analysis_outputs"
    seed: int = 42
    skip_sections: list = None               # e.g. ["C", "L"]

    # -- Plot-4 word trajectories --
    trajectory_words: list = None            # target words for lingo fig 4
    trajectory_duration: float = 1.0         # crop length for word-centred LIME

    # -- Execution modes --
    plots_only: bool = False                 # reuse saved npz, skip all analysis
    skip_lime: bool = False                  # reuse saved LIME but still plot

    def __post_init__(self):
        if self.noise_levels is None:
            self.noise_levels = [1e-4, 1e-3, 5e-3, 1e-2, 5e-2]
        if self.skip_sections is None:
            self.skip_sections = []
        if self.trajectory_words is None:
            self.trajectory_words = [
                "stop", "dark", "wash", "ask", "greasy",
                "year", "oily", "water", "carry", "suit",
            ]


PROFILES = {
    "quick": AnalysisConfig(
        n_utterances=50,
        n_lime_samples=500,
        stability_seeds=5,
        n_bootstrap=200,
        cross_method_n=10,
        ig_steps=30,
        infidelity_samples=50,
        noise_repeats=3,
        noise_levels=[1e-3, 1e-2, 5e-2],
    ),
    "standard": AnalysisConfig(
        n_utterances=200,
        n_lime_samples=1500,
    ),
    "full": AnalysisConfig(
        n_utterances=300,
        n_lime_samples=3000,
        stability_seeds=15,
        n_bootstrap=1000,
        cross_method_n=50,
        ig_steps=100,
        infidelity_samples=200,
        noise_repeats=10,
        batch_size=128,
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Dependency check
# ═══════════════════════════════════════════════════════════════════════════

def check_dependencies():
    missing = []
    for pkg in ["jax", "flax", "librosa", "sklearn", "scipy", "matplotlib", "kagglehub"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing packages: {missing}")
        print("Install with:  pip install jax flax librosa scikit-learn scipy matplotlib kagglehub")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Cochlear filter cache
# ═══════════════════════════════════════════════════════════════════════════

def _collect_word_trajectories(
    ds, encode_fn, explainer, sr_pairs, cfg, log,
):
    """Locate target words in TIMIT and build time-resolved cortical
    trajectories for the lingo_analysis Plot-4 panels.

    For each target word, the first utterance that contains it is chosen.
    A duration-cfg-sized crop is extracted, centred on the word's midpoint,
    RMS-normalised, passed through the biomimetic frontend to obtain the
    cortical-feature tensor, and explained with CorticalLIME. The per-frame
    saliency is then ``|LIME importance| × per-frame cortical energy``,
    aggregated into three linguistically-motivated filter groups.
    """
    import numpy as np
    from lingo_analysis import categorise_filters, FILTER_GROUPS

    SR = 16_000
    FRAME_SAMPLES = 80  # 5 ms at 16 kHz — matches TIMIT labels

    target_set = [w.lower() for w in cfg.trajectory_words]
    found = {}
    for utt in ds:
        for ws in utt.word_segments:
            w = ws.word.lower()
            if w in target_set and w not in found:
                found[w] = (utt, ws)
        if len(found) == len(target_set):
            break

    missing = [w for w in target_set if w not in found]
    if missing:
        log(f"  Missing target words (not found in split): {missing}")
    if not found:
        return []

    groups = categorise_filters(sr_pairs)
    filter_labels = [fl for fl, _ in FILTER_GROUPS]
    group_keys = [gk for _, gk in FILTER_GROUPS]

    n_target = int(cfg.trajectory_duration * SR)
    n_target_frames = n_target // FRAME_SAMPLES

    trajectories = []
    ordered = [w for w in target_set if w in found]
    for w in ordered:
        utt, ws = found[w]
        y = utt.audio
        # Centre the crop on the midpoint of the word.
        mid = (ws.start_sample + ws.end_sample) // 2
        s = mid - n_target // 2
        if s < 0:
            s = 0
        if s + n_target > len(y):
            s = max(0, len(y) - n_target)
        y_crop = y[s:s + n_target]
        if len(y_crop) < n_target:
            y_crop = np.pad(y_crop, (0, n_target - len(y_crop)))
        rms = float(np.sqrt(np.mean(y_crop ** 2)))
        if rms > 0:
            y_crop = y_crop / rms
        y_crop = y_crop.astype(np.float32)

        # Word-specific LIME importance (reused per word; expensive but
        # correct — ~10 calls total, which is a tiny overhead).
        res = explainer.explain(y_crop)
        imp = np.abs(np.asarray(res.importances, dtype=np.float64))

        feats = np.asarray(encode_fn(y_crop[None, :])[0])  # (F, T, S)
        # |feats| summed over the frequency axis → (T, S) per-channel energy.
        energy = np.abs(feats).sum(axis=0)  # shape (T, S)
        T = energy.shape[0]
        saliency = imp[None, :] * energy   # (T, S)

        traj = np.zeros((len(group_keys), T), dtype=np.float64)
        for gi, gk in enumerate(group_keys):
            idx = groups.get(gk, np.array([], dtype=int))
            if idx.size:
                traj[gi] = saliency[:, idx].sum(axis=1)

        # Normalise whole panel so rows are comparable.
        m = traj.max()
        if m > 0:
            traj = traj / m

        # Phoneme boundaries *inside the crop*: find all phone segments
        # whose [start, end] intersects the crop window, convert to local
        # frame indices.
        crop_start = s
        crop_end = s + n_target
        bounds: list = []
        for ps in utt.phone_segments:
            if ps.end_sample <= crop_start or ps.start_sample >= crop_end:
                continue
            local_start = max(ps.start_sample, crop_start) - crop_start
            frame = int(local_start // FRAME_SAMPLES)
            phn = ps.phone_39 if getattr(ps, "phone_39", "") else ps.phone
            if phn and phn != "sil":
                bounds.append((frame, f"/{phn}/"))
        # De-duplicate consecutive identical phones and sort by frame.
        bounds.sort(key=lambda x: x[0])
        dedup: list = []
        for b in bounds:
            if not dedup or dedup[-1][1] != b[1]:
                dedup.append(b)
        bounds = dedup

        trajectories.append(
            {"word": w, "trajectory": traj, "boundaries": bounds,
             "r2": float(res.surrogate_r2)}
        )
        log(f"  {w:10s}  frames={T}  bounds={len(bounds)}  R²={res.surrogate_r2:.3f}")

    return trajectories


def ensure_cochlear_npz():
    npz = R_CODE / "cochlear_filter_params.npz"
    if npz.is_file():
        return
    print("Building cochlear_filter_params.npz ...")
    os.chdir(R_CODE)
    import numpy as np
    from strfpy_jax import read_cochba_j
    Bs, As = read_cochba_j()
    np.savez(npz, Bs=np.asarray(Bs), As=np.asarray(As))
    print(f"  Wrote {npz}")


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run(cfg: AnalysisConfig):
    t_start = time.time()
    os.chdir(R_CODE)

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats

    from supervisedSTRF import vSupervisedSTRF
    from cortical_lime import (
        CorticalLIME, OcclusionSensitivity, CorticalIntegratedGradients,
        make_jax_callables, bootstrap_importances, build_model,
    )
    from cortical_lime_metrics import (
        deletion_curve, insertion_curve, random_baseline_curves,
        aopc, infidelity,
        explanation_stability, input_sensitivity,
        build_phoneme_profiles, phoneme_family_comparison,
        cross_method_agreement, rank_correlation_matrix,
    )
    from timit_dataset import (
        TIMITDataset, IDX_TO_PHONEME, PHONEME_FAMILIES,
        VOICING_PAIRS, PLACE_GROUPS,
    )

    warnings.filterwarnings("ignore", category=FutureWarning)

    # ── Publication style ─────────────────────────────────────────────
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
        "font.size": 10, "font.family": "serif",
        "axes.titlesize": 12, "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 9,
        "figure.constrained_layout.use": True,
    })

    FAMILY_COLORS = {
        "vowels": "#e74c3c", "stops": "#2980b9", "fricatives": "#27ae60",
        "nasals": "#8e44ad", "affricates": "#e67e22", "closures": "#95a5a6",
        "semivowels_glides": "#1abc9c", "silence": "#bdc3c7",
    }

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    log(f"CorticalLIME Analysis Pipeline")
    log(f"{'='*50}")
    log(f"Output:       {out.resolve()}")
    log(f"Profile:      n_utts={cfg.n_utterances}  n_lime={cfg.n_lime_samples}")
    log(f"Skip:         {cfg.skip_sections or '(none)'}")
    log(f"Device:       {jax.devices()}")
    log("")

    # ── Load model ────────────────────────────────────────────────────
    log("Loading model...")
    mdir = Path(cfg.model_dir)
    ckpts = sorted(
        glob.glob(str(mdir / "chkStep_*.p")),
        key=lambda p: int(re.search(r"chkStep_(\d+)\.p$", p).group(1)),
    )
    assert ckpts, f"No chkStep_*.p in {mdir}"
    ckpt = ckpts[-1]
    log(f"  Checkpoint: {ckpt}")

    with open(ckpt, "rb") as f:
        obj = pickle.load(f)
    nn_params, aud_params = (
        obj if isinstance(obj, (list, tuple))
        else (obj["nn_params"], obj["params"])
    )

    model, n_phones = build_model(nn_params, aud_params)
    encode_fn, decode_fn = make_jax_callables(model, nn_params, aud_params)
    sr_pairs = np.asarray(aud_params["sr"])
    S = sr_pairs.shape[0]
    log(f"  STRF bank: {S} channels  n_phones: {n_phones}")

    # JIT warm-up.
    _d = np.zeros((1, 16000), dtype=np.float32)
    _f = encode_fn(_d); decode_fn(_f)
    log(f"  Cortical shape: {_f.shape}  Logit shape: {decode_fn(_f).shape}")

    # ── Load TIMIT ────────────────────────────────────────────────────
    log("\nLoading TIMIT...")
    ds = TIMITDataset(
        split=cfg.timit_split,
        local_path=cfg.timit_local,
    )
    log(f"  {len(ds)} utterances in {cfg.timit_split}")
    s = ds.summary()
    log(f"  Speakers: {s['n_speakers']}  Regions: {s['n_dialect_regions']}  "
        f"Duration: {s['total_duration_min']:.1f} min")

    utts = ds.sample(cfg.n_utterances, seed=cfg.seed, exclude_sa=cfg.exclude_sa)
    log(f"  Sampled {len(utts)} utterances (SA excluded={cfg.exclude_sa})")

    # ══════════════════════════════════════════════════════════════════
    # A.  Dataset overview
    # ══════════════════════════════════════════════════════════════════
    if "A" not in cfg.skip_sections:
        log("\n── A. Dataset overview ──")
        from collections import defaultdict

        # A1: Phone distribution.
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
        fig.savefig(out / "A1_phone_distribution.png"); plt.close(fig)

        # A2: Duration histogram.
        durs = [u.duration_sec for u in ds]
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(durs, bins=30, color="#555", edgecolor="white", lw=0.3)
        ax.set(xlabel="Duration (s)", ylabel="Count", title="Utterance durations")
        fig.savefig(out / "A2_duration_hist.png"); plt.close(fig)

        # A3: Dialect + gender.
        regions = defaultdict(int)
        for u in ds:
            regions[u.dialect_region] += 1
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(sorted(regions), [regions[r] for r in sorted(regions)],
               color="#2980b9", edgecolor="white", lw=0.3)
        ax.set(xlabel="Dialect region", ylabel="Count", title="Dialect region coverage")
        fig.savefig(out / "A3_dialect_regions.png"); plt.close(fig)
        log("  Saved A1–A3")

    # ══════════════════════════════════════════════════════════════════
    # Build explainers
    # ══════════════════════════════════════════════════════════════════
    _strategy = "band_bernoulli" if cfg.band_mode else "bernoulli"
    explainer = CorticalLIME(
        encode_fn=encode_fn, decode_fn=decode_fn, sr=sr_pairs,
        strategy=_strategy, n_samples=cfg.n_lime_samples,
        keep_prob=cfg.keep_prob, kernel_width=cfg.kernel_width,
        surrogate_type=cfg.surrogate_type, surrogate_alpha=cfg.surrogate_alpha,
        batch_size=cfg.batch_size, seed=cfg.seed,
    )
    log(f"  Strategy: {_strategy}  n_samples={cfg.n_lime_samples}  "
        f"S_total={explainer.n_bands * explainer.n_strfs} "
        f"({explainer.n_bands} bands × {explainer.n_strfs} STRFs)")
    occ_explainer = OcclusionSensitivity(encode_fn, decode_fn, sr_pairs)
    ig_explainer = CorticalIntegratedGradients(
        model, nn_params, aud_params, n_steps=cfg.ig_steps,
    )

    # ══════════════════════════════════════════════════════════════════
    # B.  Single-utterance deep dive
    # ══════════════════════════════════════════════════════════════════
    if "B" not in cfg.skip_sections:
        log("\n── B. Single-utterance deep dive ──")
        wav0 = utts[0].segment_audio(1.0)
        res0 = explainer.explain(wav0)
        occ0 = occ_explainer.explain(wav0, target_class=res0.target_class)
        ig0 = ig_explainer.explain(wav0, target_class=res0.target_class)

        phn0 = IDX_TO_PHONEME.get(res0.target_class, "?")
        log(f"  Utterance: {utts[0].utterance_id}  →  /{phn0}/  "
            f"P={res0.target_prob:.4f}  R²={res0.surrogate_r2:.4f}")

        methods0 = {
            "CorticalLIME": res0.importances,
            "Occlusion": occ0["importances"],
            "Integrated Gradients": ig0["importances"],
        }

        # B1: Three methods side-by-side.
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
        for ax, (name, imp) in zip(axes, methods0.items()):
            vmax = np.max(np.abs(imp)) + 1e-12
            ax.scatter(sr_pairs[:, 1], sr_pairs[:, 0], c=imp, cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax,
                       s=60 + 400 * (np.abs(imp) / vmax),
                       edgecolors="k", linewidths=0.4, zorder=3)
            ax.axvline(0, color="grey", lw=0.5, ls=":")
            ax.set_xlabel(r"$\omega$ (Hz)"); ax.set_title(name)
        axes[0].set_ylabel(r"$\Omega$ (cyc/oct)")
        fig.suptitle(f"/{phn0}/  P={res0.target_prob:.3f}", y=1.02)
        fig.savefig(out / "B1_three_methods.png"); plt.close(fig)

        # B2: Surrogate diagnostics.
        y_true = res0.target_probs
        y_pred = res0.masks @ res0.importances + res0.intercept
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        axes[0].scatter(y_pred, y_true, c=res0.weights, cmap="viridis", s=6, alpha=0.6)
        lims = [min(y_pred.min(), y_true.min()), max(y_pred.max(), y_true.max())]
        axes[0].plot(lims, lims, "k--", lw=0.7)
        axes[0].set(xlabel="Ridge prediction", ylabel="P(class)",
                    title=f"Fidelity R²={res0.surrogate_r2:.3f}")
        axes[1].hist(res0.distances, bins=30, color="#555", edgecolor="white", lw=0.3)
        axes[1].set(xlabel="Cosine distance", title="Distance distribution")
        axes[2].hist(res0.weights, bins=30, color="#2a7f62", edgecolor="white", lw=0.3)
        axes[2].set(xlabel="Kernel weight", title="Weight distribution")
        fig.savefig(out / "B2_surrogate_diagnostics.png"); plt.close(fig)

        # B3: Bootstrap CIs.
        bm, blo, bhi = bootstrap_importances(
            res0, n_bootstrap=cfg.n_bootstrap, alpha_ci=cfg.bootstrap_alpha, seed=42,
        )
        order = np.argsort(-np.abs(bm))
        fig, ax = plt.subplots(figsize=(8, 3.8))
        x = np.arange(S)
        ax.bar(x, bm[order], color="#2a7f62", edgecolor="white", lw=0.3)
        ax.vlines(x, blo[order], bhi[order], color="k", lw=0.8)
        ax.axhline(0, color="k", lw=0.5)
        sig = (blo[order] > 0) | (bhi[order] < 0)
        for i, s_ in enumerate(sig):
            if s_:
                ax.scatter(i, bhi[order][i] + 0.002, marker="*",
                           color="goldenrod", s=25, zorder=5)
        ax.set(xlabel="STRF channel (sorted)", ylabel="Coef",
               title=f"95% bootstrap CIs — {int(sig.sum())}/{S} significant")
        fig.savefig(out / "B3_bootstrap_ci.png"); plt.close(fig)

        # Cross-method agreement.
        for (a, ia), (b, ib) in [
            (("LIME", res0.importances), ("Occ", occ0["importances"])),
            (("LIME", res0.importances), ("IG", ig0["importances"])),
        ]:
            ag = cross_method_agreement(ia, ib)
            log(f"    {a} vs {b}: ρ={ag['spearman_rho']:.3f}  "
                f"top5={ag['top5_overlap']}/5")
        log("  Saved B1–B3")

    # ══════════════════════════════════════════════════════════════════
    # Run CorticalLIME on all utterances
    # ══════════════════════════════════════════════════════════════════
    saved_npz = out / "results_raw.npz"
    if cfg.skip_lime and saved_npz.is_file():
        log(f"\n[--skip_lime] Reloading LIME results from {saved_npz}")

        class _Stub:
            pass

        with np.load(saved_npz, allow_pickle=True) as d:
            imps = np.asarray(d["importances"])
            tcs = np.asarray(d["target_classes"])
            tps = np.asarray(d["target_probs"])
            r2s = np.asarray(d["surrogate_r2s"])
            n_bands_saved = int(d["n_bands"]) if "n_bands" in d.files else 1
            band_edges_saved = (
                np.asarray(d["band_edges_hz"]) if "band_edges_hz" in d.files
                else None
            )
        all_results = []
        for i in range(len(imps)):
            r = _Stub()
            r.importances = imps[i]
            r.target_class = int(tcs[i])
            r.target_prob = float(tps[i])
            r.surrogate_r2 = float(r2s[i])
            r.n_bands = n_bands_saved
            r.band_edges_hz = band_edges_saved
            # downstream sections that need masks/probs are safely skipped
            # via --skip when --skip_lime is used.
            all_results.append(r)
        log(f"  Loaded {len(all_results)} cached LIME results "
            f"(mean R²={r2s.mean():.4f})")
    else:
        log(f"\nRunning CorticalLIME on {len(utts)} utterances...")
        all_results = []
        for i, utt in enumerate(utts):
            wav = utt.segment_audio(1.0)
            all_results.append(explainer.explain(wav))
            if (i + 1) % 25 == 0:
                mr2 = np.mean([r.surrogate_r2 for r in all_results])
                log(f"  {i+1}/{len(utts)}  mean R²={mr2:.4f}")
        mr2_final = np.mean([r.surrogate_r2 for r in all_results])
        log(f"  Done. Mean R²={mr2_final:.4f}")

    # ══════════════════════════════════════════════════════════════════
    # C.  Faithfulness
    # ══════════════════════════════════════════════════════════════════
    if "C" not in cfg.skip_sections:
        log("\n── C. Faithfulness ──")
        del_l, ins_l, del_r, ins_r = [], [], [], []
        aopc_v, infid_v = [], []
        n = len(utts)
        for i in range(n):
            wav = utts[i].segment_audio(1.0)
            feats = encode_fn(wav[None, :])[0]
            tc = all_results[i].target_class
            imp = all_results[i].importances

            dc = deletion_curve(feats, imp, decode_fn, tc)
            ic = insertion_curve(feats, imp, decode_fn, tc)
            del_l.append(dc.auc); ins_l.append(ic.auc)
            aopc_v.append(aopc(feats, imp, decode_fn, tc, K=cfg.aopc_K))
            infid_v.append(infidelity(feats, imp, decode_fn, tc,
                                      n_samples=cfg.infidelity_samples))

            rand_imp = np.random.default_rng(i).random(S).astype(np.float32)
            del_r.append(deletion_curve(feats, rand_imp, decode_fn, tc).auc)
            ins_r.append(insertion_curve(feats, rand_imp, decode_fn, tc).auc)

            if (i + 1) % 25 == 0:
                log(f"  {i+1}/{n}...")

        del_l, ins_l = np.array(del_l), np.array(ins_l)
        del_r, ins_r = np.array(del_r), np.array(ins_r)
        aopc_v, infid_v = np.array(aopc_v), np.array(infid_v)

        t_del, p_del = stats.ttest_rel(del_l, del_r)
        t_ins, p_ins = stats.ttest_rel(ins_l, ins_r)
        log(f"  Del AUC: LIME {del_l.mean():.4f}±{del_l.std():.4f}  "
            f"Random {del_r.mean():.4f}±{del_r.std():.4f}  (p={p_del:.2e})")
        log(f"  Ins AUC: LIME {ins_l.mean():.4f}±{ins_l.std():.4f}  "
            f"Random {ins_r.mean():.4f}±{ins_r.std():.4f}  (p={p_ins:.2e})")
        log(f"  AOPC:       {aopc_v.mean():.4f}±{aopc_v.std():.4f}")
        log(f"  Infidelity: {infid_v.mean():.6f}±{infid_v.std():.6f}")

        # C1: AUC distributions.
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        for ax, (cl, cr, title, dir_) in zip(axes, [
            (del_l, del_r, "Deletion AUC", "↓ better"),
            (ins_l, ins_r, "Insertion AUC", "↑ better"),
        ]):
            lo = min(cl.min(), cr.min()) - 0.02
            hi = max(cl.max(), cr.max()) + 0.02
            bins = np.linspace(lo, hi, 25)
            ax.hist(cr, bins, alpha=0.5, color="grey", label="Random", edgecolor="white", lw=0.3)
            ax.hist(cl, bins, alpha=0.7, color="#c0392b", label="CorticalLIME", edgecolor="white", lw=0.3)
            ax.set(xlabel=f"{title}  ({dir_})", ylabel="Count", title=title)
            ax.legend()
        fig.savefig(out / "C1_faithfulness_distributions.png"); plt.close(fig)

        # C2: AOPC vs Infidelity.
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(aopc_v, infid_v, s=12, alpha=0.6, color="#2980b9")
        ax.set(xlabel="AOPC (↑)", ylabel="Infidelity (↓)", title="AOPC vs Infidelity")
        fig.savefig(out / "C2_aopc_vs_infidelity.png"); plt.close(fig)
        log("  Saved C1–C2")

    # ══════════════════════════════════════════════════════════════════
    # D.  Stability
    # ══════════════════════════════════════════════════════════════════
    if "D" not in cfg.skip_sections:
        log("\n── D. Stability ──")
        wav_s = utts[0].segment_audio(1.0)
        stab = explanation_stability(
            explainer, wav_s, n_runs=cfg.stability_seeds,
        )
        log(f"  Seed ρ = {stab['mean_spearman']:.4f}±{stab['std_spearman']:.4f}")

        all_imps_s = stab["all_importances"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        im = axes[0].imshow(all_imps_s, aspect="auto", cmap="RdBu_r",
                            vmin=-np.max(np.abs(all_imps_s)),
                            vmax=np.max(np.abs(all_imps_s)))
        axes[0].set(xlabel="STRF channel", ylabel="Seed",
                    title="Importances across seeds")
        fig.colorbar(im, ax=axes[0], shrink=0.85)

        m_s = all_imps_s.mean(0)
        s_s = all_imps_s.std(0)
        o_s = np.argsort(-np.abs(m_s))
        axes[1].bar(range(S), m_s[o_s], yerr=s_s[o_s], capsize=1.5,
                    color="#2a7f62", edgecolor="white", lw=0.3)
        axes[1].axhline(0, color="k", lw=0.5)
        axes[1].set(xlabel="STRF (sorted)", ylabel="Coef",
                    title="Mean ± std across seeds")
        fig.savefig(out / "D1_stability.png"); plt.close(fig)

        # D2: Input noise sensitivity.
        sens = input_sensitivity(
            explainer, wav_s,
            noise_levels=cfg.noise_levels, n_repeats=cfg.noise_repeats, seed=0,
        )
        eps_v = [r["noise_level"] for r in sens["sensitivity"]]
        rho_v = [r["mean_rho"] for r in sens["sensitivity"]]
        rho_s = [r["std_rho"] for r in sens["sensitivity"]]
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.errorbar(eps_v, rho_v, yerr=rho_s, fmt="-o", capsize=3, color="#c0392b")
        ax.set_xscale("log")
        ax.set(xlabel="Input noise σ", ylabel="ρ (vs clean)",
               title="Input-noise robustness", ylim=(-0.1, 1.1))
        fig.savefig(out / "D2_noise_sensitivity.png"); plt.close(fig)
        log("  Saved D1–D2")

    # ══════════════════════════════════════════════════════════════════
    # E.  Per-phoneme profiles
    # ══════════════════════════════════════════════════════════════════
    log("\n── E. Per-phoneme profiles ──")
    profiles = build_phoneme_profiles(all_results, IDX_TO_PHONEME)
    log(f"  {len(profiles)} phonemes with data")

    if "E" not in cfg.skip_sections:
        top_phns = sorted(profiles.items(), key=lambda x: -x[1].n_utterances)[:9]
        n_show = min(9, len(top_phns))
        rows = (n_show + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(13, 3.5 * rows), sharey=True)
        axes_flat = axes.ravel()
        for ax, (phn, prof) in zip(axes_flat, top_phns[:n_show]):
            vmax = np.max(np.abs(prof.mean_importances)) + 1e-12
            ax.scatter(sr_pairs[:, 1], sr_pairs[:, 0],
                       c=prof.mean_importances, cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax,
                       s=30 + 250 * (np.abs(prof.mean_importances) / vmax),
                       edgecolors="k", linewidths=0.3)
            ax.axvline(0, color="grey", lw=0.5, ls=":")
            ax.set_title(f"/{phn}/  n={prof.n_utterances}  R²={prof.mean_r2:.2f}",
                         fontsize=10)
        for ax in axes_flat[n_show:]:
            ax.set_visible(False)
        fig.suptitle("Mean CorticalLIME per predicted phoneme", y=1.01, fontsize=13)
        fig.savefig(out / "E1_phoneme_profiles.png"); plt.close(fig)

        # E2: Magnitude ranking.
        mean_abs = {p: float(np.mean(np.abs(pf.mean_importances)))
                    for p, pf in profiles.items() if pf.n_utterances >= 3}
        sorted_p = sorted(mean_abs, key=lambda p: -mean_abs[p])[:20]
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.bar(range(len(sorted_p)), [mean_abs[p] for p in sorted_p],
               color="#2980b9", edgecolor="white", lw=0.3)
        ax.set_xticks(range(len(sorted_p)))
        ax.set_xticklabels([f"/{p}/" for p in sorted_p], rotation=45,
                           ha="right", fontsize=8)
        ax.set_ylabel("Mean |importance|")
        ax.set_title("Phonemes by mean |STRF importance|")
        fig.savefig(out / "E2_importance_magnitude.png"); plt.close(fig)
        log("  Saved E1–E2")

    # ══════════════════════════════════════════════════════════════════
    # F.  Manner of articulation
    # ══════════════════════════════════════════════════════════════════
    if "F" not in cfg.skip_sections:
        log("\n── F. Manner of articulation ──")
        comps = phoneme_family_comparison(profiles, PHONEME_FAMILIES)
        for pair, res in comps.items():
            log(f"    {pair}: {res['n_significant']}/{S} sig channels")

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for fname, members in PHONEME_FAMILIES.items():
            rows_ = [profiles[p].mean_importances for p in members if p in profiles]
            if len(rows_) < 2:
                continue
            fm = np.mean(rows_, axis=0)
            fs = np.std(rows_, axis=0)
            o = np.argsort(-np.abs(fm))
            c = FAMILY_COLORS.get(fname, "grey")
            ax.plot(range(S), fm[o], "-", lw=1.5, color=c, label=fname)
            ax.fill_between(range(S), fm[o] - fs[o], fm[o] + fs[o],
                            alpha=0.12, color=c)
        ax.axhline(0, color="k", lw=0.5)
        ax.set(xlabel="STRF channel (sorted)", ylabel="Mean coef",
               title="Manner-of-articulation profiles")
        ax.legend(ncol=2, fontsize=8)
        fig.savefig(out / "F1_manner_profiles.png"); plt.close(fig)

        # F2: Effect sizes for key pairs.
        key_pairs = [p for p in ["stops_vs_vowels", "stops_vs_fricatives",
                                  "fricatives_vs_vowels", "vowels_vs_nasals"]
                     if p in comps]
        if key_pairs:
            fig, axes = plt.subplots(1, len(key_pairs),
                                     figsize=(5 * len(key_pairs), 4), sharey=True)
            if len(key_pairs) == 1:
                axes = [axes]
            for ax, pair in zip(axes, key_pairs):
                cd = comps[pair]["effect_sizes"]
                cdmax = np.max(np.abs(cd)) + 0.1
                ax.scatter(sr_pairs[:, 1], sr_pairs[:, 0], c=cd, cmap="PiYG",
                           vmin=-cdmax, vmax=cdmax,
                           s=40 + 300 * (np.abs(cd) / cdmax),
                           edgecolors="k", linewidths=0.3)
                ax.axvline(0, color="grey", lw=0.5, ls=":")
                ax.set_xlabel(r"$\omega$ (Hz)")
                fa, fb = pair.split("_vs_")
                ax.set_title(f"Cohen's d: {fa} − {fb}", fontsize=10)
            axes[0].set_ylabel(r"$\Omega$ (cyc/oct)")
            fig.savefig(out / "F2_manner_effect_sizes.png"); plt.close(fig)
        log("  Saved F1–F2")

    # ══════════════════════════════════════════════════════════════════
    # G.  Voicing contrasts
    # ══════════════════════════════════════════════════════════════════
    if "G" not in cfg.skip_sections:
        log("\n── G. Voicing contrasts ──")
        pairs_wd = [(v, uv) for v, uv in VOICING_PAIRS
                    if v in profiles and uv in profiles]
        if pairs_wd:
            n_vp = len(pairs_wd)
            fig, axes = plt.subplots(1, n_vp, figsize=(4 * n_vp, 4), sharey=True)
            if n_vp == 1:
                axes = [axes]
            for ax, (v, uv) in zip(axes, pairs_wd):
                diff = profiles[v].mean_importances - profiles[uv].mean_importances
                vmax = np.max(np.abs(diff)) + 1e-12
                ax.scatter(sr_pairs[:, 1], sr_pairs[:, 0], c=diff, cmap="PiYG",
                           vmin=-vmax, vmax=vmax,
                           s=40 + 300 * (np.abs(diff) / vmax),
                           edgecolors="k", linewidths=0.3)
                ax.axvline(0, color="grey", lw=0.5, ls=":")
                ax.set_xlabel(r"$\omega$"); ax.set_title(f"/{v}/ − /{uv}/")
            axes[0].set_ylabel(r"$\Omega$")
            fig.suptitle("Voicing contrasts (voiced − voiceless)", y=1.02)
            fig.savefig(out / "G1_voicing_contrasts.png"); plt.close(fig)
            log(f"  {n_vp} pairs. Saved G1")
        else:
            log("  No voicing pairs with data.")

    # ══════════════════════════════════════════════════════════════════
    # H.  Place of articulation
    # ══════════════════════════════════════════════════════════════════
    if "H" not in cfg.skip_sections:
        log("\n── H. Place of articulation ──")
        place_colors = {"labial": "#e74c3c", "alveolar": "#2980b9",
                        "velar": "#27ae60", "palatal": "#8e44ad",
                        "dental": "#e67e22"}
        fig, ax = plt.subplots(figsize=(7, 4))
        for pname, members in PLACE_GROUPS.items():
            rows_ = [profiles[p].mean_importances for p in members if p in profiles]
            if not rows_:
                continue
            fm = np.mean(rows_, axis=0)
            o = np.argsort(-np.abs(fm))
            ax.plot(range(S), fm[o], "-o", ms=2, lw=1.2,
                    color=place_colors.get(pname, "grey"), label=pname)
        ax.axhline(0, color="k", lw=0.5)
        ax.set(xlabel="STRF channel (sorted)", ylabel="Mean coef",
               title="Place-of-articulation profiles")
        ax.legend()
        fig.savefig(out / "H1_place_profiles.png"); plt.close(fig)
        log("  Saved H1")

    # ══════════════════════════════════════════════════════════════════
    # I.  Filter utilisation
    # ══════════════════════════════════════════════════════════════════
    if "I" not in cfg.skip_sections:
        log("\n── I. Filter utilisation ──")
        all_imps_mat = np.stack([r.importances for r in all_results])
        mean_abs = np.mean(np.abs(all_imps_mat), axis=0)
        std_across = np.std(all_imps_mat, axis=0)
        cv = std_across / (mean_abs + 1e-12)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        order_i = np.argsort(-mean_abs)
        axes[0].bar(range(S), mean_abs[order_i], color="#2a7f62",
                    edgecolor="white", lw=0.3)
        axes[0].set(xlabel="STRF (sorted)", ylabel="Mean |coef|",
                    title="Per-channel mean |importance|")

        vmax_i = np.max(mean_abs) + 1e-12
        sc = axes[1].scatter(sr_pairs[:, 1], sr_pairs[:, 0], c=cv, cmap="inferno",
                             s=60 + 400 * (mean_abs / vmax_i),
                             edgecolors="k", linewidths=0.4)
        axes[1].axvline(0, color="grey", lw=0.5, ls=":")
        axes[1].set_xlabel(r"$\omega$"); axes[1].set_ylabel(r"$\Omega$")
        axes[1].set_title("CV of importance (high = class-specific)")
        fig.colorbar(sc, ax=axes[1], label="CV", shrink=0.85)
        fig.savefig(out / "I1_filter_utilisation.png"); plt.close(fig)

        corr = np.corrcoef(all_imps_mat.T)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set(xlabel="STRF", ylabel="STRF", title="Inter-channel importance correlation")
        fig.colorbar(im, ax=ax, shrink=0.85)
        fig.savefig(out / "I2_channel_correlation.png"); plt.close(fig)
        log("  Saved I1–I2")

    # ══════════════════════════════════════════════════════════════════
    # J.  Confidence vs. explanation quality
    # ══════════════════════════════════════════════════════════════════
    if "J" not in cfg.skip_sections:
        log("\n── J. Confidence vs R² ──")
        ents, r2s, probs_j = [], [], []
        for utt, res in zip(utts, all_results):
            # Compute prediction entropy from stored full prob vector.
            pv = res.full_probs
            ent = float(-np.sum(pv * np.log(pv + 1e-12)))
            ents.append(ent); r2s.append(res.surrogate_r2)
            probs_j.append(res.target_prob)

        ents, r2s, probs_j = np.array(ents), np.array(r2s), np.array(probs_j)
        rho_ent, _ = stats.spearmanr(ents, r2s)
        rho_prob, _ = stats.spearmanr(probs_j, r2s)
        log(f"  Entropy vs R²: ρ={rho_ent:.3f}")
        log(f"  P(class) vs R²: ρ={rho_prob:.3f}")

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].scatter(ents, r2s, s=15, alpha=0.6, color="#2980b9")
        axes[0].set(xlabel="Entropy", ylabel="R²",
                    title=f"Entropy vs R²  (ρ={rho_ent:.3f})")
        axes[1].scatter(probs_j, r2s, s=15, alpha=0.6, color="#27ae60")
        axes[1].set(xlabel="P(class)", ylabel="R²",
                    title=f"Confidence vs R²  (ρ={rho_prob:.3f})")
        fig.savefig(out / "J1_confidence_vs_r2.png"); plt.close(fig)
        log("  Saved J1")

    # ══════════════════════════════════════════════════════════════════
    # K.  Rate vs Scale decomposition
    # ══════════════════════════════════════════════════════════════════
    if "K" not in cfg.skip_sections:
        log("\n── K. Rate vs Scale ──")
        all_imps_mat = np.stack([r.importances for r in all_results])
        rates = sr_pairs[:, 1]; scales = sr_pairs[:, 0]
        rate_thr = np.median(np.abs(rates))
        scale_thr = np.median(scales)
        hi_rate = np.abs(rates) > rate_thr
        hi_scale = scales > scale_thr

        fhr = np.abs(all_imps_mat[:, hi_rate]).sum(1) / (np.abs(all_imps_mat).sum(1) + 1e-12)
        fhs = np.abs(all_imps_mat[:, hi_scale]).sum(1) / (np.abs(all_imps_mat).sum(1) + 1e-12)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(fhr, bins=25, color="#c0392b", edgecolor="white", lw=0.3)
        axes[0].axvline(0.5, color="k", lw=0.8, ls="--")
        axes[0].set(xlabel=f"Frac |imp| high-rate (|ω|>{rate_thr:.1f})",
                    ylabel="Count", title="Rate contribution")
        axes[1].hist(fhs, bins=25, color="#2980b9", edgecolor="white", lw=0.3)
        axes[1].axvline(0.5, color="k", lw=0.8, ls="--")
        axes[1].set(xlabel=f"Frac |imp| high-scale (Ω>{scale_thr:.1f})",
                    ylabel="Count", title="Scale contribution")
        fig.savefig(out / "K1_rate_vs_scale.png"); plt.close(fig)

        abs_imp_mean = np.mean(np.abs(all_imps_mat), axis=0)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        s1 = axes[0].scatter(np.abs(rates), abs_imp_mean, s=40, c=scales,
                             cmap="viridis", edgecolors="k", linewidths=0.3)
        axes[0].set(xlabel="|Rate| (Hz)", ylabel="Mean |imp|",
                    title="Importance vs rate")
        fig.colorbar(s1, ax=axes[0], label="Scale")
        s2 = axes[1].scatter(scales, abs_imp_mean, s=40, c=np.abs(rates),
                             cmap="magma", edgecolors="k", linewidths=0.3)
        axes[1].set(xlabel="Scale (cyc/oct)", ylabel="Mean |imp|",
                    title="Importance vs scale")
        fig.colorbar(s2, ax=axes[1], label="|Rate|")
        fig.savefig(out / "K2_marginal_importance.png"); plt.close(fig)
        log(f"  Mean frac high-rate: {fhr.mean():.3f}  high-scale: {fhs.mean():.3f}")
        log("  Saved K1–K2")

    # ══════════════════════════════════════════════════════════════════
    # L.  Cross-method agreement (population)
    # ══════════════════════════════════════════════════════════════════
    if "L" not in cfg.skip_sections:
        log("\n── L. Cross-method agreement (population) ──")
        n_cm = min(cfg.cross_method_n, len(utts))
        rhos_cm = []
        for i in range(n_cm):
            wav = utts[i].segment_audio(1.0)
            tc = all_results[i].target_class
            occ_r = occ_explainer.explain(wav, target_class=tc)
            rho, _ = stats.spearmanr(all_results[i].importances, occ_r["importances"])
            rhos_cm.append(rho)
            if (i + 1) % 10 == 0:
                log(f"  {i+1}/{n_cm}...")
        rhos_cm = np.array(rhos_cm)
        log(f"  LIME vs Occ: ρ = {rhos_cm.mean():.3f}±{rhos_cm.std():.3f}")

        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(rhos_cm, bins=20, color="#8e44ad", edgecolor="white", lw=0.3)
        ax.axvline(rhos_cm.mean(), color="k", lw=1.2, ls="--",
                   label=f"mean={rhos_cm.mean():.3f}")
        ax.set(xlabel="Spearman ρ (LIME vs Occ)", ylabel="Count",
               title=f"Cross-method agreement (n={n_cm})")
        ax.legend()
        fig.savefig(out / "L1_cross_method_agreement.png"); plt.close(fig)
        log("  Saved L1")

    # ══════════════════════════════════════════════════════════════════
    # Save results
    # ══════════════════════════════════════════════════════════════════
    save_dict = dict(
        importances=np.stack([r.importances for r in all_results]).astype(np.float32),
        target_classes=np.array([r.target_class for r in all_results]),
        target_probs=np.array([r.target_prob for r in all_results]),
        surrogate_r2s=np.array([r.surrogate_r2 for r in all_results]),
        sr_pairs=sr_pairs,
        utterance_ids=np.array([u.utterance_id for u in utts]),
        # band-mode metadata
        n_bands=np.int32(getattr(all_results[0], "n_bands", 1)),
        band_edges_hz=(
            getattr(all_results[0], "band_edges_hz", None)
            if getattr(all_results[0], "n_bands", 1) > 1
            else np.zeros((0, 2))
        ),
    )
    if cfg.save_masks and hasattr(all_results[0], "masks") \
            and all_results[0].masks is not None:
        # uint8 keeps perfect fidelity for Bernoulli (0/1) masks at 4× compression.
        try:
            mks = np.stack([r.masks for r in all_results]).astype(np.uint8)
            tps = np.stack([r.target_probs for r in all_results]).astype(np.float32)
            save_dict["masks"] = mks
            save_dict["per_mask_target_probs"] = tps
            log(f"  Persisting masks: {mks.shape} uint8 "
                f"({mks.nbytes / 1e6:.1f} MB)")
        except Exception as e:
            log(f"  WARNING: could not persist masks: {e}")
    np.savez(out / "results_raw.npz", **save_dict)

    # ══════════════════════════════════════════════════════════════════
    # N.  Per-word cortical trajectories (for lingo_analysis Plot 4)
    # ══════════════════════════════════════════════════════════════════
    log("\n── N. Per-word cortical trajectories ──")
    traj_out = _collect_word_trajectories(
        ds, encode_fn, explainer, sr_pairs, cfg, log,
    )
    if traj_out:
        np.savez(
            out / "trajectories.npz",
            trajectories=np.array(traj_out, dtype=object),
        )
        log(f"  Saved {len(traj_out)} word trajectories → trajectories.npz")
    else:
        log("  No target words located — skipping trajectory save.")

    # ══════════════════════════════════════════════════════════════════
    # M.  Skeptic-proof sanity dashboard (consolidated 2 × 2 PNG)
    # ══════════════════════════════════════════════════════════════════
    log("\n── M. Sanity-check dashboard ──")
    try:
        from cortical_lime_analysis import sanity_check_dashboard
        n_dash = min(20, len(utts))
        cm_wavs = [utts[i].segment_audio(1.0) for i in range(n_dash)]
        cm_imps = [all_results[i].importances for i in range(n_dash)]
        cm_targs = [all_results[i].target_class for i in range(n_dash)]
        sanity_check_dashboard(
            out / "M0_sanity_dashboard.png",
            explainer=explainer, occ_explainer=occ_explainer,
            encode_fn=encode_fn, decode_fn=decode_fn,
            stability_wav=utts[0].segment_audio(1.0),
            cross_method_wavs=cm_wavs,
            cross_method_imps=cm_imps,
            cross_method_targets=cm_targs,
            r2s=np.array([r.surrogate_r2 for r in all_results]),
            probs=np.array([r.target_prob for r in all_results]),
            n_stability_seeds=10, log=log,
        )
    except Exception as e:
        log(f"  Sanity dashboard failed: {e}")

    # ══════════════════════════════════════════════════════════════════
    # O.  lingo_analysis publication figures
    # ══════════════════════════════════════════════════════════════════
    log("\n── O. lingo_analysis figures ──")
    try:
        from lingo_analysis import render_all as _render_lingo
        lingo_out = out / "figures_lingo"
        written = _render_lingo(
            str(out / "results_raw.npz"),
            str(out / "trajectories.npz") if traj_out else None,
            str(lingo_out),
        )
        for k, v in written.items():
            log(f"  {k}: {v}")
    except Exception as e:
        log(f"  lingo_analysis failed: {e}")

    elapsed = time.time() - t_start
    log(f"\n{'='*50}")
    log(f"Total time: {elapsed/60:.1f} min")
    log(f"Figures:    {out.resolve()}/")
    log(f"Raw data:   {out / 'results_raw.npz'}")

    # Write summary log.
    with open(out / "run_summary.txt", "w") as f:
        f.write("\n".join(log_lines))
    print(f"Summary:    {out / 'run_summary.txt'}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="CorticalLIME full analysis pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--profile", choices=["quick", "standard", "full"],
                        default="standard",
                        help="Preset configuration profile.")
    parser.add_argument("--n_utterances", type=int, default=None)
    parser.add_argument("--n_lime_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--timit_local", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip", type=str, nargs="*", default=None,
                        help="Sections to skip (e.g. --skip C L).")
    parser.add_argument("--plots_only", action="store_true",
                        help="Skip everything; reload saved results_raw.npz "
                             "and trajectories.npz and re-render the "
                             "lingo_analysis figures only.")
    parser.add_argument("--skip_lime", action="store_true",
                        help="Reuse a previously saved results_raw.npz "
                             "instead of re-running CorticalLIME.")
    args = parser.parse_args()

    # Start from profile, then override with CLI args.
    cfg = PROFILES[args.profile]
    if args.n_utterances is not None:
        cfg.n_utterances = args.n_utterances
    if args.n_lime_samples is not None:
        cfg.n_lime_samples = args.n_lime_samples
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.model_dir is not None:
        cfg.model_dir = args.model_dir
    if args.timit_local is not None:
        cfg.timit_local = args.timit_local
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.seed is not None:
        cfg.seed = args.seed
    if args.skip is not None:
        cfg.skip_sections = args.skip
    cfg.plots_only = args.plots_only
    cfg.skip_lime = args.skip_lime

    if cfg.plots_only:
        _plots_only(cfg)
        return

    check_dependencies()
    ensure_cochlear_npz()
    run(cfg)


def _plots_only(cfg: AnalysisConfig):
    """Re-render lingo_analysis figures from a previously saved run.

    Expects ``results_raw.npz`` (and optionally ``trajectories.npz``) to
    already exist in ``cfg.output_dir``. Performs no LIME / no model / no
    dataset loading — purely a figure refresh.
    """
    from pathlib import Path
    out = Path(cfg.output_dir)
    results = out / "results_raw.npz"
    trajs = out / "trajectories.npz"
    if not results.is_file():
        print(f"[plots_only] missing {results}; run the full pipeline first.")
        sys.exit(1)
    from lingo_analysis import render_all as _render_lingo
    written = _render_lingo(
        str(results),
        str(trajs) if trajs.is_file() else None,
        str(out / "figures_lingo"),
    )
    print("[plots_only] wrote:")
    for k, v in written.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
