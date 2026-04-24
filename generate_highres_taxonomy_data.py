#!/usr/bin/env python3
"""Phonetic Sniper — ultra-high-resolution data extractor for the
phonetic-taxonomy dendrogram (Figure 3).

Why this script exists
----------------------
The dendrogram in ``generate_heavy_taxonomy.py`` is only as good as the
per-phoneme rows it clusters.  When those rows come from the ordinary
``run.py`` LIME sweep, they suffer from three quality problems:

  1. **Coverage.** The default sweep only touches whichever phones the
     random TIMIT sampling happens to encounter, so several manner
     classes — affricates ``{ch, jh}``, liquids/glides ``{l, r, w, y, hh}``
     — may be missing or under-sampled.
  2. **Token quality.** A 1 s window centred on a phone segment captures
     the surrounding syllable and contaminates the LIME importance with
     coarticulation and silence.
  3. **Statistical resolution.** ``n_samples=2000`` perturbations is
     enough for a single phoneme card but too noisy to define a stable
     centroid in 330-D space.

This script fixes all three at once by being a "phonetic sniper":

  - Walks **every TIMIT-39 phoneme** in 6 manner families (Vowels,
    Stops, Fricatives, Nasals, **Affricates**, **Liquids/Glides**) and
    requires **N = 15 perfect tokens** per phoneme.
  - "Perfect" = (a) duration inside a per-phoneme biologically-plausible
    window, and (b) the frozen frontend's argmax prediction matches the
    ground-truth label for that token.
  - Crops the audio surgically: ``[start, end] ± 5 ms`` — never a fixed
    1 s window.
  - Runs CorticalLIME at **n_samples = 5000** in band-mode for fine
    statistical resolution.
  - Aggregates the |importance| of all N tokens per phoneme and saves
    the result as a balanced ``(39, n_bands × n_strfs)`` matrix that
    ``generate_heavy_taxonomy.py`` can cluster with confidence.

Output
------
``analysis_outputs2/highres_taxonomy_data.npz`` with keys::

    mean_importances : (P, n_bands*n_strfs) float32
    phoneme_labels   : (P,) string  — TIMIT-39 phone, manner-ordered
    families         : (P,) string  — manner-of-articulation family
    n_bands          : int32
    n_strfs          : int32
    n_tokens         : (P,) int32   — actually-collected count per phoneme

Usage
-----
::

    ./generate_highres_taxonomy_data.py
    ./generate_highres_taxonomy_data.py --tokens_per_phone 20 --n_samples 6000
    ./generate_highres_taxonomy_data.py --phones s aa t   # subset for debugging
"""

from __future__ import annotations

import argparse
import gc
import glob
import os as _os
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ────────────────────────────────────────────────────────────────────────
# Project imports — mirror the path resolution used by the other
# standalone scripts so this runs from anywhere on disk.  strfpy_jax
# (transitively imported via supervisedSTRF) opens ``./cochba.txt`` at
# module-load time, hence the chdir-before-import dance.
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
_os.chdir(str(_R_CODE))

from cortical_lime import CorticalLIME, build_model, make_jax_callables  # noqa: E402
from timit_dataset import TIMITDataset  # noqa: E402
from lingo_analysis import (  # noqa: E402
    PHONE_61_TO_39, _TIMIT_61,
)


# ────────────────────────────────────────────────────────────────────────
# Project-canonical default paths.
# ────────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path("/Users/eminent/Projects/Cortical_Front")
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "r_code" / "analysis_outputs2"
_DEFAULT_MODEL_DIR = (
    _PROJECT_ROOT / "r_code" / "nishit_trained_models"
                  / "main_jax_phoneme_rec_timit"
)


# ────────────────────────────────────────────────────────────────────────
# 6-family TIMIT-39 phoneme inventory.
# ────────────────────────────────────────────────────────────────────────
# This is intentionally broader than ``lingo_analysis.PHONEMES_BY_FAMILY``
# (which only carries 4 families) — Affricates and Liquids/Glides are
# essential for a complete phonetic-taxonomy dendrogram.
PHONEME_FAMILIES_39: Dict[str, List[str]] = {
    "Vowels": [
        "iy", "ih", "eh", "ae", "aa", "ah", "uw", "uh", "er",
        "ey", "ay", "oy", "aw", "ow",
    ],
    "Stops":           ["b", "d", "g", "p", "t", "k"],
    "Fricatives":      ["s", "sh", "z", "f", "th", "v", "dh"],
    "Nasals":          ["m", "n", "ng"],
    "Affricates":      ["ch", "jh"],
    "Liquids/Glides":  ["l", "r", "w", "y", "hh"],
}

# Flattened phone → family lookup, used during plotting.
FAMILY_OF_39: Dict[str, str] = {
    p: fam for fam, ps in PHONEME_FAMILIES_39.items() for p in ps
}

# Canonical manner-ordered list of 39 TIMIT-39 phones.
ORDERED_PHONES_39: List[str] = [
    p for fam in PHONEME_FAMILIES_39.values() for p in fam
]


# ────────────────────────────────────────────────────────────────────────
# Per-phoneme biologically-plausible duration windows (ms).
# ────────────────────────────────────────────────────────────────────────
# Centred on published acoustic-phonetic statistics (Klatt 1976; Crystal
# & House 1988; TIMIT corpus stats).  Tokens outside the window are
# rejected: too-short ⇒ clipped artefact; too-long ⇒ multi-phone label.
PHONE_DUR_WINDOW_MS: Dict[str, Tuple[float, float]] = {
    # ── Vowels (mid → long) ──
    "iy": (60.0, 250.0), "ih": (50.0, 220.0), "eh": (50.0, 220.0),
    "ae": (60.0, 260.0), "aa": (60.0, 260.0), "ah": (40.0, 200.0),
    "uw": (60.0, 250.0), "uh": (40.0, 200.0), "er": (50.0, 250.0),
    "ey": (60.0, 260.0), "ay": (70.0, 280.0), "oy": (70.0, 280.0),
    "aw": (70.0, 280.0), "ow": (60.0, 260.0),
    # ── Stops (short bursts) ──
    "b":  (20.0, 130.0), "d":  (20.0, 130.0), "g":  (20.0, 130.0),
    "p":  (25.0, 150.0), "t":  (25.0, 150.0), "k":  (25.0, 150.0),
    # ── Fricatives (sustained turbulence) ──
    "s":  (60.0, 220.0), "sh": (60.0, 220.0), "z":  (50.0, 200.0),
    "f":  (50.0, 200.0), "th": (50.0, 200.0), "v":  (40.0, 180.0),
    "dh": (30.0, 150.0),
    # ── Nasals ──
    "m":  (40.0, 200.0), "n":  (40.0, 200.0), "ng": (40.0, 200.0),
    # ── Affricates ──
    "ch": (50.0, 200.0), "jh": (50.0, 200.0),
    # ── Liquids / glides / aspiration ──
    "l":  (40.0, 200.0), "r":  (40.0, 200.0),
    "w":  (40.0, 180.0), "y":  (40.0, 180.0),
    "hh": (25.0, 150.0),
}
DEFAULT_DUR_WINDOW_MS = (40.0, 250.0)

# TIMIT-61 phone string  →  1-indexed model class id.  We use this so
# CorticalLIME explains the **true** class for each token rather than
# whatever the model happens to argmax to.
PHONE61_TO_CLASS_IDX: Dict[str, int] = {
    p: (i + 1) for i, p in enumerate(_TIMIT_61)
}


# ────────────────────────────────────────────────────────────────────────
# Frontend loader (mirrors generate_hero_cochleagrams.py)
# ────────────────────────────────────────────────────────────────────────

def load_frontend(model_dir: Path):
    """Locate the latest checkpoint in ``model_dir`` and build encode /
    decode callables."""
    ckpts = sorted(
        glob.glob(str(model_dir / "chkStep_*.p")),
        key=lambda p: int(re.search(r"chkStep_(\d+)\.p$", p).group(1)),
    )
    if not ckpts:
        raise FileNotFoundError(f"No chkStep_*.p in {model_dir}")
    with open(ckpts[-1], "rb") as f:
        obj = pickle.load(f)
    nn_params, aud_params = (
        obj if isinstance(obj, (list, tuple))
        else (obj["nn_params"], obj["params"])
    )
    model, n_phones = build_model(nn_params, aud_params)
    encode_fn, decode_fn = make_jax_callables(model, nn_params, aud_params)
    sr_pairs = np.asarray(aud_params["sr"])
    print(f"  Checkpoint: {ckpts[-1]}")
    print(f"  STRF bank:  {sr_pairs.shape[0]} channels   n_phones: {n_phones}")
    return encode_fn, decode_fn, sr_pairs


# ────────────────────────────────────────────────────────────────────────
# Token sniping
# ────────────────────────────────────────────────────────────────────────

def _phone_label_39(ps) -> str:
    """Return the TIMIT-39 label for a phone segment."""
    label = getattr(ps, "phone_39", None)
    if label:
        return label
    return PHONE_61_TO_39.get(ps.phone, ps.phone)


def collect_perfect_tokens(
    ds: TIMITDataset,
    target_phones: Sequence[str],
    tokens_per_phone: int,
    encode_fn,
    decode_fn,
    *,
    sample_rate: int = 16_000,
    pad_ms: float = 5.0,
    require_correct_pred: bool = True,
    verbose: bool = True,
) -> Dict[str, List[Dict]]:
    """Walk TIMIT once; for each target phoneme collect up to N tokens
    that satisfy:

      - duration inside ``PHONE_DUR_WINDOW_MS[phone]``
      - if ``require_correct_pred``: the frozen frontend's argmax over
        the surgically-cropped audio matches the ground-truth class.

    Returns
    -------
    tokens_by_phone : Dict[str, List[token_record]]
        token_record carries 'audio', 'utt_id', 'phone61', 'class_idx',
        'duration_ms', 'pred_idx', 'pred_match'.
    """
    pad_samples = int(round(pad_ms * 1e-3 * sample_rate))
    targets = set(target_phones)

    tokens_by_phone: Dict[str, List[Dict]] = {p: [] for p in target_phones}
    phones_full: set = set()  # phones for which we already have N tokens

    if verbose:
        print(f"\nScanning TIMIT TEST split for {len(targets)} target phones, "
              f"need {tokens_per_phone} perfect tokens each "
              f"(pad ±{pad_ms:.0f} ms, require_correct_pred={require_correct_pred})…")

    for utt_idx, utt in enumerate(ds):
        if phones_full == targets:
            break
        for ps in utt.phone_segments:
            p39 = _phone_label_39(ps)
            if p39 not in targets or p39 in phones_full:
                continue
            lo_ms, hi_ms = PHONE_DUR_WINDOW_MS.get(p39, DEFAULT_DUR_WINDOW_MS)
            dur_ms = (ps.end_sample - ps.start_sample) * 1000.0 / sample_rate
            if not (lo_ms <= dur_ms <= hi_ms):
                continue

            # Surgical crop with the requested padding.
            s = max(0, int(ps.start_sample) - pad_samples)
            e = min(len(utt.audio), int(ps.end_sample) + pad_samples)
            y = np.asarray(utt.audio[s:e], dtype=np.float32)
            rms = float(np.sqrt(np.mean(y ** 2)))
            if rms <= 1e-6:
                continue
            y_norm = y / rms

            # Resolve the true model class from the TIMIT-61 surface form.
            class_idx = PHONE61_TO_CLASS_IDX.get(ps.phone, None)
            if class_idx is None:
                continue

            # Cheap sanity gate: classify the crop and (optionally) reject
            # if the model argmax doesn't agree with the truth.  This is
            # what makes a token "perfect".
            try:
                cortical = np.asarray(encode_fn(y_norm[None, :]))
                logits = np.asarray(decode_fn(cortical))
                if logits.ndim == 4:
                    logits = logits.mean(axis=2)
                # softmax-then-time-average matches CorticalLIME's logic.
                e_l = np.exp(logits - logits.max(axis=-1, keepdims=True))
                probs = (e_l / e_l.sum(axis=-1, keepdims=True)).mean(axis=1)
                pred_idx = int(np.argmax(probs[0]))
            except Exception as ex:
                if verbose:
                    print(f"    [warn] forward pass failed on "
                          f"{utt.utterance_id}/{ps.phone}: {ex}")
                continue
            pred_match = (pred_idx == class_idx)
            if require_correct_pred and not pred_match:
                continue

            tokens_by_phone[p39].append({
                "audio": y_norm,
                "utt_id": str(utt.utterance_id),
                "phone61": str(ps.phone),
                "class_idx": int(class_idx),
                "duration_ms": float(dur_ms),
                "pred_idx": int(pred_idx),
                "pred_match": bool(pred_match),
            })
            if len(tokens_by_phone[p39]) >= tokens_per_phone:
                phones_full.add(p39)
                if verbose:
                    print(f"  ✓ /{p39:<3}/ filled "
                          f"({tokens_per_phone} tokens) at utt #{utt_idx}")

    if verbose:
        missing = [p for p in target_phones
                   if len(tokens_by_phone[p]) < tokens_per_phone]
        if missing:
            print("\n  Phones that did NOT fill their quota:")
            for p in missing:
                print(f"    /{p:<3}/  got {len(tokens_by_phone[p])}/"
                      f"{tokens_per_phone}")
        else:
            print("\n  All target phonemes filled their quotas.")
    return tokens_by_phone


# ────────────────────────────────────────────────────────────────────────
# CorticalLIME with OOM-safe retry
# ────────────────────────────────────────────────────────────────────────

def _try_clear_jax_caches() -> None:
    """Best-effort cache clearing for JAX/Flax OOM recovery."""
    try:
        import jax  # noqa: WPS433
        if hasattr(jax, "clear_caches"):
            jax.clear_caches()
        elif hasattr(jax, "clear_backends"):
            jax.clear_backends()
    except Exception:
        pass
    gc.collect()


def explain_with_oom_retry(
    explainer: CorticalLIME,
    audio: np.ndarray,
    target_class: int,
    *,
    fallback_batch_sizes: Sequence[int] = (32, 16, 8),
    verbose_prefix: str = "",
) -> Optional[object]:
    """Run ``explainer.explain`` and retry with progressively smaller
    decoder batch sizes if JAX raises an OOM-style error.  Returns
    ``None`` if every retry fails."""
    original_batch = int(getattr(explainer, "batch_size", 64))
    attempts = [original_batch] + [
        b for b in fallback_batch_sizes if b < original_batch
    ]
    last_err: Optional[BaseException] = None
    for bs in attempts:
        explainer.batch_size = bs
        try:
            return explainer.explain(audio, target_class=target_class)
        except Exception as ex:  # noqa: BLE001 — JAX/XLA OOMs vary by version
            last_err = ex
            msg = str(ex).lower()
            looks_oom = any(tok in msg for tok in (
                "out of memory", "oom", "resource_exhausted", "cuda_error",
                "xla_runtime_error", "memory",
            ))
            if not looks_oom:
                # Surface non-OOM errors immediately.
                explainer.batch_size = original_batch
                raise
            print(f"{verbose_prefix}  [OOM] batch={bs} → clearing caches "
                  f"and retrying smaller…")
            _try_clear_jax_caches()
    explainer.batch_size = original_batch
    print(f"{verbose_prefix}  [FAIL] every retry OOM'd: {last_err}")
    return None


# ────────────────────────────────────────────────────────────────────────
# Main pipeline
# ────────────────────────────────────────────────────────────────────────

def run_highres_pipeline(
    ds: TIMITDataset,
    encode_fn,
    decode_fn,
    sr_pairs: np.ndarray,
    *,
    target_phones: Sequence[str],
    tokens_per_phone: int,
    n_samples: int,
    pad_ms: float,
    keep_prob: float,
    kernel_width: float,
    surrogate_alpha: float,
    batch_size: int,
    seed: int,
    require_correct_pred: bool,
) -> Tuple[np.ndarray, List[str], List[str], np.ndarray, int, int]:
    """End-to-end: collect perfect tokens, run high-res LIME, aggregate.

    Returns
    -------
    mean_importances : (P, n_bands*n_strfs) float32
    phoneme_labels   : list of length P
    families         : list of length P
    n_tokens         : (P,) int32 — collected count per phoneme
    n_bands          : int
    n_strfs          : int
    """
    print("\nBuilding band-mode CorticalLIME (n_samples="
          f"{n_samples}, batch_size={batch_size})…")
    explainer = CorticalLIME(
        encode_fn=encode_fn, decode_fn=decode_fn, sr=sr_pairs,
        strategy="band_bernoulli",
        n_samples=n_samples, keep_prob=keep_prob,
        kernel_width=kernel_width,
        surrogate_type="ridge", surrogate_alpha=surrogate_alpha,
        batch_size=batch_size, seed=seed,
    )
    n_bands = int(getattr(explainer, "n_bands", 1))
    n_strfs = int(getattr(explainer, "n_strfs", sr_pairs.shape[0]))
    n_features = n_bands * n_strfs
    print(f"  Explainer ready: n_bands={n_bands}, n_strfs={n_strfs}, "
          f"feature_dim={n_features}.")

    # ── 1. Collect tokens. ──────────────────────────────────────────
    tokens_by_phone = collect_perfect_tokens(
        ds, target_phones, tokens_per_phone,
        encode_fn=encode_fn, decode_fn=decode_fn,
        pad_ms=pad_ms,
        require_correct_pred=require_correct_pred,
    )

    # ── 2. Run LIME on every collected token. ───────────────────────
    rows: List[np.ndarray] = []
    phoneme_labels: List[str] = []
    families: List[str] = []
    n_tokens_per_phone: List[int] = []

    print("\nRunning ultra-high-resolution LIME on collected tokens…")
    for p in target_phones:
        toks = tokens_by_phone.get(p, [])
        if not toks:
            print(f"\n[/{p}/] 0 tokens — phoneme dropped from output matrix.")
            continue
        fam = FAMILY_OF_39.get(p, "?")
        print(f"\n[/{p}/  family={fam}]  N={len(toks)} tokens")

        per_token_imp: List[np.ndarray] = []
        for i, tok in enumerate(toks, start=1):
            # New seed per token so we don't waste 5000 reps on identical
            # mask draws across the N tokens of a phoneme.
            explainer.seed = seed + (1009 * (i - 1)) + hash(p) % 9973
            res = explain_with_oom_retry(
                explainer, tok["audio"], tok["class_idx"],
                verbose_prefix=f"  [{p} {i:>2}/{len(toks)}]",
            )
            if res is None:
                continue
            imp = np.abs(np.asarray(res.importances, dtype=np.float64))
            per_token_imp.append(imp)
            print(
                f"  [{p} {i:>2}/{len(toks)}] "
                f"utt={tok['utt_id']:<22} dur={tok['duration_ms']:6.1f} ms  "
                f"P={float(res.target_prob):.3f}  "
                f"R²={float(res.surrogate_r2):.3f}"
            )

        if not per_token_imp:
            print(f"  /{p}/ produced 0 successful LIME runs — dropped.")
            continue
        # Mean of |importances| over the N tokens of this phoneme.
        row = np.mean(np.stack(per_token_imp, axis=0), axis=0)
        if row.size != n_features:
            # Defensive reshape — should always match in band mode.
            print(f"  [warn] /{p}/ row dim={row.size} != expected "
                  f"{n_features}; padding/truncating.")
            fixed = np.zeros(n_features, dtype=np.float64)
            k = min(row.size, n_features)
            fixed[:k] = row[:k]
            row = fixed
        rows.append(row.astype(np.float32))
        phoneme_labels.append(p)
        families.append(fam)
        n_tokens_per_phone.append(len(per_token_imp))

    if not rows:
        raise RuntimeError("No phonemes survived — taxonomy matrix empty.")

    mean_importances = np.vstack(rows)
    return (
        mean_importances, phoneme_labels, families,
        np.asarray(n_tokens_per_phone, dtype=np.int32),
        n_bands, n_strfs,
    )


# ────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_dir", default=str(_DEFAULT_MODEL_DIR),
                   help="Directory containing chkStep_*.p")
    p.add_argument("--timit_local", default=None,
                   help="Optional local TIMIT root (skip Kaggle download).")
    p.add_argument("--output_dir", default=str(_DEFAULT_OUTPUT_DIR),
                   help="Where to write highres_taxonomy_data.npz.")
    p.add_argument(
        "--phones", nargs="+", default=ORDERED_PHONES_39,
        help="TIMIT-39 phones to extract (defaults to all 39 across 6 "
             "manner families).",
    )
    p.add_argument("--tokens_per_phone", type=int, default=15,
                   help="N perfect tokens to collect per phoneme.")
    p.add_argument("--pad_ms", type=float, default=5.0,
                   help="Symmetric padding (ms) added to phone bounds.")
    p.add_argument("--n_samples", type=int, default=5000,
                   help="Perturbation masks per CorticalLIME run.")
    p.add_argument("--keep_prob", type=float, default=0.85)
    p.add_argument("--kernel_width", type=float, default=0.25)
    p.add_argument("--surrogate_alpha", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--allow_misclassified", action="store_true",
        help="Accept tokens whose argmax prediction disagrees with truth. "
             "Off by default — disabling the check produces a noisier "
             "matrix.",
    )
    p.add_argument(
        "--output_name", default="highres_taxonomy_data.npz",
        help="Output .npz filename (saved into --output_dir).",
    )
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output_name

    print("=" * 72)
    print("PHONETIC SNIPER  —  Ultra-high-resolution taxonomy data extractor")
    print("=" * 72)

    print("\nLoading frontend…")
    encode_fn, decode_fn, sr_pairs = load_frontend(Path(args.model_dir))

    print("\nLoading TIMIT TEST split…")
    ds = TIMITDataset(split="TEST", local_path=args.timit_local)
    print(f"  {len(ds)} utterances")

    (mean_importances, phoneme_labels, families,
     n_tokens, n_bands, n_strfs) = run_highres_pipeline(
        ds, encode_fn, decode_fn, sr_pairs,
        target_phones=args.phones,
        tokens_per_phone=args.tokens_per_phone,
        n_samples=args.n_samples,
        pad_ms=args.pad_ms,
        keep_prob=args.keep_prob,
        kernel_width=args.kernel_width,
        surrogate_alpha=args.surrogate_alpha,
        batch_size=args.batch_size,
        seed=args.seed,
        require_correct_pred=not args.allow_misclassified,
    )

    print("\n" + "─" * 72)
    print("Aggregation summary")
    print("─" * 72)
    print(f"  matrix shape   : {mean_importances.shape}")
    print(f"  phonemes kept  : {len(phoneme_labels)}")
    print(f"  n_bands × n_strfs = {n_bands} × {n_strfs} = "
          f"{n_bands * n_strfs}")
    by_fam: Dict[str, int] = {}
    for f in families:
        by_fam[f] = by_fam.get(f, 0) + 1
    for fam in PHONEME_FAMILIES_39:
        if fam in by_fam:
            print(f"    {fam:<16}: {by_fam[fam]} phones")

    np.savez(
        out_path,
        mean_importances=mean_importances,
        phoneme_labels=np.array(phoneme_labels),
        families=np.array(families),
        n_tokens=n_tokens,
        n_bands=np.int32(n_bands),
        n_strfs=np.int32(n_strfs),
    )
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
