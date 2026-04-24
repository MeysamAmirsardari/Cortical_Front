#!/usr/bin/env python3
"""Surgical hero-figure data extractor.

For each target phoneme (default: /s/, /aa/, /iy/, /t/), this script:

  1. Walks the TIMIT TEST split and locates the *first* clean instance of
     the phone whose duration falls inside a sane window (rejects 5 ms
     clipped artefacts and >300 ms outliers).
  2. Crops the utterance to **exactly** the phone's [start, end] bounds
     plus a tight 10 ms pad on each side — never a fixed 0.3 s window.
  3. Computes a 25 ms / 5 ms-hop magnitude STFT, converts to dB, and
     applies a 60 dB dynamic-range floor so background mud disappears.
  4. Runs ``CorticalLIME`` on **the same exact crop**, so the heatmap
     in the Figure-1 bottom row pairs cell-for-cell with the cochleagram
     in the top row.
  5. Persists everything (cochleagrams, LIME importances, per-phoneme
     durations) to ``hero_cochleagrams.npz`` and renders a PNG via
     ``lingo_analysis.plot_acoustic_vs_cortical_hero``.

Usage::

    python generate_hero_cochleagrams.py \
        --model_dir r_code/checkpoints \
        --output_dir analysis_outputs2 \
        --phones s aa iy t
"""

from __future__ import annotations

import argparse
import glob
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

# ────────────────────────────────────────────────────────────────────────
# Project imports
# ────────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cortical_lime import CorticalLIME, build_model, make_jax_callables  # noqa: E402
from timit_dataset import TIMITDataset  # noqa: E402
from lingo_analysis import (  # noqa: E402
    PHONE_61_TO_39, plot_acoustic_vs_cortical_hero,
    set_academic_style, load_lime_results,
)


# ────────────────────────────────────────────────────────────────────────
# Acceptance criteria for "clean" phoneme tokens
# ────────────────────────────────────────────────────────────────────────
# Per-phoneme acceptable duration window (ms).  We reject clipped artefacts
# (<25 ms) and absurdly long outliers (anything over the upper bound).
PHONE_DUR_WINDOW_MS: Dict[str, Tuple[float, float]] = {
    "s":  (60.0, 220.0),
    "sh": (60.0, 220.0),
    "f":  (50.0, 200.0),
    "z":  (50.0, 200.0),
    "t":  (25.0, 150.0),
    "k":  (25.0, 150.0),
    "p":  (25.0, 150.0),
    "d":  (20.0, 130.0),
    "b":  (20.0, 130.0),
    "g":  (20.0, 130.0),
    "aa": (60.0, 250.0),
    "iy": (60.0, 250.0),
    "ih": (50.0, 220.0),
    "eh": (50.0, 220.0),
    "ae": (60.0, 250.0),
    "uw": (60.0, 250.0),
    "m":  (40.0, 200.0),
    "n":  (40.0, 200.0),
    "ng": (40.0, 200.0),
}
DEFAULT_DUR_WINDOW_MS = (40.0, 250.0)


# ────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────

def load_frontend(model_dir: Path):
    """Locate the latest checkpoint in ``model_dir`` and build encode /
    decode callables. Mirrors the loader in ``run.py``."""
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


def find_clean_token(
    ds: TIMITDataset,
    phn39: str,
    sample_rate: int = 16_000,
) -> Optional[Tuple[object, object]]:
    """Return ``(utterance, phone_segment)`` for the first acceptable
    realisation of ``phn39``. Acceptability = duration inside the per-phone
    window in ``PHONE_DUR_WINDOW_MS``."""
    lo_ms, hi_ms = PHONE_DUR_WINDOW_MS.get(phn39, DEFAULT_DUR_WINDOW_MS)
    lo, hi = lo_ms * 1e-3 * sample_rate, hi_ms * 1e-3 * sample_rate
    best = None
    best_score = np.inf
    for utt in ds:
        for ps in utt.phone_segments:
            label = ps.phone_39 or PHONE_61_TO_39.get(ps.phone, ps.phone)
            if label != phn39:
                continue
            dur = ps.end_sample - ps.start_sample
            if not (lo <= dur <= hi):
                continue
            # Prefer tokens close to the *centre* of the acceptable window —
            # avoids the lazy first-match bias toward minimum-duration clips.
            target = 0.5 * (lo + hi)
            score = abs(dur - target)
            if score < best_score:
                best, best_score = (utt, ps), score
                # Early exit on a near-perfect match.
                if score < 0.05 * (hi - lo):
                    return best
    return best


def crop_with_padding(
    audio: np.ndarray, start: int, end: int,
    pad_samples: int,
) -> Tuple[np.ndarray, int, int]:
    s = max(0, start - pad_samples)
    e = min(len(audio), end + pad_samples)
    return audio[s:e].astype(np.float32), s, e


def stft_db(
    y: np.ndarray, sr_hz: int,
    nperseg_ms: float = 25.0, hop_ms: float = 5.0,
    fmin: float = 60.0, fmax: float = 8000.0,
    db_floor: float = 60.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Magnitude STFT in dB, restricted to [fmin, fmax] and floored to
    ``max - db_floor`` so the background goes black."""
    nperseg = max(64, int(nperseg_ms * 1e-3 * sr_hz))
    noverlap = nperseg - int(hop_ms * 1e-3 * sr_hz)
    f, t, S = spectrogram(
        y.astype(np.float64), fs=sr_hz,
        nperseg=nperseg, noverlap=noverlap,
        scaling="spectrum", window="hann",
    )
    S_db = 10.0 * np.log10(S + 1e-12)
    sel = (f >= fmin) & (f <= fmax)
    S_db = S_db[sel]
    f = f[sel]
    # Per-panel dynamic-range floor — black anything more than db_floor below
    # the panel's own peak.  Crystal-clear formant structure.
    S_db = np.clip(S_db, S_db.max() - db_floor, None)
    return S_db.astype(np.float32), f.astype(np.float32), t.astype(np.float32)


def explain_crop(
    explainer: CorticalLIME,
    y_crop: np.ndarray,
    n_pad_to: int = 16_000,
) -> Tuple[np.ndarray, float, float, int]:
    """RMS-normalise + LIME-explain ``y_crop``. Pads to ``n_pad_to`` only
    if the encoder requires a fixed length (most do not)."""
    y = np.asarray(y_crop, dtype=np.float32)
    rms = float(np.sqrt(np.mean(y ** 2)))
    if rms > 0:
        y = y / rms
    try:
        res = explainer.explain(y)
    except Exception:
        # Encoder may require a fixed length — fall back to centred pad.
        if len(y) < n_pad_to:
            pad = n_pad_to - len(y)
            left = pad // 2
            y = np.pad(y, (left, pad - left))
        res = explainer.explain(y)
    return (
        np.asarray(res.importances, dtype=np.float32),
        float(res.target_prob),
        float(res.surrogate_r2),
        int(res.target_class),
    )


# ────────────────────────────────────────────────────────────────────────
# Main pipeline
# ────────────────────────────────────────────────────────────────────────

def build_payload(
    ds: TIMITDataset,
    explainer: CorticalLIME,
    phones: Sequence[str],
    pad_ms: float = 10.0,
    sample_rate: int = 16_000,
) -> Dict[str, np.ndarray]:
    """Walk TIMIT, crop / spectrogrammise / explain each target phone,
    and produce the dict that ``np.savez`` will serialise."""
    pad_samples = int(round(pad_ms * 1e-3 * sample_rate))
    payload: Dict[str, np.ndarray] = {}

    # n_bands / band_edges_hz are introspected from the explainer for
    # downstream reshape on the LIME side.
    n_bands = int(getattr(explainer, "n_bands", 1))
    n_strfs = int(getattr(explainer, "n_strfs", 0))
    if hasattr(explainer, "band_edges_hz") and explainer.band_edges_hz is not None:
        payload["band_edges_hz"] = np.asarray(
            explainer.band_edges_hz, dtype=np.float32,
        )
    payload["n_bands"] = np.int32(n_bands)

    print(f"\nExtracting {len(phones)} hero tokens "
          f"(pad ±{pad_ms:.0f} ms, n_bands={n_bands}, n_strfs={n_strfs}):")
    for phn in phones:
        token = find_clean_token(ds, phn, sample_rate=sample_rate)
        if token is None:
            print(f"  /{phn:<3}/  ✗ no clean token — skipped")
            continue
        utt, ps = token
        phn_dur_s = (ps.end_sample - ps.start_sample) / sample_rate

        y_crop, s_idx, e_idx = crop_with_padding(
            utt.audio, ps.start_sample, ps.end_sample, pad_samples,
        )
        crop_dur_s = len(y_crop) / sample_rate

        S_db, freqs, _ = stft_db(y_crop, sr_hz=sample_rate)

        imp_vec, p_class, r2, tc = explain_crop(explainer, y_crop)
        # Reshape importance to (n_bands, n_strfs) when in band-mode.
        if n_bands > 1 and imp_vec.size == n_bands * n_strfs:
            imp_2d = imp_vec.reshape(n_bands, n_strfs)
        else:
            imp_2d = imp_vec[None, :] if n_bands == 1 else imp_vec.reshape(
                -1, n_strfs)

        payload[f"{phn}__cochleagram"] = S_db
        payload[f"{phn}__freqs_hz"] = freqs
        payload[f"{phn}__duration_s"] = np.float32(crop_dur_s)
        payload[f"{phn}__phone_dur_s"] = np.float32(phn_dur_s)
        payload[f"{phn}__pad_ms"] = np.float32(pad_ms)
        payload[f"{phn}__lime_importance"] = imp_2d.astype(np.float32)
        payload[f"{phn}__target_prob"] = np.float32(p_class)
        payload[f"{phn}__surrogate_r2"] = np.float32(r2)
        payload[f"{phn}__utterance_id"] = np.array(utt.utterance_id)
        payload[f"{phn}__phone_label"] = np.array(ps.phone)

        print(f"  /{phn:<3}/  ✓ utt={utt.utterance_id:<22}  "
              f"phone={phn_dur_s*1000:5.1f} ms  crop={crop_dur_s*1000:5.1f} ms  "
              f"P={p_class:.3f}  R²={r2:.3f}")
    return payload


def render_png(
    npz_path: Path,
    results_path: Optional[Path],
    out_png: Path,
    phones: Sequence[str],
) -> None:
    """Render the publication PNG via ``lingo_analysis``."""
    set_academic_style()
    if results_path is None or not results_path.exists():
        # When there is no results_raw.npz we still need a LimeResults
        # shell carrying sr_pairs / n_bands / band_edges so the lingo
        # plot function can size the heatmap.  We can fabricate a
        # minimal one from the cochleagram npz alone.
        with np.load(npz_path, allow_pickle=True) as d:
            n_bands = int(d["n_bands"]) if "n_bands" in d.files else 1
            band_edges = (
                np.asarray(d["band_edges_hz"]) if "band_edges_hz" in d.files
                else None
            )
        raise RuntimeError(
            "results_raw.npz is required to size the LIME panel — "
            "run `python run.py` once first, then this script."
        )

    res = load_lime_results(str(results_path))
    cochs = _load_cochleagrams_with_lime(npz_path)
    fig = plot_acoustic_vs_cortical_hero(
        res, cochs, phones=phones,
    )
    fig.savefig(out_png)
    plt.close(fig)
    print(f"\n  Rendered Figure 1 → {out_png}")


def _load_cochleagrams_with_lime(path: Path) -> Dict[str, Dict]:
    """Loader that also surfaces ``lime_importance`` into each per-phone dict
    so the plot function can use the surgically-matched heatmap."""
    with np.load(path, allow_pickle=True) as d:
        keys = [k for k in d.files if k.endswith("__cochleagram")]
        out: Dict[str, Dict] = {}
        for k in keys:
            phn = k.replace("__cochleagram", "")
            entry: Dict = {
                "cochleagram": np.asarray(d[k]),
                "freqs_hz": np.asarray(d[f"{phn}__freqs_hz"]),
                "duration_s": float(d[f"{phn}__duration_s"]),
            }
            for opt in ("lime_importance", "phone_dur_s", "pad_ms",
                        "target_prob", "surrogate_r2"):
                key = f"{phn}__{opt}"
                if key in d.files:
                    entry[opt] = np.asarray(d[key])
            out[phn] = entry
    return out


# Project-canonical default paths (mirror run.py output layout).
_PROJECT_ROOT = Path("/Users/eminent/Projects/Cortical_Front")
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "r_code" / "analysis_outputs2"
_DEFAULT_RESULTS = _DEFAULT_OUTPUT_DIR / "results_raw.npz"
_DEFAULT_MODEL_DIR = _PROJECT_ROOT / "r_code" / "checkpoints"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_dir", default=str(_DEFAULT_MODEL_DIR),
                   help="Directory containing chkStep_*.p")
    p.add_argument("--timit_local", default=None,
                   help="Optional local TIMIT root (skip Kaggle download).")
    p.add_argument("--output_dir", default=str(_DEFAULT_OUTPUT_DIR),
                   help="Where to write hero_cochleagrams.npz / PNG.")
    p.add_argument("--phones", nargs="+",
                   default=["s", "aa", "iy", "t"],
                   help="TIMIT-39 phones to extract, in order.")
    p.add_argument("--pad_ms", type=float, default=10.0,
                   help="Symmetric padding (ms) added to phone bounds.")
    p.add_argument("--n_lime_samples", type=int, default=4000,
                   help="Perturbation masks per CorticalLIME run.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results", default=str(_DEFAULT_RESULTS),
                   help="Path to results_raw.npz (used to size the LIME "
                        "heatmap when rendering the PNG).")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading frontend...")
    encode_fn, decode_fn, sr_pairs = load_frontend(Path(args.model_dir))

    print("\nBuilding band-mode CorticalLIME explainer...")
    explainer = CorticalLIME(
        encode_fn=encode_fn, decode_fn=decode_fn, sr=sr_pairs,
        strategy="band_bernoulli",
        n_samples=args.n_lime_samples, keep_prob=0.85,
        kernel_width=0.25, surrogate_type="ridge", surrogate_alpha=1.0,
        batch_size=args.batch_size, seed=args.seed,
    )

    print("\nLoading TIMIT TEST split...")
    ds = TIMITDataset(split="TEST", local_path=args.timit_local)
    print(f"  {len(ds)} utterances")

    payload = build_payload(
        ds, explainer, phones=args.phones, pad_ms=args.pad_ms,
    )

    npz_path = out / "hero_cochleagrams.npz"
    np.savez(npz_path, **payload)
    print(f"\nSaved cochleagrams → {npz_path}")

    results_path = Path(args.results)
    png_path = out / "fig1_acoustic_vs_cortical_hero.png"
    try:
        render_png(npz_path, results_path, png_path, phones=args.phones)
    except RuntimeError as e:
        print(f"\n  PNG render skipped: {e}")


if __name__ == "__main__":
    main()
