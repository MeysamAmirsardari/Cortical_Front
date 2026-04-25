#!/usr/bin/env python3
"""Surgical hero-figure data extractor — *biomimetic* cochleagram edition.

For each target phoneme this script:

  1. Walks the TIMIT TEST split and locates the *first* clean instance of
     the phone whose duration falls inside a sane per-phone window.
  2. Crops the utterance to **exactly** the phone's [start, end] bounds
     plus a tight 10 ms pad on each side — never a fixed 0.3 s window.
  3. Runs that exact crop through the **trained model's own NSL
     biomimetic cochlear filterbank** (``supervisedSTRF.wav2aud``).
     The image you see in Figure 1, top row, is therefore literally
     what the cortex stage in the same network sees — *not* a generic
     scipy STFT.  Frequency axis is derived from the NSL formula
     ``f[k] = 440·2^((k − 31 + octave_shift)/24)``.
  4. For visualisation only, the linear cochleagram is converted to dB
     and clipped to a 60 dB dynamic range so formant structure pops.
     The raw linear cochleagram is also persisted for downstream use.
  5. Runs ``CorticalLIME`` on the **same exact crop** so the heatmap
     in the Figure-1 bottom row pairs cell-for-cell with the cochleagram
     in the top row.
  6. Persists everything (model cochleagrams, NSL frequency axis, LIME
     importances, per-phoneme durations) to ``hero_cochleagrams.npz``
     and renders the publication PNGs via
     ``lingo_analysis.plot_acoustic_vs_cortical_hero``.

Why no scipy/librosa STFT
-------------------------
The whole point of the paper is to interpret the *biomimetic auditory
neural network*: the cochlear filterbank, the lateral inhibitory
network, the leaky integrator, then the STRFs.  Comparing the model's
LIME-based saliency to a *generic* STFT on the same waveform is
scientifically invalid because (a) the frequency axes differ
(linear-Hz STFT vs. log-spaced NSL filterbank), (b) the dynamic range
is not the same (different compression, LIN, leaky integration), and
(c) interpretability claims about "what the model sees" require that
the picture come from the model itself.  This script enforces that.

Usage::

    python generate_hero_cochleagrams.py
    python generate_hero_cochleagrams.py --phones s aa iy t
"""

from __future__ import annotations

import argparse
import glob
import os as _os
import pickle
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

# ────────────────────────────────────────────────────────────────────────
# Project imports — mirror run.py's path resolution so r_code modules
# (supervisedSTRF, etc.) are discoverable wherever this script lives.
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

# strfpy_jax (imported transitively via supervisedSTRF) opens
# `./cochba.txt` with a relative path at *module-load* time, so we must
# chdir into r_code/ before any project import touches it.
_os.chdir(str(_R_CODE))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from cortical_lime import CorticalLIME, build_model, make_jax_callables  # noqa: E402
from timit_dataset import TIMITDataset  # noqa: E402
from lingo_analysis import (  # noqa: E402
    PHONE_61_TO_39, plot_acoustic_vs_cortical_hero,
    set_academic_style, load_lime_results,
)


# ────────────────────────────────────────────────────────────────────────
# Acceptance criteria for "clean" phoneme tokens
# ────────────────────────────────────────────────────────────────────────
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
# NSL biomimetic frontend — frequency axis & cochleagram callable
# ────────────────────────────────────────────────────────────────────────

def nsl_center_freqs(
    n_filterbank_channels: int = 128,
    octave_shift: int = 0,
) -> np.ndarray:
    """Centre frequencies (Hz) of the NSL cochlear filterbank used by the
    diffAudNeuro / supervisedSTRF frontend.

    The NSL filterbank is log-spaced according to::

        f[k] = 440 · 2^((k − 31 + octave_shift) / 24),   k = 0 … N−1

    With the default ``octave_shift=0`` and ``N=128`` channels this
    spans approximately **180 Hz – 7040 Hz**, 24 channels per octave.
    """
    k = np.arange(int(n_filterbank_channels))
    return 440.0 * 2.0 ** ((k - 31 + int(octave_shift)) / 24.0)


def cochleagram_freq_axis(n_output_channels: int) -> np.ndarray:
    """Return the frequency (Hz) axis aligned with the **observed**
    output cochleagram dimension ``n_output_channels``.

    The NSL pipeline applies a first-difference filter (or, in the
    update_lin=True branch used by this checkpoint, a length-2
    convolution with VALID padding), reducing 128 raw filterbank
    channels to 127 output bins.  Each output bin sits *between* two
    adjacent filterbank channels, so we report the geometric mean of
    those neighbours — the exact tonotopic centre of the differenced
    band.  When ``n_output_channels == 128`` we just return the raw
    filterbank centres.
    """
    raw = nsl_center_freqs(max(128, int(n_output_channels) + 1))
    if n_output_channels == raw.shape[0]:
        return raw[:n_output_channels]
    # n_output = n_filterbank − 1 ⇒ geometric mean of adjacent channels.
    n = int(n_output_channels)
    return np.sqrt(raw[:n] * raw[1:n + 1])


def load_frontend(model_dir: Path):
    """Locate the latest checkpoint in ``model_dir`` and build encode /
    decode / **cochleagram** callables.

    The cochleagram callable wraps the *non-vmapped* twin of
    ``supervisedSTRF`` so we can drive the model's own ``wav2aud``
    method with a single waveform.  This is the same trick that
    ``make_jax_callables`` uses for the decoder; it works because
    ``vmap(..., variable_axes={'params': None})`` keeps the parameter
    tree structurally identical between the vmapped and non-vmapped
    classes.
    """
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

    # ── Build the cochleagram callable from the same checkpoint. ──
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
    fac = aud_params["compression_params"]
    alpha = aud_params["alpha"] if "alpha" in aud_params else 0.9922179

    @jax.jit
    def _cochlea(x):
        # wav2aud returns (T_frames, F) — exactly what the cortex stage
        # of the SAME network consumes.  No external STFT.
        return base_model.apply(
            nn_params, x, fac, alpha, method=base_model.wav2aud,
        )

    def cochlea_fn(wav_np: np.ndarray) -> np.ndarray:
        return np.asarray(_cochlea(jnp.asarray(wav_np)))

    print(f"  Checkpoint: {ckpts[-1]}")
    print(f"  STRF bank:  {sr_pairs.shape[0]} channels   n_phones: {n_phones}")
    print(f"  Cochleagram: NSL biomimetic filterbank, "
          f"update_lin={model.update_lin}, alpha={float(alpha):.6f}.")
    return encode_fn, decode_fn, sr_pairs, cochlea_fn


# ────────────────────────────────────────────────────────────────────────
# Token sniping
# ────────────────────────────────────────────────────────────────────────

def find_clean_token(
    ds: TIMITDataset,
    phn39: str,
    sample_rate: int = 16_000,
) -> Optional[Tuple[object, object]]:
    """Return ``(utterance, phone_segment)`` for the first acceptable
    realisation of ``phn39``."""
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
            target = 0.5 * (lo + hi)
            score = abs(dur - target)
            if score < best_score:
                best, best_score = (utt, ps), score
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


# ────────────────────────────────────────────────────────────────────────
# Cochleagram → dB picture
# ────────────────────────────────────────────────────────────────────────

def cochleagram_db_for_display(
    coch_lin_TF: np.ndarray,
    db_floor: float = 60.0,
) -> np.ndarray:
    """Convert the model's *linear-energy* cochleagram into the dB image
    we plot in the figure.

    Conventions
    -----------
    Follows the NSL toolbox ``aud2dB`` convention: normalise to the
    per-panel peak, then take ``20·log10``.  The result is in [-∞, 0]
    dB; we clip to ``[-db_floor, 0]`` so the colourmap has a sane,
    panel-agnostic range that surfaces formant structure clearly.

    Parameters
    ----------
    coch_lin_TF : (T, F) array, output of ``model.wav2aud``.  May contain
        tiny negatives from the LIN convolution + leaky integration —
        we rectify before normalising.
    db_floor : dynamic-range below the per-panel peak that we display
        (e.g. 60 ⇒ everything below the peak − 60 dB is clipped).

    Returns
    -------
    (T, F) float32 array in dB, in the closed range ``[-db_floor, 0]``.
    """
    coch_pos = np.clip(coch_lin_TF, 0.0, None).astype(np.float64)
    peak = float(coch_pos.max())
    if peak <= 0.0:
        # Pathological all-zero panel — return a flat floor rather than -inf.
        return np.full(coch_pos.shape, -db_floor, dtype=np.float32)
    # Normalise relative to the panel peak, then convert to dB.  The +1e-12
    # guard avoids -inf at exact zeros without distorting the dB above floor.
    coch_db = 20.0 * np.log10(coch_pos / peak + 1e-12)
    coch_db = np.clip(coch_db, -db_floor, 0.0)
    return coch_db.astype(np.float32)


# ────────────────────────────────────────────────────────────────────────
# LIME runner (unchanged — the same surgical crop)
# ────────────────────────────────────────────────────────────────────────

def explain_crop(
    explainer: CorticalLIME,
    y_crop: np.ndarray,
    n_pad_to: int = 16_000,
) -> Tuple[np.ndarray, float, float, int]:
    """RMS-normalise + LIME-explain ``y_crop``."""
    y = np.asarray(y_crop, dtype=np.float32)
    rms = float(np.sqrt(np.mean(y ** 2)))
    if rms > 0:
        y = y / rms
    try:
        res = explainer.explain(y)
    except Exception:
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
    cochlea_fn: Callable[[np.ndarray], np.ndarray],
    phones: Sequence[str],
    pad_ms: float = 10.0,
    sample_rate: int = 16_000,
    db_floor: float = 60.0,
) -> Dict[str, np.ndarray]:
    """Walk TIMIT, crop / model-cochleagrammise / LIME-explain each
    target phone, and produce the dict that ``np.savez`` will serialise."""
    pad_samples = int(round(pad_ms * 1e-3 * sample_rate))
    payload: Dict[str, np.ndarray] = {}

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

        # ── 1. Surgical crop (exactly what LIME will explain) ──
        y_crop, s_idx, e_idx = crop_with_padding(
            utt.audio, ps.start_sample, ps.end_sample, pad_samples,
        )
        crop_dur_s = len(y_crop) / sample_rate

        # RMS-normalise so the model frontend sees the same scale it was
        # trained on (and the same scale CorticalLIME will use a moment
        # later).  Without this the cochleagram for a quiet TIMIT clip
        # collapses to ≈0 → flat panels in the figure.
        y_norm = y_crop.astype(np.float32)
        rms = float(np.sqrt(np.mean(y_norm ** 2)))
        if rms > 0:
            y_norm = y_norm / rms

        # ── 2. Cochleagram from the MODEL's own NSL frontend ──
        # Note: each unique crop length triggers a one-off JAX recompile
        # of wav2aud; for the ~20 hero phones this is a few minutes total.
        coch_TF_lin = cochlea_fn(y_norm)                              # (T, F)
        coch_TF_db = cochleagram_db_for_display(
            coch_TF_lin, db_floor=db_floor,
        )
        # The downstream plot expects (F, T): rows = frequency, cols = time.
        coch_FT_db = coch_TF_db.T
        freqs = cochleagram_freq_axis(coch_TF_lin.shape[1]).astype(np.float32)

        # ── 3. CorticalLIME on the SAME crop (no audio changes) ──
        # explain_crop will RMS-renormalise internally; safe to pass
        # either y_crop or y_norm — passing y_norm makes provenance
        # explicit.
        imp_vec, p_class, r2, tc = explain_crop(explainer, y_norm)
        if n_bands > 1 and imp_vec.size == n_bands * n_strfs:
            imp_2d = imp_vec.reshape(n_bands, n_strfs)
        else:
            imp_2d = imp_vec[None, :] if n_bands == 1 else imp_vec.reshape(
                -1, n_strfs)

        payload[f"{phn}__cochleagram"] = coch_FT_db                     # (F, T) dB
        payload[f"{phn}__cochleagram_linear"] = coch_TF_lin.T.astype(np.float32)
        payload[f"{phn}__freqs_hz"] = freqs
        payload[f"{phn}__duration_s"] = np.float32(crop_dur_s)
        payload[f"{phn}__phone_dur_s"] = np.float32(phn_dur_s)
        payload[f"{phn}__pad_ms"] = np.float32(pad_ms)
        payload[f"{phn}__lime_importance"] = imp_2d.astype(np.float32)
        payload[f"{phn}__target_prob"] = np.float32(p_class)
        payload[f"{phn}__surrogate_r2"] = np.float32(r2)
        payload[f"{phn}__utterance_id"] = np.array(utt.utterance_id)
        payload[f"{phn}__phone_label"] = np.array(ps.phone)
        payload[f"{phn}__cochleagram_source"] = np.array(
            "model_wav2aud_NSL")

        print(f"  /{phn:<3}/  ✓ utt={utt.utterance_id:<22}  "
              f"phone={phn_dur_s*1000:5.1f} ms  crop={crop_dur_s*1000:5.1f} ms  "
              f"coch=({coch_FT_db.shape[0]}F × {coch_FT_db.shape[1]}T)  "
              f"P={p_class:.3f}  R²={r2:.3f}")
    payload["frontend"] = np.array(
        "supervisedSTRF.wav2aud (NSL biomimetic)")
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
    """Loader that surfaces ``lime_importance`` and the linear cochleagram
    into each per-phone dict so the plot function can use the surgically
    matched heatmap and (optionally) the raw model output."""
    with np.load(path, allow_pickle=True) as d:
        keys = [k for k in d.files if k.endswith("__cochleagram")]
        out: Dict[str, Dict] = {}
        for k in keys:
            phn = k.replace("__cochleagram", "")
            entry: Dict = {
                "cochleagram": np.asarray(d[k]),                    # (F, T) dB
                "freqs_hz": np.asarray(d[f"{phn}__freqs_hz"]),
                "duration_s": float(d[f"{phn}__duration_s"]),
            }
            for opt in ("cochleagram_linear", "lime_importance",
                        "phone_dur_s", "pad_ms",
                        "target_prob", "surrogate_r2"):
                key = f"{phn}__{opt}"
                if key in d.files:
                    entry[opt] = np.asarray(d[key])
            out[phn] = entry
    return out


# ────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────

# Project-canonical default paths (mirror run.py output layout).
_PROJECT_ROOT = Path("/Users/eminent/Projects/Cortical_Front")
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "r_code" / "analysis_outputs2"
_DEFAULT_RESULTS = _DEFAULT_OUTPUT_DIR / "results_raw.npz"
_DEFAULT_MODEL_DIR = (
    _PROJECT_ROOT / "r_code" / "nishit_trained_models"
                  / "main_jax_phoneme_rec_timit"
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_dir", default=str(_DEFAULT_MODEL_DIR),
                   help="Directory containing chkStep_*.p")
    p.add_argument("--timit_local", default=None,
                   help="Optional local TIMIT root (skip Kaggle download).")
    p.add_argument("--output_dir", default=str(_DEFAULT_OUTPUT_DIR),
                   help="Where to write hero_cochleagrams.npz / PNG.")
    p.add_argument(
        "--phones", nargs="+",
        default=[
            # Fricatives
            "s", "sh", "f", "z",
            # Stops
            "t", "k", "p", "b",
            # Low / open vowels
            "aa", "ae", "ah", "uw",
            # High / close vowels
            "iy", "ih", "eh", "uh",
            # Nasals + glide
            "m", "n", "ng", "er",
        ],
        help="TIMIT-39 phones to extract, in order. They are rendered "
             "in groups of --panels_per_fig per output PNG.",
    )
    p.add_argument("--panels_per_fig", type=int, default=4,
                   help="Phonemes per row in each rendered PNG.")
    p.add_argument("--pad_ms", type=float, default=10.0,
                   help="Symmetric padding (ms) added to phone bounds.")
    p.add_argument("--db_floor", type=float, default=60.0,
                   help="Dynamic-range floor (dB) for the cochleagram "
                        "image — a viewing parameter only; the linear "
                        "cochleagram is also persisted unchanged.")
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
    encode_fn, decode_fn, sr_pairs, cochlea_fn = load_frontend(
        Path(args.model_dir),
    )

    print("\nBuilding band-mode CorticalLIME explainer...")
    explainer = CorticalLIME(
        encode_fn=encode_fn, decode_fn=decode_fn, sr=sr_pairs,
        strategy="band_bernoulli",
        n_samples=args.n_lime_samples, keep_prob=0.85,
        kernel_width=0.25, surrogate_type="ridge", surrogate_alpha=0.1,
        batch_size=args.batch_size, seed=args.seed,
    )

    print("\nLoading TIMIT TEST split...")
    ds = TIMITDataset(split="TEST", local_path=args.timit_local)
    print(f"  {len(ds)} utterances")

    payload = build_payload(
        ds, explainer, cochlea_fn,
        phones=args.phones, pad_ms=args.pad_ms,
        db_floor=args.db_floor,
    )

    npz_path = out / "hero_cochleagrams.npz"
    np.savez(npz_path, **payload)
    print(f"\nSaved cochleagrams → {npz_path}")

    extracted = [
        p for p in args.phones
        if f"{p}__cochleagram" in payload
    ]
    if not extracted:
        print("  No phones extracted — nothing to render.")
        return

    results_path = Path(args.results)
    panels = max(1, int(args.panels_per_fig))
    n_figs = (len(extracted) + panels - 1) // panels
    print(f"\nRendering {n_figs} figure(s)  ({panels} panels each)…")
    for i in range(n_figs):
        chunk = extracted[i * panels : (i + 1) * panels]
        suffix = f"_{i + 1:02d}" if n_figs > 1 else ""
        png_path = out / f"fig1_acoustic_vs_cortical_hero{suffix}.png"
        try:
            render_png(npz_path, results_path, png_path, phones=chunk)
        except RuntimeError as e:
            print(f"  PNG render skipped for chunk {chunk}: {e}")


if __name__ == "__main__":
    main()
