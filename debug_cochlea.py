#!/usr/bin/env python3
"""Quick debug probe: load the trained frontend, run a single TIMIT clip
through cochlea_fn, and dump statistics + a tiny PNG so we can see what
the cochleagram tensor actually looks like."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_R_CODE = (_SCRIPT_DIR / "r_code").resolve()
sys.path.insert(0, str(_R_CODE))
sys.path.insert(0, str(_SCRIPT_DIR))
os.chdir(str(_R_CODE))

# Reuse the exact loader from generate_hero_cochleagrams.py.
from generate_hero_cochleagrams import (   # noqa: E402
    load_frontend, cochleagram_db_for_display, cochleagram_freq_axis,
    find_clean_token, crop_with_padding, PHONE_DUR_WINDOW_MS,
)
from timit_dataset import TIMITDataset  # noqa: E402

OUT = Path("/Users/eminent/Projects/Cortical_Front/r_code/analysis_outputs2")
OUT.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path(
    "/Users/eminent/Projects/Cortical_Front/r_code/"
    "nishit_trained_models/main_jax_phoneme_rec_timit"
)


def _stats(name, x):
    x = np.asarray(x)
    finite = np.isfinite(x)
    print(f"  {name:<28s} shape={x.shape}  dtype={x.dtype}")
    print(f"    finite={finite.all()}  min={np.nanmin(x):+.6g}  "
          f"max={np.nanmax(x):+.6g}  mean={np.nanmean(x):+.6g}  "
          f"std={np.nanstd(x):+.6g}")
    nz = (np.abs(x) > 1e-12).sum()
    print(f"    nonzero count = {nz}/{x.size}  "
          f"({100.0 * nz / x.size:.2f}%)")


def main():
    print("=" * 64)
    print("DEBUG PROBE — cochlea_fn end-to-end")
    print("=" * 64)
    print("\n[1/4] Loading frontend…")
    encode_fn, decode_fn, sr_pairs, cochlea_fn = load_frontend(MODEL_DIR)

    print("\n[2/4] Loading TIMIT TEST split…")
    ds = TIMITDataset(split="TEST")
    print(f"  {len(ds)} utterances")

    print("\n[3/4] Probing with three increasingly demanding stimuli.")

    # ── (a) full-utterance, no cropping. This is what run.py uses. ──
    utt = ds[0]
    y_utt = np.asarray(utt.audio, dtype=np.float32)
    rms_utt = float(np.sqrt(np.mean(y_utt ** 2)))
    y_utt_norm = y_utt / max(rms_utt, 1e-12)
    print(f"\n— (a) Full utterance {utt.utterance_id}: "
          f"len={len(y_utt)} samples, RMS={rms_utt:.4f}")
    _stats("y_utt_norm (audio in)", y_utt_norm)
    coch_a = cochlea_fn(y_utt_norm)
    _stats("cochleagram (a) linear", coch_a)
    db_a = cochleagram_db_for_display(coch_a, db_floor=60.0)
    _stats("cochleagram (a) dB",     db_a)

    # ── (b) the surgical /s/ crop the hero script uses. ──
    token = find_clean_token(ds, "s")
    if token is None:
        print("  ✗ no clean /s/ token found"); return
    utt_s, ps_s = token
    pad = int(round(10e-3 * 16000))
    y_s, _, _ = crop_with_padding(
        utt_s.audio, ps_s.start_sample, ps_s.end_sample, pad,
    )
    rms_s = float(np.sqrt(np.mean(y_s ** 2)))
    y_s_norm = y_s.astype(np.float32) / max(rms_s, 1e-12)
    print(f"\n— (b) /s/ crop from {utt_s.utterance_id}: "
          f"len={len(y_s)} samples, dur={len(y_s) * 1000.0 / 16000:.1f} ms, "
          f"RMS={rms_s:.4f}")
    _stats("y_s_norm", y_s_norm)
    coch_b = cochlea_fn(y_s_norm)
    _stats("cochleagram (b) linear", coch_b)
    db_b = cochleagram_db_for_display(coch_b, db_floor=60.0)
    _stats("cochleagram (b) dB",     db_b)

    # ── (c) cross-check: encode_fn should be strf(wav2aud(x)). ──
    cortical = encode_fn(y_s_norm[None, :])
    print(f"\n— (c) encode_fn (cortical) on the same /s/ clip:")
    _stats("cortical (B,F,T,S)", cortical)

    print("\n[3.5] Trying alternative dB conventions on the /s/ crop:")
    # Option A: ABS value (what NSL toolbox uses for display).
    abs_b = np.abs(coch_b)
    db_abs = 20.0 * np.log10(abs_b / max(abs_b.max(), 1e-12) + 1e-12)
    db_abs = np.clip(db_abs, -60.0, 0.0)
    _stats("dB( |coch| / max|coch| )", db_abs)
    # Option B: shift to non-negative, then dB.
    shifted = coch_b - coch_b.min()
    db_shift = 20.0 * np.log10(shifted / max(shifted.max(), 1e-12) + 1e-12)
    db_shift = np.clip(db_shift, -60.0, 0.0)
    _stats("dB( shift-to-zero )",      db_shift)
    # Option C: negate (since loudest seems to be near zero, quietest near -75).
    neg = -coch_b
    neg = np.clip(neg, 0.0, None)
    db_neg = 20.0 * np.log10(neg / max(neg.max(), 1e-12) + 1e-12)
    db_neg = np.clip(db_neg, -60.0, 0.0)
    _stats("dB( -coch, rectified )",   db_neg)

    print("\n[4/4] Saving a tiny diagnostic PNG…")
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5),
                             gridspec_kw={"hspace": 0.45, "wspace": 0.30})
    freqs_a = cochleagram_freq_axis(coch_a.shape[1])
    t_a = np.arange(coch_a.shape[0]) * 5.0          # 5 ms hop
    freqs_b = cochleagram_freq_axis(coch_b.shape[1])
    t_b = np.arange(coch_b.shape[0]) * 5.0

    axes[0, 0].imshow(coch_a.T, origin="lower", aspect="auto",
                      extent=[t_a[0], t_a[-1], freqs_a[0], freqs_a[-1]],
                      cmap="magma")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("(a) Full utterance — LINEAR cochleagram")
    axes[0, 0].set_ylabel("Hz"); axes[0, 0].set_xlabel("ms")

    axes[0, 1].imshow(db_a.T, origin="lower", aspect="auto",
                      extent=[t_a[0], t_a[-1], freqs_a[0], freqs_a[-1]],
                      cmap="magma", vmin=-60, vmax=0)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("(a) Full utterance — dB")

    axes[1, 0].imshow(coch_b.T, origin="lower", aspect="auto",
                      extent=[t_b[0], t_b[-1], freqs_b[0], freqs_b[-1]],
                      cmap="magma")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("(b) /s/ crop — LINEAR cochleagram")
    axes[1, 0].set_ylabel("Hz"); axes[1, 0].set_xlabel("ms")

    axes[1, 1].imshow(db_b.T, origin="lower", aspect="auto",
                      extent=[t_b[0], t_b[-1], freqs_b[0], freqs_b[-1]],
                      cmap="magma", vmin=-60, vmax=0)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("(b) /s/ crop — dB")

    out_png = OUT / "debug_cochlea.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"  → {out_png}")


if __name__ == "__main__":
    main()
