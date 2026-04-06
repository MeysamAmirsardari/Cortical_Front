#!/usr/bin/env python3
"""
Run phoneme recognition on a WAV (or other librosa-readable) file using a
trained STRF phoneme model checkpoint.

Training saves checkpoints as pickle: [nn_params, params] (see STRF_phonemeRecognition.py).

Default model directory (relative to repo root): models/main_jax_phoneme_rec_timit
Looks for chkStep_*.p with the highest step, or use --checkpoint (repo-relative or absolute).

The model expects 16 kHz mono and (by default) exactly 1 s of audio (16000 samples);
longer files are center-cropped, shorter files are zero-padded.

Requires: jax, flax, librosa, numpy.
Run from anywhere; the script temporarily uses the repo root for cochba / npz paths.
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import re
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import librosa
import numpy as np

# Repo root (directory containing this script)
REPO_ROOT = Path(__file__).resolve().parent


def resolve_in_repo(path_str: str) -> Path:
    """Project paths: relative to repo root; absolute paths unchanged."""
    p = Path(path_str)
    return p.resolve() if p.is_absolute() else (REPO_ROOT / p).resolve()


# Same 61-phone inventory and index order as precompute_timit_labels.py (labels are 1..61)
TIMIT_PHONEMES = [
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


def ensure_cochlear_npz() -> None:
    """supervisedSTRF loads ./cochlear_filter_params.npz from CWD; build from cochba if missing."""
    npz_path = REPO_ROOT / "cochlear_filter_params.npz"
    if npz_path.is_file():
        return
    os.chdir(REPO_ROOT)
    from strfpy_jax import read_cochba_j

    Bs, As = read_cochba_j()
    np.savez(npz_path, Bs=np.asarray(Bs), As=np.asarray(As))
    print(f"Wrote {npz_path} from cochba.txt (needed by supervisedSTRF).", file=sys.stderr)


def load_checkpoint(path: Path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, (list, tuple)) and len(obj) == 2:
        return obj[0], obj[1]
    if isinstance(obj, dict) and "nn_params" in obj and "params" in obj:
        return obj["nn_params"], obj["params"]
    raise ValueError(
        f"Unexpected checkpoint format in {path}: expected [nn_params, params] or dict with those keys."
    )


def find_latest_checkpoint(model_dir: Path) -> Path:
    pattern = str(model_dir / "chkStep_*.p")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"No chkStep_*.p under {model_dir}. Train first or pass --checkpoint explicitly."
        )

    def step_key(p: str) -> int:
        m = re.search(r"chkStep_(\d+)\.p$", p)
        return int(m.group(1)) if m else -1

    return Path(max(files, key=step_key))


def infer_dense_num_classes(nn_params) -> int | None:
    """Infer classifier output size from the last Flax Dense kernel."""
    try:
        from flax.traverse_util import flatten_dict
    except ImportError:
        return None
    params = nn_params.get("params", nn_params)
    flat = flatten_dict(params)
    best_idx = -1
    best_c = None
    for key, val in flat.items():
        if key[-1] != "kernel" or not hasattr(val, "shape") or len(val.shape) < 2:
            continue
        path_str = "/".join(str(k) for k in key[:-1])
        if "Dense" not in path_str:
            continue
        m = re.search(r"Dense_(\d+)", path_str)
        idx = int(m.group(1)) if m else 0
        if idx >= best_idx:
            best_idx = idx
            best_c = int(val.shape[-1])
    return best_c


def idx_to_phone(idx: int) -> str:
    """Map logits index to TIMIT symbol; training uses 1..61 for phones, 0 unused/pad."""
    if idx == 0:
        return "<blank>"
    if 1 <= idx <= len(TIMIT_PHONEMES):
        return TIMIT_PHONEMES[idx - 1]
    return f"<unk:{idx}>"


def collapse_repeated(seq: list[int]) -> list[int]:
    if not seq:
        return []
    out = [seq[0]]
    for x in seq[1:]:
        if x != out[-1]:
            out.append(x)
    return out


def prepare_audio(path: str, target_sr: int, n_samples: int, crop: str) -> tuple[np.ndarray, float]:
    """Load audio, resample to target_sr, return (waveform float32, duration_sec)."""
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    dur = float(len(y) / target_sr)
    if len(y) >= n_samples:
        if crop == "center":
            start = (len(y) - n_samples) // 2
            y = y[start : start + n_samples]
        elif crop == "start":
            y = y[:n_samples]
        else:
            raise ValueError(f"Unknown crop mode: {crop}")
    else:
        y = np.pad(y, (0, n_samples - len(y)), mode="constant")
    return y.astype(np.float32), dur


def main():
    parser = argparse.ArgumentParser(description="Phoneme inference with STRF TIMIT checkpoint.")
    parser.add_argument("--audio", required=True, help="Path to WAV/audio file.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models/main_jax_phoneme_rec_timit",
        help="Directory containing chkStep_*.p; relative to repo root unless absolute.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Explicit path to .p checkpoint; default: latest chkStep_*.p in model_dir.",
    )
    parser.add_argument(
        "--n_phones",
        type=int,
        default=61,
        help="Number of phoneme classes (not counting the +1 logit in the model). "
        "TIMIT setup uses 61; use 62 only if your checkpoint was trained that way.",
    )
    parser.add_argument("--decoder_type", type=str, default="cnn")
    parser.add_argument("--encoder_type", type=str, default="strf")
    parser.add_argument("--input_type", type=str, default="audio")
    parser.add_argument("--compression_method", type=str, default="identity")
    parser.add_argument("--update_lin", action="store_true")
    parser.add_argument("--use_class", action="store_true")
    parser.add_argument("--conv_feats", nargs="+", type=int, default=[10, 20, 40])
    parser.add_argument("--pooling_stride", type=int, default=2)
    parser.add_argument(
        "--crop",
        choices=["center", "start"],
        default="center",
        help="How to pick 1 s from longer audio.",
    )
    parser.add_argument(
        "--skip_blank",
        action="store_true",
        help="Remove <blank> (index 0) from collapsed output.",
    )
    args = parser.parse_args()

    model_dir = resolve_in_repo(args.model_dir)
    ckpt_path = (
        resolve_in_repo(args.checkpoint)
        if args.checkpoint
        else find_latest_checkpoint(model_dir)
    )

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Imports that pull in strfpy_jax / supervisedSTRF after optional npz creation
    os.chdir(REPO_ROOT)
    ensure_cochlear_npz()

    from supervisedSTRF import vSupervisedSTRF

    nn_params, aud_params = load_checkpoint(ckpt_path)
    print(f"Loaded checkpoint: {ckpt_path}", file=sys.stderr)

    inferred_c = infer_dense_num_classes(nn_params)
    if inferred_c is not None and inferred_c != args.n_phones + 1:
        print(
            f"Warning: checkpoint Dense output dim is {inferred_c}, but n_phones+1 would be "
            f"{args.n_phones + 1}. Pass --n_phones {inferred_c - 1} if inference looks wrong.",
            file=sys.stderr,
        )

    model = vSupervisedSTRF(
        n_phones=args.n_phones,
        input_type=args.input_type,
        update_lin=args.update_lin,
        use_class=args.use_class,
        encoder_type=args.encoder_type,
        decoder_type=args.decoder_type,
        compression_method=args.compression_method,
        conv_feats=args.conv_feats,
        pooling_stride=args.pooling_stride,
    )

    target_sr = 16000
    n_samples = 16000
    audio_path = Path(args.audio)
    audio_path = audio_path.resolve() if audio_path.is_absolute() else (Path.cwd() / audio_path).resolve()
    wav, duration = prepare_audio(str(audio_path), target_sr, n_samples, args.crop)
    # Match training RMS normalization
    rms = np.sqrt(np.mean(wav**2))
    if rms > 0:
        wav = wav / rms

    x = jnp.asarray(wav)[None, :]  # (1, 16000)

    def forward(nn_p, x_b, ap):
        logits = model.apply(nn_p, x_b, ap)
        if logits.ndim == 4:
            logits = jnp.mean(logits, axis=2)
        return logits

    logits = forward(nn_params, x, aud_params)
    pred = np.asarray(jnp.argmax(logits, axis=-1)[0])
    collapsed = collapse_repeated(pred.tolist())

    if args.skip_blank:
        collapsed = [i for i in collapsed if i != 0]

    phones = [idx_to_phone(i) for i in collapsed]
    # Readable one-line output
    print(" ".join(phones))
    print(f"# source_duration_s={duration:.3f} used_samples={n_samples} crop={args.crop}", file=sys.stderr)


if __name__ == "__main__":
    main()
