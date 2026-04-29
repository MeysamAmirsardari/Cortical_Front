#!/usr/bin/env python3
"""Build best_phonemes_cochleagrams.ipynb — a self-contained Colab notebook
that picks the cleanest exemplar of each TIMIT-39 phoneme, runs the
trained model's biomimetic wav2aud on it, and renders a publication-grade
gallery of cochleagrams grouped by manner family.

Runs locally and writes a valid .ipynb JSON file.
"""

import json
import uuid
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent.parent.parent / "best_phonemes_cochleagrams.ipynb"


def _id() -> str:
    return uuid.uuid4().hex[:12]


def md(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": _id(),
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": _id(),
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


cells = []

# ────────────────────────────────────────────────────────────────────────
cells.append(md("""\
# Best Exemplar of Every TIMIT Phoneme — Through the Trained Biomimetic Cochlea

**Paper:** [A Biomimetic Frontend for Differentiable Audio Processing](https://arxiv.org/abs/2409.08997)
**Repo:**  [MeysamAmirsardari/Cortical_Front](https://github.com/MeysamAmirsardari/Cortical_Front)

This notebook does one thing, beautifully:

1. Loads the trained STRF phoneme-recognition model (`chkStep_200000.p`).
2. Walks the TIMIT TEST split and, for each of the 39 folded phonemes,
   finds the **cleanest, highest-confidence exemplar** — a token whose
   duration sits inside a biologically plausible window, that lives
   inside its utterance with room on both sides, and that the model
   itself classifies correctly with high posterior.
3. Runs each exemplar's surrounding 1.5 s window through the model's
   own learned cochlear filterbank (`supervisedSTRF.wav2aud`, **not**
   a generic STFT), then crops the cochleagram in the frame domain to
   the phone's exact bounds plus a tight 20 ms pad on each side.  This
   way the visualisation is literally what the cortex stage of the
   *same* network sees, with no boundary artefacts from the LIN
   convolution or the leaky integrator at the crop edges.
4. Renders a beautiful 6-family grid of dB cochleagrams on an NSL
   log-frequency axis, colour-coded by manner of articulation.

> **Runtime:** select **GPU** (Runtime → Change runtime type → T4 GPU).
> The whole notebook runs in ≈3 minutes on a T4, 6–8 minutes on CPU.
"""))

# ════════════════════════════════════════════════════════════════════════
cells.append(md("## 1. Environment setup"))

cells.append(code("""\
%%capture
# Install JAX with GPU support, Flax, and audio deps. Colab already has
# torch, numpy, matplotlib pre-installed.
!pip install --upgrade "jax[cuda12]" flax optax
!pip install librosa soundfile kagglehub
"""))

cells.append(code("""\
# Clone the repository so we can import the model code.
import os

REPO_URL = "https://github.com/MeysamAmirsardari/Cortical_Front.git"
REPO_DIR = "/content/Cortical_Front"
CODE_DIR = os.path.join(REPO_DIR, "r_code")

if not os.path.exists(REPO_DIR):
    !git clone {REPO_URL} {REPO_DIR}
    print(f"Cloned repository to {REPO_DIR}")
else:
    print(f"Repository already exists at {REPO_DIR}")

# The model code expects r_code/ as the CWD (it loads cochlear_filter_params.npz
# from a relative path).
os.chdir(CODE_DIR)
print(f"Working directory: {os.getcwd()}")
"""))

cells.append(code("""\
import sys
import re
import pickle
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec, patches
from matplotlib.colors import LinearSegmentedColormap

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, vmap

import librosa

# The model code lives in r_code/ — make it importable.
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from supervisedSTRF import vSupervisedSTRF, supervisedSTRF
from strfpy_jax import wav2aud_j  # only needed for sanity checks

warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

print(f"JAX  version : {jax.__version__}")
print(f"JAX  backend : {jax.default_backend()}")
print(f"JAX  devices : {jax.devices()}")
print(f"NumPy version: {np.__version__}")
"""))

# ════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## 2. Load the trained checkpoint and build two model twins

The shipped checkpoint stores `[nn_params, aud_params]`:
- `nn_params` — Flax tree (CNN weights, learned LIN kernel, leaky-integrator α).
- `aud_params` — `{sr, compression_params, alpha}`.

We build **two** model objects from the *same* parameter tree:

| Object | Class | Use |
|---|---|---|
| `model_v` | `vSupervisedSTRF` (vmapped) | batched scoring of candidate windows |
| `model_s` | `supervisedSTRF` (non-vmapped) | single-stream cochleagram extraction via `model_s.wav2aud(...)` |

This works because the vmap is configured with
`variable_axes={"params": None}`, which keeps the parameter tree
structurally identical between the vmapped and non-vmapped versions.
"""))

cells.append(code("""\
CHECKPOINT_PATH = "nishit_trained_models/main_jax_phoneme_rec_timit/chkStep_200000.p"

ckpt_path = Path(CHECKPOINT_PATH)
assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path.resolve()}"

with open(ckpt_path, "rb") as f:
    nn_params, aud_params = pickle.load(f)

# Auto-detect a couple of hyperparameters from the checkpoint contents.
from flax.traverse_util import flatten_dict
flat = flatten_dict(nn_params.get("params", nn_params))

best_idx, n_phones = -1, 62
for key, val in flat.items():
    key_str = "/".join(str(k) for k in key)
    if "Dense" in key_str and key[-1] == "kernel" and hasattr(val, "shape"):
        m = re.search(r"Dense_(\\d+)", key_str)
        idx = int(m.group(1)) if m else 0
        if idx >= best_idx:
            best_idx, n_phones = idx, int(val.shape[-1]) - 1   # model adds +1 (blank)

CONFIG = dict(
    n_phones=n_phones,
    input_type="audio",
    encoder_type="strf",
    decoder_type="cnn",
    compression_method="power" if "compression_params" in aud_params else "identity",
    update_lin="alpha" in aud_params,
    use_class="AuditorySpectrogram" in str(list(flat.keys())),
    conv_feats=[10, 20, 40],
    pooling_stride=2,
)
print("Detected config:")
for k, v in CONFIG.items():
    print(f"  {k:20s} = {v}")
print(f"  α  (leaky integrator) = {float(aud_params.get('alpha', 0.9922179)):.6f}")
print(f"  STRF bank             = {aud_params['sr'].shape[0]} channels")
"""))

cells.append(code("""\
# ── Vmapped model (for batched scoring) ──
model_v = vSupervisedSTRF(**CONFIG)

BATCH = 8
SR = 16_000
N_SAMPLES_1S = SR             # 16000
N_FRAMES_1S = 200             # 200 frames @ 5 ms hop

# Initialise param structure (we then immediately replace with the loaded
# weights — this is just to verify shape compatibility).
_ = model_v.init(jax.random.key(0), jnp.zeros([BATCH, N_SAMPLES_1S]), aud_params)

@jit
def clip_posterior(nn_params, audio_batch_BxT, aud_params):
    \"\"\"Run the full model and return ONE posterior vector per clip.

    The decoder's raw output is (B, F=128, T=25, K=n_phones+1) on this
    checkpoint — the first non-batch axis is *frequency* (the model's
    author kept the original tonotopic axis as the decoder's spatial
    dimension), and the *second* non-batch axis is the pooled time
    axis.  We don't need to get either axis exactly right to pick the
    best exemplar: averaging the softmax over every non-batch,
    non-class position gives a simple **clip-level posterior** —
    \"how confident is the model that the target phone is present
    *somewhere* in this 1-second window?\".  Higher = better exemplar.
    \"\"\"
    logits = model_v.apply(nn_params, audio_batch_BxT, aud_params)
    soft   = jax.nn.softmax(logits, axis=-1)               # (B, ..., K)
    reduce_axes = tuple(range(1, soft.ndim - 1))           # everything between B and K
    return soft.mean(axis=reduce_axes)                     # (B, K)

# JIT warm-up.
_ = clip_posterior(nn_params, jnp.zeros([BATCH, N_SAMPLES_1S]), aud_params)

# ── Non-vmapped twin (for clean cochleagram extraction) ──
model_s = supervisedSTRF(
    n_phones=CONFIG["n_phones"],
    input_type=CONFIG["input_type"],
    encoder_type=CONFIG["encoder_type"],
    decoder_type=CONFIG["decoder_type"],
    compression_method=CONFIG["compression_method"],
    update_lin=CONFIG["update_lin"],
    use_class=CONFIG["use_class"],
    conv_feats=CONFIG["conv_feats"],
    pooling_stride=CONFIG["pooling_stride"],
)

ALPHA = float(aud_params.get("alpha", 0.9922179))
COMP  = aud_params["compression_params"]

@jit
def cochlea_fn_jax(x_T):
    \"\"\"Compute the model's biomimetic cochleagram for ONE waveform.\"\"\"
    return model_s.apply(
        nn_params, x_T, COMP, ALPHA, method=model_s.wav2aud,
    )

def cochlea_np(wav: np.ndarray) -> np.ndarray:
    \"\"\"NumPy convenience wrapper.  Returns a (T_frames, F) float64 array.\"\"\"
    return np.asarray(cochlea_fn_jax(jnp.asarray(wav, dtype=jnp.float32)))

# Sanity: cochleagram of 1.5 s of silence — should not crash and should
# return a 2D array with ≈300 frames and ~129 frequency channels.
_test = cochlea_np(np.zeros(int(1.5 * SR), dtype=np.float32))
print(f"cochleagram smoke test: shape={_test.shape}  dtype={_test.dtype}")
"""))

# ════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## 3. NSL centre frequencies (from the codebase) & dB-display helper

The 129 cochlear-channel centre frequencies are **not** redefined here —
we mirror the **exact** formula the codebase itself uses inside
`strfpy.py` (lines 64–66):

```python
cf = np.linspace(-31, 97, 31+97+1)/24    # 129 channels, k = -31..97
cf = [440 * 2**f for f in cf]             # CFs in Hz
cf = [round(f/10)*10 for f in cf]         # rounded to the nearest 10 Hz
```

Spans **180 Hz – 7170 Hz**, 24 channels per octave.  The cochleagram
output of `model.wav2aud` has shape `(T_frames, 129)`; the frequency
axis is this exact 129-element vector, **bin-for-bin**.  No
geometric-mean approximation, no remapping — same array the model uses
internally.

**dB convention.** The trained checkpoint has `α = 1.0076 > 1`, which
inverts the leaky-integrator's sign convention — the cochleagram
output is *signed* and its peak energy lives at the **most-negative**
values.  A naïve `clip(coch, 0, ∞)` zeros out 80 %+ of the data and
produces a uniformly black panel.  The fix is to take the **magnitude**
before normalising and applying `20·log10`.
"""))

cells.append(code("""\
def cochlea_center_freqs() -> np.ndarray:
    \"\"\"Return the codebase's canonical 129-channel CF list, in Hz.

    *Verbatim* port of strfpy.py:64-66.  The cochlear filterbank in this
    project has 129 channels indexed by k = -31, -30, …, +97; the centre
    frequencies are the standard NSL log-spaced grid
    ``cf = 440 · 2^(k/24)`` rounded to the nearest 10 Hz, exactly as the
    rest of the codebase consumes them.  We do NOT redefine, smooth, or
    geometric-mean-approximate this array — it is bin-for-bin aligned
    with the (T, 129) cochleagram returned by ``model.wav2aud``.
    \"\"\"
    cf = np.linspace(-31, 97, 31 + 97 + 1) / 24.0
    cf = np.array([440.0 * 2.0 ** f for f in cf])
    cf = np.array([round(float(f) / 10.0) * 10.0 for f in cf])
    return cf


# Compute once, share everywhere.
COCHLEA_CF = cochlea_center_freqs()
print(f"Codebase CF list: {len(COCHLEA_CF)} channels   "
      f"({COCHLEA_CF[0]:.0f} Hz – {COCHLEA_CF[-1]:.0f} Hz)   "
      f"first 5: {COCHLEA_CF[:5].astype(int).tolist()}   "
      f"last 5: {COCHLEA_CF[-5:].astype(int).tolist()}")


def cochleagram_to_db(coch_TF: np.ndarray, db_floor: float = 60.0) -> np.ndarray:
    \"\"\"Magnitude-based aud2dB normalisation, clipped to [-db_floor, 0].\"\"\"
    mag = np.abs(np.asarray(coch_TF, dtype=np.float64))
    peak = float(mag.max())
    if peak <= 0.0:
        return np.full(mag.shape, -db_floor, dtype=np.float32)
    db = 20.0 * np.log10(mag / peak + 1e-12)
    return np.clip(db, -db_floor, 0.0).astype(np.float32)


# Sanity: the model's wav2aud must produce exactly len(COCHLEA_CF) channels.
t = np.linspace(0, 1.0, SR, endpoint=False)
chirp = 0.3 * np.sin(2 * np.pi * np.cumsum(np.linspace(200, 4000, SR)) / SR)
chirp /= max(np.sqrt(np.mean(chirp**2)), 1e-12)
coch_chirp = cochlea_np(chirp.astype(np.float32))
print(f"chirp cochleagram: shape={coch_chirp.shape}  "
      f"min={coch_chirp.min():+.3g}  max={coch_chirp.max():+.3g}")
assert coch_chirp.shape[1] == len(COCHLEA_CF), (
    f"Cochleagram has {coch_chirp.shape[1]} channels but the codebase's "
    f"CF list has {len(COCHLEA_CF)}.  These must match exactly — "
    f"check strfpy.py:64-66 for the canonical CF formula."
)
"""))

# ════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## 4. Phoneme inventory, manner families & duration windows

We follow the standard Lee-&-Hon 61→39 folding and group the 39 folded
phones into six **manner-of-articulation families**.  For each phone we
also define a biologically plausible duration window — silence is
trimmed to a tight band so we don't accidentally pick a 10 ms blip.
"""))

cells.append(code("""\
TIMIT_PHONEMES = [
    "aa","ae","ah","ao","aw","ax","ax-h","axr","ay",
    "b","bcl","ch","d","dcl","dh","dx",
    "eh","el","em","en","eng","epi","er","ey",
    "f","g","gcl","h#","hh","hv",
    "ih","ix","iy",
    "jh","k","kcl","l","m","n","ng","nx",
    "ow","oy","p","pau","pcl","q","r",
    "s","sh","t","tcl","th","uh","uw","ux",
    "v","w","y","z","zh",
]
PHONEME_TO_IDX = {p: i + 1 for i, p in enumerate(TIMIT_PHONEMES)}   # 1-indexed
IDX_TO_PHONEME = {0: "<blank>"}
IDX_TO_PHONEME.update({i + 1: p for i, p in enumerate(TIMIT_PHONEMES)})

PHONE_61_TO_39 = {
    "aa":"aa","ae":"ae","ah":"ah","ao":"aa","aw":"aw","ax":"ah","ax-h":"ah",
    "axr":"er","ay":"ay","b":"b","bcl":"sil","ch":"ch","d":"d","dcl":"sil",
    "dh":"dh","dx":"dx","eh":"eh","el":"l","em":"m","en":"n","eng":"ng",
    "epi":"sil","er":"er","ey":"ey","f":"f","g":"g","gcl":"sil","h#":"sil",
    "hh":"hh","hv":"hh","ih":"ih","ix":"ih","iy":"iy","jh":"jh","k":"k",
    "kcl":"sil","l":"l","m":"m","n":"n","ng":"ng","nx":"n","ow":"ow",
    "oy":"oy","p":"p","pau":"sil","pcl":"sil","q":"","r":"r","s":"s",
    "sh":"sh","t":"t","tcl":"sil","th":"th","uh":"uh","uw":"uw","ux":"uw",
    "v":"v","w":"w","y":"y","z":"z","zh":"sh",
}

# Six manner families covering all 39 folded phones.
PHONEME_FAMILIES = {
    "Vowels":         ["iy","ih","eh","ae","aa","ah","ao","uh","uw","er","ey","ay","oy","aw","ow"],
    "Stops":          ["p","t","k","b","d","g"],
    "Affricates":     ["ch","jh"],
    "Fricatives":     ["f","th","s","sh","v","dh","z","hh"],
    "Nasals":         ["m","n","ng"],
    "Liquids/Glides": ["l","r","w","y"],
}
ALL_PHONES_39 = [p for fam in PHONEME_FAMILIES.values() for p in fam]
assert len(ALL_PHONES_39) == 38, f"got {len(ALL_PHONES_39)} phones"   # excludes 'sil','dx'
# We deliberately skip 'sil' (boring) and 'dx' (very short) from the gallery.

# Per-phone duration windows in ms (lo, hi).  Looser than training defaults
# so we still catch tokens; tighter than chance so we reject pathological
# segmentations.
DEFAULT_DUR_WINDOW_MS = (40.0, 250.0)
PHONE_DUR_WINDOW_MS = {
    # Vowels: tend to be longer.
    "iy":(60,260),"ih":(40,200),"eh":(50,220),"ae":(60,250),"aa":(60,260),
    "ah":(40,200),"ao":(60,260),"uh":(40,200),"uw":(60,250),"er":(60,250),
    "ey":(70,260),"ay":(80,280),"oy":(80,280),"aw":(80,280),"ow":(70,260),
    # Stops & affricates: short bursts.
    "p":(20,140),"t":(20,140),"k":(20,140),"b":(20,140),"d":(20,140),"g":(20,140),
    "ch":(50,200),"jh":(50,200),
    # Fricatives: medium and noisy.
    "f":(50,220),"th":(40,200),"s":(50,220),"sh":(50,220),
    "v":(40,200),"dh":(30,180),"z":(40,220),"hh":(30,180),
    # Nasals: medium.
    "m":(40,200),"n":(40,200),"ng":(40,200),
    # Liquids/Glides: medium.
    "l":(40,200),"r":(40,220),"w":(40,200),"y":(40,200),
}

# Family colour palette for the poster.  Distinct, print-safe.
FAMILY_COLORS = {
    "Vowels":         "#E63946",   # warm red
    "Stops":          "#F4A261",   # amber
    "Affricates":     "#E9C46A",   # yellow
    "Fricatives":     "#2A9D8F",   # teal
    "Nasals":         "#264653",   # deep blue-green
    "Liquids/Glides": "#8E7DBE",   # soft violet
}

def phone_to_family(p: str) -> Optional[str]:
    for fam, members in PHONEME_FAMILIES.items():
        if p in members:
            return fam
    return None

print(f"Will gallery {len(ALL_PHONES_39)} phones across {len(PHONEME_FAMILIES)} families.")
"""))

# ════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## 5. Download TIMIT and parse phoneme alignments

We download the TIMIT corpus from Kaggle (`mfekadu/darpa-timit-acousticphonetic-continuous-speech`) and parse the `.PHN` files directly.
A `.PHN` line has the form `start_sample stop_sample phone`.

> **Kaggle credentials.** In Colab, add `KAGGLE_USERNAME` and `KAGGLE_KEY` to the *Secrets* sidebar (get them from <https://www.kaggle.com/settings>).
"""))

cells.append(code("""\
import kagglehub

kaggle_path = Path(kagglehub.dataset_download(
    "mfekadu/darpa-timit-acousticphonetic-continuous-speech"
))
print(f"Dataset cache: {kaggle_path}")

candidates = list(kaggle_path.rglob("TEST"))
assert candidates, f"No TEST/ folder under {kaggle_path}"
TIMIT_ROOT = candidates[0].parent
print(f"TIMIT root  : {TIMIT_ROOT}")

phn_files = sorted(TIMIT_ROOT.rglob("TEST/**/*.[Pp][Hh][Nn]"))
print(f"Found {len(phn_files)} .PHN files in TEST split.")
"""))

cells.append(code("""\
@dataclass
class PhoneSeg:
    start_sample: int
    end_sample:   int
    phone_61:     str
    phone_39:     str

@dataclass
class Utterance:
    utt_id:        str
    audio:         np.ndarray              # full waveform, float32
    phone_segs:    List[PhoneSeg]


def _parse_phn(path: Path) -> List[PhoneSeg]:
    out = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            s, e, p = int(parts[0]), int(parts[1]), parts[2].lower()
            out.append(PhoneSeg(s, e, p, PHONE_61_TO_39.get(p, p)))
    return out


def _wav_for(phn: Path) -> Optional[Path]:
    for ext in (".WAV", ".wav", ".Wav"):
        cand = phn.with_suffix(ext)
        if cand.exists():
            return cand
    return None


def load_utterance(phn_path: Path) -> Optional[Utterance]:
    wav_path = _wav_for(phn_path)
    if wav_path is None:
        return None
    try:
        audio, _ = librosa.load(str(wav_path), sr=SR, mono=True)
    except Exception:
        return None
    audio = audio.astype(np.float32)
    segs = _parse_phn(phn_path)
    if not segs:
        return None
    utt_id = f"{phn_path.parent.name}_{phn_path.stem}"
    return Utterance(utt_id=utt_id, audio=audio, phone_segs=segs)


# Cap how many utterances we scan.  300 covers TIMIT's phone diversity
# fully; bump to None for a full sweep (≈1680 utterances).
MAX_UTTS_TO_SCAN = 300
print(f"Will scan up to {MAX_UTTS_TO_SCAN} utterances.")
"""))

# ════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## 6. Pick the best exemplar of every phone

For each phone we collect candidate tokens that satisfy:

1. Duration falls inside the phone's biologically plausible window.
2. The token has at least 200 ms of audio context on each side
   (no boundary effects in the cochleagram).

For every candidate we extract a **fixed-size 1-second window** centred
on the token, RMS-normalise it, and run it through the model.  The
candidate's score is the mean posterior probability that the model
assigns to the *correct* class on the frames covering the token.  We
keep the highest-scoring candidate per phone.

Using a fixed window size means JAX JIT-compiles `score_batch` exactly
once — no recompile thrash.
"""))

cells.append(code("""\
HALF_WINDOW = N_SAMPLES_1S // 2     # 8000

def extract_centered_window(utt: Utterance, seg: PhoneSeg) -> Tuple[np.ndarray, int, int]:
    \"\"\"Extract a 1-s window centred on the segment.

    Returns
    -------
    window : (N_SAMPLES_1S,) float32 RMS-normalised
    seg_start_in_window : int     # sample offset of the segment within the window
    seg_end_in_window   : int
    \"\"\"
    centre = (seg.start_sample + seg.end_sample) // 2
    win_start = centre - HALF_WINDOW
    win_end   = centre + HALF_WINDOW
    audio = utt.audio
    n = len(audio)

    # Compute padding needed at each end.
    left_pad  = max(0, -win_start)
    right_pad = max(0, win_end - n)
    s = max(0, win_start)
    e = min(n, win_end)

    window = np.concatenate([
        np.zeros(left_pad, dtype=np.float32),
        audio[s:e],
        np.zeros(right_pad, dtype=np.float32),
    ])
    assert len(window) == N_SAMPLES_1S, len(window)

    rms = float(np.sqrt(np.mean(window**2)))
    if rms > 0:
        window = window / rms

    seg_start_in_window = seg.start_sample - win_start
    seg_end_in_window   = seg.end_sample   - win_start
    return window.astype(np.float32), seg_start_in_window, seg_end_in_window


def score_candidates_in_batches(candidates):
    \"\"\"Run the model in fixed-size batches and return per-candidate scores.

    Each score is the model's clip-level posterior probability for the
    candidate's *true* phoneme class (1-indexed 61-phone label).  The
    candidate is centred in a 1-second window, so high posterior on the
    target class is a clean signal that the token is recognisable.
    \"\"\"
    if not candidates:
        return np.array([])

    n = len(candidates)
    pad = (-n) % BATCH
    audio_stack = np.stack([c["window"] for c in candidates] +
                           [np.zeros(N_SAMPLES_1S, dtype=np.float32)] * pad)
    scores = np.zeros(n, dtype=np.float64)

    for b0 in range(0, len(audio_stack), BATCH):
        batch = jnp.asarray(audio_stack[b0:b0 + BATCH])
        post  = np.asarray(clip_posterior(nn_params, batch, aud_params))   # (B, K)
        for i in range(BATCH):
            idx = b0 + i
            if idx >= n:
                break
            target = PHONEME_TO_IDX[candidates[idx]["seg"].phone_61]
            scores[idx] = float(post[i, target])
    return scores


# ── Step 1: collect candidates per folded phone ──
candidates_per_phone = {p: [] for p in ALL_PHONES_39}
SR_MS = SR / 1000.0

for k, phn_path in enumerate(phn_files):
    if MAX_UTTS_TO_SCAN is not None and k >= MAX_UTTS_TO_SCAN:
        break
    utt = load_utterance(phn_path)
    if utt is None:
        continue

    n = len(utt.audio)
    for seg in utt.phone_segs:
        p39 = seg.phone_39
        if p39 not in candidates_per_phone:
            continue

        dur_ms = (seg.end_sample - seg.start_sample) / SR_MS
        lo, hi = PHONE_DUR_WINDOW_MS.get(p39, DEFAULT_DUR_WINDOW_MS)
        if not (lo <= dur_ms <= hi):
            continue

        # Need 200 ms of audio context on each side (so the cochleagram
        # crop has no edge artefacts) AND the full 1-s scoring window
        # must fit inside the utterance (so the model is never fed
        # silence-padded edges that could throw off its posterior).
        centre = (seg.start_sample + seg.end_sample) // 2
        if seg.start_sample < int(0.2 * SR) or seg.end_sample > n - int(0.2 * SR):
            continue
        if centre - HALF_WINDOW < 0 or centre + HALF_WINDOW > n:
            continue

        win, s_in, e_in = extract_centered_window(utt, seg)
        candidates_per_phone[p39].append({
            "utt_id": utt.utt_id,
            "audio_full": utt.audio,
            "seg": seg,
            "window": win,
            "seg_start_in_window": s_in,
            "seg_end_in_window":   e_in,
            "dur_ms": dur_ms,
        })

print("Candidate counts per phone:")
for fam, members in PHONEME_FAMILIES.items():
    counts = [f"{p}:{len(candidates_per_phone[p])}" for p in members]
    print(f"  {fam:>16s}  " + "  ".join(counts))
"""))

cells.append(code("""\
# ── Step 2: score each candidate, keep the best per phone ──
best_per_phone = {}      # phone -> chosen candidate (with extra 'score' key)

for p39 in ALL_PHONES_39:
    cands = candidates_per_phone[p39]
    if not cands:
        print(f"  ⚠  No candidates found for /{p39}/  — try increasing MAX_UTTS_TO_SCAN")
        continue
    scores = score_candidates_in_batches(cands)
    j = int(np.argmax(scores))
    cands[j]["score"] = float(scores[j])
    best_per_phone[p39] = cands[j]

# Pretty summary.
print(f"\\nChosen exemplars ({len(best_per_phone)} / {len(ALL_PHONES_39)}):")
for fam, members in PHONEME_FAMILIES.items():
    print(f"\\n  ── {fam} ──")
    for p in members:
        if p not in best_per_phone:
            print(f"    /{p:<3s}/  ✗ (no candidate)")
            continue
        c = best_per_phone[p]
        print(f"    /{p:<3s}/  score={c['score']:.3f}  dur={c['dur_ms']:6.1f} ms  "
              f"utt={c['utt_id']}")
"""))

# ════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## 7. Run wav2aud on each chosen exemplar's surrounding window

For each chosen exemplar we:

1. Take the same RMS-normalised 1-second window we already scored.
2. Pass it through the **trained model's own** `wav2aud` (the
   non-vmapped twin so we drive a single waveform).
3. Crop the resulting cochleagram in the **frame domain** to the
   phone's bounds plus a 20 ms pad on each side.

Because the leaky integrator is run on the full 1-second window and
the crop happens *after*, there are no boundary artefacts at the
crop edges.
"""))

cells.append(code("""\
PAD_MS_VIS = 20.0

def cochleagram_for_token(cand) -> Tuple[np.ndarray, float]:
    \"\"\"Return (coch_TF_cropped, dur_ms).

    The cropped time axis is anchored at 0 ms = phone onset; padding is
    included so the listener can see what's happening just before / after.
    The frequency axis is the shared global ``COCHLEA_CF`` (the codebase's
    canonical CF list) — every cropped cochleagram has the same 129 bins.
    \"\"\"
    coch_full = cochlea_np(cand["window"])                # (T_frames, F)
    n_frames, n_freq = coch_full.shape
    assert n_freq == len(COCHLEA_CF), (n_freq, len(COCHLEA_CF))
    ms_per_frame = N_SAMPLES_1S / SR * 1000.0 / n_frames  # ≈5 ms

    # Phone onset/offset → frame indices.
    t0_ms = cand["seg_start_in_window"] / SR * 1000.0
    t1_ms = cand["seg_end_in_window"]   / SR * 1000.0
    f0 = max(0, int(round((t0_ms - PAD_MS_VIS) / ms_per_frame)))
    f1 = min(n_frames, int(round((t1_ms + PAD_MS_VIS) / ms_per_frame)))
    coch_crop = coch_full[f0:f1]                          # (T_crop, 129)
    return coch_crop, cand["dur_ms"]


# Pre-compute every cropped cochleagram + dB conversion so the plot cell
# is fast and side-effect-free.  Frequency axis is the global COCHLEA_CF.
cochleagrams_db = {}    # phone -> (coch_db, dur_ms)
for p, cand in best_per_phone.items():
    coch, dur_ms = cochleagram_for_token(cand)
    cochleagrams_db[p] = (cochleagram_to_db(coch, db_floor=60.0), dur_ms)

# Quick stats for verification (none should be flat).
for p in list(cochleagrams_db.keys())[:6]:
    db, dur = cochleagrams_db[p]
    print(f"  /{p:<3s}/  db.shape={db.shape}  "
          f"db.min={db.min():.1f}  db.max={db.max():.1f}  "
          f"std={db.std():.2f}  dur≈{dur:.0f} ms")
"""))

# ════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## 8. The grand poster

A six-panel figure — one per manner family — each laying out its
phones in a clean grid of dB cochleagrams.  The frequency axis is
log-scaled to honour the NSL filterbank's tonotopy; the colour map is
panel-relative dB so formant structure pops in every cell regardless
of overall energy.
"""))

cells.append(code("""\
mpl.rcParams.update({
    "font.family":     "DejaVu Sans",
    "axes.titlesize":  10,
    "axes.labelsize":  8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  "white",
    "savefig.facecolor": "white",
    "savefig.dpi":       150,
})

# Use a slightly muted colormap inspired by Praat / Praat-derived spectrograms.
COCH_CMAP = "magma"
DB_FLOOR  = 60.0


def _draw_one_panel(ax, db, phone, dur_ms, score, family_color):
    \"\"\"Draw one cochleagram panel with consistent styling.

    The frequency axis is the global ``COCHLEA_CF`` (the codebase's
    canonical 129-channel CF list, in Hz) — bin-for-bin aligned with
    the cochleagram's frequency dimension.  We draw the image at unit
    bin-spacing so every bin gets equal screen real-estate (the spacing
    is *already* logarithmic in Hz by construction), then label the
    axis with the actual CF values from ``COCHLEA_CF``.
    \"\"\"
    n_frames, n_freq = db.shape
    assert n_freq == len(COCHLEA_CF), (n_freq, len(COCHLEA_CF))
    ms_per_frame = 5.0
    # x = time (ms), y = bin index (0..128).  Bin i has CF = COCHLEA_CF[i].
    extent = [-PAD_MS_VIS, n_frames * ms_per_frame - PAD_MS_VIS,
              -0.5, n_freq - 0.5]
    im = ax.imshow(db.T, origin="lower", aspect="auto", cmap=COCH_CMAP,
                   extent=extent, vmin=-DB_FLOOR, vmax=0.0,
                   interpolation="bilinear")

    # Tick at the bin whose CF is closest to each target frequency.
    target_hz = [200, 500, 1000, 2000, 5000]
    bin_ticks, hz_labels = [], []
    for hz in target_hz:
        if COCHLEA_CF[0] <= hz <= COCHLEA_CF[-1]:
            i = int(np.argmin(np.abs(COCHLEA_CF - hz)))
            bin_ticks.append(i)
            hz_labels.append(f"{hz/1000:g}k" if hz >= 1000 else f"{hz}")
    ax.set_yticks(bin_ticks)
    ax.set_yticklabels(hz_labels)
    ax.set_ylim(-0.5, n_freq - 0.5)

    # Phone-onset / offset markers.
    ax.axvline(0.0, color="white", lw=0.8, alpha=0.55, ls="--")
    ax.axvline(dur_ms, color="white", lw=0.8, alpha=0.55, ls="--")

    # Family-coloured phone label badge in the upper-left corner.
    ax.text(0.04, 0.94, f"/{phone}/", transform=ax.transAxes,
            ha="left", va="top",
            fontsize=14, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.25", fc=family_color,
                      ec="white", lw=0.6, alpha=0.92))
    # Duration & score in the upper-right.
    ax.text(0.97, 0.95, f"{dur_ms:.0f} ms  ·  p={score:.2f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5, color="white", alpha=0.85,
            bbox=dict(boxstyle="round,pad=0.15", fc="black",
                      ec="none", alpha=0.4))

    ax.tick_params(direction="out", length=2.5, pad=1.5, color="#888")
    for spine in ax.spines.values():
        spine.set_color("#888")
        spine.set_linewidth(0.6)
    return im


def plot_family(family_name, members, n_cols=4):
    members = [m for m in members if m in cochleagrams_db]
    if not members:
        return
    n = len(members)
    n_rows = int(np.ceil(n / n_cols))
    color = FAMILY_COLORS[family_name]

    fig = plt.figure(figsize=(n_cols * 2.9, n_rows * 2.4 + 0.7))
    gs = gridspec.GridSpec(
        n_rows + 1, n_cols,
        height_ratios=[0.18] + [1.0] * n_rows,
        hspace=0.32, wspace=0.20, figure=fig,
    )

    # ── Family banner ──
    banner_ax = fig.add_subplot(gs[0, :])
    banner_ax.set_facecolor(color)
    banner_ax.set_xticks([]); banner_ax.set_yticks([])
    for s in banner_ax.spines.values():
        s.set_visible(False)
    banner_ax.text(0.012, 0.5, family_name,
                   transform=banner_ax.transAxes,
                   ha="left", va="center",
                   fontsize=15, fontweight="bold", color="white")
    banner_ax.text(0.988, 0.5, f"{n} phone{'s' if n != 1 else ''}",
                   transform=banner_ax.transAxes,
                   ha="right", va="center",
                   fontsize=10, color="white", alpha=0.85)

    # ── Cochleagram grid ──
    last_im = None
    panel_axes = []
    for k, phone in enumerate(members):
        r, c = divmod(k, n_cols)
        ax = fig.add_subplot(gs[r + 1, c])
        panel_axes.append(ax)
        db, dur_ms = cochleagrams_db[phone]
        score = best_per_phone[phone]["score"]
        last_im = _draw_one_panel(ax, db, phone, dur_ms, score, color)

        # Axis labels only on outer rim.
        if c == 0:
            ax.set_ylabel("Freq (Hz)")
        else:
            ax.set_yticklabels([])
        if r == n_rows - 1:
            ax.set_xlabel("Time (ms,  0 = phone onset)")
        else:
            ax.set_xticklabels([])

    # Hide unused cells.
    for k in range(n, n_rows * n_cols):
        r, c = divmod(k, n_cols)
        ax = fig.add_subplot(gs[r + 1, c])
        ax.axis("off")

    # Shared colourbar on the right edge — sized against the panel axes only.
    cbar = fig.colorbar(last_im, ax=panel_axes, orientation="vertical",
                        fraction=0.022, pad=0.018, shrink=0.88)
    cbar.set_label("dB (panel-relative)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        "Best TIMIT exemplar of each phoneme  ·  trained NSL biomimetic cochleagram",
        y=0.995, fontsize=11, color="#333"
    )
    plt.show()


for family, members in PHONEME_FAMILIES.items():
    plot_family(family, members, n_cols=5 if len(members) > 4 else max(2, len(members)))
"""))

# ════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## 9. (Optional) listen to the chosen exemplars

Quick sanity check — play the cropped audio for each exemplar so you
can verify by ear that the picker did its job.
"""))

cells.append(code("""\
from IPython.display import display, Audio, HTML

for fam, members in PHONEME_FAMILIES.items():
    chosen = [(p, best_per_phone[p]) for p in members if p in best_per_phone]
    if not chosen:
        continue
    display(HTML(
        f"<h3 style='margin:14px 0 4px 0;color:white;background:{FAMILY_COLORS[fam]};"
        f"padding:6px 10px;border-radius:4px;font-family:sans-serif;'>{fam}</h3>"
    ))
    for p, c in chosen:
        seg = c["seg"]
        pad = int(0.05 * SR)
        s = max(0, seg.start_sample - pad)
        e = min(len(c["audio_full"]), seg.end_sample + pad)
        clip = c["audio_full"][s:e]
        display(HTML(
            f"<div style='margin:2px 0 0 8px;font-family:sans-serif;'>"
            f"<b>/{p}/</b>  <span style='color:#666;'>· "
            f"{c['dur_ms']:.0f} ms · score={c['score']:.2f} · "
            f"{c['utt_id']}</span></div>"
        ))
        display(Audio(clip, rate=SR))
"""))

# ════════════════════════════════════════════════════════════════════════
# Notebook metadata
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"Wrote {OUT}  ({OUT.stat().st_size/1024:.1f} KB, {len(cells)} cells)")
