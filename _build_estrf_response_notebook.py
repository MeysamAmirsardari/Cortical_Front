#!/usr/bin/env python3
"""Build estrf_responses_along_utterances.ipynb — a self-contained Colab
notebook that:

  1. Loads the trained STRF phoneme-recognition model.
  2. Reconstructs the per-phoneme **effective STRF (eSTRF)** kernels
     from precomputed CorticalLIME signed aggregates (inlined as a
     base64 blob — no external download).
  3. Picks a few TIMIT TEST utterances rich in each target phoneme.
  4. For each (utterance, phoneme-eSTRF) pair, applies the eSTRF to the
     **full-utterance cochleagram** (model's own wav2aud) → 1-D
     "response of the brain to that template" curve over time.
  5. Renders one beautiful, time-aligned figure per example with:
       * the eSTRF kernel itself (top inset)
       * the auditory spectrogram (model cochleagram)
       * the TIMIT phoneme strip (colour-coded by manner family)
       * the eSTRF response curve r(t)
     all sharing the SAME time axis.

Run locally to (re)build the .ipynb JSON.
"""

import json
import uuid
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent.parent.parent / "estrf_responses_along_utterances.ipynb"

# Inline precomputed eSTRF coefficient aggregates so the notebook is fully
# self-contained on Colab (no need to upload results_raw.npz).
_B64_PATH = Path(__file__).resolve().parent / "_estrf_aggregates_b64.txt"
_B64_TEXT = _B64_PATH.read_text().strip()


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
# Effective STRFs Riding Along TIMIT — One Filter, One Sentence, One Time-Series

**Paper:** [A Biomimetic Frontend for Differentiable Audio Processing](https://arxiv.org/abs/2409.08997)
**Repo:**  [MeysamAmirsardari/Cortical_Front](https://github.com/MeysamAmirsardari/Cortical_Front)

This notebook does something delightful: it takes the **effective STRF
(eSTRF)** of a phoneme — the modulation-energy receptive field that
CorticalLIME found explains the trained model's confidence in that
phoneme — and slides it across whole TIMIT utterances to produce a
single-channel "**this template's response over time**" curve.

For each example you'll see, in one tightly time-aligned figure:

1. **The eSTRF kernel itself** (inset, top-left) — a 2-D template in
   (time, log-frequency), red = excitatory, blue = inhibitory.  Built
   from the model's own STRF parameters `(scale Ω, rate ω)` weighted by
   the signed mean LIME coefficients per linguistic band.
2. **The auditory spectrogram** of the utterance — the model's *own*
   cochleagram (`supervisedSTRF.wav2aud`, NOT a generic mel/STFT), so
   the picture's frequency axis matches the eSTRF's domain bin-for-bin.
3. **The TIMIT phoneme strip** — coloured boxes per manner family,
   placed at the exact time bounds from the `.PHN` alignment.
4. **The eSTRF response** $r(t) = \\sum_{\\tau, f} |\\text{coch}|(t-\\tau, f)\\,\\text{eSTRF}(\\tau, f)$
   — a 1-D power-in-time curve, filled in the family colour, with
   peak frames marked.

Everything is shown for a representative phoneme of every manner family,
two examples each, so you can watch the eSTRF for /s/ light up on /s/,
the eSTRF for /aa/ light up on /aa/, and so on — *the model's
linguistic intuition rendered as a trace through speech*.

> **Runtime:** select **GPU** (Runtime → Change runtime type → T4 GPU).
> Whole notebook runs in ≈4 min on a T4, ≈8 min on CPU.
"""))

# ════════════════════════════════════════════════════════════════════════
cells.append(md("## 1. Environment setup"))

cells.append(code("""\
%%capture
# JAX with GPU support, Flax, audio deps. Colab already has torch,
# numpy, matplotlib pre-installed.
!pip install --upgrade "jax[cuda12]" flax optax
!pip install librosa soundfile kagglehub
"""))

cells.append(code("""\
import os

REPO_URL = "https://github.com/MeysamAmirsardari/Cortical_Front.git"
REPO_DIR = "/content/Cortical_Front"
CODE_DIR = os.path.join(REPO_DIR, "r_code")

if not os.path.exists(REPO_DIR):
    !git clone {REPO_URL} {REPO_DIR}
    print(f"Cloned repository to {REPO_DIR}")
else:
    print(f"Repository already exists at {REPO_DIR}")

# The model code expects r_code/ as the CWD.
os.chdir(CODE_DIR)
print(f"Working directory: {os.getcwd()}")
"""))

cells.append(code("""\
import sys
import re
import io
import base64
import pickle
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec, patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
from scipy.signal import correlate2d
from scipy.ndimage import gaussian_filter

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit

import librosa

if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from supervisedSTRF import supervisedSTRF

warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=FutureWarning)

print(f"JAX  version : {jax.__version__}")
print(f"JAX  backend : {jax.default_backend()}")
print(f"JAX  devices : {jax.devices()}")
print(f"NumPy version: {np.__version__}")
"""))

# ════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## 2. Load the trained checkpoint and build the cochleagram callable

The shipped checkpoint stores `[nn_params, aud_params]`.  We rebuild a
non-vmapped `supervisedSTRF` instance and call its `wav2aud` method —
the **model's own** biomimetic cochlear filterbank, lateral inhibition,
compression, and leaky integrator.  No external STFT, no librosa
mel-spec.
"""))

cells.append(code("""\
CHECKPOINT_PATH = "nishit_trained_models/main_jax_phoneme_rec_timit/chkStep_200000.p"

ckpt_path = Path(CHECKPOINT_PATH)
assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path.resolve()}"

with open(ckpt_path, "rb") as f:
    nn_params, aud_params = pickle.load(f)

ALPHA = float(aud_params["alpha"])
COMP  = aud_params["compression_params"]
SR_PAIRS_MODEL = np.asarray(aud_params["sr"], dtype=np.float64)   # (n_strfs, 2)
SR    = 16_000

# Build the non-vmapped twin.
model_s = supervisedSTRF(
    n_phones=62,
    input_type="audio",
    encoder_type="strf",
    decoder_type="cnn",
    compression_method="power",
    update_lin=True,
    use_class=False,
    conv_feats=[10, 20, 40],
    pooling_stride=2,
)


@jit
def _cochlea_jax(x):
    return model_s.apply(nn_params, x, COMP, ALPHA, method=model_s.wav2aud)


def cochlea_np(wav: np.ndarray) -> np.ndarray:
    \"\"\"Return the model's cochleagram, shape (T_frames, 129) float64.\"\"\"
    return np.asarray(_cochlea_jax(jnp.asarray(wav, dtype=jnp.float32)))


# Smoke test.
_t = cochlea_np(np.zeros(int(1.5 * SR), dtype=np.float32))
print(f"cochleagram smoke test: shape={_t.shape}  dtype={_t.dtype}")
print(f"α (leaky integrator)  = {ALPHA:.6f}")
print(f"STRF bank             = {SR_PAIRS_MODEL.shape[0]} channels   "
      f"(scale Ω, rate ω)")
"""))

# ════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## 3. Canonical centre frequencies & dB-display helper

The 129 cochlear-channel centre frequencies come **straight from the
codebase** (`strfpy.py:64-66`):

```python
cf = np.linspace(-31, 97, 129) / 24       # 129 channels, k = -31..+97
cf = [440 * 2**f for f in cf]              # NSL log-spacing
cf = [round(f/10)*10 for f in cf]          # rounded to 10 Hz
```

Spans **180 Hz – 7250 Hz**, 24 channels per octave, bin-for-bin
aligned with `wav2aud`'s output.

The trained checkpoint has `α = 1.0076 > 1`, which inverts the leaky
integrator's sign convention — the cochleagram output is *signed* and
peak energy lives at the **most-negative** values.  For both the dB
display **and** the eSTRF response we use the magnitude `|coch|`.
"""))

cells.append(code("""\
def cochlea_center_freqs() -> np.ndarray:
    \"\"\"Codebase's canonical 129-channel CF list (Hz).  Verbatim port of
    strfpy.py:64-66.  Bin i ↔ COCHLEA_CF[i].\"\"\"
    cf = np.linspace(-31, 97, 31 + 97 + 1) / 24.0
    cf = np.array([440.0 * 2.0 ** f for f in cf])
    cf = np.array([round(float(f) / 10.0) * 10.0 for f in cf])
    return cf


COCHLEA_CF = cochlea_center_freqs()
N_FREQ = len(COCHLEA_CF)
assert N_FREQ == 129
print(f"COCHLEA_CF: {N_FREQ} channels   ({COCHLEA_CF[0]:.0f} Hz – {COCHLEA_CF[-1]:.0f} Hz)")


def cochleagram_to_db(coch_TF: np.ndarray, db_floor: float = 60.0) -> np.ndarray:
    \"\"\"Magnitude-based aud2dB normalisation, clipped to [-db_floor, 0].\"\"\"
    mag = np.abs(np.asarray(coch_TF, dtype=np.float64))
    peak = float(mag.max())
    if peak <= 0.0:
        return np.full(mag.shape, -db_floor, dtype=np.float32)
    db = 20.0 * np.log10(mag / peak + 1e-12)
    return np.clip(db, -db_floor, 0.0).astype(np.float32)
"""))

# ════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## 4. Decode the inlined per-phoneme LIME aggregates

A small (~17 KB) base64 blob below contains the **signed mean LIME
coefficients per phoneme** computed from `results_raw.npz` on a
high-resolution CorticalLIME run (band-mode, 5 linguistic bands × 40
STRF channels = 200 coefficients per phoneme).

We decode it into a `dict[phone] -> (200,)` array, plus the model's
`sr_pairs` and the linguistic `band_edges_hz`.  Inlining means the
notebook is fully self-contained — no second download, no Drive mount.
"""))

cells.append(code("""\
ESTRF_AGG_B64 = \"\"\"
""" + _B64_TEXT + """
\"\"\"

_blob = base64.b64decode("".join(ESTRF_AGG_B64.split()))
with np.load(io.BytesIO(_blob), allow_pickle=False) as _d:
    AGG_PHONES     = [str(p) for p in np.asarray(_d["phones"])]
    AGG_COEFS      = np.asarray(_d["coefs"], dtype=np.float64)        # (P, 200)
    AGG_SR_PAIRS   = np.asarray(_d["sr_pairs"], dtype=np.float64)     # (40, 2)
    AGG_BAND_EDGES = np.asarray(_d["band_edges_hz"], dtype=np.float64)# (5, 2)
    N_BANDS        = int(_d["n_bands"])

# Sanity: the LIME aggregates' STRF channels should match the model's.
assert AGG_SR_PAIRS.shape == SR_PAIRS_MODEL.shape, (
    AGG_SR_PAIRS.shape, SR_PAIRS_MODEL.shape
)
print(f"Decoded {len(AGG_PHONES)} per-phoneme LIME aggregates "
      f"({AGG_COEFS.shape[1]} coefs each = {N_BANDS} bands × "
      f"{AGG_SR_PAIRS.shape[0]} STRFs).")
print("Available phones:", " ".join(f"/{p}/" for p in AGG_PHONES))
print("Linguistic bands:")
for i, (lo, hi) in enumerate(AGG_BAND_EDGES):
    print(f"  band {i}:  {lo:6.0f} – {hi:6.0f} Hz")
"""))

# ════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## 5. Reconstruct each phoneme's eSTRF — directly on the cochleagram grid

For each `(band b, STRF channel k)` we stamp a 2-D Gabor *envelope*
kernel at the band's geometric-mean frequency.  The envelope widths
are channel-adaptive — narrow for high-rate / high-scale STRFs,
broad for low-rate / low-scale STRFs (the constant-Q property real
cortical wavelets have).  The signed LIME weight `w_{b,k}` modulates
the kernel:

$$\\text{eSTRF}(t, f) = \\sum_{b,k} w_{b,k} \\cdot \\exp\\!\\left(-\\tfrac{t^2}{2\\sigma_t(\\omega_k)^2}\\right) \\cdot \\exp\\!\\left(-\\tfrac{(f - f_b)^2}{2\\sigma_f(\\Omega_k)^2}\\right)$$

Crucially we build the eSTRF **directly on the cochleagram's grid** —
60 time-frames at 5 ms hop (300 ms wide), 129 frequency bins at the
COCHLEA_CF positions.  This means the eSTRF is bin-aligned with the
cochleagram and we can apply it via plain 2-D correlation.
"""))

cells.append(code("""\
BASE_HZ        = 125.0          # reference for octave conversions
KERNEL_T_S     = 0.30           # total kernel width in time (s)
KERNEL_N_T     = 60             # 60 frames × 5 ms = 300 ms
SMOOTH_SIGMA   = 0.8            # light Gaussian post-smoothing


def _gabor_envelope(omega: float, Omega: float,
                    t_s: np.ndarray, f_oct: np.ndarray,
                    cycles: float = 0.5,
                    omega_floor: float = 2.0,
                    Omega_floor: float = 0.5) -> np.ndarray:
    \"\"\"2-D Gaussian envelope of a Gabor wavelet — phase-invariant template
    for a magnitude / power read-out cortex, with channel-adaptive widths.\"\"\"
    sw = max(abs(float(omega)), omega_floor)
    sW = max(abs(float(Omega)), Omega_floor)
    sigma_t = cycles / sw
    sigma_f = cycles / sW
    return np.outer(np.exp(-(t_s / sigma_t) ** 2 / 2),
                    np.exp(-(f_oct / sigma_f) ** 2 / 2))


def reconstruct_estrf_on_cochleagram(coef_vec: np.ndarray) -> np.ndarray:
    \"\"\"Synthesise the modulation-energy eSTRF for one phoneme on the
    cochleagram's exact grid (60 frames × 129 freq bins).\"\"\"
    n_strfs = AGG_SR_PAIRS.shape[0]
    coef_2d = coef_vec.reshape(N_BANDS, n_strfs)
    band_centres_oct = np.log2(
        np.sqrt(np.maximum(AGG_BAND_EDGES[:, 0], 1.0) * AGG_BAND_EDGES[:, 1])
        / BASE_HZ
    )
    t_kern = np.linspace(-KERNEL_T_S / 2.0, KERNEL_T_S / 2.0, KERNEL_N_T)
    f_oct  = np.log2(COCHLEA_CF / BASE_HZ)

    estrf = np.zeros((KERNEL_N_T, N_FREQ), dtype=np.float64)
    for b in range(N_BANDS):
        for k in range(n_strfs):
            w = coef_2d[b, k]
            if w == 0.0:
                continue
            estrf += w * _gabor_envelope(
                omega=AGG_SR_PAIRS[k, 1],
                Omega=AGG_SR_PAIRS[k, 0],
                t_s=t_kern,
                f_oct=f_oct - band_centres_oct[b],
            )
    if SMOOTH_SIGMA > 0:
        estrf = gaussian_filter(estrf, sigma=SMOOTH_SIGMA)
    return estrf


# Pre-compute every available phoneme's eSTRF.
ESTRF: Dict[str, np.ndarray] = {
    p: reconstruct_estrf_on_cochleagram(AGG_COEFS[i])
    for i, p in enumerate(AGG_PHONES)
}

# Sanity print.
for p in ("aa", "iy", "s", "n", "r"):
    if p in ESTRF:
        e = ESTRF[p]
        print(f"  /{p:<3s}/   shape={e.shape}   "
              f"range=[{e.min():+.4f}, {e.max():+.4f}]   "
              f"||e||₂={np.linalg.norm(e):.3f}")
"""))

# ════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## 6. The eSTRF response: convolve the kernel against $|\\text{coch}|$

Given an eSTRF kernel $K(\\tau, f)$ (60 × 129) and the cochleagram
magnitude $|C|(t, f)$ (T × 129), the response is

$$r(t) = \\sum_{\\tau=0}^{n_t - 1} \\sum_{f=0}^{F-1} |C|(t + \\tau - n_t/2,\\, f) \\cdot \\tilde K(\\tau, f), \\qquad \\tilde K = (K - \\bar K) \\,\\big/\\, \\|K - \\bar K\\|_2$$

i.e. a 2-D cross-correlation that **sums over the entire frequency
axis** (the kernel covers all 129 bins) and slides only along time.

**Why the $K - \\bar K$?**  Most LIME coefficients are positive (a STRF
channel firing makes the model *more* confident in the target phoneme),
so a raw $K$ has a positive mean.  A raw correlation $r(t) = \\sum |C|
\\cdot K$ is then dominated by $\\bar K \\cdot \\text{total\\_energy}(t)$
— the response becomes "wherever speech is loud" rather than "wherever
the kernel pattern is present."  Subtracting the kernel mean turns the
filter into a true **template-matching** detector: positive $r(t)$ now
means "this region's spectro-temporal *shape* matches the eSTRF",
independent of overall loudness.  The $\\|\\cdot\\|_2$ normalisation
makes responses across different probe phonemes directly comparable.
"""))

cells.append(code("""\
def _normalised_template(estrf_kern: np.ndarray) -> np.ndarray:
    \"\"\"Mean-subtract and unit-norm the eSTRF so the response is a
    template-match score, not an energy envelope.\"\"\"
    K = np.asarray(estrf_kern, dtype=np.float64)
    K = K - K.mean()
    n = float(np.linalg.norm(K))
    return K / n if n > 0 else K


def apply_estrf(coch_TF_signed: np.ndarray, estrf_kern: np.ndarray) -> np.ndarray:
    \"\"\"Slide the *normalised* eSTRF template over |cochleagram| → 1-D
    response of length T (same as the cochleagram's time axis).\"\"\"
    K_tilde = _normalised_template(estrf_kern)
    n_t, n_f = K_tilde.shape
    assert coch_TF_signed.shape[1] == n_f, (coch_TF_signed.shape, K_tilde.shape)
    coch_pos = np.abs(coch_TF_signed)
    pad = n_t // 2
    coch_padded = np.pad(coch_pos, ((pad, n_t - 1 - pad), (0, 0)), mode="constant")
    out = correlate2d(coch_padded, K_tilde, mode="valid")
    assert out.shape[0] == coch_TF_signed.shape[0], (out.shape, coch_TF_signed.shape)
    return out[:, 0]
"""))

# ════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## 7. Phoneme inventory, families & a tasteful colour palette
"""))

cells.append(code("""\
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

PHONEME_FAMILIES = {
    "Vowels":         ["iy","ih","eh","ae","aa","ah","ao","uh","uw","er",
                       "ey","ay","oy","aw","ow"],
    "Stops":          ["p","t","k","b","d","g"],
    "Affricates":     ["ch","jh"],
    "Fricatives":     ["f","th","s","sh","v","dh","z","hh"],
    "Nasals":         ["m","n","ng"],
    "Liquids/Glides": ["l","r","w","y"],
}

FAMILY_COLORS = {
    "Vowels":         "#E63946",
    "Stops":          "#F4A261",
    "Affricates":     "#E9C46A",
    "Fricatives":     "#2A9D8F",
    "Nasals":         "#264653",
    "Liquids/Glides": "#8E7DBE",
    "Silence":        "#888888",
}

def family_of(p39: str) -> Optional[str]:
    for fam, members in PHONEME_FAMILIES.items():
        if p39 in members:
            return fam
    if p39 in ("sil", "", "dx"):
        return "Silence"
    return None

# Which phoneme to use as the eSTRF "probe" for each family.
# These are the most discriminative tokens we have LIME aggregates for.
FAMILY_PROBE = {
    "Vowels":         ["iy", "aa"],          # high-front vs low-back
    "Stops":          ["t", "k"],
    "Fricatives":     ["s", "sh", "f"],
    "Nasals":         ["n", "m"],
    "Liquids/Glides": ["r", "l", "w"],
}
# (Affricates: no LIME aggregates available for /ch/ or /jh/ in this run.)

print("Probe phonemes per family:")
for fam, probes in FAMILY_PROBE.items():
    avail = [p for p in probes if p in ESTRF]
    print(f"  {fam:>16s}  →  {' '.join(f'/{p}/' for p in avail)}")
"""))

# ════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## 8. Download TIMIT and pick utterances rich in each probe phoneme

For each probe phoneme, we want utterances that contain **multiple
clean instances** of it (so the eSTRF response curve has obvious peaks
to compare against the phoneme strip), plus enough other content to
form a meaningful "trace through speech".
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
    audio:         np.ndarray
    phone_segs:    List[PhoneSeg]
    transcript:    str = ""


def _wav_for(phn: Path) -> Optional[Path]:
    for ext in (".WAV", ".wav", ".Wav"):
        cand = phn.with_suffix(ext)
        if cand.exists():
            return cand
    return None


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


def _parse_txt(phn_path: Path) -> str:
    txt_path = phn_path.with_suffix(".TXT")
    if not txt_path.exists():
        txt_path = phn_path.with_suffix(".txt")
    if not txt_path.exists():
        return ""
    with open(txt_path) as f:
        line = f.readline().strip()
    parts = line.split(None, 2)
    return parts[2] if len(parts) >= 3 else ""


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
    return Utterance(utt_id=utt_id, audio=audio,
                     phone_segs=segs, transcript=_parse_txt(phn_path))


# We only need a small handful of utterances per probe — scanning ~150
# is plenty.
MAX_UTTS_TO_SCAN = 200
all_utts: List[Utterance] = []
for k, phn in enumerate(phn_files[:MAX_UTTS_TO_SCAN]):
    u = load_utterance(phn)
    if u is not None:
        all_utts.append(u)
print(f"Loaded {len(all_utts)} utterances.")


def pick_utterances_for_probe(probe_phone: str, n_picks: int = 2,
                              min_instances: int = 3,
                              max_dur_s: float = 4.5) -> List[Utterance]:
    \"\"\"Pick utterances containing many clean instances of probe_phone.\"\"\"
    scored = []
    for u in all_utts:
        if len(u.audio) > int(max_dur_s * SR):
            continue
        # Count clean instances (≥30 ms duration so we don't count blips).
        n_inst = sum(1 for ps in u.phone_segs
                     if ps.phone_39 == probe_phone
                     and (ps.end_sample - ps.start_sample) >= int(0.03 * SR))
        if n_inst < min_instances:
            continue
        # Variety bonus: we like utterances with ≥6 distinct phones too.
        n_distinct = len({ps.phone_39 for ps in u.phone_segs})
        scored.append((n_inst * 10 + n_distinct, u))
    scored.sort(key=lambda x: -x[0])
    return [u for _, u in scored[:n_picks]]


# Quick sanity print.
for fam, probes in FAMILY_PROBE.items():
    for probe in probes:
        if probe not in ESTRF:
            continue
        chosen = pick_utterances_for_probe(probe, n_picks=2)
        ids = ", ".join(u.utt_id for u in chosen) or "(none)"
        print(f"  /{probe}/  → {ids}")
"""))

# ════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## 9. The plot — eSTRF kernel + cochleagram + phoneme strip + response

A four-row layout, all sharing the **same time axis** so you can read
the eSTRF response curve directly against the phonemes that triggered
each peak.

* **Row 0 (top-left inset):** the eSTRF kernel itself, on its own
  ±150 ms time axis and a log-Hz y-axis.  Diverging colour map
  (RdBu_r), zero-centred — red excitatory, blue inhibitory.  A small
  text panel on the right gives the phoneme, the family, and the
  utterance ID.
* **Row 1:** the model's auditory spectrogram (`wav2aud`) in dB,
  panel-relative magma.
* **Row 2:** the TIMIT phoneme strip — coloured boxes per family with
  white phone labels.
* **Row 3:** the eSTRF response $r(t)$, filled in the family colour,
  zero-line drawn, peak frames marked with thin verticals.
"""))

cells.append(code("""\
mpl.rcParams.update({
    "font.family":     "DejaVu Sans",
    "axes.titlesize":  10,
    "axes.labelsize":  9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  "white",
    "savefig.facecolor": "white",
    "savefig.dpi":       150,
})


def _fmt_hz(hz: float) -> str:
    return f"{hz/1000:g}k" if hz >= 1000 else f"{int(hz)}"


def _draw_phone_strip(ax, segs: List[PhoneSeg], total_ms: float):
    \"\"\"TIMIT phonemes as colour-coded boxes with white labels.\"\"\"
    ax.set_xlim(0, total_ms)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#bbb")
    ax.tick_params(axis="x", colors="#666", length=2.5)

    for ps in segs:
        t0 = ps.start_sample / SR * 1000.0
        t1 = ps.end_sample   / SR * 1000.0
        fam = family_of(ps.phone_39)
        color = FAMILY_COLORS.get(fam, "#bbb")
        # Slight rounding for elegance.
        ax.add_patch(FancyBboxPatch(
            (t0, 0.05), max(t1 - t0, 0.5), 0.90,
            boxstyle="round,pad=0.0,rounding_size=1.2",
            linewidth=0.0, facecolor=color, alpha=0.92,
            transform=ax.transData, clip_on=True,
        ))
        if (t1 - t0) >= 25:    # only label boxes that have room
            ax.text((t0 + t1) / 2, 0.5, ps.phone_39,
                    ha="center", va="center",
                    fontsize=7.5, color="white", fontweight="bold",
                    transform=ax.transData, clip_on=True)


def _draw_estrf_kernel(ax, kernel: np.ndarray, vmax: float):
    \"\"\"Draw the eSTRF kernel inset with log-Hz y-axis and ms x-axis.\"\"\"
    n_t, n_f = kernel.shape
    t_ms_extent = [-KERNEL_T_S / 2 * 1000, KERNEL_T_S / 2 * 1000]
    extent = [t_ms_extent[0], t_ms_extent[1], -0.5, n_f - 0.5]
    im = ax.imshow(
        kernel.T, origin="lower", aspect="auto", cmap="RdBu_r",
        extent=extent, vmin=-vmax, vmax=vmax,
        interpolation="bilinear",
    )
    ax.axvline(0, color="black", lw=0.7, ls=":", alpha=0.7)
    ax.set_xlabel("Lag (ms)", fontsize=8)
    ax.set_ylabel("Hz", fontsize=8)

    # CF-aware y-ticks.
    target_hz = [200, 500, 1000, 2000, 5000]
    tk_bins, tk_lab = [], []
    for hz in target_hz:
        if COCHLEA_CF[0] <= hz <= COCHLEA_CF[-1]:
            i = int(np.argmin(np.abs(COCHLEA_CF - hz)))
            tk_bins.append(i); tk_lab.append(_fmt_hz(hz))
    ax.set_yticks(tk_bins)
    ax.set_yticklabels(tk_lab, fontsize=7)
    ax.tick_params(axis="x", labelsize=7, length=2.5, colors="#444")
    ax.tick_params(axis="y", colors="#444")
    for spine in ax.spines.values():
        spine.set_color("#888"); spine.set_linewidth(0.6)
    return im


def plot_estrf_along_utterance(utt: Utterance, probe_phone: str,
                               family: str, save_path: Optional[str] = None):
    \"\"\"Render one publication-grade figure for (utterance, probe-eSTRF).\"\"\"
    color = FAMILY_COLORS[family]
    estrf = ESTRF[probe_phone]

    # Cochleagram of the FULL utterance.
    coch = cochlea_np(utt.audio.astype(np.float32))           # (T_frames, 129)
    n_frames = coch.shape[0]
    ms_per_frame = len(utt.audio) / SR * 1000.0 / n_frames    # ≈5 ms
    total_ms = n_frames * ms_per_frame
    coch_db = cochleagram_to_db(coch, db_floor=60.0)

    # Apply the (mean-subtracted, unit-norm) eSTRF.
    response = apply_estrf(coch, estrf)                       # (T_frames,)

    # Local maxima — peaks above the 80 th percentile, dominant in a ±70 ms window.
    win_frames = int(round(70.0 / ms_per_frame))
    thresh = np.quantile(response, 0.80)
    peaks = []
    for i in range(n_frames):
        lo = max(0, i - win_frames); hi = min(n_frames, i + win_frames + 1)
        if response[i] >= thresh and response[i] == response[lo:hi].max():
            peaks.append(i)

    # Frames covered by the probe phoneme (for ground-truth markers under r(t)).
    probe_frames = []
    for ps in utt.phone_segs:
        if ps.phone_39 != probe_phone:
            continue
        f0 = int(round(ps.start_sample / SR * 1000.0 / ms_per_frame))
        f1 = int(round(ps.end_sample   / SR * 1000.0 / ms_per_frame))
        probe_frames.append((f0, f1))

    # ── Layout: a comfortably proportioned 4-row grid with a generous top
    # row for the kernel inset + caption and proper margins so nothing clips.
    fig = plt.figure(figsize=(14.0, 8.6))
    outer = gridspec.GridSpec(
        4, 1, height_ratios=[1.45, 1.55, 0.34, 1.20],
        hspace=0.30, figure=fig,
        left=0.085, right=0.975, top=0.905, bottom=0.075,
    )

    # ─── Row 0: eSTRF kernel inset + caption ───
    top = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[0],
        width_ratios=[0.46, 0.54], wspace=0.20,
    )
    ax_kern = fig.add_subplot(top[0])
    vmax_k = float(np.max(np.abs(estrf))) + 1e-12
    im_k = _draw_estrf_kernel(ax_kern, estrf, vmax_k)
    cb = fig.colorbar(im_k, ax=ax_kern, fraction=0.045, pad=0.02)
    cb.set_label("eSTRF weight (signed)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax_kern.set_title(
        f"Effective STRF for /{probe_phone}/   ·   {family}",
        fontsize=11, fontweight="bold", color=color, pad=8,
    )

    # Caption panel (reads like a small poster legend).
    ax_cap = fig.add_subplot(top[1])
    ax_cap.axis("off")
    n_inst = sum(1 for ps in utt.phone_segs if ps.phone_39 == probe_phone)

    # Family-coloured header bar.
    ax_cap.add_patch(plt.Rectangle(
        (0.00, 0.86), 1.0, 0.13, transform=ax_cap.transAxes,
        facecolor=color, edgecolor="none", clip_on=False,
    ))
    ax_cap.text(0.018, 0.925, family.upper(),
                ha="left", va="center", color="white",
                fontsize=11, fontweight="bold",
                transform=ax_cap.transAxes)
    ax_cap.text(0.985, 0.925, f"probe   /{probe_phone}/",
                ha="right", va="center", color="white",
                fontsize=11, fontweight="bold",
                transform=ax_cap.transAxes)

    # Two-column metadata block.
    meta_y = 0.74
    line_h = 0.115
    rows = [
        ("Utterance",              utt.utt_id),
        ("Duration",               f"{len(utt.audio)/SR*1000:.0f} ms   ·   {n_frames} frames"),
        (f"Instances of /{probe_phone}/", str(n_inst)),
        ("Response peaks (≥ p80)", str(len(peaks))),
    ]
    for k, (lbl, val) in enumerate(rows):
        y = meta_y - k * line_h
        ax_cap.text(0.020, y, lbl, ha="left", va="center",
                    fontsize=9.5, color="#666",
                    transform=ax_cap.transAxes)
        ax_cap.text(0.34, y, val, ha="left", va="center",
                    fontsize=9.5, color="#222", fontweight="bold",
                    transform=ax_cap.transAxes)

    if utt.transcript:
        words = utt.transcript.strip().rstrip(".")
        # Wrap to ~70 chars.
        wrapped = words if len(words) < 80 else words[:77] + "…"
        ax_cap.text(0.020, 0.13,
                    f"\\u201c{wrapped}\\u201d",
                    ha="left", va="center",
                    fontsize=10, color="#444", style="italic",
                    transform=ax_cap.transAxes)

    # ─── Row 1: cochleagram (auditory spectrogram) ───
    ax_coch = fig.add_subplot(outer[1])
    coch_extent = [0, total_ms, -0.5, N_FREQ - 0.5]
    ax_coch.imshow(
        coch_db.T, origin="lower", aspect="auto", cmap="magma",
        extent=coch_extent, vmin=-60.0, vmax=0.0,
        interpolation="bilinear",
    )
    target_hz = [200, 500, 1000, 2000, 5000]
    tk_bins, tk_lab = [], []
    for hz in target_hz:
        if COCHLEA_CF[0] <= hz <= COCHLEA_CF[-1]:
            i = int(np.argmin(np.abs(COCHLEA_CF - hz)))
            tk_bins.append(i); tk_lab.append(_fmt_hz(hz))
    ax_coch.set_yticks(tk_bins)
    ax_coch.set_yticklabels(tk_lab)
    ax_coch.set_ylim(-0.5, N_FREQ - 0.5)
    ax_coch.set_ylabel("Frequency (Hz)")
    ax_coch.set_xlim(0, total_ms)
    ax_coch.set_xticklabels([])
    ax_coch.tick_params(axis="y", colors="#444")
    for spine in ax_coch.spines.values():
        spine.set_color("#888"); spine.set_linewidth(0.6)

    # Highlight probe-phoneme intervals on the cochleagram with a thin top bar.
    for f0, f1 in probe_frames:
        ax_coch.add_patch(plt.Rectangle(
            (f0 * ms_per_frame, N_FREQ - 4),
            (f1 - f0) * ms_per_frame, 4,
            facecolor=color, edgecolor="none", alpha=0.85, clip_on=True,
        ))
    # Faint white verticals at peak frames.
    for f in peaks:
        ax_coch.axvline(f * ms_per_frame, color="white", lw=0.5, alpha=0.30)
    ax_coch.text(0.005, 0.965, "Auditory spectrogram   (model wav2aud, dB)",
                 transform=ax_coch.transAxes, ha="left", va="top",
                 fontsize=8.5, color="white", alpha=0.95,
                 bbox=dict(boxstyle="round,pad=0.22", fc="black",
                           ec="none", alpha=0.50))

    # ─── Row 2: phoneme strip ───
    ax_strip = fig.add_subplot(outer[2])
    _draw_phone_strip(ax_strip, utt.phone_segs, total_ms)
    ax_strip.set_xticklabels([])

    # ─── Row 3: eSTRF response ───
    ax_resp = fig.add_subplot(outer[3])
    t_axis = np.arange(n_frames) * ms_per_frame
    # Centre the curve on its median so the zero line is meaningful.
    r = response - np.median(response)
    rmin, rmax = float(r.min()), float(r.max())
    pad_y = 0.10 * max(rmax - rmin, 1e-6)
    y_lo, y_hi = rmin - pad_y, rmax + pad_y * 2.0

    # Ground-truth shading (where the probe phoneme actually lives in the utt).
    for f0, f1 in probe_frames:
        ax_resp.axvspan(f0 * ms_per_frame, f1 * ms_per_frame,
                        facecolor=color, alpha=0.10, edgecolor="none", zorder=0)

    # The response curve itself.
    ax_resp.fill_between(t_axis, 0, r, where=(r > 0),
                         color=color, alpha=0.55, linewidth=0.0, zorder=2)
    ax_resp.fill_between(t_axis, 0, r, where=(r < 0),
                         color=color, alpha=0.18, linewidth=0.0, zorder=2)
    ax_resp.plot(t_axis, r, color=color, linewidth=1.4, zorder=3)
    ax_resp.axhline(0, color="#555", lw=0.7, alpha=0.65)
    for f in peaks:
        ax_resp.axvline(f * ms_per_frame, color=color, lw=0.5,
                        alpha=0.55, ls=":", zorder=1)
    ax_resp.set_xlim(0, total_ms); ax_resp.set_ylim(y_lo, y_hi)
    ax_resp.set_xlabel("Time (ms)")
    ax_resp.set_ylabel(f"r(t)  for  /{probe_phone}/", color="#222")
    ax_resp.tick_params(axis="y", colors="#444")
    for spine in ax_resp.spines.values():
        spine.set_color("#888"); spine.set_linewidth(0.6)

    # Tiny legend in the response panel: shaded = ground-truth probe regions.
    ax_resp.text(
        0.005, 0.96,
        f"shaded = ground-truth /{probe_phone}/ intervals   ·   "
        f"dotted lines = response peaks",
        transform=ax_resp.transAxes, ha="left", va="top",
        fontsize=7.5, color="#444",
        bbox=dict(boxstyle="round,pad=0.22", fc="white",
                  ec="#ccc", lw=0.5, alpha=0.85),
    )

    fig.suptitle(
        "eSTRF response along a TIMIT utterance   ·   trained biomimetic frontend",
        y=0.975, fontsize=12.5, color="#222", fontweight="bold",
    )
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.show()
"""))

# ════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## 10. Render every (family × probe × utterance) example

For each manner family we plot **two utterance examples** for each
probe phoneme.  Watch how the response curve correlates with the
phoneme strip — the eSTRF for /s/ flares on /s/, the eSTRF for /aa/
hugs the open-vowel patches, and so on.
"""))

cells.append(code("""\
N_UTTERANCES_PER_PROBE = 2

for family, probes in FAMILY_PROBE.items():
    available = [p for p in probes if p in ESTRF]
    if not available:
        print(f"  ── {family} ──   (no LIME aggregates available)")
        continue
    print(f"\\n══════  {family}  ══════")
    for probe in available:
        utts = pick_utterances_for_probe(probe, n_picks=N_UTTERANCES_PER_PROBE)
        if not utts:
            print(f"  /{probe}/   no suitable utterance found.")
            continue
        for utt in utts:
            print(f"  /{probe}/   →   {utt.utt_id}")
            plot_estrf_along_utterance(utt, probe, family)
"""))

# ════════════════════════════════════════════════════════════════════════
cells.append(md("""\
## 11. (Optional) bring your own utterance

Drop a `.wav` here (mono, 16 kHz preferred) and paste a `.PHN`-style
`start  end  phone` alignment file in the cell below to get the same
analysis on your own clip.  TIMIT phones (61-symbol set) are auto-folded
to the 39-class inventory.
"""))

cells.append(code("""\
# Example: re-run the most striking probe on the first /s/-rich utterance.
import random
random.seed(0)
fam = "Fricatives"
probe = "s"
candidates = pick_utterances_for_probe(probe, n_picks=4)
if candidates:
    chosen = random.choice(candidates)
    print(f"Bonus render:  /{probe}/  on  {chosen.utt_id}")
    print(f"  transcript: \\u201c{chosen.transcript}\\u201d")
    plot_estrf_along_utterance(chosen, probe, fam)
"""))

# ════════════════════════════════════════════════════════════════════════
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"Wrote {OUT}  ({OUT.stat().st_size/1024:.1f} KB, {len(cells)} cells)")
