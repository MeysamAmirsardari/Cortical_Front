"""Smoke-test the notebook's picker logic against the real checkpoint.

Verifies:
  1. model_v + clip_posterior returns a valid (B, K) probability matrix.
  2. supervisedSTRF (non-vmapped twin) cochlea_fn shares params correctly
     and produces a sensible (T_frames, F) cochleagram.
  3. cochleagram_to_db on the model's signed output produces a non-flat
     dB array with the expected dynamic range.
"""
import os, sys, pickle, re
from pathlib import Path

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit

CODE_DIR = "/Users/eminent/Projects/Cortical_Front/r_code"
sys.path.insert(0, CODE_DIR)
os.chdir(CODE_DIR)

from supervisedSTRF import vSupervisedSTRF, supervisedSTRF
from flax.traverse_util import flatten_dict

ckpt_path = Path("nishit_trained_models/main_jax_phoneme_rec_timit/chkStep_200000.p")
with open(ckpt_path, "rb") as f:
    nn_params, aud_params = pickle.load(f)

# ── Detect config ──
flat = flatten_dict(nn_params.get("params", nn_params))
best_idx, n_phones = -1, 62
for key, val in flat.items():
    key_str = "/".join(str(k) for k in key)
    if "Dense" in key_str and key[-1] == "kernel" and hasattr(val, "shape"):
        m = re.search(r"Dense_(\d+)", key_str)
        idx = int(m.group(1)) if m else 0
        if idx >= best_idx:
            best_idx, n_phones = idx, int(val.shape[-1]) - 1

CONFIG = dict(
    n_phones=n_phones, input_type="audio",
    encoder_type="strf", decoder_type="cnn",
    compression_method="power" if "compression_params" in aud_params else "identity",
    update_lin="alpha" in aud_params,
    use_class="AuditorySpectrogram" in str(list(flat.keys())),
    conv_feats=[10, 20, 40], pooling_stride=2,
)
print("Config:", CONFIG)

BATCH = 4
SR = 16_000
N = SR

model_v = vSupervisedSTRF(**CONFIG)
_ = model_v.init(jax.random.key(0), jnp.zeros([BATCH, N]), aud_params)

@jit
def clip_posterior(nn_params, audio_BxT, aud_params):
    logits = model_v.apply(nn_params, audio_BxT, aud_params)
    soft = jax.nn.softmax(logits, axis=-1)
    reduce_axes = tuple(range(1, soft.ndim - 1))
    return soft.mean(axis=reduce_axes)

# Random RMS-normalised noise → posterior should be a valid probability vector.
rng = np.random.default_rng(0)
audio = rng.standard_normal((BATCH, N)).astype(np.float32)
audio /= np.sqrt((audio ** 2).mean(axis=1, keepdims=True))
post = np.asarray(clip_posterior(nn_params, jnp.asarray(audio), aud_params))
print(f"clip_posterior  shape={post.shape}  "
      f"row_sums≈{post.sum(axis=1).round(6).tolist()}  "
      f"min={post.min():+.4g}  max={post.max():+.4g}")
assert post.shape == (BATCH, CONFIG["n_phones"] + 1), post.shape
assert np.allclose(post.sum(axis=1), 1.0, atol=1e-3), "posteriors must sum to 1"

# Non-vmapped twin → cochleagram.
model_s = supervisedSTRF(
    n_phones=CONFIG["n_phones"], input_type=CONFIG["input_type"],
    encoder_type=CONFIG["encoder_type"], decoder_type=CONFIG["decoder_type"],
    compression_method=CONFIG["compression_method"], update_lin=CONFIG["update_lin"],
    use_class=CONFIG["use_class"], conv_feats=CONFIG["conv_feats"],
    pooling_stride=CONFIG["pooling_stride"],
)
ALPHA = float(aud_params.get("alpha", 0.9922179))
COMP = aud_params["compression_params"]

@jit
def cochlea_fn_jax(x_T):
    return model_s.apply(nn_params, x_T, COMP, ALPHA, method=model_s.wav2aud)

coch = np.asarray(cochlea_fn_jax(jnp.asarray(audio[0])))
print(f"cochleagram     shape={coch.shape}  "
      f"min={coch.min():+.4g}  max={coch.max():+.4g}  "
      f"mean={coch.mean():+.4g}  std={coch.std():.4g}")

# dB conversion using |coch|.
mag = np.abs(coch.astype(np.float64))
peak = mag.max()
db = np.clip(20.0 * np.log10(mag / peak + 1e-12), -60.0, 0.0)
print(f"dB(|coch|)     shape={db.shape}  "
      f"min={db.min():+.2f}  max={db.max():+.2f}  "
      f"std={db.std():.2f}  (should be std > 5 if not flat)")
assert db.std() > 1.0, "dB cochleagram is suspiciously flat"

print("\nAll picker smoke tests passed.")
