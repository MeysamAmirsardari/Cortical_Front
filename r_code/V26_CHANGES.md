# V26 Changes - Original JAX Code with TIMIT & WSJ Support

## Overview

V26 is the original JAX/Flax code with **minimal changes** to support:
1. **TIMIT phoneme recognition** 
2. **WSJ source separation** (with MUSDB18 noise)

## Environment

```bash
conda activate diffaud_jax
```

## Changes Made

### `STRF_phonemeRecognition.py`
- Added support for empty `--home_dir ""` to use absolute paths in file lists

### `STRF_SpeechSeparation_stft.py`
- Added `load_audio_universal()` function to handle WSJ NIST SPHERE (.wv1) files
- Added `get_audio_duration()` function for SPHERE format
- Modified `CocktailAudioDataset` to use these helper functions
- Added support for empty `--home_dir ""` for absolute paths

### `precompute_timit_labels.py` (NEW)
- Script to pre-compute TIMIT labels in the format expected by the original code

---

## TIMIT Phoneme Recognition

### Step 1: Pre-compute TIMIT Labels

```bash
conda activate diffaud_jax
cd /mnt/localssd/diffaudio/diff_aud/v26-jax-diffAudNeuro-main

python precompute_timit_labels.py \
    --timit_root /mnt/localssd/diffaudio/timit_dataset/timit/TIMIT \
    --split TRAIN

python precompute_timit_labels.py \
    --timit_root /mnt/localssd/diffaudio/timit_dataset/timit/TIMIT \
    --split TEST
```

This creates:
- `precomputed_labels/timit_train_audio.txt` - list of audio paths
- `precomputed_labels/timit_train_labels.txt` - list of label file paths
- `precomputed_labels/TRAIN/*.pt` - frame-level phoneme labels

### Step 2: Train Phoneme Recognition

```bash
conda activate diffaud_jax
cd /mnt/localssd/diffaudio/diff_aud/v26-jax-diffAudNeuro-main

python STRF_phonemeRecognition.py \
    --num_strfs 40 \
    --strf_init_method random \
    --spec_type linAud \
    --n_phones 62 \
    --decoder_type cnn \
    --encoder_type strf \
    --input_type audio \
    --compression_method power \
    --update_lin \
    --conv_feats 10 20 40 \
    --pooling_stride 2 \
    --update_sr 1 \
    --loss_fct xe \
    --lr_v 0.001 \
    --lr_sr 0.001 \
    --num_steps 200000 \
    --minibatch_size 4 \
    --home_dir "" \
    --metadata_dir /mnt/localssd/diffaudio/timit_dataset/precomputed_labels/timit_train_audio.txt \
    --label_dir /mnt/localssd/diffaudio/timit_dataset/precomputed_labels/timit_train_labels.txt \
    --noise_condition clean \
    --model_name timit_phoneme_jax
```

---

## WSJ Source Separation

### Prerequisites

1. **sph2pipe** for WSJ SPHERE format:
```bash
# Already compiled at /mnt/localssd/diffaudio/tools/sph2pipe
# If not present:
cd /mnt/localssd/diffaudio/tools
wget https://www.openslr.org/resources/3/sph2pipe_v2.5.tar.gz
tar -xzf sph2pipe_v2.5.tar.gz
cd sph2pipe_v2.5
gcc -o sph2pipe *.c -lm
cp sph2pipe ../
```

2. **File lists** (already created):
- `/mnt/localssd/diffaudio/wsj_dataset/speaker_002_train_1000.txt` - WSJ clean speech
- `/mnt/localssd/diffaudio/wsj_dataset/musdb18_train_music.txt` - MUSDB18 music noise

### Train Source Separation

```bash
conda activate diffaud_jax
cd /mnt/localssd/diffaudio/diff_aud/v26-jax-diffAudNeuro-main

python STRF_SpeechSeparation_stft.py \
    --num_strfs 40 \
    --strf_init_method log \
    --update_sr 1 \
    --input_type audio \
    --encoder_type strf \
    --conv_features 20 40 10 1 \
    --lr_v 0.001 \
    --lr_sr 0.001 \
    --num_steps 200000 \
    --minibatch_size 4 \
    --loss L1 \
    --snr 0.0 \
    --home_dir "" \
    --clean_dir /mnt/localssd/diffaudio/wsj_dataset/speaker_002_train_1000.txt \
    --noise_dir /mnt/localssd/diffaudio/wsj_dataset/musdb18_train_music.txt \
    --model_name wsj_source_sep_jax
```

---

## Paper Settings Reference

### Phoneme Recognition (Section 3.1)
| Setting | Value |
|---------|-------|
| STRFs | 40, random init |
| CNN Decoder | [10, 20, 40] features |
| Pooling stride | 2 |
| Learning rate | 0.001 |
| Batch size | 4 |
| Loss | Cross-entropy (XE) |
| Compression | Power law |
| Update LIN | Yes |

### Source Separation (Section 3.2)
| Setting | Value |
|---------|-------|
| STRFs | 40, **log-spaced init** |
| CNN | [20, 40, 10, 1] features |
| Learning rate | 0.001 |
| Batch size | 4 |
| Loss | L1 (waveform + multi-scale STFT) |
| SNR | 0 dB |
| Training data | WSJ S002 (female), 1000 utterances |
| Noise | MUSDB18 music (100 tracks) |

