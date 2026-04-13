"""
TIMIT dataset loader for CorticalLIME analysis.

Downloads the TIMIT corpus from Kaggle (mfekadu/darpa-timit-acousticphonetic-
continuous-speech) and provides a clean iterator over utterances with
frame-level phoneme labels, word boundaries, speaker metadata, and
standard 61→39 phone folding for evaluation.

The loader mirrors the training-time conventions of the diffAudNeuro
phoneme recognition pipeline (STRF_phonemeRecognition.py):
  - 16 kHz mono audio
  - 5 ms frame hop → 80 samples/frame → 200 frames/second
  - 1-indexed phoneme labels (0 = blank/pad)
  - RMS normalisation per segment

Usage
-----
    from timit_dataset import TIMITDataset

    ds = TIMITDataset(split="TEST")           # downloads on first call
    utt = ds[0]                               # TIMITUtterance dataclass
    wav = utt.segment_audio(duration=1.0)     # 1-sec centre crop, RMS-normed
    labels = utt.segment_labels(duration=1.0) # matching frame labels
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import librosa
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

SR = 16_000
FRAME_MS = 5
FRAME_SAMPLES = SR * FRAME_MS // 1000  # 80

# 61-phone TIMIT inventory (same ordering as precompute_timit_labels.py).
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

PHONEME_TO_IDX = {p: i + 1 for i, p in enumerate(TIMIT_PHONEMES)}  # 1-indexed
IDX_TO_PHONEME = {0: "<blank>"}
IDX_TO_PHONEME.update({i + 1: p for i, p in enumerate(TIMIT_PHONEMES)})

# Standard 61→39 folding (Lee & Hon, 1989).
PHONE_61_TO_39 = {
    "aa": "aa", "ae": "ae", "ah": "ah", "ao": "aa", "aw": "aw",
    "ax": "ah", "ax-h": "ah", "axr": "er", "ay": "ay",
    "b": "b", "bcl": "sil", "ch": "ch",
    "d": "d", "dcl": "sil", "dh": "dh", "dx": "dx",
    "eh": "eh", "el": "l", "em": "m", "en": "n", "eng": "ng",
    "epi": "sil", "er": "er", "ey": "ey",
    "f": "f", "g": "g", "gcl": "sil",
    "h#": "sil", "hh": "hh", "hv": "hh",
    "ih": "ih", "ix": "ih", "iy": "iy",
    "jh": "jh", "k": "k", "kcl": "sil",
    "l": "l", "m": "m", "n": "n", "ng": "ng", "nx": "n",
    "ow": "ow", "oy": "oy", "p": "p", "pau": "sil", "pcl": "sil",
    "q": "",  # glottal stop — excluded from evaluation
    "r": "r", "s": "s", "sh": "sh",
    "t": "t", "tcl": "sil", "th": "th",
    "uh": "uh", "uw": "uw", "ux": "uw",
    "v": "v", "w": "w", "y": "y", "z": "z", "zh": "sh",
}

# Phoneme families (linguistically motivated groupings for analysis).
PHONEME_FAMILIES = {
    "vowels": [
        "aa", "ae", "ah", "ao", "aw", "ax", "ax-h", "axr", "ay",
        "eh", "er", "ey", "ih", "ix", "iy", "ow", "oy", "uh", "uw", "ux",
    ],
    "stops": ["b", "d", "g", "p", "t", "k"],
    "closures": ["bcl", "dcl", "gcl", "pcl", "tcl", "kcl"],
    "fricatives": ["f", "s", "sh", "z", "zh", "v", "th", "dh"],
    "affricates": ["ch", "jh"],
    "nasals": ["m", "n", "ng", "nx", "em", "en", "eng"],
    "semivowels_glides": ["l", "el", "r", "w", "y", "hh", "hv"],
    "silence": ["h#", "pau", "epi"],
}

# Voicing cognate pairs (voiced, voiceless).
VOICING_PAIRS = [
    ("b", "p"), ("d", "t"), ("g", "k"),
    ("z", "s"), ("v", "f"), ("dh", "th"),
    ("jh", "ch"),
]

# Place-of-articulation groups (within stops + fricatives).
PLACE_GROUPS = {
    "labial":   ["b", "p", "f", "v", "m"],
    "alveolar": ["d", "t", "s", "z", "n", "l"],
    "velar":    ["g", "k", "ng"],
    "palatal":  ["sh", "zh", "ch", "jh"],
    "dental":   ["th", "dh"],
}


# ═══════════════════════════════════════════════════════════════════════════
# Parsing helpers
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PhoneSegment:
    """A single phone segment from a .PHN file."""
    start_sample: int
    end_sample: int
    phone: str
    phone_idx: int              # 1-indexed
    phone_39: str               # folded label
    start_frame: int = 0
    end_frame: int = 0

    @property
    def duration_samples(self) -> int:
        return self.end_sample - self.start_sample

    @property
    def duration_sec(self) -> float:
        return self.duration_samples / SR


@dataclass
class WordSegment:
    """A single word from a .WRD file."""
    start_sample: int
    end_sample: int
    word: str


def _parse_phn(path: Path) -> list[PhoneSegment]:
    segments = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            start, end = int(parts[0]), int(parts[1])
            phone = parts[2].lower()
            idx = PHONEME_TO_IDX.get(phone, 0)
            p39 = PHONE_61_TO_39.get(phone, "")
            sf = start // FRAME_SAMPLES
            ef = end // FRAME_SAMPLES
            segments.append(PhoneSegment(start, end, phone, idx, p39, sf, ef))
    return segments


def _parse_wrd(path: Path) -> list[WordSegment]:
    segments = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            segments.append(WordSegment(int(parts[0]), int(parts[1]),
                                        " ".join(parts[2:])))
    return segments


def _parse_txt(path: Path) -> str:
    with open(path, "r") as f:
        line = f.readline().strip()
    parts = line.split(maxsplit=2)
    return parts[2] if len(parts) >= 3 else line


def _find_file(parent: Path, stem: str, ext: str) -> Optional[Path]:
    """Case-insensitive file lookup."""
    for variant in [ext.upper(), ext.lower(), ext.capitalize()]:
        p = parent / (stem + variant)
        if p.exists():
            return p
    # Glob fallback.
    pat = stem + "".join(f"[{c.lower()}{c.upper()}]" for c in ext.lstrip("."))
    hits = list(parent.glob("*" + pat))
    return hits[0] if hits else None


# ═══════════════════════════════════════════════════════════════════════════
# Utterance container
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TIMITUtterance:
    """A single TIMIT utterance with all available metadata."""
    utterance_id: str                       # e.g. "DR1_FCJF0_SA1"
    wav_path: Path
    phn_path: Path
    wrd_path: Optional[Path]
    txt_path: Optional[Path]

    split: str                              # "TRAIN" or "TEST"
    dialect_region: str                     # e.g. "DR1"
    speaker_id: str                         # e.g. "FCJF0"
    sentence_type: str                      # "SA", "SI", "SX"
    gender: str                             # "M" or "F" (first char of speaker_id)

    phone_segments: list[PhoneSegment] = field(default_factory=list)
    word_segments: list[WordSegment] = field(default_factory=list)
    sentence_text: str = ""

    _audio: Optional[np.ndarray] = field(default=None, repr=False)

    # ── Audio loading ─────────────────────────────────────────────────

    @property
    def audio(self) -> np.ndarray:
        """Full waveform, 16 kHz float32 (lazy-loaded, cached)."""
        if self._audio is None:
            y, _ = librosa.load(str(self.wav_path), sr=SR, mono=True)
            self._audio = y.astype(np.float32)
        return self._audio

    @property
    def n_samples(self) -> int:
        return len(self.audio)

    @property
    def n_frames(self) -> int:
        return self.n_samples // FRAME_SAMPLES

    @property
    def duration_sec(self) -> float:
        return self.n_samples / SR

    def release_audio(self):
        """Free cached audio to save memory."""
        self._audio = None

    # ── Frame-level labels ────────────────────────────────────────────

    def frame_labels(self, fold_to_39: bool = False) -> np.ndarray:
        """Dense frame-level labels, shape (n_frames,), 1-indexed."""
        labels = np.ones(self.n_frames, dtype=np.int64)  # default silence
        for seg in self.phone_segments:
            idx = seg.phone_idx
            if fold_to_39 and seg.phone_39:
                idx = PHONEME_TO_IDX.get(seg.phone_39, idx)
            labels[seg.start_frame:min(seg.end_frame, self.n_frames)] = idx
        return labels

    def dominant_phone(self, fold_to_39: bool = False) -> tuple[str, int]:
        """Most-frequently-occurring phone (by frame count), excluding silence."""
        labels = self.frame_labels(fold_to_39)
        sil_indices = {PHONEME_TO_IDX.get(s, -1)
                       for s in ("h#", "pau", "epi", "sil")}
        mask = np.array([l not in sil_indices for l in labels])
        if not mask.any():
            return "sil", 0
        non_sil = labels[mask]
        vals, counts = np.unique(non_sil, return_counts=True)
        best = vals[np.argmax(counts)]
        return IDX_TO_PHONEME.get(int(best), "?"), int(best)

    # ── Segmenting ────────────────────────────────────────────────────

    def segment_audio(
        self,
        duration: float = 1.0,
        mode: str = "center",
        rms_norm: bool = True,
    ) -> np.ndarray:
        """Extract a fixed-length segment, zero-padded if needed.

        mode : "center" | "start" | "random"
        """
        n_target = int(duration * SR)
        y = self.audio
        if len(y) >= n_target:
            if mode == "center":
                s = (len(y) - n_target) // 2
            elif mode == "start":
                s = 0
            elif mode == "random":
                s = np.random.randint(0, len(y) - n_target + 1)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            y = y[s:s + n_target]
        else:
            y = np.pad(y, (0, n_target - len(y)))
        if rms_norm:
            rms = np.sqrt(np.mean(y ** 2))
            if rms > 0:
                y = y / rms
        return y.astype(np.float32)

    def segment_labels(
        self,
        duration: float = 1.0,
        mode: str = "center",
        fold_to_39: bool = False,
    ) -> np.ndarray:
        """Frame labels matching segment_audio (same crop window)."""
        n_target_samples = int(duration * SR)
        n_target_frames = n_target_samples // FRAME_SAMPLES
        labels = self.frame_labels(fold_to_39)
        n = len(labels)
        if n >= n_target_frames:
            if mode == "center":
                s = (n - n_target_frames) // 2
            elif mode == "start":
                s = 0
            elif mode == "random":
                s = np.random.randint(0, n - n_target_frames + 1)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            labels = labels[s:s + n_target_frames]
        else:
            labels = np.pad(labels, (0, n_target_frames - n), constant_values=1)
        return labels

    # ── Phone sequence ────────────────────────────────────────────────

    def phone_sequence(self, fold_to_39: bool = False, skip_silence: bool = False) -> list[str]:
        """Ordered phone string sequence for this utterance."""
        seq = []
        for seg in self.phone_segments:
            p = seg.phone_39 if fold_to_39 else seg.phone
            if skip_silence and p in ("sil", "h#", "pau", "epi", ""):
                continue
            seq.append(p)
        return seq

    def phone_durations(self) -> dict[str, list[float]]:
        """Per-phone duration distributions (in ms)."""
        durs: dict[str, list[float]] = {}
        for seg in self.phone_segments:
            durs.setdefault(seg.phone, []).append(seg.duration_sec * 1000)
        return durs


# ═══════════════════════════════════════════════════════════════════════════
# Dataset class
# ═══════════════════════════════════════════════════════════════════════════

class TIMITDataset:
    """Full TIMIT dataset loaded from the Kaggle mirror.

    Parameters
    ----------
    split : "TEST", "TRAIN", or "ALL"
    kaggle_slug : Kaggle dataset identifier.
    local_path : Override automatic download; point to an existing TIMIT root.
    min_duration : Skip utterances shorter than this (seconds).
    """

    def __init__(
        self,
        split: str = "TEST",
        kaggle_slug: str = "mfekadu/darpa-timit-acousticphonetic-continuous-speech",
        local_path: Optional[str] = None,
        min_duration: float = 0.3,
    ):
        self.split = split.upper()
        self.timit_root = self._resolve_root(kaggle_slug, local_path)
        self.utterances: list[TIMITUtterance] = []
        self._index_utterances(min_duration)

    # ── Download / locate ─────────────────────────────────────────────

    @staticmethod
    def _resolve_root(slug: str, local_path: Optional[str]) -> Path:
        if local_path is not None:
            root = Path(local_path)
            if not root.is_dir():
                raise FileNotFoundError(f"Local path not found: {root}")
            return root

        import kagglehub
        dl_path = Path(kagglehub.dataset_download(slug))
        # The download may nest: find the dir that contains TEST/.
        candidates = list(dl_path.rglob("TEST"))
        if not candidates:
            raise FileNotFoundError(
                f"No TEST/ directory found under {dl_path}. "
                "Check the Kaggle dataset structure."
            )
        return candidates[0].parent

    # ── Indexing ──────────────────────────────────────────────────────

    def _index_utterances(self, min_dur: float):
        splits = ["TEST", "TRAIN"] if self.split == "ALL" else [self.split]

        for split_name in splits:
            phn_files = sorted(
                self.timit_root.rglob(
                    f"{split_name}/**/*.[Pp][Hh][Nn]"
                )
            )
            for phn_path in phn_files:
                wav_path = _find_file(phn_path.parent, phn_path.stem, ".wav")
                if wav_path is None:
                    continue

                # Quick duration check without loading audio.
                try:
                    dur = librosa.get_duration(path=str(wav_path))
                except Exception:
                    continue
                if dur < min_dur:
                    continue

                wrd_path = _find_file(phn_path.parent, phn_path.stem, ".wrd")
                txt_path = _find_file(phn_path.parent, phn_path.stem, ".txt")

                # Parse path structure: .../SPLIT/DR?/SPEAKER_ID/SENTENCE.*
                parts = phn_path.relative_to(self.timit_root).parts
                # Find split position.
                try:
                    si = next(i for i, p in enumerate(parts)
                             if p.upper() in ("TEST", "TRAIN"))
                except StopIteration:
                    continue

                dialect = parts[si + 1] if si + 1 < len(parts) else "?"
                speaker = parts[si + 2] if si + 2 < len(parts) else "?"
                sentence = phn_path.stem.upper()

                # Sentence type: SA (dialect), SI (diverse), SX (compact).
                stype_match = re.match(r"(SA|SI|SX)", sentence)
                stype = stype_match.group(1) if stype_match else "?"

                gender = speaker[0].upper() if speaker else "?"

                utt_id = f"{dialect}_{speaker}_{sentence}"

                phone_segments = _parse_phn(phn_path)
                word_segments = _parse_wrd(wrd_path) if wrd_path else []
                text = _parse_txt(txt_path) if txt_path else ""

                self.utterances.append(TIMITUtterance(
                    utterance_id=utt_id,
                    wav_path=wav_path,
                    phn_path=phn_path,
                    wrd_path=wrd_path,
                    txt_path=txt_path,
                    split=split_name,
                    dialect_region=dialect,
                    speaker_id=speaker,
                    sentence_type=stype,
                    gender=gender,
                    phone_segments=phone_segments,
                    word_segments=word_segments,
                    sentence_text=text,
                ))

    # ── Accessors ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.utterances)

    def __getitem__(self, idx) -> TIMITUtterance:
        return self.utterances[idx]

    def __iter__(self):
        return iter(self.utterances)

    # ── Filtering ─────────────────────────────────────────────────────

    def filter(
        self,
        dialect_regions: Optional[Sequence[str]] = None,
        genders: Optional[Sequence[str]] = None,
        sentence_types: Optional[Sequence[str]] = None,
        speakers: Optional[Sequence[str]] = None,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
    ) -> list[TIMITUtterance]:
        """Return a filtered subset (does not modify the dataset in place)."""
        out = []
        for u in self.utterances:
            if dialect_regions and u.dialect_region not in dialect_regions:
                continue
            if genders and u.gender not in genders:
                continue
            if sentence_types and u.sentence_type not in sentence_types:
                continue
            if speakers and u.speaker_id not in speakers:
                continue
            if min_duration and u.duration_sec < min_duration:
                continue
            if max_duration and u.duration_sec > max_duration:
                continue
            out.append(u)
        return out

    def sample(
        self,
        n: int,
        seed: int = 42,
        exclude_sa: bool = True,
    ) -> list[TIMITUtterance]:
        """Randomly sample n utterances.

        exclude_sa : skip SA sentences (they are repeated across speakers,
                     inflating apparent coverage).
        """
        pool = self.utterances
        if exclude_sa:
            pool = [u for u in pool if u.sentence_type != "SA"]
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
        return [pool[i] for i in idx]

    # ── Summary statistics ────────────────────────────────────────────

    def summary(self) -> dict:
        n = len(self.utterances)
        durs = [u.duration_sec for u in self.utterances]
        speakers = {u.speaker_id for u in self.utterances}
        regions = {u.dialect_region for u in self.utterances}
        genders = {}
        for u in self.utterances:
            genders[u.gender] = genders.get(u.gender, 0) + 1
        stypes = {}
        for u in self.utterances:
            stypes[u.sentence_type] = stypes.get(u.sentence_type, 0) + 1
        return dict(
            n_utterances=n,
            n_speakers=len(speakers),
            n_dialect_regions=len(regions),
            dialect_regions=sorted(regions),
            gender_counts=genders,
            sentence_type_counts=stypes,
            duration_mean=float(np.mean(durs)) if durs else 0,
            duration_std=float(np.std(durs)) if durs else 0,
            duration_min=float(np.min(durs)) if durs else 0,
            duration_max=float(np.max(durs)) if durs else 0,
            total_duration_min=float(np.sum(durs) / 60) if durs else 0,
        )

    def phone_distribution(self, fold_to_39: bool = False) -> dict[str, int]:
        """Count total frames per phone across all utterances."""
        counts: dict[str, int] = {}
        for u in self.utterances:
            for seg in u.phone_segments:
                p = seg.phone_39 if fold_to_39 else seg.phone
                if not p:
                    continue
                n_frames = seg.end_frame - seg.start_frame
                counts[p] = counts.get(p, 0) + n_frames
        return dict(sorted(counts.items(), key=lambda x: -x[1]))
