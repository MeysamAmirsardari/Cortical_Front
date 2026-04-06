"""
Pre-compute TIMIT labels in the format expected by STRF_phonemeRecognition.py

The original code expects:
1. metadata_dir: text file with paths to audio files
2. label_dir: text file with paths to .pt label files (torch tensors)

Each .pt file contains frame-level phoneme labels (200 frames for 1 second at 5ms/frame).

Usage:
    conda activate diffaud_jax
    python precompute_timit_labels.py --timit_root /mnt/localssd/diffaudio/timit_dataset/timit/TIMIT
"""

import argparse
import os
from pathlib import Path
import numpy as np
import torch

# TIMIT 61 phonemes mapped to indices (1-indexed, 0 reserved for blank/padding)
TIMIT_PHONEMES = [
    'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay',
    'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx',
    'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey',
    'f', 'g', 'gcl', 'h#', 'hh', 'hv',
    'ih', 'ix', 'iy',
    'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx',
    'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r',
    's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux',
    'v', 'w', 'y', 'z', 'zh'
]
PHONEME_TO_IDX = {p: i+1 for i, p in enumerate(TIMIT_PHONEMES)}  # 1-indexed


def parse_phn_file(phn_path, sr=16000, frame_ms=5):
    """Parse TIMIT .PHN file and return frame-level labels."""
    frame_samples = int(sr * frame_ms / 1000)  # 80 samples per frame
    
    labels = []
    with open(phn_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                start_sample = int(parts[0])
                end_sample = int(parts[1])
                phoneme = parts[2].lower()
                
                start_frame = start_sample // frame_samples
                end_frame = end_sample // frame_samples
                
                idx = PHONEME_TO_IDX.get(phoneme, 1)  # Default to 1 if unknown
                labels.append((start_frame, end_frame, idx))
    
    return labels


def create_frame_labels(phn_labels, num_frames=200):
    """Create dense frame-level label array."""
    frame_labels = np.ones(num_frames, dtype=np.int64)  # Default to 1 (silence)
    
    for start_frame, end_frame, idx in phn_labels:
        for f in range(start_frame, min(end_frame, num_frames)):
            if f >= 0 and f < num_frames:
                frame_labels[f] = idx
    
    return frame_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timit_root', required=True, type=str,
                        help='Path to TIMIT root (containing TRAIN and TEST dirs)')
    parser.add_argument('--output_dir', default=None, type=str,
                        help='Output directory for labels (default: same as timit_root)')
    parser.add_argument('--split', default='TRAIN', type=str,
                        help='TRAIN or TEST')
    args = parser.parse_args()
    
    timit_root = Path(args.timit_root)
    output_dir = Path(args.output_dir) if args.output_dir else timit_root.parent / 'precomputed_labels'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    split_dir = timit_root / args.split
    
    # Find all PHN files
    phn_files = list(split_dir.rglob('*.PHN'))
    print(f"Found {len(phn_files)} .PHN files in {split_dir}")
    
    audio_paths = []
    label_paths = []
    
    for phn_path in phn_files:
        # Corresponding WAV file
        wav_path = phn_path.with_suffix('.WAV')
        if not wav_path.exists():
            wav_path = phn_path.with_suffix('.wav')
        
        if not wav_path.exists():
            print(f"Warning: No WAV file for {phn_path}")
            continue
        
        # Parse PHN file
        phn_labels = parse_phn_file(phn_path)
        
        # Get total duration in frames
        # TIMIT files are typically 16kHz, we need enough frames for at least 1 second
        max_frame = max(end for _, end, _ in phn_labels) if phn_labels else 0
        
        # Only use files with at least 215 frames (1 second + buffer for random start)
        if max_frame < 215:
            continue
        
        # Create dense frame labels for entire file
        frame_labels = create_frame_labels(phn_labels, num_frames=max_frame + 1)
        
        # Save label file
        rel_path = phn_path.relative_to(timit_root)
        label_file = output_dir / args.split / rel_path.with_suffix('.pt')
        label_file.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(torch.tensor(frame_labels), label_file)
        
        audio_paths.append(str(wav_path))
        label_paths.append(str(label_file))
    
    # Write file lists
    audio_list_file = output_dir / f'timit_{args.split.lower()}_audio.txt'
    label_list_file = output_dir / f'timit_{args.split.lower()}_labels.txt'
    
    with open(audio_list_file, 'w') as f:
        f.write('\n'.join(audio_paths))
    
    with open(label_list_file, 'w') as f:
        f.write('\n'.join(label_paths))
    
    print(f"Created {len(audio_paths)} audio/label pairs")
    print(f"Audio list: {audio_list_file}")
    print(f"Label list: {label_list_file}")
    print(f"\nTo train phoneme recognition:")
    print(f"  python STRF_phonemeRecognition.py \\")
    print(f"    --metadata_dir {audio_list_file} \\")
    print(f"    --label_dir {label_list_file} \\")
    print(f"    --n_phones 62 \\")  # 61 phonemes + 1 blank
    print(f"    ... other args ...")


if __name__ == '__main__':
    main()

