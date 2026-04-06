import argparse
import pickle
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
import torch
from torch.utils import data
from tqdm import tqdm
import editdistance

from strfpy import *
from strfpy_jax import *
from supervisedSTRF import *

print(f"Evaluating on {jax.default_backend()}...")

# TIMIT 61 to 39 phone folding map (for standard evaluation)
PHONE_61_TO_39 = {
    'aa': 'aa', 'ae': 'ae', 'ah': 'ah', 'ao': 'aa', 'aw': 'aw',
    'ax': 'ah', 'ax-h': 'ah', 'axr': 'er', 'ay': 'ay', 'b': 'b',
    'bcl': 'sil', 'ch': 'ch', 'd': 'd', 'dcl': 'sil', 'dh': 'dh',
    'dx': 'dx', 'eh': 'eh', 'el': 'l', 'em': 'm', 'en': 'n',
    'eng': 'ng', 'epi': 'sil', 'er': 'er', 'ey': 'ey', 'f': 'f',
    'g': 'g', 'gcl': 'sil', 'h#': 'sil', 'hh': 'hh', 'hv': 'hh',
    'ih': 'ih', 'ix': 'ih', 'iy': 'iy', 'jh': 'jh', 'k': 'k',
    'kcl': 'sil', 'l': 'l', 'm': 'm', 'n': 'n', 'ng': 'ng',
    'nx': 'n', 'ow': 'ow', 'oy': 'oy', 'p': 'p', 'pau': 'sil',
    'pcl': 'sil', 'q': '', 'r': 'r', 's': 's', 'sh': 'sh',
    't': 't', 'tcl': 'sil', 'th': 'th', 'uh': 'uh', 'uw': 'uw',
    'ux': 'uw', 'v': 'v', 'w': 'w', 'y': 'y', 'z': 'z', 'zh': 'sh'
}

def calculate_per(predictions, targets, phone_61_list, use_39_fold=True):
    """
    Calculate Phoneme Error Rate (PER)
    
    Args:
        predictions: list of predicted phone sequences (each is a list of phone indices)
        targets: list of target phone sequences (each is a list of phone indices)
        phone_61_list: list of 61 phone strings (index to phone name mapping)
        use_39_fold: if True, fold to 39 phones before calculating PER
    
    Returns:
        per: Phoneme Error Rate as a percentage
        substitutions, deletions, insertions: error counts
    """
    if use_39_fold:
        # Create 39 phone vocabulary
        phone_39_set = sorted(set(PHONE_61_TO_39.values()) - {''})
        phone_39_to_idx = {p: i for i, p in enumerate(phone_39_set)}
        
        def fold_sequence(seq_61_idx):
            """Convert sequence of 61-phone indices to 39-phone indices"""
            seq_39_idx = []
            for idx_61 in seq_61_idx:
                if idx_61 >= len(phone_61_list):
                    continue  # skip padding or invalid indices
                phone_61 = phone_61_list[idx_61]
                phone_39 = PHONE_61_TO_39.get(phone_61, '')
                if phone_39 and phone_39 in phone_39_to_idx:
                    seq_39_idx.append(phone_39_to_idx[phone_39])
            return seq_39_idx
        
        # Fold both predictions and targets
        predictions = [fold_sequence(seq) for seq in predictions]
        targets = [fold_sequence(seq) for seq in targets]
    
    # Calculate edit distance for all sequences
    total_distance = 0
    total_length = 0
    
    for pred, tgt in zip(predictions, targets):
        distance = editdistance.eval(pred, tgt)
        total_distance += distance
        total_length += len(tgt)
    
    per = (total_distance / total_length) * 100 if total_length > 0 else 0.0
    
    return per, total_distance, total_length


def collapse_repeated_predictions(predictions):
    """
    Collapse repeated predictions (like CTC decoding)
    e.g., [1, 1, 2, 2, 2, 3] -> [1, 2, 3]
    """
    collapsed = []
    for seq in predictions:
        if len(seq) == 0:
            collapsed.append([])
            continue
        new_seq = [seq[0]]
        for phone in seq[1:]:
            if phone != new_seq[-1]:
                new_seq.append(phone)
        collapsed.append(new_seq)
    return collapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', required=True, type=str, help='Path to checkpoint file')
    parser.add_argument('--model_dir', required=True, type=str, help='Path to model directory containing init.p')
    parser.add_argument('--num_strfs', required=True, type=int)
    parser.add_argument('--n_phones', required=True, type=int)
    parser.add_argument('--decoder_type', required=True, type=str)
    parser.add_argument('--encoder_type', required=True, type=str)
    parser.add_argument('--input_type', default='audio', type=str)
    parser.add_argument('--compression_method', default='power', type=str)
    parser.add_argument('--update_lin', action='store_true')
    parser.add_argument('--use_class', action='store_true')
    parser.add_argument('--conv_feats', required=True, nargs="+", type=int)
    parser.add_argument('--pooling_stride', default=2, type=int)
    parser.add_argument('--home_dir', type=str, default='')
    parser.add_argument('--metadata_dir', required=True, type=str, help='Test set audio file list')
    parser.add_argument('--label_dir', required=True, type=str, help='Test set label file list')
    parser.add_argument('--noise_condition', default='clean', type=str)
    parser.add_argument('--SNR', default=0, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--use_39_fold', action='store_true', help='Fold to 39 phones for evaluation')
    
    config = parser.parse_args()
    
    # Load cochlear filter parameters
    Bs, As = read_cochba_j()
    
    @jit
    def wav2aud_lin(x):
        '''Output size: 200 x 128'''
        return wav2aud_j(x, 5, 8, -2, 0, As, Bs, fft=True, return_stage=5).T
    batch_wav2aud_lin = vmap(wav2aud_lin)
    
    # Initialize model
    model = vSupervisedSTRF(
        n_phones=config.n_phones,
        input_type=config.input_type,
        update_lin=config.update_lin,
        use_class=config.use_class,
        encoder_type=config.encoder_type,
        decoder_type=config.decoder_type,
        compression_method=config.compression_method,
        conv_feats=config.conv_feats,
        pooling_stride=config.pooling_stride
    )
    
    # Load initial parameters to get structure
    with open(f'{config.model_dir}/init.p', 'rb') as f:
        init_data = pickle.load(f)
        params = init_data['params']
    
    # Initialize model parameters
    if config.input_type == 'audio':
        nn_params = model.init(jax.random.key(0), jnp.ones([config.batch_size, 16000]), params)
    elif config.input_type == 'spec':
        nn_params = model.init(jax.random.key(0), jnp.ones([config.batch_size, 200, 128]), params)
    else:
        raise KeyError(f"Unknown input_type: {config.input_type}")
    
    # Load checkpoint
    with open(config.checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
        nn_params = checkpoint['nn_params']
        params = checkpoint['params']
    
    print(f"Loaded checkpoint from {config.checkpoint_path}")
    
    # Load test dataset
    class SupervisedAudioDataset(data.Dataset):
        def __init__(self, home_dir, metadata_dir, label_dir, collapse_alignment, noise_condition='clean', snr=0):
            self.home_dir = home_dir
            if home_dir:
                self.filepaths = [home_dir + line.strip() for line in open(metadata_dir)]
                self.labelpath = [home_dir + line.strip() for line in open(label_dir)]
            else:
                self.filepaths = [line.strip() for line in open(metadata_dir)]
                self.labelpath = [line.strip() for line in open(label_dir)]
            self.collapse_alignment = collapse_alignment
            self.noise_condition = noise_condition
            self.snr = snr

        def __len__(self):
            return len(self.filepaths)

        def __getitem__(self, index):
            audiopath = self.filepaths[index]
            labelpath = self.labelpath[index]
            audio = torch.load(audiopath)
            label = torch.load(labelpath)
            
            if self.collapse_alignment:
                label = [label[0]] + [label[i] for i in range(1, len(label)) if label[i] != label[i-1]]
            
            y_pad = np.array([len(label), 1])
            y = np.array(label)
            audio = np.array(audio)
            
            if self.noise_condition != 'clean':
                raise NotImplementedError
            
            return audio, y, y_pad
    
    test_set = SupervisedAudioDataset(
        home_dir=config.home_dir,
        metadata_dir=config.metadata_dir,
        label_dir=config.label_dir,
        collapse_alignment=0,  # Don't collapse for evaluation
        noise_condition=config.noise_condition,
        snr=config.SNR
    )
    
    # Load phone vocabulary (from init.p or create it)
    # For TIMIT, we use the standard 61 phones
    TIMIT_61_PHONES = ['h#', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay',
                       'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em',
                       'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'hh', 'hv',
                       'ih', 'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng',
                       'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 'sh',
                       't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']
    
    print(f"Evaluating on {len(test_set)} test samples...")
    
    # Run evaluation
    all_predictions = []
    all_targets = []
    
    test_loader = data.DataLoader(test_set, batch_size=config.batch_size, shuffle=False, num_workers=4)
    
    @jit
    def predict_batch(nn_params, s, sr):
        """Get predictions for a batch"""
        logits = model.apply(nn_params, s, sr)
        # If 4D (batch, time, pooled, n_phones), average over pooled dimension
        if len(logits.shape) == 4:
            logits = jnp.mean(logits, axis=2)
        # Get argmax predictions
        predictions = jnp.argmax(logits, axis=-1)  # (batch, time)
        return predictions
    
    with torch.no_grad():
        for batch_idx, (audio, labels, _) in enumerate(tqdm(test_loader)):
            audio_np = audio.numpy()
            labels_np = labels.numpy()
            
            # Convert audio to spectrogram if needed
            if config.input_type == 'audio':
                s = jnp.array(audio_np)
            else:
                raise NotImplementedError
            
            # Get predictions
            pred = predict_batch(nn_params, s, params)
            pred_np = np.array(pred)
            
            # Convert to lists of phone sequences
            for i in range(len(audio_np)):
                pred_seq = pred_np[i].tolist()
                target_seq = labels_np[i].tolist()
                
                all_predictions.append(pred_seq)
                all_targets.append(target_seq)
    
    # Collapse repeated predictions (like CTC decoding)
    all_predictions = collapse_repeated_predictions(all_predictions)
    
    # Calculate PER
    per, distance, total_length = calculate_per(
        all_predictions, all_targets, TIMIT_61_PHONES, use_39_fold=config.use_39_fold
    )
    
    print(f"\n{'='*60}")
    print(f"Evaluation Results:")
    print(f"{'='*60}")
    print(f"Total samples: {len(test_set)}")
    print(f"Total phonemes: {total_length}")
    print(f"Total edit distance: {distance}")
    if config.use_39_fold:
        print(f"PER (39 phones): {per:.2f}%")
    else:
        print(f"PER (61 phones): {per:.2f}%")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

