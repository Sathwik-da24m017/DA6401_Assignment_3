import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import pandas as pd
from pathlib import Path

# Special tokens used for padding and decoding boundaries
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"

# ------------------------------
# Custom PyTorch Dataset wrapper
# ------------------------------
class Seq2SeqDataset(Dataset):
    def __init__(self, input_sequences, target_sequences):
        self.input_sequences = input_sequences
        self.target_sequences = target_sequences

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, idx):
        return self.input_sequences[idx], self.target_sequences[idx]

# ------------------------------
# Load TSV data into (input, target) pairs
# ------------------------------
def load_data(file_path):
    df = pd.read_csv(file_path, sep="\t", header=None, names=["target", "input", "attestations"])
    df.dropna(subset=["input", "target"], inplace=True)
    return list(zip(df["input"].astype(str), df["target"].astype(str)))

# ------------------------------
# Build vocabulary and mappings from training pairs
# ------------------------------
def build_vocab(pairs, is_input=True):
    counter = Counter()
    for src, tgt in pairs:
        seq = src if is_input else tgt
        counter.update(seq)

    # Add special tokens to sorted character set
    vocab = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN] + sorted(counter)
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for char, idx in char2idx.items()}
    return vocab, char2idx, idx2char

# ------------------------------
# Encode a character sequence into index tokens
# ------------------------------
def encode_sequence(seq, char2idx, add_sos_eos=False):
    encoded = [char2idx[c] for c in seq]
    if add_sos_eos:
        return [char2idx[SOS_TOKEN]] + encoded + [char2idx[EOS_TOKEN]]
    return encoded

# ------------------------------
# Convert full dataset into PyTorch tensors
# ------------------------------
def tensorify(pairs, input2idx, target2idx):
    input_seqs = [torch.tensor(encode_sequence(src, input2idx), dtype=torch.long) for src, _ in pairs]
    target_seqs = [torch.tensor(encode_sequence(tgt, target2idx, add_sos_eos=True), dtype=torch.long) for _, tgt in pairs]
    return input_seqs, target_seqs

# ------------------------------
# Return a collate function for a specific padding index
# ------------------------------
def collate_fn(input_pad_idx, target_pad_idx):
    def fn(batch):
        input_batch, target_batch = zip(*batch)
        input_batch = pad_sequence(input_batch, batch_first=True, padding_value=input_pad_idx)
        target_batch = pad_sequence(target_batch, batch_first=True, padding_value=target_pad_idx)
        return input_batch, target_batch
    return fn

# ------------------------------
# Wrapper to create DataLoaders for train/dev/test
# ------------------------------
def get_dataloaders(train_pairs, dev_pairs, test_pairs, input2idx, target2idx, batch_size=64):
    train_input, train_target = tensorify(train_pairs, input2idx, target2idx)
    dev_input, dev_target     = tensorify(dev_pairs, input2idx, target2idx)
    test_input, test_target   = tensorify(test_pairs, input2idx, target2idx)

    train_loader = DataLoader(
        Seq2SeqDataset(train_input, train_target),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn(input2idx[PAD_TOKEN], target2idx[PAD_TOKEN])
    )

    dev_loader = DataLoader(
        Seq2SeqDataset(dev_input, dev_target),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn(input2idx[PAD_TOKEN], target2idx[PAD_TOKEN])
    )

    test_loader = DataLoader(
        Seq2SeqDataset(test_input, test_target),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn(input2idx[PAD_TOKEN], target2idx[PAD_TOKEN])
    )

    return train_loader, dev_loader, test_loader