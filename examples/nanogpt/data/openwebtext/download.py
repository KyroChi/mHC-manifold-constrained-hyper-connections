"""
Download and prepare OpenWebText dataset to shared location.

This script downloads OpenWebText from HuggingFace, tokenizes it using GPT-2 tokenizer,
and saves it to /weka/kyle/data/openwebtext (or OWT_DATA_DIR if set).

Usage:
    python download.py
    OWT_DATA_DIR=/path/to/data python download.py
"""

import os
import sys
import json
import tiktoken
import numpy as np
import torch
from datasets import load_dataset

# Target data directory (shared location)
TARGET_DATA_DIR = os.environ.get("OWT_DATA_DIR", "/weka/kyle/data/openwebtext")

# Create data directory if it doesn't exist
os.makedirs(TARGET_DATA_DIR, exist_ok=True)

# Check if data already exists
train_path = os.path.join(TARGET_DATA_DIR, "train.bin")
val_path = os.path.join(TARGET_DATA_DIR, "val.bin")
meta_path = os.path.join(TARGET_DATA_DIR, "meta.json")

if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(meta_path):
    print(f"OpenWebText data already exists at {TARGET_DATA_DIR}")
    print("Skipping download. To re-download, delete the existing files first.")
    sys.exit(0)

# GPT-2 tokenizer
print("Initializing GPT-2 tokenizer...")
enc = tiktoken.get_encoding("gpt2")
vocab_size = 50257  # GPT-2 vocab size

print(f"Downloading OpenWebText dataset from HuggingFace...")
print(f"Data will be saved to: {TARGET_DATA_DIR}")
print("This may take a while (dataset is ~38GB)...")

# Load OpenWebText dataset from HuggingFace
# This downloads the dataset and tokenizes it
try:
    dataset = load_dataset("openwebtext", split="train")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Make sure you have 'datasets' and 'tiktoken' installed:")
    print("  uv sync --all-extras")
    sys.exit(1)

print(f"Tokenizing {len(dataset):,} documents...")

# Tokenize all documents
all_tokens = []
for i, item in enumerate(dataset):
    text = item["text"]
    tokens = enc.encode(text)
    all_tokens.extend(tokens)
    
    if (i + 1) % 10000 == 0:
        print(f"  Processed {i + 1:,} documents, {len(all_tokens):,} tokens so far...")

# Convert to numpy array for efficient splitting
all_tokens = np.array(all_tokens, dtype=np.uint16)

# Split into train and val (90/10 split)
n = int(0.9 * len(all_tokens))
train_tokens = all_tokens[:n]
val_tokens = all_tokens[n:]

# Convert to torch tensors
train_data = torch.from_numpy(train_tokens.astype(np.int64))
val_data = torch.from_numpy(val_tokens.astype(np.int64))

# Save tokenized data
print(f"\nSaving train data ({len(train_data):,} tokens) to {train_path}...")
torch.save(train_data, train_path)

print(f"Saving val data ({len(val_data):,} tokens) to {val_path}...")
torch.save(val_data, val_path)

# Save metadata
meta = {
    "vocab_size": vocab_size,
    "tokenizer": "gpt2",
    "train_tokens": len(train_data),
    "val_tokens": len(val_data),
}

with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nDone!")
print(f"  Train tokens: {len(train_data):,}")
print(f"  Val tokens: {len(val_data):,}")
print(f"  Total tokens: {len(train_data) + len(val_data):,}")
print(f"  Vocab size: {vocab_size}")
print(f"\nData saved to: {TARGET_DATA_DIR}")
print("You can now use OWT_DATA_DIR={} in your training scripts".format(TARGET_DATA_DIR))

