#!/usr/bin/env python3
"""
Script 00: Generate ESM2-650M Embeddings for Design Variants

One-time data preparation script. Generates mean-pooled ESM2-650M embeddings
for the ~5,000 designed signal peptide variants in the Grasso library that
lack precomputed PLM embeddings.

Output: data/design_esm2-650M_embeddings.parquet
    Columns: sequence (str), embedding (list[float64]), WA (float64)

Dependencies: torch, transformers (pip install torch transformers)
Runtime: ~30-60 min CPU (i7-1360P), ~5-10 min GPU (RTX)
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from transformers import EsmTokenizer, EsmModel

# ── Paths ─────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
XLSX_PATH = DATA_DIR / 'sb2c00328_si_011.xlsx'
OUTPUT_PATH = DATA_DIR / 'design_esm2-650M_embeddings.parquet'

MODEL_NAME = 'facebook/esm2_t33_650M_UR50D'
BATCH_SIZE = 8
MAX_LENGTH = 512


def load_design_sequences():
    """Load design variant sequences with WA from the xlsx."""
    print("Loading Library_w_Bins_and_WA from xlsx...")
    df = pd.read_excel(XLSX_PATH, sheet_name='Library_w_Bins_and_WA')
    print(f"  Total rows: {len(df)}")

    # Design variants: Set is NaN, WA not NaN, SP_aa not NaN
    mask = df['Set'].isna() & df['WA'].notna() & df['SP_aa'].notna()
    df_design = df[mask].copy()
    print(f"  Design variants (Set=NaN, WA+SP_aa not NaN): {len(df_design)}")

    # Deduplicate by SP_aa, computing mean WA per unique sequence
    grouped = df_design.groupby('SP_aa')['WA'].mean().reset_index()
    grouped.columns = ['sequence', 'WA']
    print(f"  Unique sequences after dedup: {len(grouped)}")

    return grouped


def generate_embeddings(sequences, model, tokenizer, device):
    """Generate mean-pooled ESM2 embeddings for a list of sequences."""
    print(f"\nGenerating embeddings for {len(sequences)} sequences...")
    embeddings = []

    for i in range(0, len(sequences), BATCH_SIZE):
        batch = sequences[i:i + BATCH_SIZE]

        tokens = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}

        with torch.no_grad():
            outputs = model(**tokens)
            attention_mask = tokens['attention_mask']
            hidden = outputs.last_hidden_state

            # Mean pooling with attention mask
            masked = hidden * attention_mask.unsqueeze(-1)
            pooled = masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            embeddings.extend(pooled.cpu().numpy())

        done = min(i + BATCH_SIZE, len(sequences))
        if (i // BATCH_SIZE + 1) % 25 == 0 or done == len(sequences):
            print(f"  {done}/{len(sequences)} sequences processed")

    embeddings = np.array(embeddings)
    print(f"  Embedding shape: {embeddings.shape}")
    return embeddings


def main():
    print("=" * 60)
    print("  Script 00: Generate ESM2-650M Design Embeddings")
    print("=" * 60)

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU (this will take 30-60 minutes)")

    # Load sequences
    df = load_design_sequences()

    # Load model
    print(f"\nLoading ESM-2 model: {MODEL_NAME}...")
    tokenizer = EsmTokenizer.from_pretrained(MODEL_NAME)
    model = EsmModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    print(f"  Embedding dimension: {model.config.hidden_size}")

    # Generate embeddings
    sequences = df['sequence'].tolist()
    embeddings = generate_embeddings(sequences, model, tokenizer, device)

    # Cast to float64 to match existing parquet convention
    embeddings = embeddings.astype(np.float64)

    # Build output DataFrame
    df_out = pd.DataFrame({
        'sequence': sequences,
        'embedding': [emb.tolist() for emb in embeddings],
        'WA': df['WA'].values.astype(np.float64),
    })

    # Save
    df_out.to_parquet(OUTPUT_PATH)
    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"  Rows: {len(df_out)}")
    print(f"  Columns: {list(df_out.columns)}")

    # Verify round-trip
    print("\nVerifying round-trip...")
    df_check = pd.read_parquet(OUTPUT_PATH)
    emb_check = np.stack(df_check['embedding'].values)
    print(f"  Reloaded shape: {emb_check.shape}")
    assert emb_check.shape[1] == model.config.hidden_size, \
        f"Expected dim {model.config.hidden_size}, got {emb_check.shape[1]}"
    print("  Round-trip verification passed!")

    print(f"\n{'=' * 60}")
    print(f"  Done. {len(df_out)} design embeddings saved.")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
