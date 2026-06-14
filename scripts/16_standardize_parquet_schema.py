#!/usr/bin/env python3
"""
Script 16: Standardize Parquet Schema

Consolidates all datasets into a unified schema:

  | Column         | Type      | Notes                            |
  |----------------|-----------|----------------------------------|
  | sequence       | str       | Amino acid sequence              |
  | embedding      | float32[] | 1280-dim ESM2-650M               |
  | WA             | float64   | Weighted average                 |
  | BIN01-BIN10    | float64   | Bin probabilities (NaN if N/A)   |
  | gene           | str       | Gene name (NaN if N/A)           |
  | set            | str       | train / test / design            |
  | dataset_source | str       | grasso / wu / xue / zhang_p43 / zhang_pglvm |

Datasets standardized:
  - Grasso train (3095); has bins + gene from xlsx merge
  - Grasso test (1326); same
  - Grasso design (4911); no bins, gene from xlsx
  - Wu (81); 3 cols only
  - Xue (322); same
  - Zhang-P43 (114); same
  - Zhang-PglVM (114); same

Outputs:
  - data/unified/{dataset_source}.parquet; per-dataset files
  - data/unified/all_datasets_esm2_650M.parquet; combined file
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Paths ─────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / 'data'
UNIFIED_DIR = DATA_DIR / 'unified'

BIN_COLUMNS_LONG = [f'Perc_unambiguousReads_BIN{i:02d}_bin' for i in range(1, 11)]
BIN_COLUMNS_SHORT = [f'BIN{i:02d}' for i in range(1, 11)]

STANDARD_COLUMNS = ['sequence', 'embedding', 'WA'] + BIN_COLUMNS_SHORT + ['gene', 'set', 'dataset_source']


def load_xlsx_gene_map():
    """Build sequence → gene mapping from the Library xlsx sheet."""
    xlsx = DATA_DIR / 'sb2c00328_si_011.xlsx'
    df = pd.read_excel(xlsx, sheet_name='Library_w_Bins_and_WA')
    gene_map = dict(zip(df['SP_aa'].values, df['gene'].values))
    return gene_map


def standardize_grasso_split(parquet_name, set_label, gene_map):
    """Standardize a Grasso train/test parquet file."""
    df = pd.read_parquet(DATA_DIR / parquet_name)

    result = pd.DataFrame()
    result['sequence'] = df['sequence']
    result['embedding'] = df['embedding']
    result['WA'] = df['WA'].astype(np.float64)

    for short, long in zip(BIN_COLUMNS_SHORT, BIN_COLUMNS_LONG):
        if long in df.columns:
            result[short] = df[long].astype(np.float64)
        else:
            result[short] = np.nan

    result['gene'] = result['sequence'].map(gene_map)
    result['set'] = set_label
    result['dataset_source'] = 'grasso'

    return result[STANDARD_COLUMNS]


def standardize_grasso_design(gene_map):
    """Standardize the Grasso design variants parquet."""
    df = pd.read_parquet(DATA_DIR / 'design_esm2-650M_embeddings.parquet')

    result = pd.DataFrame()
    result['sequence'] = df['sequence']
    result['embedding'] = df['embedding']
    result['WA'] = df['WA'].astype(np.float64)

    for short in BIN_COLUMNS_SHORT:
        result[short] = np.nan

    result['gene'] = result['sequence'].map(gene_map)
    result['set'] = 'design'
    result['dataset_source'] = 'grasso'

    return result[STANDARD_COLUMNS]


def standardize_external(filename, source_name):
    """Standardize an external dataset parquet (3-column: sequence, embedding, WA)."""
    df = pd.read_parquet(DATA_DIR / filename)

    result = pd.DataFrame()
    result['sequence'] = df['sequence']
    result['embedding'] = df['embedding']
    result['WA'] = df['WA'].astype(np.float64)

    for short in BIN_COLUMNS_SHORT:
        result[short] = np.nan

    result['gene'] = np.nan
    result['set'] = 'external'
    result['dataset_source'] = source_name

    return result[STANDARD_COLUMNS]


def main():
    UNIFIED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading gene map from xlsx...")
    gene_map = load_xlsx_gene_map()
    print(f"  {len(gene_map)} sequence→gene mappings loaded")

    all_dfs = []

    # ── Grasso train/test ─────────────────────────────────────────────────
    # Use ESM2-650M embeddings (shared representation across all external datasets)
    for parquet, set_label in [('trainAA_esm2-650M.parquet', 'train'),
                                ('testAA_esm2-650M.parquet', 'test')]:
        print(f"\n  Standardizing Grasso {set_label}...")
        df = standardize_grasso_split(parquet, set_label, gene_map)
        n_gene = df['gene'].notna().sum()
        print(f"    {len(df)} rows, {n_gene} with gene annotation")
        df.to_parquet(UNIFIED_DIR / f'grasso_{set_label}.parquet', index=False)
        all_dfs.append(df)

    # ── Grasso design ─────────────────────────────────────────────────────
    print(f"\n  Standardizing Grasso design variants...")
    df_design = standardize_grasso_design(gene_map)
    n_gene = df_design['gene'].notna().sum()
    print(f"    {len(df_design)} rows, {n_gene} with gene annotation")
    df_design.to_parquet(UNIFIED_DIR / 'grasso_design.parquet', index=False)
    all_dfs.append(df_design)

    # ── External datasets ─────────────────────────────────────────────────
    external = [
        ('wu_esm_embeddings.parquet', 'wu'),
        ('xue_esm_embeddings.parquet', 'xue'),
        ('zhang_p43_esm_embeddings.parquet', 'zhang_p43'),
        ('zhang_pglvm_esm_embeddings.parquet', 'zhang_pglvm'),
    ]
    for filename, source in external:
        print(f"\n  Standardizing {source}...")
        df_ext = standardize_external(filename, source)
        print(f"    {len(df_ext)} rows")
        df_ext.to_parquet(UNIFIED_DIR / f'{source}.parquet', index=False)
        all_dfs.append(df_ext)

    # ── Combined file ─────────────────────────────────────────────────────
    print(f"\n  Combining all datasets...")
    df_all = pd.concat(all_dfs, ignore_index=True)
    print(f"    Total: {len(df_all)} rows")
    print(f"    By source: {df_all['dataset_source'].value_counts().to_dict()}")
    print(f"    By set:    {df_all['set'].value_counts().to_dict()}")

    combined_path = UNIFIED_DIR / 'all_datasets_esm2_650M.parquet'
    df_all.to_parquet(combined_path, index=False)

    # ── Verification ──────────────────────────────────────────────────────
    print(f"\n  Verifying round-trip load...")
    df_check = pd.read_parquet(combined_path)
    assert len(df_check) == len(df_all), f"Row count mismatch: {len(df_check)} vs {len(df_all)}"
    assert list(df_check.columns) == STANDARD_COLUMNS, f"Column mismatch"

    # Verify each per-dataset file
    for f in UNIFIED_DIR.glob('*.parquet'):
        if f.name == 'all_datasets_esm2_650M.parquet':
            continue
        df_part = pd.read_parquet(f)
        assert list(df_part.columns) == STANDARD_COLUMNS, f"Schema mismatch in {f.name}"

    print(f"\n  All files saved to {UNIFIED_DIR}/")
    print(f"  Schema verified: {STANDARD_COLUMNS}")

    # Print summary table
    print(f"\n  {'Source':<15} {'Set':<10} {'Rows':>6} {'Bins?':>6} {'Gene?':>6}")
    print(f"  {'-'*48}")
    for source in df_all['dataset_source'].unique():
        for s in df_all[df_all['dataset_source'] == source]['set'].unique():
            mask = (df_all['dataset_source'] == source) & (df_all['set'] == s)
            sub = df_all[mask]
            has_bins = sub['BIN01'].notna().any()
            has_gene = sub['gene'].notna().any()
            print(f"  {source:<15} {s:<10} {len(sub):>6} {'yes' if has_bins else 'no':>6} {'yes' if has_gene else 'no':>6}")

    print("\nDone.")


if __name__ == '__main__':
    main()
