#!/usr/bin/env python3
"""
Script 17: Gene-Stratified Evaluation (Leave-One-Gene-Out CV)

Tests whether the model generalizes across genes by holding out ALL sequences
for each gene during training and evaluating on the held-out gene.

132 of 134 genes overlap between train and test sets (98.5%). This experiment
isolates gene-level generalization from within-gene interpolation.

Approach:
  1. Combine train + test data (with valid bins) from Ginkgo-AA0
  2. For each gene, hold out ALL its sequences, train on rest
  3. Use best vector NN architecture (256,256,128, dropout=0.35, focal, seed=42)
  4. Scaler fit inside each fold
  5. Compare: LOGO MSE vs standard test MSE (0.964, fair 1-seed)
  6. Also report: MSE for 2 test-only genes vs 132 shared genes

Uses 1 seed per fold (seed=42) to keep runtime manageable (~134 trainings).
Reports bootstrap CIs on aggregated predictions.

Outputs:
  - results/gene_stratified_evaluation_results.json
  - figures/gene_stratified_evaluation.png
"""
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as sp_stats

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_loading import load_plm_with_bins, load_plm_embeddings
from src.models import FocalLoss

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE / 'results'
FIGURES_DIR = BASE / 'figures'
DATA_DIR = BASE / 'data'

# ── Constants ─────────────────────────────────────────────────────────────
SEED = 42
BIN_CENTERS = np.arange(1, 11)
BIN_COLUMNS = [f'Perc_unambiguousReads_BIN{i:02d}_bin' for i in range(1, 11)]
N_BOOTSTRAP = 10_000
STANDARD_TEST_MSE = 0.964  # fair single-seed standard comparator for LOGO


def build_model(input_dim, seed=SEED):
    """Best vector NN: (256,256,128), dropout=0.35, focal loss."""
    tf.random.set_seed(seed)
    np.random.seed(seed)

    inputs = keras.Input(shape=(input_dim,))
    x = inputs
    for units in (256, 256, 128):
        x = layers.Dense(units)(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-4),
                  loss=FocalLoss(alpha=0.25, gamma=2.0))
    return model


def train_predict(X_train, y_bins, X_test, input_dim, seed=SEED):
    """Train model and return WA predictions."""
    model = build_model(input_dim, seed=seed)
    callbacks = [keras.callbacks.ReduceLROnPlateau(
        monitor='loss', factor=0.5, patience=10, min_lr=1e-6, verbose=0)]
    model.fit(X_train, y_bins, epochs=300, batch_size=32,
              callbacks=callbacks, verbose=0)
    pred = model.predict(X_test, verbose=0) @ BIN_CENTERS
    del model
    keras.backend.clear_session()
    return pred


def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def bootstrap_ci(y_true, y_pred, n_boot=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    mse_b = np.empty(n_boot)
    sp_b = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        mse_b[b] = np.mean((yt - yp) ** 2)
        sp_b[b] = sp_stats.spearmanr(yt, yp).statistic

    mse_pt = mse(y_true, y_pred)
    return {
        'mse': {'point': mse_pt,
                'ci_lo': float(np.percentile(mse_b, 2.5)),
                'ci_hi': float(np.percentile(mse_b, 97.5))},
        'spearman': {'point': float(sp_stats.spearmanr(y_true, y_pred).statistic),
                     'ci_lo': float(np.percentile(sp_b, 2.5)),
                     'ci_hi': float(np.percentile(sp_b, 97.5))},
    }


def load_data_with_genes():
    """Load Ginkgo-AA0 embeddings and merge gene labels from xlsx."""
    # Load embeddings (train + test)
    X_train_raw, _, y_train_wa_raw, _, y_train_bins_raw, _, meta_train = \
        load_plm_with_bins('ginkgo-AA0-650M')
    _, X_test_raw, _, y_test_raw, meta_test = load_plm_embeddings('ginkgo-AA0-650M')

    # Load Ginkgo parquets for sequence-to-gene mapping
    df_train_pq = pd.read_parquet(DATA_DIR / 'trainAA_ginkgo-AA0-650M.parquet')
    df_test_pq = pd.read_parquet(DATA_DIR / 'testAA_ginkgo-AA0-650M.parquet')

    # Load gene labels from xlsx
    xlsx_path = DATA_DIR / 'sb2c00328_si_011.xlsx'
    df_lib = pd.read_excel(xlsx_path, sheet_name='Library_w_Bins_and_WA',
                           usecols=['SP_aa', 'gene', 'Set', 'WA'])

    # Map sequences to genes
    seq_to_gene = dict(zip(df_lib['SP_aa'], df_lib['gene']))

    # Get gene labels for train set (using load_plm_with_bins which filters NaN bins)
    # The filtered df has fewer rows - need to match by filtering same way
    train_valid = df_train_pq[BIN_COLUMNS].notna().all(axis=1)
    df_train_valid = df_train_pq[train_valid].reset_index(drop=True)
    train_genes = df_train_valid['sequence'].map(seq_to_gene).values

    # Test set (all rows, no bin filtering needed for evaluation)
    test_genes = df_test_pq['sequence'].map(seq_to_gene).values

    # Also get test bins for combined LOGO eval
    test_valid = df_test_pq[BIN_COLUMNS].notna().all(axis=1)
    df_test_valid = df_test_pq[test_valid].reset_index(drop=True)
    X_test_bins = np.stack(df_test_valid['embedding'].values)
    y_test_bins_wa = df_test_valid['WA'].values.astype(np.float64)
    y_test_bins = df_test_valid[BIN_COLUMNS].values.astype(np.float64)
    test_bins_genes = df_test_valid['sequence'].map(seq_to_gene).values

    return {
        'X_train': X_train_raw, 'y_train_wa': y_train_wa_raw,
        'y_train_bins': y_train_bins_raw, 'train_genes': train_genes,
        'X_test': X_test_raw, 'y_test_wa': y_test_raw, 'test_genes': test_genes,
        'X_test_bins': X_test_bins, 'y_test_bins_wa': y_test_bins_wa,
        'y_test_bins': y_test_bins, 'test_bins_genes': test_bins_genes,
        'dim': X_train_raw.shape[1],
        'n_train': len(y_train_wa_raw), 'n_test': len(y_test_raw),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    print("Loading Ginkgo-AA0 data with gene labels...")
    data = load_data_with_genes()
    print(f"  Train (valid bins): {data['n_train']}, Test: {data['n_test']}, Dim: {data['dim']}")

    # Combine train + test (with valid bins) for LOGO CV
    X_all = np.vstack([data['X_train'], data['X_test_bins']])
    y_all_wa = np.concatenate([data['y_train_wa'], data['y_test_bins_wa']])
    y_all_bins = np.vstack([data['y_train_bins'], data['y_test_bins']])
    genes_all = np.concatenate([data['train_genes'], data['test_bins_genes']])

    unique_genes = np.unique(genes_all)
    n_genes = len(unique_genes)
    print(f"  Combined: {len(X_all)} samples, {n_genes} genes")

    # ══════════════════════════════════════════════════════════════════════
    # Leave-One-Gene-Out Cross-Validation
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  LEAVE-ONE-GENE-OUT CV ({n_genes} genes, seed={SEED})")
    print(f"{'='*60}")

    all_preds = np.full(len(X_all), np.nan)
    gene_results = {}

    for i, gene in enumerate(unique_genes):
        t0 = time.time()
        test_mask = genes_all == gene
        train_mask = ~test_mask

        n_test_gene = int(test_mask.sum())
        n_train_gene = int(train_mask.sum())

        # Scale inside fold
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_all[train_mask])
        X_te = scaler.transform(X_all[test_mask])

        y_tr_bins = y_all_bins[train_mask]
        y_te_wa = y_all_wa[test_mask]

        pred = train_predict(X_tr, y_tr_bins, X_te, data['dim'])
        all_preds[test_mask] = pred

        gene_mse = mse(y_te_wa, pred)
        gene_results[gene] = {
            'n_samples': n_test_gene,
            'mse': gene_mse,
            'mean_wa': float(np.mean(y_te_wa)),
        }

        elapsed = time.time() - t0
        status = f"Gene {i+1:>3}/{n_genes}: {gene:<12} n={n_test_gene:>3}, MSE={gene_mse:.4f}  ({elapsed:.1f}s)"
        print(f"  {status}")

    # ── Overall LOGO metrics ──────────────────────────────────────────────
    assert not np.any(np.isnan(all_preds)), "Some samples were not predicted!"

    logo_mse = mse(y_all_wa, all_preds)
    logo_ci = bootstrap_ci(y_all_wa, all_preds)

    print(f"\n{'='*60}")
    print(f"  LOGO CV RESULTS")
    print(f"{'='*60}")
    print(f"  LOGO MSE:     {logo_mse:.4f}  [{logo_ci['mse']['ci_lo']:.4f}, {logo_ci['mse']['ci_hi']:.4f}]")
    print(f"  LOGO Spearman: {logo_ci['spearman']['point']:.4f}")
    print(f"  Standard MSE:  {STANDARD_TEST_MSE} (5-seed ensemble, full data)")
    print(f"  Ratio:         {logo_mse / STANDARD_TEST_MSE:.3f}x")

    # ── Per-gene MSE statistics ───────────────────────────────────────────
    gene_mses = np.array([gene_results[g]['mse'] for g in unique_genes])
    gene_sizes = np.array([gene_results[g]['n_samples'] for g in unique_genes])

    # Weighted by gene size
    weighted_mse = float(np.average(gene_mses, weights=gene_sizes))

    print(f"\n  Per-gene MSE statistics:")
    print(f"    Mean (unweighted):  {np.mean(gene_mses):.4f}")
    print(f"    Mean (weighted):    {weighted_mse:.4f}")
    print(f"    Median:             {np.median(gene_mses):.4f}")
    print(f"    Std:                {np.std(gene_mses):.4f}")
    print(f"    Min:                {np.min(gene_mses):.4f} ({unique_genes[np.argmin(gene_mses)]})")
    print(f"    Max:                {np.max(gene_mses):.4f} ({unique_genes[np.argmax(gene_mses)]})")

    # ── Shared vs test-only genes ─────────────────────────────────────────
    train_gene_set = set(data['train_genes'])
    test_gene_set = set(data['test_genes'])
    shared_genes = train_gene_set & test_gene_set
    test_only_genes = test_gene_set - train_gene_set

    print(f"\n  Gene overlap:")
    print(f"    Shared genes: {len(shared_genes)}")
    print(f"    Test-only genes: {test_only_genes}")

    # Standard test-set predictions (for shared vs test-only comparison)
    # Retrain on all training data (like the standard pipeline)
    print(f"\n  Retraining standard model for shared/test-only comparison...")
    scaler = StandardScaler()
    X_tr_std = scaler.fit_transform(data['X_train'])
    X_te_std = scaler.transform(data['X_test'])
    pred_std = train_predict(X_tr_std, data['y_train_bins'], X_te_std, data['dim'])
    std_mse = mse(data['y_test_wa'], pred_std)

    shared_mask = np.array([g in shared_genes for g in data['test_genes']])
    testonly_mask = np.array([g in test_only_genes for g in data['test_genes']])

    if shared_mask.sum() > 0:
        shared_mse = mse(data['y_test_wa'][shared_mask], pred_std[shared_mask])
        print(f"    Shared genes MSE:    {shared_mse:.4f} (n={shared_mask.sum()})")
    if testonly_mask.sum() > 0:
        testonly_mse = mse(data['y_test_wa'][testonly_mask], pred_std[testonly_mask])
        print(f"    Test-only genes MSE: {testonly_mse:.4f} (n={testonly_mask.sum()})")
    print(f"    Overall standard MSE: {std_mse:.4f} (n={len(data['y_test_wa'])}, 1-seed)")

    # ── Save results ──────────────────────────────────────────────────────
    elapsed_min = (time.time() - t_total) / 60
    results = {
        'logo_cv': {
            'overall_mse': logo_mse,
            'bootstrap_ci': logo_ci,
            'n_samples': len(y_all_wa),
            'n_genes': n_genes,
            'seed': SEED,
        },
        'per_gene': gene_results,
        'gene_mse_stats': {
            'mean_unweighted': float(np.mean(gene_mses)),
            'mean_weighted': weighted_mse,
            'median': float(np.median(gene_mses)),
            'std': float(np.std(gene_mses)),
            'min': float(np.min(gene_mses)),
            'max': float(np.max(gene_mses)),
            'min_gene': str(unique_genes[np.argmin(gene_mses)]),
            'max_gene': str(unique_genes[np.argmax(gene_mses)]),
        },
        'standard_comparison': {
            'standard_test_mse_comparator_1seed': STANDARD_TEST_MSE,
            'standard_test_mse_1seed': std_mse,
            'shared_genes_mse': shared_mse if shared_mask.sum() > 0 else None,
            'shared_genes_n': int(shared_mask.sum()),
            'testonly_genes_mse': testonly_mse if testonly_mask.sum() > 0 else None,
            'testonly_genes_n': int(testonly_mask.sum()),
            'testonly_gene_names': sorted(list(test_only_genes)),
        },
        'ratio_logo_to_standard': logo_mse / STANDARD_TEST_MSE,
        'elapsed_minutes': elapsed_min,
    }

    def convert(obj):
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)): return [convert(i) for i in obj]
        elif isinstance(obj, (np.integer,)): return int(obj)
        elif isinstance(obj, (np.floating,)): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    with open(RESULTS_DIR / 'gene_stratified_evaluation_results.json', 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\n  Results saved to gene_stratified_evaluation_results.json")

    # ── Figure ────────────────────────────────────────────────────────────
    # Canonical Tufte figure generated by scripts/replot_figures.py (replot_17)

    print(f"\nDone in {elapsed_min:.1f} min.")
    return results


def make_figure(gene_results, unique_genes, logo_mse, logo_ci):
    """Gene-stratified evaluation figure."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    gene_mses = np.array([gene_results[g]['mse'] for g in unique_genes])
    gene_sizes = np.array([gene_results[g]['n_samples'] for g in unique_genes])

    # ── (A) Per-gene MSE histogram ────────────────────────────────────────
    ax = axes[0]
    ax.hist(gene_mses, bins=25, color='#2E6B8A', edgecolor='white', alpha=0.85)
    ax.axvline(x=logo_mse, color='red', linewidth=1.5, linestyle='-',
               label=f'LOGO MSE = {logo_mse:.3f}')
    ax.axvline(x=STANDARD_TEST_MSE, color='green', linewidth=1.5, linestyle='--',
               label=f'Standard MSE = {STANDARD_TEST_MSE}')
    ax.set_xlabel('Per-Gene MSE', fontsize=9)
    ax.set_ylabel('Number of Genes', fontsize=9)
    ax.legend(fontsize=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(-0.15, 1.05, 'A', transform=ax.transAxes, fontsize=12, fontweight='bold')

    # ── (B) MSE vs gene size ──────────────────────────────────────────────
    ax = axes[1]
    ax.scatter(gene_sizes, gene_mses, s=15, alpha=0.6, color='#2E6B8A', edgecolors='none')
    ax.axhline(y=logo_mse, color='red', linewidth=0.8, linestyle='-', alpha=0.7)
    ax.set_xlabel('Gene Sample Size', fontsize=9)
    ax.set_ylabel('Per-Gene MSE', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(-0.15, 1.05, 'B', transform=ax.transAxes, fontsize=12, fontweight='bold')

    # ── (C) LOGO vs Standard comparison bar ───────────────────────────────
    ax = axes[2]
    bars = ax.bar([0, 1], [STANDARD_TEST_MSE, logo_mse],
                  color=['#4A9E6D', '#C45B3E'], width=0.5, edgecolor='none')
    # CI error bar on LOGO
    ci_lo = logo_ci['mse']['ci_lo']
    ci_hi = logo_ci['mse']['ci_hi']
    ax.errorbar(1, logo_mse, yerr=[[logo_mse - ci_lo], [ci_hi - logo_mse]],
                fmt='none', ecolor='black', capsize=5, capthick=1, linewidth=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Standard\n(5-seed)', 'LOGO CV\n(1-seed)'], fontsize=9)
    ax.set_ylabel('Test MSE', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(-0.15, 1.05, 'C', transform=ax.transAxes, fontsize=12, fontweight='bold')

    # Add value labels on bars
    for bar, val in zip(bars, [STANDARD_TEST_MSE, logo_mse]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    fig_path = FIGURES_DIR / 'gene_stratified_evaluation.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved to {fig_path}")


if __name__ == '__main__':
    main()
