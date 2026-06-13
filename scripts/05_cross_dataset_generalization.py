#!/usr/bin/env python3
"""
Script 05: Cross-Dataset Generalization Analysis

Evaluates whether models trained on Grasso et al. data generalize to external
signal peptide datasets with different organisms, promoters, and activity scales.

Strategy:
  - Train best RF and NN on Grasso ESM2-650M embeddings (the shared representation)
  - Predict on 4 external datasets (Wu, Xue, Zhang-P43, Zhang-PglVM)
  - Use Spearman rank correlation as primary metric (scale-invariant)
  - Wu is binary (0/1) → also compute AUC-ROC
  - Zhang datasets share the same 114 sequences with different promoters →
    compare predicted rankings to test promoter-independence

External datasets:
  - Wu et al.:         81 SPs, binary WA (0/1), B. subtilis
  - Xue et al.:       322 SPs, WA 0–10437, continuous
  - Zhang et al. P43: 114 SPs, WA 0–193, B. subtilis P43 promoter
  - Zhang et al. PglVM: 114 SPs, WA 0–327, same SPs, PglVM promoter

All external datasets have ESM2-650M embeddings only (1280d).
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy import stats

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import SignalPeptideRegressorNN
from src.evaluation import compute_metrics, format_metrics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Configuration ──────────────────────────────────────────────────────────
RANDOM_STATE = 42
N_SEEDS = 5
SEEDS = [42, 123, 456, 789, 1024]
DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results'
FIGURES_DIR = Path(__file__).resolve().parent.parent / 'figures'

# Best ESM2-650M RF hyperparameters (from Script 02, 100-iter search)
BEST_RF_PARAMS = dict(
    n_estimators=300,
    max_depth=25,
    min_samples_split=0.001,
    min_samples_leaf=4,
    max_features=0.5,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# Best ESM2-650M NN hyperparameters (from Script 03, 40-config search)
BEST_NN_PARAMS = dict(
    hidden_layers=(256, 128),
    dropout=0.4,
    l2_reg=0.01,
    learning_rate=1e-3,
    batch_size=64,
    epochs=200,
    random_state=RANDOM_STATE,
)

# External datasets
EXTERNAL_DATASETS = {
    'Wu': {
        'file': 'wu_esm_embeddings.parquet',
        'is_binary': True,
        'description': '81 SPs, binary WA (functional/non-functional)',
    },
    'Xue': {
        'file': 'xue_esm_embeddings.parquet',
        'is_binary': False,
        'description': '322 SPs, WA 0–10437 (enzyme activity units)',
    },
    'Zhang-P43': {
        'file': 'zhang_p43_esm_embeddings.parquet',
        'is_binary': False,
        'description': '114 SPs, WA 0–193 (P43 promoter)',
    },
    'Zhang-PglVM': {
        'file': 'zhang_pglvm_esm_embeddings.parquet',
        'is_binary': False,
        'description': '114 SPs, WA 0–327 (PglVM promoter)',
    },
}


def load_external_dataset(filename):
    """Load an external ESM embedding parquet file."""
    path = DATA_DIR / filename
    df = pd.read_parquet(path)
    X = np.stack(df['embedding'].values)
    y = df['WA'].values.astype(np.float64)
    sequences = df['sequence'].values if 'sequence' in df.columns else None
    return X, y, sequences


def evaluate_on_external(y_true, y_pred, is_binary=False):
    """Compute generalization metrics (scale-invariant)."""
    result = {}

    # Spearman rank correlation (primary metric — scale-invariant)
    sp_rho, sp_p = stats.spearmanr(y_true, y_pred)
    result['spearman_rho'] = float(sp_rho)
    result['spearman_p'] = float(sp_p)

    # Pearson correlation
    pe_r, pe_p = stats.pearsonr(y_true, y_pred)
    result['pearson_r'] = float(pe_r)
    result['pearson_p'] = float(pe_p)

    result['n_samples'] = len(y_true)

    # For binary datasets, also compute AUC-ROC
    if is_binary:
        auc = roc_auc_score(y_true, y_pred)
        result['auc_roc'] = float(auc)

    return result


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(RANDOM_STATE)

    # ── 1. Load Grasso ESM2-650M training data ───────────────────────────
    print("Loading Grasso ESM2-650M train/test data...")
    df_train = pd.read_parquet(DATA_DIR / 'trainAA_esm2-650M.parquet')
    df_test = pd.read_parquet(DATA_DIR / 'testAA_esm2-650M.parquet')

    X_train_raw = np.stack(df_train['embedding'].values)
    X_test_raw = np.stack(df_test['embedding'].values)
    y_train = df_train['WA'].values.astype(np.float64)
    y_test = df_test['WA'].values.astype(np.float64)

    print(f"  Train: {X_train_raw.shape[0]} samples, {X_train_raw.shape[1]}d")
    print(f"  Test:  {X_test_raw.shape[0]} samples")

    # ── 2. Preprocess (StandardScaler, no log transform) ─────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # ── 3. Train RF ──────────────────────────────────────────────────────
    print(f"\nTraining RF with best ESM2-650M params...")
    rf = RandomForestRegressor(**BEST_RF_PARAMS)
    rf.fit(X_train, y_train)

    # Sanity check on Grasso test set
    y_pred_rf_test = rf.predict(X_test)
    rf_test_metrics = compute_metrics(y_test, y_pred_rf_test)
    print(f"  RF Grasso test MSE: {rf_test_metrics['mse']:.4f} (expected ~1.19)")

    # ── 4. Train NN ensemble (multi-seed) ───────────────────────────────
    print(f"\nTraining {N_SEEDS}-seed NN ensemble with best ESM2-650M params...")
    nn_models = []
    nn_test_preds_each = []

    for i, seed in enumerate(SEEDS):
        print(f"  Seed {seed} ({i+1}/{N_SEEDS})...", end=' ')
        rng = np.random.default_rng(seed)
        n = len(y_train)
        indices = rng.permutation(n)
        n_val = int(n * 0.2)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        nn_params = {**BEST_NN_PARAMS, 'random_state': seed}
        nn = SignalPeptideRegressorNN(**nn_params)
        nn.fit(
            X_train[train_idx], y_train[train_idx],
            X_val=X_train[val_idx], y_val=y_train[val_idx],
            verbose=0,
        )
        nn_models.append(nn)

        pred = nn.predict(X_test)
        nn_test_preds_each.append(pred)
        seed_mse = float(np.mean((y_test - pred) ** 2))
        print(f"test MSE = {seed_mse:.4f}")

    # Average predictions across seeds
    y_pred_nn_test = np.mean(nn_test_preds_each, axis=0)
    nn_test_metrics = compute_metrics(y_test, y_pred_nn_test)
    print(f"  Ensemble NN Grasso test MSE: {nn_test_metrics['mse']:.4f} (averaged over {N_SEEDS} seeds)")

    # ── 5. Evaluate on external datasets ─────────────────────────────────
    print(f"\n{'='*60}")
    print("  Cross-Dataset Generalization Results")
    print(f"{'='*60}")

    results = {
        'grasso_sanity_check': {
            'rf_test_mse': rf_test_metrics['mse'],
            'nn_test_mse': nn_test_metrics['mse'],
            'nn_seeds': SEEDS,
            'nn_n_seeds': N_SEEDS,
        },
        'external_datasets': {},
    }

    all_external_results = []

    for name, spec in EXTERNAL_DATASETS.items():
        print(f"\n--- {name}: {spec['description']} ---")
        X_ext, y_ext, seq_ext = load_external_dataset(spec['file'])
        X_ext_scaled = scaler.transform(X_ext)

        # RF predictions
        y_pred_rf = rf.predict(X_ext_scaled)
        rf_metrics = evaluate_on_external(y_ext, y_pred_rf, spec['is_binary'])

        # NN predictions (ensemble average)
        y_pred_nn = np.mean([m.predict(X_ext_scaled) for m in nn_models], axis=0)
        nn_metrics = evaluate_on_external(y_ext, y_pred_nn, spec['is_binary'])

        rf_sig = '*' if rf_metrics['spearman_p'] < 0.05 else ' '
        nn_sig = '*' if nn_metrics['spearman_p'] < 0.05 else ' '

        print(f"  RF  Spearman: {rf_metrics['spearman_rho']:+.3f} (p={rf_metrics['spearman_p']:.3f}){rf_sig}  "
              f"Pearson: {rf_metrics['pearson_r']:+.3f}  "
              f"n={rf_metrics['n_samples']}", end='')
        if 'auc_roc' in rf_metrics:
            print(f"  AUC-ROC: {rf_metrics['auc_roc']:.3f}", end='')
        print()

        print(f"  NN  Spearman: {nn_metrics['spearman_rho']:+.3f} (p={nn_metrics['spearman_p']:.3f}){nn_sig}  "
              f"Pearson: {nn_metrics['pearson_r']:+.3f}  "
              f"n={nn_metrics['n_samples']}", end='')
        if 'auc_roc' in nn_metrics:
            print(f"  AUC-ROC: {nn_metrics['auc_roc']:.3f}", end='')
        print()

        dataset_result = {
            'description': spec['description'],
            'is_binary': spec['is_binary'],
            'n_samples': len(y_ext),
            'wa_range': [float(y_ext.min()), float(y_ext.max())],
            'rf_metrics': rf_metrics,
            'nn_metrics': nn_metrics,
            'rf_predictions': y_pred_rf.tolist(),
            'nn_predictions': y_pred_nn.tolist(),
            'actual_wa': y_ext.tolist(),
        }
        results['external_datasets'][name] = dataset_result
        all_external_results.append((name, rf_metrics, nn_metrics, spec))

    # ── 6. Zhang cross-promoter analysis ─────────────────────────────────
    # The P43 and PglVM files contain the same 114 sequences in DIFFERENT
    # row order. We must match by sequence before comparing predictions.
    print(f"\n{'='*60}")
    print("  Zhang Cross-Promoter Consistency Analysis")
    print(f"{'='*60}")

    df_p43 = pd.read_parquet(DATA_DIR / 'zhang_p43_esm_embeddings.parquet')
    df_pglvm = pd.read_parquet(DATA_DIR / 'zhang_pglvm_esm_embeddings.parquet')

    # Predict on each (re-use scaler from above)
    X_p43 = scaler.transform(np.stack(df_p43['embedding'].values))
    X_pglvm = scaler.transform(np.stack(df_pglvm['embedding'].values))

    df_p43['pred_rf'] = rf.predict(X_p43)
    df_p43['pred_nn'] = np.mean([m.predict(X_p43) for m in nn_models], axis=0)
    df_pglvm['pred_rf'] = rf.predict(X_pglvm)
    df_pglvm['pred_nn'] = np.mean([m.predict(X_pglvm) for m in nn_models], axis=0)

    # Merge on sequence to align rows
    merged = df_p43[['sequence', 'WA', 'pred_rf', 'pred_nn']].merge(
        df_pglvm[['sequence', 'WA', 'pred_rf', 'pred_nn']],
        on='sequence', suffixes=('_p43', '_pglvm'),
    )
    print(f"  Matched sequences: {len(merged)} / 114")

    # Predictions should be nearly identical (same sequence → same embedding)
    rf_rank_corr, _ = stats.spearmanr(merged['pred_rf_p43'], merged['pred_rf_pglvm'])
    nn_rank_corr, _ = stats.spearmanr(merged['pred_nn_p43'], merged['pred_nn_pglvm'])
    print(f"  RF prediction consistency (P43 vs PglVM): Spearman = {rf_rank_corr:.4f}")
    print(f"  NN prediction consistency (P43 vs PglVM): Spearman = {nn_rank_corr:.4f}")

    # Actual WA correlation between promoters (sequence-matched)
    actual_cross_corr, _ = stats.spearmanr(merged['WA_p43'], merged['WA_pglvm'])
    print(f"  Actual WA cross-promoter correlation:      Spearman = {actual_cross_corr:.4f}")
    print(f"  (Same sequences → predictions should be ~1.0; actual WA shows promoter transfer)")

    results['zhang_cross_promoter'] = {
        'rf_prediction_consistency': float(rf_rank_corr),
        'nn_prediction_consistency': float(nn_rank_corr),
        'actual_wa_correlation': float(actual_cross_corr),
        'n_matched': len(merged),
        'note': 'Same 114 sequences measured under P43 and PglVM promoters. '
                'Model predictions depend only on SP sequence embedding, so '
                'prediction consistency should be ~1.0. The actual WA correlation '
                'indicates how much SP efficiency transfers across promoters.',
    }

    # ── 7. Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Dataset':<15} {'N':>4} {'RF Spearman':>12} {'p':>8} {'NN Spearman':>12} {'p':>8}")
    print(f"  {'-'*15} {'-'*4} {'-'*12} {'-'*8} {'-'*12} {'-'*8}")
    for name, rf_m, nn_m, spec in all_external_results:
        print(f"  {name:<15} {rf_m['n_samples']:>4} {rf_m['spearman_rho']:>+12.3f} {rf_m['spearman_p']:>8.3f} "
              f"{nn_m['spearman_rho']:>+12.3f} {nn_m['spearman_p']:>8.3f}")

    # ── 8. Figure ────────────────────────────────────────────────────────
    _make_figure(all_external_results, results, merged)

    # ── 9. Save ──────────────────────────────────────────────────────────
    # Remove prediction arrays for cleaner JSON (they're large)
    results_clean = json.loads(json.dumps(results))
    for ds in results_clean['external_datasets'].values():
        ds.pop('rf_predictions', None)
        ds.pop('nn_predictions', None)
        ds.pop('actual_wa', None)

    out_path = RESULTS_DIR / 'cross_dataset_results.json'
    with open(out_path, 'w') as f:
        json.dump(results_clean, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


def _make_figure(all_external_results, results, zhang_merged):
    """Generate cross-dataset generalization figure."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Panel A: Spearman correlations by dataset ────────────────────────
    ax = axes[0]
    datasets = [name for name, _, _, _ in all_external_results]
    rf_spearman = [rf_m['spearman_rho'] for _, rf_m, _, _ in all_external_results]
    nn_spearman = [nn_m['spearman_rho'] for _, _, nn_m, _ in all_external_results]

    x = np.arange(len(datasets))
    width = 0.35

    bars_rf = ax.bar(x - width/2, rf_spearman, width, label='RF', color='steelblue', alpha=0.85)
    bars_nn = ax.bar(x + width/2, nn_spearman, width, label='NN', color='darkorange', alpha=0.85)

    ax.set_ylabel('Spearman Rank Correlation')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim(-0.5, 0.5)
    ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='--')

    # ── Panel B: Zhang cross-promoter scatter (sequence-matched) ─────────
    ax = axes[1]
    from scipy.stats import rankdata

    # Use the sequence-matched merged dataframe
    p43_wa = zhang_merged['WA_p43'].values
    pglvm_wa = zhang_merged['WA_pglvm'].values

    p43_wa_ranks = rankdata(p43_wa)
    pglvm_wa_ranks = rankdata(pglvm_wa)
    pred_rf_ranks = rankdata(zhang_merged['pred_rf_p43'].values)

    ax.scatter(p43_wa_ranks, pglvm_wa_ranks, alpha=0.5, s=30, color='steelblue',
               label='Actual WA: P43 vs PglVM')
    ax.scatter(p43_wa_ranks, pred_rf_ranks, alpha=0.5, s=30, color='darkorange', marker='^',
               label=f'RF predicted vs P43 actual')

    lims = [0, len(p43_wa) + 1]
    ax.plot(lims, lims, 'k--', lw=0.8, alpha=0.5)
    ax.set_xlabel('Zhang-P43 Actual WA Rank')
    ax.set_ylabel('Rank')
    ax.text(0.05, 0.95, '(B)', transform=ax.transAxes, fontsize=10, va='top')
    ax.legend(fontsize=8, loc='upper left')

    plt.tight_layout()

    fig_path = FIGURES_DIR / 'cross_dataset_generalization.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to {fig_path}")


if __name__ == '__main__':
    main()
