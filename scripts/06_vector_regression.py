#!/usr/bin/env python3
"""
Script 06: Vector Regression with Focal Loss

Predicts 10-dimensional bin probability distributions instead of scalar WA.
Matches Schrier's approach (MSE 0.95 in Wolfram).

Architecture (matching Schrier's Wolfram network 4):
  - Dense(256) -> LeakyReLU -> Dropout(0.2)
  - Dense(256) -> LeakyReLU -> Dropout(0.2)
  - Dense(10, softmax)
  - No BatchNorm

Loss functions tested:
  1. Categorical cross-entropy (baseline)
  2. Focal loss (alpha=0.25, gamma=2.0)

WA conversion: wa_pred = predicted_probs @ [1, 2, ..., 10]

5-seed ensemble per embedding type per loss function.
"""
import sys
import json
import time
import numpy as np
from pathlib import Path

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_loading import load_plm_with_bins, BIN_COLUMNS
from src.models import SignalPeptideVectorNN
from src.evaluation import compute_metrics, format_metrics

from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Configuration ──────────────────────────────────────────────────────────
SEEDS = [42, 123, 456, 789, 1024]
N_SEEDS = len(SEEDS)
VAL_FRACTION = 0.2
BIN_CENTERS = np.arange(1, 11)  # [1, 2, 3, ..., 10]

EMBEDDING_TYPES = {
    'ESM2-650M': 'esm2-650M',
    'ESM2-3B': 'esm2-3B',
    'Ginkgo-AA0': 'ginkgo-AA0-650M',
}

LOSS_TYPES = ['categorical_crossentropy', 'focal']

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results'
FIGURES_DIR = Path(__file__).resolve().parent.parent / 'figures'


def bins_to_wa(probs):
    """Convert 10-dim bin probabilities to weighted average."""
    return probs @ BIN_CENTERS


def verify_bin_integrity(y_bins, y_wa, label):
    """Verify bin probabilities are valid and report WA consistency."""
    # Check sums
    sums = y_bins.sum(axis=1)
    max_sum_err = np.max(np.abs(sums - 1.0))
    assert max_sum_err < 0.01, f"{label}: bin sums deviate by {max_sum_err:.4f} (max)"
    print(f"  {label}: bin sum max deviation = {max_sum_err:.6f} (OK, < 0.01)")

    # Check WA consistency (informational — WA was computed from raw reads,
    # not from binned percentages, so some discrepancy is expected)
    wa_from_bins = bins_to_wa(y_bins)
    wa_diff = np.abs(wa_from_bins - y_wa)
    max_wa_err = np.max(wa_diff)
    mean_wa_err = np.mean(wa_diff)
    median_wa_err = np.median(wa_diff)
    n_close = np.sum(wa_diff < 0.05)
    print(f"  {label}: WA from bins — median diff = {median_wa_err:.4f}, "
          f"mean = {mean_wa_err:.4f}, max = {max_wa_err:.4f}, "
          f"exact match (<0.05): {n_close}/{len(y_wa)} ({100*n_close/len(y_wa):.1f}%)")


def train_ensemble(X_train, y_train_bins, X_test, loss_type):
    """Train 5-seed ensemble and return averaged test predictions.

    The StandardScaler is fit per fold (on each seed's training partition
    only) so the validation rows used for early stopping never leak into
    the feature normalization.
    """
    all_preds = []

    for i, seed in enumerate(SEEDS):
        rng = np.random.default_rng(seed)
        n = len(y_train_bins)
        indices = rng.permutation(n)
        n_val = int(n * VAL_FRACTION)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        # Fit scaler inside fold to avoid validation data leakage
        fold_scaler = StandardScaler()
        fold_scaler.fit(X_train[train_idx])
        X_tr = fold_scaler.transform(X_train[train_idx])
        X_val = fold_scaler.transform(X_train[val_idx])
        X_te = fold_scaler.transform(X_test)

        model = SignalPeptideVectorNN(
            hidden_layers=(256, 256),
            dropout=0.2,
            learning_rate=5e-4,
            batch_size=32,
            epochs=200,
            loss=loss_type,
            random_state=seed,
        )
        model.fit(
            X_tr, y_train_bins[train_idx],
            X_val=X_val, y_val_bins=y_train_bins[val_idx],
            verbose=0,
        )

        pred = model.predict(X_te)
        all_preds.append(pred)

        print(f"    Seed {seed} ({i+1}/{N_SEEDS}) done")

    avg_preds = np.mean(all_preds, axis=0)
    return avg_preds, all_preds


def validate_predictions(probs, label):
    """Assert predicted probabilities are valid."""
    assert np.all(probs >= -1e-6), f"{label}: negative probabilities found"
    sums = probs.sum(axis=1)
    max_sum_err = np.max(np.abs(sums - 1.0))
    assert max_sum_err < 0.01, f"{label}: predicted prob sums deviate by {max_sum_err:.4f}"
    print(f"    {label}: predictions valid (sums within {max_sum_err:.6f} of 1.0)")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    # ── 1. Load data and verify bin integrity ──────────────────────────────
    print("="*60)
    print("  Loading data and verifying bin integrity")
    print("="*60)

    all_data = {}
    for emb_name, model_name in EMBEDDING_TYPES.items():
        print(f"\n  Loading {emb_name}...")
        X_train, X_test, y_train_wa, y_test_wa, y_train_bins, y_test_bins, meta = \
            load_plm_with_bins(model_name)
        print(f"    Train: {meta['n_train']} samples, {meta['n_features']}d"
              f" (dropped {meta.get('n_train_dropped_nan_bins', 0)} with NaN bins)")
        print(f"    Test:  {meta['n_test']} samples"
              f" (dropped {meta.get('n_test_dropped_nan_bins', 0)} with NaN bins)")

        verify_bin_integrity(y_train_bins, y_train_wa, f"{emb_name} train")
        verify_bin_integrity(y_test_bins, y_test_wa, f"{emb_name} test")

        all_data[emb_name] = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train_wa': y_train_wa,
            'y_test_wa': y_test_wa,
            'y_train_bins': y_train_bins,
            'y_test_bins': y_test_bins,
            'meta': meta,
        }

    # ── 2. Train and evaluate ──────────────────────────────────────────────
    for emb_name, data in all_data.items():
        X_train = data['X_train']
        X_test = data['X_test']
        y_train_wa = data['y_train_wa']
        y_test_wa = data['y_test_wa']
        y_train_bins = data['y_train_bins']
        y_test_bins = data['y_test_bins']

        for loss_type in LOSS_TYPES:
            key = f"{emb_name}_{loss_type}"
            print(f"\n{'='*60}")
            print(f"  {emb_name} + {loss_type}")
            print(f"{'='*60}")

            t0 = time.time()
            avg_preds, all_preds = train_ensemble(
                X_train, y_train_bins, X_test, loss_type
            )
            elapsed = time.time() - t0

            # Validate predictions
            validate_predictions(avg_preds, f"{emb_name}+{loss_type} ensemble")

            # Convert to WA
            wa_pred_ensemble = bins_to_wa(avg_preds)

            # Per-seed MSEs
            seed_mses = []
            for i, pred in enumerate(all_preds):
                wa_pred_i = bins_to_wa(pred)
                mse_i = float(np.mean((y_test_wa - wa_pred_i) ** 2))
                seed_mses.append(mse_i)
                print(f"    Seed {SEEDS[i]} test MSE: {mse_i:.4f}")

            # Ensemble metrics
            metrics = compute_metrics(y_test_wa, wa_pred_ensemble)
            print(f"\n  Ensemble ({N_SEEDS} seeds):")
            print(format_metrics(metrics, prefix=f'{emb_name} + {loss_type}'))

            results[key] = {
                'embedding': emb_name,
                'loss': loss_type,
                'n_features': data['meta']['n_features'],
                'n_train': data['meta']['n_train'],
                'n_test': data['meta']['n_test'],
                'test_metrics': metrics,
                'seed_mses': seed_mses,
                'seeds': SEEDS,
                'elapsed_seconds': round(elapsed, 1),
                'architecture': {
                    'hidden_layers': [256, 256],
                    'dropout': 0.2,
                    'learning_rate': 5e-4,
                    'batch_size': 32,
                    'activation': 'LeakyReLU',
                    'output': 'softmax(10)',
                },
            }

    # ── 3. Summary table ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SUMMARY: Vector Regression Results")
    print(f"{'='*60}")
    print(f"  {'Config':<30} {'Test MSE':>10} {'Test R²':>8} {'Spearman':>10} {'Time':>8}")
    print(f"  {'-'*30} {'-'*10} {'-'*8} {'-'*10} {'-'*8}")

    for key, r in results.items():
        m = r['test_metrics']
        print(f"  {key:<30} {m['mse']:>10.4f} {m['r2']:>8.4f} "
              f"{m['spearman_rho']:>10.4f} {r['elapsed_seconds']:>7.1f}s")

    print(f"\n  prior single-run (retracted): MSE = 0.95")
    print(f"  Script 03 best (NN Ginkgo-AA0): MSE = 1.0497")

    # ── 4. Figure ──────────────────────────────────────────────────────────
    _make_figure(results)

    # ── 5. Save JSON ───────────────────────────────────────────────────────
    out_path = RESULTS_DIR / 'vector_regression_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


def _make_figure(results):
    """Generate comparison bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Group by embedding type
    emb_names = list(EMBEDDING_TYPES.keys())
    n_emb = len(emb_names)

    # For each embedding: CE, Focal, plus reference lines
    x = np.arange(n_emb)
    width = 0.3

    ce_mses = []
    focal_mses = []
    for emb_name in emb_names:
        ce_key = f"{emb_name}_categorical_crossentropy"
        focal_key = f"{emb_name}_focal"
        ce_mses.append(results[ce_key]['test_metrics']['mse'])
        focal_mses.append(results[focal_key]['test_metrics']['mse'])

    bars_ce = ax.bar(x - width/2, ce_mses, width, label='Cross-Entropy', color='steelblue', alpha=0.85)
    bars_focal = ax.bar(x + width/2, focal_mses, width, label='Focal Loss', color='darkorange', alpha=0.85)

    # Reference lines
    ax.axhline(y=1.0497, color='gray', linewidth=1.0, linestyle='--', alpha=0.8,
               label='NN Regression Best (1.05)')
    ax.axhline(y=0.95, color='green', linewidth=1.0, linestyle=':', alpha=0.8,
               label='prior single-run (0.95, retracted)')

    # Labels
    ax.set_ylabel('Test MSE')
    ax.set_xticks(x)
    ax.set_xticklabels(emb_names)
    ax.legend(loc='upper right')

    ax.set_ylim(0, max(ce_mses + focal_mses) * 1.15)

    plt.tight_layout()
    fig_path = FIGURES_DIR / 'vector_regression_comparison.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to {fig_path}")


if __name__ == '__main__':
    main()
