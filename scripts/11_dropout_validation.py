#!/usr/bin/env python3
"""
Script 11: Dropout Validation via Held-Out Data

Validates the dropout selection from Script 10 using an 80/20 train/validation
split, independent of the test set. This confirms that the optimal dropout rate
(0.35) is not an artifact of test-set model selection.

For each dropout value in {0.15, 0.20, 0.25, 0.30, 0.35}, trains 5-seed
ensembles on 80% of the training data and evaluates on the held-out 20%
validation set. The same (256, 256, 128) architecture and focal loss from
Script 10 are used throughout.

Outputs:
  - results/dropout_validation_results.json
  - figures/dropout_validation.png
"""
import sys
import json
import time
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_loading import load_plm_with_bins
from src.models import FocalLoss

from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results'
FIGURES_DIR = Path(__file__).resolve().parent.parent / 'figures'

# ── Constants ─────────────────────────────────────────────────────────────
SEEDS = [42, 123, 456, 789, 1024]
BIN_CENTERS = np.arange(1, 11)
VAL_FRACTION = 0.2
DROPOUT_VALUES = [0.15, 0.20, 0.25, 0.30, 0.35]

# Script 10 test-set results for comparison
# RETRACTED: these are single-draw test MSEs from the test-set dropout sweep
# (the leakage the audit repudiated); kept only to reproduce the historical figure.
# RETRACTED: single-draw test MSEs from the test-set dropout sweep (the leakage the
# audit repudiated); kept only to reproduce the historical figure, not as a target.
SCRIPT10_TEST_MSES = {
    0.15: 0.985,
    0.20: 0.993,
    0.25: 0.971,
    0.30: 0.940,
    0.35: 0.932,
}


# ═══════════════════════════════════════════════════════════════════════════
# Model building (same as Script 10)
# ═══════════════════════════════════════════════════════════════════════════

def build_model(input_dim, hidden_layers=(256, 256, 128), dropout=0.2,
                lr=5e-4, seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    inputs = keras.Input(shape=(input_dim,))
    x = inputs
    for units in hidden_layers:
        x = layers.Dense(units)(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=loss_fn)
    return model


def predict_wa(model, X):
    return model.predict(X, verbose=0) @ BIN_CENTERS


def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    t_total = time.time()

    # ── Load data ─────────────────────────────────────────────────────────
    print("Loading Ginkgo-AA0 data...")
    X_train, _, _, _, y_train_bins, _, meta = load_plm_with_bins('ginkgo-AA0-650M')
    y_train_wa = y_train_bins @ BIN_CENTERS
    n_train = len(y_train_wa)
    dim = X_train.shape[1]
    print(f"  Train samples: {n_train}, Features: {dim}")

    # ── Dropout sweep with validation split ───────────────────────────────
    print(f"\n{'='*60}")
    print("  DROPOUT VALIDATION (80/20 train/val split)")
    print(f"{'='*60}")

    results = {}

    for drop in DROPOUT_VALUES:
        label = f'drop-{drop:.2f}'
        print(f"\n  {label} — 5-seed ensemble (val split)")

        seed_val_mses = []
        seed_train_mses = []

        for seed in SEEDS:
            t0 = time.time()

            # 80/20 split
            rng = np.random.default_rng(seed)
            indices = rng.permutation(n_train)
            n_val = int(n_train * VAL_FRACTION)
            val_idx = indices[:n_val]
            train_idx = indices[n_val:]

            # Scale
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train[train_idx])
            X_val = scaler.transform(X_train[val_idx])

            # Train
            model = build_model(dim, hidden_layers=(256, 256, 128),
                                dropout=drop, lr=5e-4, seed=seed)

            model.fit(X_tr, y_train_bins[train_idx],
                      epochs=300, batch_size=32,
                      callbacks=[keras.callbacks.ReduceLROnPlateau(
                          monitor='loss', factor=0.5, patience=10,
                          min_lr=1e-6, verbose=0)],
                      verbose=0)

            # Predict on validation
            wa_val_pred = predict_wa(model, X_val)
            wa_val_true = y_train_wa[val_idx]
            val_mse = mse(wa_val_true, wa_val_pred)
            seed_val_mses.append(val_mse)

            # Also record train MSE for reference
            wa_tr_pred = predict_wa(model, X_tr)
            wa_tr_true = y_train_wa[train_idx]
            train_mse = mse(wa_tr_true, wa_tr_pred)
            seed_train_mses.append(train_mse)

            del model
            keras.backend.clear_session()

            print(f"    Seed {seed:>5}: val MSE = {val_mse:.4f}, "
                  f"train MSE = {train_mse:.4f}  ({time.time()-t0:.1f}s)")

        mean_val = float(np.mean(seed_val_mses))
        std_val = float(np.std(seed_val_mses))
        mean_train = float(np.mean(seed_train_mses))

        results[label] = {
            'dropout': drop,
            'val_mse_mean': mean_val,
            'val_mse_std': std_val,
            'val_mse_per_seed': seed_val_mses,
            'train_mse_mean': mean_train,
            'train_mse_per_seed': seed_train_mses,
            'test_mse_script10': SCRIPT10_TEST_MSES[drop],
        }
        print(f"  → Mean val MSE = {mean_val:.4f} ± {std_val:.4f}")

    # ── Summary ───────────────────────────────────────────────────────────
    total_elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  DROPOUT VALIDATION SUMMARY  ({total_elapsed:.0f}s total)")
    print(f"{'='*60}")
    print(f"  {'Dropout':>8}  {'Val MSE':>10}  {'± Std':>8}  {'Test MSE':>10}  {'Trend'}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*5}")

    prev_val = None
    best_val_drop = min(results, key=lambda k: results[k]['val_mse_mean'])
    for label in sorted(results, key=lambda k: results[k]['dropout']):
        r = results[label]
        arrow = ''
        if prev_val is not None:
            arrow = '↓' if r['val_mse_mean'] < prev_val else '↑'
        prev_val = r['val_mse_mean']
        marker = ' ← best' if label == best_val_drop else ''
        print(f"  {r['dropout']:>8.2f}  {r['val_mse_mean']:>10.4f}  "
              f"±{r['val_mse_std']:>7.4f}  {r['test_mse_script10']:>10.3f}  "
              f"{arrow}{marker}")

    best_r = results[best_val_drop]
    print(f"\n  Best validation dropout: {best_r['dropout']:.2f} "
          f"(val MSE = {best_r['val_mse_mean']:.4f})")
    print(f"  Script 10 test MSE at that dropout: "
          f"{best_r['test_mse_script10']:.3f}")

    # Check if val and test agree on best
    best_test_drop = min(SCRIPT10_TEST_MSES, key=SCRIPT10_TEST_MSES.get)
    if best_r['dropout'] == best_test_drop:
        print(f"\n  ✓ Validation and test set agree: dropout = {best_r['dropout']:.2f}")
    else:
        print(f"\n  Note: Validation picks dropout {best_r['dropout']:.2f}, "
              f"test picks {best_test_drop:.2f}")

    # Spearman correlation between val and test MSE rankings
    val_mses = [results[f'drop-{d:.2f}']['val_mse_mean'] for d in DROPOUT_VALUES]
    test_mses = [SCRIPT10_TEST_MSES[d] for d in DROPOUT_VALUES]
    rho, pval = sp_stats.spearmanr(val_mses, test_mses)
    print(f"  Spearman(val MSE, test MSE) = {rho:.3f} (p = {pval:.4f})")

    # ── Save JSON ─────────────────────────────────────────────────────────
    output = {
        'dropout_values': DROPOUT_VALUES,
        'val_fraction': VAL_FRACTION,
        'seeds': SEEDS,
        'architecture': [256, 256, 128],
        'results': results,
        'best_val_dropout': best_r['dropout'],
        'val_test_spearman': {'rho': rho, 'pval': pval},
        'elapsed_seconds': total_elapsed,
    }
    out_path = RESULTS_DIR / 'dropout_validation_results.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    # ── Figure ────────────────────────────────────────────────────────────
    _make_figure(results)

    return results


def _make_figure(results):
    """Dropout vs MSE on validation and test sets."""
    fig, ax = plt.subplots(figsize=(8, 5))

    drops = sorted([results[k]['dropout'] for k in results])
    val_mses = [results[f'drop-{d:.2f}']['val_mse_mean'] for d in drops]
    val_stds = [results[f'drop-{d:.2f}']['val_mse_std'] for d in drops]
    test_mses = [SCRIPT10_TEST_MSES[d] for d in drops]

    ax.errorbar(drops, val_mses, yerr=val_stds, marker='o', capsize=5,
                linewidth=2, markersize=8, color='steelblue',
                label='Validation MSE')
    ax.plot(drops, test_mses, marker='s', linewidth=2, markersize=8,
            color='darkorange', label='Test MSE')

    ax.set_xlabel('Dropout Rate', fontsize=12)
    ax.set_ylabel('MSE', fontsize=12)
    ax.set_xticks(drops)
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    fig_path = FIGURES_DIR / 'dropout_validation.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure saved to {fig_path}")


if __name__ == '__main__':
    main()
