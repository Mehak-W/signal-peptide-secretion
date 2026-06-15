#!/usr/bin/env python3
"""
Script 18: Linear Probe + Ridge + XGBoost Baselines

Establishes how much predictive power comes from the PLM embedding itself
vs the MLP head by comparing:

  1. Linear probe: Input(1280) → Dense(10, softmax) with focal loss, 5-seed ensemble
  2. Ridge regression: sklearn Ridge, alpha via 5-fold CV, predicts scalar WA
  3. XGBoost: XGBRegressor with hyperparameter search, predicts scalar WA

All evaluated on same Ginkgo-AA0 embeddings and 1326 test samples with bootstrap CIs.

Outputs:
  - results/linear_baseline_results.json
  - figures/linear_baseline.png
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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score

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

# ── Constants ─────────────────────────────────────────────────────────────
SEEDS = [42, 123, 456, 789, 1024]
BIN_CENTERS = np.arange(1, 11)
N_BOOTSTRAP = 10_000
BEST_NN_MSE = 0.957  # reproducible 5-seed value, full data (Section 3.4)
NET4_REFERENCE_MSE = 0.953  # single-run reference value for the optimization sweeps


def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def bootstrap_ci(y_true, y_pred, n_boot=N_BOOTSTRAP, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    mse_b = np.empty(n_boot)
    sp_b = np.empty(n_boot)
    r2_b = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        mse_b[b] = np.mean((yt - yp) ** 2)
        sp_b[b] = sp_stats.spearmanr(yt, yp).statistic
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2_b[b] = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    mse_pt = mse(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2_pt = 1 - ss_res / ss_tot

    return {
        'mse': {'point': mse_pt,
                'ci_lo': float(np.percentile(mse_b, 2.5)),
                'ci_hi': float(np.percentile(mse_b, 97.5))},
        'r2': {'point': float(r2_pt),
               'ci_lo': float(np.percentile(r2_b, 2.5)),
               'ci_hi': float(np.percentile(r2_b, 97.5))},
        'spearman': {'point': float(sp_stats.spearmanr(y_true, y_pred).statistic),
                     'ci_lo': float(np.percentile(sp_b, 2.5)),
                     'ci_hi': float(np.percentile(sp_b, 97.5))},
    }


# ═══════════════════════════════════════════════════════════════════════════
# 1. Linear Probe (softmax vector regression, no hidden layers)
# ═══════════════════════════════════════════════════════════════════════════

def run_linear_probe(X_tr, y_bins, X_te, y_test):
    """Linear probe: Input → Dense(10, softmax), focal loss, 5-seed ensemble."""
    print(f"\n{'='*60}")
    print("  1. LINEAR PROBE (Input → softmax(10), focal loss)")
    print(f"{'='*60}")

    preds = []
    for seed in SEEDS:
        t0 = time.time()
        tf.random.set_seed(seed)
        np.random.seed(seed)

        inputs = keras.Input(shape=(X_tr.shape[1],))
        outputs = layers.Dense(10, activation='softmax')(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-4),
                      loss=FocalLoss(alpha=0.25, gamma=2.0))

        callbacks = [keras.callbacks.ReduceLROnPlateau(
            monitor='loss', factor=0.5, patience=10, min_lr=1e-6, verbose=0)]
        model.fit(X_tr, y_bins, epochs=300, batch_size=32,
                  callbacks=callbacks, verbose=0)

        pred = model.predict(X_te, verbose=0) @ BIN_CENTERS
        preds.append(pred)
        mse_i = mse(y_test, pred)
        print(f"    Seed {seed:>5}: MSE = {mse_i:.4f}  ({time.time()-t0:.1f}s)")
        del model
        keras.backend.clear_session()

    y_ens = np.mean(preds, axis=0)
    mse_ens = mse(y_test, y_ens)
    ci = bootstrap_ci(y_test, y_ens)
    print(f"  -> Linear Probe Ensemble MSE = {mse_ens:.4f}")
    return y_ens, ci, [mse(y_test, p) for p in preds]


# ═══════════════════════════════════════════════════════════════════════════
# 2. Ridge Regression
# ═══════════════════════════════════════════════════════════════════════════

def run_ridge(X_tr, y_train_wa, X_te, y_test):
    """Ridge regression with alpha selection via 5-fold CV."""
    print(f"\n{'='*60}")
    print("  2. RIDGE REGRESSION (sklearn RidgeCV)")
    print(f"{'='*60}")

    t0 = time.time()
    alphas = np.logspace(-3, 5, 50)
    ridge = RidgeCV(alphas=alphas, cv=5, scoring='neg_mean_squared_error')
    ridge.fit(X_tr, y_train_wa)

    y_pred = ridge.predict(X_te)
    mse_val = mse(y_test, y_pred)
    ci = bootstrap_ci(y_test, y_pred)

    print(f"    Best alpha: {ridge.alpha_:.4f}")
    print(f"    Test MSE:   {mse_val:.4f}")
    print(f"    ({time.time()-t0:.1f}s)")

    return y_pred, ci, {'best_alpha': float(ridge.alpha_)}


# ═══════════════════════════════════════════════════════════════════════════
# 3. XGBoost
# ═══════════════════════════════════════════════════════════════════════════

def run_xgboost(X_tr, y_train_wa, X_te, y_test):
    """XGBoost with hyperparameter search."""
    print(f"\n{'='*60}")
    print("  3. XGBOOST (hyperparameter search)")
    print(f"{'='*60}")

    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("  XGBoost not installed. Skipping.")
        return None, None, None

    t0 = time.time()

    # Hyperparameter grid
    param_grid = [
        {'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.1,
         'subsample': 0.8, 'colsample_bytree': 0.8},
        {'n_estimators': 500, 'max_depth': 8, 'learning_rate': 0.05,
         'subsample': 0.8, 'colsample_bytree': 0.7},
        {'n_estimators': 300, 'max_depth': 10, 'learning_rate': 0.1,
         'subsample': 0.9, 'colsample_bytree': 0.8},
        {'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.05,
         'subsample': 0.7, 'colsample_bytree': 0.6},
        {'n_estimators': 1000, 'max_depth': 4, 'learning_rate': 0.01,
         'subsample': 0.8, 'colsample_bytree': 0.8},
        {'n_estimators': 300, 'max_depth': 25, 'learning_rate': 0.1,
         'subsample': 0.8, 'colsample_bytree': 0.5},
    ]

    best_mse = float('inf')
    best_params = None
    best_model = None

    for params in param_grid:
        xgb = XGBRegressor(**params, random_state=42, n_jobs=-1, verbosity=0)
        cv_scores = cross_val_score(xgb, X_tr, y_train_wa,
                                     cv=5, scoring='neg_mean_squared_error')
        cv_mse = -cv_scores.mean()
        print(f"    depth={params['max_depth']:>2}, n_est={params['n_estimators']:>4}, "
              f"lr={params['learning_rate']:.3f} → CV MSE = {cv_mse:.4f}")

        if cv_mse < best_mse:
            best_mse = cv_mse
            best_params = params.copy()

    # Retrain best model on full training data
    print(f"\n    Best params: depth={best_params['max_depth']}, "
          f"n_est={best_params['n_estimators']}, lr={best_params['learning_rate']:.3f}")
    best_model = XGBRegressor(**best_params, random_state=42, n_jobs=-1, verbosity=0)
    best_model.fit(X_tr, y_train_wa)

    y_pred = best_model.predict(X_te)
    test_mse = mse(y_test, y_pred)
    ci = bootstrap_ci(y_test, y_pred)

    print(f"    Test MSE: {test_mse:.4f}")
    print(f"    ({time.time()-t0:.1f}s)")

    return y_pred, ci, {'best_params': best_params, 'cv_mse': best_mse}


# ═══════════════════════════════════════════════════════════════════════════
# 4. Best NN (for direct comparison, same pipeline as Script 10)
# ═══════════════════════════════════════════════════════════════════════════

def run_best_nn(X_tr, y_bins, X_te, y_test):
    """Best vector NN: (256,256,128), dropout=0.35, focal, 5-seed."""
    print(f"\n{'='*60}")
    print("  4. BEST NN (256,256,128, drop=0.35, focal, 5-seed)")
    print(f"{'='*60}")

    preds = []
    for seed in SEEDS:
        t0 = time.time()
        tf.random.set_seed(seed)
        np.random.seed(seed)

        inputs = keras.Input(shape=(X_tr.shape[1],))
        x = inputs
        for units in (256, 256, 128):
            x = layers.Dense(units)(x)
            x = layers.LeakyReLU()(x)
            x = layers.Dropout(0.35)(x)
        outputs = layers.Dense(10, activation='softmax')(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-4),
                      loss=FocalLoss(alpha=0.25, gamma=2.0))

        callbacks = [keras.callbacks.ReduceLROnPlateau(
            monitor='loss', factor=0.5, patience=10, min_lr=1e-6, verbose=0)]
        model.fit(X_tr, y_bins, epochs=300, batch_size=32,
                  callbacks=callbacks, verbose=0)

        pred = model.predict(X_te, verbose=0) @ BIN_CENTERS
        preds.append(pred)
        mse_i = mse(y_test, pred)
        print(f"    Seed {seed:>5}: MSE = {mse_i:.4f}  ({time.time()-t0:.1f}s)")
        del model
        keras.backend.clear_session()

    y_ens = np.mean(preds, axis=0)
    mse_ens = mse(y_test, y_ens)
    ci = bootstrap_ci(y_test, y_ens)
    print(f"  -> Best NN Ensemble MSE = {mse_ens:.4f}")
    return y_ens, ci, [mse(y_test, p) for p in preds]


def make_figure(results):
    """Baseline comparison figure."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    models = ['linear_probe', 'ridge', 'xgboost', 'best_nn']
    labels = ['Linear\nProbe', 'Ridge', 'XGBoost', 'Best NN\n(256,256,128)']
    colors = ['#D4A843', '#4A9E6D', '#8B5CF6', '#2E6B8A']

    # Filter out None results
    valid = [(m, l, c) for m, l, c in zip(models, labels, colors) if results.get(m) is not None]
    models, labels, colors = zip(*valid)

    # ── (A) MSE comparison with CIs ──────────────────────────────────────
    ax = axes[0]
    mse_vals = [results[m]['bootstrap_ci']['mse']['point'] for m in models]
    ci_lo = [results[m]['bootstrap_ci']['mse']['ci_lo'] for m in models]
    ci_hi = [results[m]['bootstrap_ci']['mse']['ci_hi'] for m in models]
    yerr_lo = [v - lo for v, lo in zip(mse_vals, ci_lo)]
    yerr_hi = [hi - v for v, hi in zip(mse_vals, ci_hi)]

    x = np.arange(len(models))
    ax.bar(x, mse_vals, color=colors, width=0.6, edgecolor='none')
    ax.errorbar(x, mse_vals, yerr=[yerr_lo, yerr_hi], fmt='none', ecolor='black',
                capsize=5, capthick=1, linewidth=1)
    ax.axhline(y=NET4_REFERENCE_MSE, color='gray', linewidth=0.8, linestyle='--', alpha=0.7)
    ax.text(len(models)-0.6, NET4_REFERENCE_MSE + 0.01, 'net4 single run (~0.95)', fontsize=7, color='gray')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Test MSE', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(-0.15, 1.05, 'A', transform=ax.transAxes, fontsize=12, fontweight='bold')

    # Add value labels
    for xi, val in zip(x, mse_vals):
        ax.text(xi, val + max(yerr_hi) + 0.02, f'{val:.3f}',
                ha='center', va='bottom', fontsize=8)

    # ── (B) Spearman comparison with CIs ─────────────────────────────────
    ax = axes[1]
    sp_vals = [results[m]['bootstrap_ci']['spearman']['point'] for m in models]
    sp_lo = [results[m]['bootstrap_ci']['spearman']['ci_lo'] for m in models]
    sp_hi = [results[m]['bootstrap_ci']['spearman']['ci_hi'] for m in models]
    yerr_lo = [v - lo for v, lo in zip(sp_vals, sp_lo)]
    yerr_hi = [hi - v for v, hi in zip(sp_vals, sp_hi)]

    ax.bar(x, sp_vals, color=colors, width=0.6, edgecolor='none')
    ax.errorbar(x, sp_vals, yerr=[yerr_lo, yerr_hi], fmt='none', ecolor='black',
                capsize=5, capthick=1, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Spearman ρ', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(-0.15, 1.05, 'B', transform=ax.transAxes, fontsize=12, fontweight='bold')

    plt.tight_layout()
    fig_path = FIGURES_DIR / 'linear_baseline.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved to {fig_path}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    # ── Load data ─────────────────────────────────────────────────────────
    print("Loading Ginkgo-AA0 data...")
    X_train, _, y_train_wa, _, y_train_bins, _, meta = load_plm_with_bins('ginkgo-AA0-650M')
    _, X_test_full, _, y_test, meta_full = load_plm_embeddings('ginkgo-AA0-650M')

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test_full)
    dim = X_tr.shape[1]
    print(f"  Train: {meta['n_train']}, Test: {meta_full['n_test']}, Dim: {dim}")

    all_results = {}

    # ── Run all baselines ─────────────────────────────────────────────────
    y_lp, ci_lp, seed_mses_lp = run_linear_probe(X_tr, y_train_bins, X_te, y_test)
    all_results['linear_probe'] = {
        'bootstrap_ci': ci_lp,
        'seed_mses': seed_mses_lp,
        'description': 'Input(1280) → Dense(10, softmax), focal loss, 5-seed',
    }

    y_ridge, ci_ridge, info_ridge = run_ridge(X_tr, y_train_wa, X_te, y_test)
    all_results['ridge'] = {
        'bootstrap_ci': ci_ridge,
        'info': info_ridge,
        'description': 'Ridge regression (alpha via 5-fold CV)',
    }

    y_xgb, ci_xgb, info_xgb = run_xgboost(X_tr, y_train_wa, X_te, y_test)
    if y_xgb is not None:
        all_results['xgboost'] = {
            'bootstrap_ci': ci_xgb,
            'info': info_xgb,
            'description': 'XGBoost regressor (hyperparameter search)',
        }

    y_nn, ci_nn, seed_mses_nn = run_best_nn(X_tr, y_train_bins, X_te, y_test)
    all_results['best_nn'] = {
        'bootstrap_ci': ci_nn,
        'seed_mses': seed_mses_nn,
        'description': '(256,256,128) dropout=0.35, focal, 5-seed',
    }

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = (time.time() - t_total) / 60
    print(f"\n{'='*60}")
    print(f"  SUMMARY  ({elapsed:.1f} min)")
    print(f"{'='*60}")
    print(f"\n  {'Model':<25} {'MSE':>8} {'MSE 95% CI':>22} {'Spearman':>10} {'R²':>8}")
    print(f"  {'-'*75}")
    for name, label in [('linear_probe', 'Linear Probe'),
                        ('ridge', 'Ridge Regression'),
                        ('xgboost', 'XGBoost'),
                        ('best_nn', 'Best NN (256,256,128)')]:
        if name not in all_results:
            continue
        ci = all_results[name]['bootstrap_ci']
        print(f"  {label:<25} {ci['mse']['point']:>8.4f} "
              f"[{ci['mse']['ci_lo']:.3f}, {ci['mse']['ci_hi']:.3f}] "
              f"{ci['spearman']['point']:>10.4f} "
              f"{ci['r2']['point']:>8.4f}")

    # Embedding contribution analysis
    if 'linear_probe' in all_results and 'best_nn' in all_results:
        lp_mse = all_results['linear_probe']['bootstrap_ci']['mse']['point']
        nn_mse = all_results['best_nn']['bootstrap_ci']['mse']['point']
        # Mean prediction baseline MSE
        mean_pred = np.full_like(y_test, y_train_wa.mean())
        baseline_mse = mse(y_test, mean_pred)

        improvement_lp = baseline_mse - lp_mse
        improvement_nn = baseline_mse - nn_mse
        total_improvement = improvement_nn
        embedding_pct = 100 * improvement_lp / total_improvement if total_improvement > 0 else 0
        mlp_pct = 100 - embedding_pct

        print(f"\n  Embedding contribution analysis:")
        print(f"    Mean-prediction baseline MSE: {baseline_mse:.4f}")
        print(f"    Linear probe improvement:     {improvement_lp:.4f} ({embedding_pct:.1f}% of total)")
        print(f"    MLP additional improvement:   {improvement_nn - improvement_lp:.4f} ({mlp_pct:.1f}% of total)")

        all_results['embedding_contribution'] = {
            'baseline_mse': baseline_mse,
            'embedding_pct': embedding_pct,
            'mlp_pct': mlp_pct,
        }

    all_results['elapsed_minutes'] = elapsed

    # Save
    def convert(obj):
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)): return [convert(i) for i in obj]
        elif isinstance(obj, (np.integer,)): return int(obj)
        elif isinstance(obj, (np.floating,)): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    with open(RESULTS_DIR / 'linear_baseline_results.json', 'w') as f:
        json.dump(convert(all_results), f, indent=2)
    print(f"\n  Results saved.")

    # ── Figure ────────────────────────────────────────────────────────────
    make_figure(all_results)

    print(f"\nDone in {elapsed:.1f} min.")
    return all_results


if __name__ == '__main__':
    main()
