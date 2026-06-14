#!/usr/bin/env python3
"""
Script 01: Grasso et al. Baseline Reproduction

Reproduces the Grasso et al. Random Forest model with exact published parameters:
  - 75 trees, max_depth=25, min_samples_split=0.001, min_samples_leaf=0.0001
  - 156 physicochemical features loaded directly from xlsx
  - StandardScaler on features, log1p on target if |skew| > 1.0
  - Quality filters: SP length 10-40aa, WA 1.0-10.0
  - Original train/test split from 'Set' column

Expected: MSE ~1.19 (within 5% of reported 1.22)
"""
import sys
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_loading import (
    load_physicochemical, preprocess, inverse_transform_target, GRASSO_FEATURES
)
from src.evaluation import compute_metrics, format_metrics

# ── Configuration ──────────────────────────────────────────────────────────
RANDOM_STATE = 42
RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results'
FIGURES_DIR = Path(__file__).resolve().parent.parent / 'figures'

# Exact Grasso et al. RF hyperparameters
GRASSO_RF_PARAMS = dict(
    n_estimators=75,
    max_depth=25,
    min_samples_split=0.001,
    min_samples_leaf=0.0001,
    max_features=156,        # all features
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

GRASSO_BASELINE_MSE = 1.22


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(RANDOM_STATE)

    # ── 1. Load data from xlsx (single source of truth) ────────────────
    print("Loading physicochemical features from xlsx...")
    X_train, X_test, y_train, y_test, meta = load_physicochemical()

    print(f"  Features:  {meta['n_features']} (expected 156)")
    print(f"  Train set: {meta['n_train']}")
    print(f"  Test set:  {meta['n_test']}")
    assert meta['n_features'] == 156, f"Expected 156 features, got {meta['n_features']}"

    # ── 2. Preprocess ──────────────────────────────────────────────────
    print("\nPreprocessing...")
    X_train_p, X_test_p, y_train_p, y_test_p, prep_info = preprocess(
        X_train, X_test, y_train, y_test,
        scale_features=True,
        log_transform_target='auto',
    )
    print(f"  Scaled:          {prep_info['scaled']}")
    print(f"  Log transformed: {prep_info['log_transformed']}")
    if 'skewness' in prep_info:
        print(f"  Target skewness: {prep_info['skewness']:.3f}")

    # ── 3. Train exact Grasso RF ───────────────────────────────────────
    print(f"\nTraining Grasso RF with params: {GRASSO_RF_PARAMS}")
    rf = RandomForestRegressor(**GRASSO_RF_PARAMS)
    rf.fit(X_train_p, y_train_p)

    # ── 4. Predict and inverse-transform ───────────────────────────────
    y_pred_train_p = rf.predict(X_train_p)
    y_pred_test_p = rf.predict(X_test_p)

    y_pred_train = inverse_transform_target(y_pred_train_p, prep_info['log_transformed'])
    y_pred_test = inverse_transform_target(y_pred_test_p, prep_info['log_transformed'])

    # ── 5. Evaluate ────────────────────────────────────────────────────
    train_metrics = compute_metrics(y_train, y_pred_train)
    test_metrics = compute_metrics(y_test, y_pred_test)

    print("\n" + format_metrics(train_metrics, prefix='TRAIN'))
    print("\n" + format_metrics(test_metrics, prefix='TEST'))

    pct_diff = (test_metrics['mse'] - GRASSO_BASELINE_MSE) / GRASSO_BASELINE_MSE * 100
    print(f"\n  Grasso baseline MSE: {GRASSO_BASELINE_MSE}")
    print(f"  My test MSE:        {test_metrics['mse']:.4f}")
    print(f"  Difference:          {pct_diff:+.1f}%")

    # ── 6. Save results ───────────────────────────────────────────────
    results = {
        'script': '01_grasso_reproduction',
        'description': 'Exact Grasso RF reproduction from xlsx',
        'rf_params': {k: v for k, v in GRASSO_RF_PARAMS.items() if k != 'n_jobs'},
        'preprocessing': {
            'scaler': 'StandardScaler',
            'log_transform': prep_info['log_transformed'],
            'skewness': prep_info.get('skewness'),
        },
        'data': {
            'n_features': meta['n_features'],
            'n_train': meta['n_train'],
            'n_test': meta['n_test'],
        },
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'grasso_baseline_mse': GRASSO_BASELINE_MSE,
        'pct_difference': round(pct_diff, 2),
    }

    out_path = RESULTS_DIR / 'grasso_reproduction.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # ── 7. Figure: predicted vs actual ─────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, y_true, y_pred, title in [
        (axes[0], y_train, y_pred_train, 'Train'),
        (axes[1], y_test, y_pred_test, 'Test'),
    ]:
        ax.scatter(y_true, y_pred, alpha=0.3, s=10, color='steelblue')
        lims = [min(y_true.min(), y_pred.min()) - 0.5,
                max(y_true.max(), y_pred.max()) + 0.5]
        ax.plot(lims, lims, 'r--', lw=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel('Actual WA')
        ax.set_ylabel('Predicted WA')
        ax.text(0.05, 0.95, title, transform=ax.transAxes,
                fontsize=10, va='top')

    plt.tight_layout()

    fig_path = FIGURES_DIR / 'grasso_reproduction.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to {fig_path}")

    # ── 8. Verification ───────────────────────────────────────────────
    if 1.10 <= test_metrics['mse'] <= 1.30:
        print("\n[PASS] MSE within expected range (1.10-1.30)")
    else:
        print(f"\n[WARN] MSE {test_metrics['mse']:.4f} outside expected range (1.10-1.30)")

    return results


if __name__ == '__main__':
    main()
