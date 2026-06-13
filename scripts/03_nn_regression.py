#!/usr/bin/env python3
"""
Script 03: Neural Network Regression Search

Regression NN with dimension-aware hyperparameter search per feature type.
  - Dense(units, relu) -> BatchNorm -> Dropout -> ... -> Dense(1, linear)
  - Loss: MSE
  - Output: continuous prediction (no binning or classification)

Dimension-aware search grids:
  - PhysChem (156d): layers [[128], [128,64], [256,128]]
  - ESM-2/Ginkgo (1280d): layers [[256], [512,256], [256,128]]
  - ESM-2 3B (2560d): layers [[512], [512,256], [1024,512]]

Shared search space:
  - dropout: [0.2, 0.3, 0.4, 0.5]
  - l2_reg: [1e-4, 1e-3, 1e-2]
  - learning_rate: [1e-3, 5e-4, 1e-4]
  - batch_size: [32, 64]
  - log_transform: [True, False]

40 random configs per feature type, 80/20 train/val split.
EarlyStopping(patience=15), ReduceLROnPlateau(patience=7).
"""
import sys
import json
import time
import numpy as np
from pathlib import Path

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_loading import (
    load_all_feature_types, preprocess, inverse_transform_target, FEATURE_TYPES
)
from src.models import SignalPeptideRegressorNN
from src.evaluation import compute_metrics, format_metrics

# ── Configuration ──────────────────────────────────────────────────────────
RANDOM_STATE = 42
N_CONFIGS = 40
VAL_FRACTION = 0.2
RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results'

# Dimension-aware layer architectures
LAYER_GRIDS = {
    'PhysChem': [[128], [128, 64], [256, 128]],
    'ESM2-650M': [[256], [512, 256], [256, 128]],
    'ESM2-3B': [[512], [512, 256], [1024, 512]],
    'Ginkgo-AA0': [[256], [512, 256], [256, 128]],
}

# Shared hyperparameter search space
SHARED_GRID = {
    'dropout': [0.2, 0.3, 0.4, 0.5],
    'l2_reg': [1e-4, 1e-3, 1e-2],
    'learning_rate': [1e-3, 5e-4, 1e-4],
    'batch_size': [32, 64],
    'log_transform': [True, False],
}


def sample_configs(feature_type, n_configs, rng):
    """Sample n_configs random hyperparameter configurations."""
    layers_options = LAYER_GRIDS[feature_type]

    # Build all possible combinations
    keys = list(SHARED_GRID.keys())
    values = [SHARED_GRID[k] for k in keys]

    configs = []
    for _ in range(n_configs):
        config = {}
        config['hidden_layers'] = layers_options[rng.integers(len(layers_options))].copy()
        for k, v in SHARED_GRID.items():
            config[k] = v[rng.integers(len(v))]
        configs.append(config)

    return configs


def train_val_split(X, y, val_fraction, rng):
    """Split training data into train and validation sets."""
    n = len(y)
    indices = rng.permutation(n)
    n_val = int(n * val_fraction)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    return (X[train_idx], y[train_idx], X[val_idx], y[val_idx])


def search_one_feature_type(name, X_train, X_test, y_train, y_test, rng):
    """Run NN hyperparameter search for one feature type."""
    print(f"\n{'='*60}")
    print(f"  NN Search: {name}  ({X_train.shape[1]} features)")
    print(f"{'='*60}")

    configs = sample_configs(name, N_CONFIGS, rng)
    best_val_mse = float('inf')
    best_config = None
    best_test_metrics = None
    best_train_metrics = None
    all_trials = []

    for i, config in enumerate(configs):
        log_transform = config.pop('log_transform')

        # Split BEFORE scaling so the validation set — which SELECTS the best
        # config below — never informs the scaler. Fitting the scaler on the
        # full training set first leaks the validation rows' statistics into
        # their own normalization and biases the selection metric.
        X_tr_raw, y_tr_raw, X_val_raw, y_val_raw = train_val_split(
            X_train, y_train, VAL_FRACTION, rng)

        # Fit preprocessing on the training partition only; apply to val + test.
        X_tr, X_val, y_tr, y_val, prep_info = preprocess(
            X_tr_raw, X_val_raw, y_tr_raw, y_val_raw,
            scale_features=True,
            log_transform_target=log_transform,
        )
        scaler = prep_info['scaler']
        X_te_p = scaler.transform(X_test) if scaler is not None else X_test.copy()

        # Build and train model
        model = SignalPeptideRegressorNN(
            hidden_layers=tuple(config['hidden_layers']),
            dropout=config['dropout'],
            l2_reg=config['l2_reg'],
            learning_rate=config['learning_rate'],
            batch_size=config['batch_size'],
            epochs=200,
            random_state=RANDOM_STATE,
        )

        t0 = time.time()
        model.fit(X_tr, y_tr, X_val=X_val, y_val=y_val, verbose=0)
        elapsed = time.time() - t0

        # Evaluate on validation set
        y_val_pred_p = model.predict(X_val)
        y_val_pred = inverse_transform_target(y_val_pred_p, prep_info['log_transformed'])
        y_val_actual = inverse_transform_target(y_val, prep_info['log_transformed'])
        val_metrics = compute_metrics(y_val_actual, y_val_pred)

        # Evaluate on test set
        y_test_pred_p = model.predict(X_te_p)
        y_test_pred = inverse_transform_target(y_test_pred_p, prep_info['log_transformed'])
        test_metrics = compute_metrics(y_test, y_test_pred)

        # Evaluate on train set
        y_train_pred_p = model.predict(X_tr)
        y_train_pred = inverse_transform_target(y_train_pred_p, prep_info['log_transformed'])
        y_train_actual = inverse_transform_target(y_tr, prep_info['log_transformed'])
        train_metrics = compute_metrics(y_train_actual, y_train_pred)

        trial = {
            'config': {**config, 'log_transform': log_transform},
            'val_mse': val_metrics['mse'],
            'test_mse': test_metrics['mse'],
            'train_time_seconds': round(elapsed, 1),
        }
        all_trials.append(trial)

        marker = ''
        if val_metrics['mse'] < best_val_mse:
            best_val_mse = val_metrics['mse']
            best_config = {**config, 'log_transform': log_transform}
            best_test_metrics = test_metrics
            best_train_metrics = train_metrics
            marker = ' *BEST*'

        print(f"  [{i+1:2d}/{N_CONFIGS}] val_mse={val_metrics['mse']:.4f}  "
              f"test_mse={test_metrics['mse']:.4f}  "
              f"layers={config['hidden_layers']}  "
              f"drop={config['dropout']}  lr={config['learning_rate']}  "
              f"log={log_transform}  ({elapsed:.1f}s){marker}")

    print(f"\n  Best config: {best_config}")
    print(format_metrics(best_test_metrics, prefix=f'{name} TEST (best)'))

    return {
        'feature_type': name,
        'n_features': X_train.shape[1],
        'best_config': _serialize(best_config),
        'best_val_mse': float(best_val_mse),
        'train_metrics': best_train_metrics,
        'test_metrics': best_test_metrics,
        'n_configs': N_CONFIGS,
        'val_fraction': VAL_FRACTION,
        'all_trials': [_serialize(t) for t in all_trials],
    }


def _serialize(obj):
    """Make objects JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)

    print("Loading all feature types...")
    all_data = load_all_feature_types()

    results = {}
    for name in LAYER_GRIDS:  # iterate in defined order
        X_train, X_test, y_train, y_test, meta = all_data[name]
        result = search_one_feature_type(name, X_train, X_test, y_train, y_test, rng)
        results[name] = result

    # Summary
    print("\n" + "="*60)
    print("  SUMMARY: NN Regression Search Results")
    print("="*60)
    print(f"  {'Feature Type':<15} {'Val MSE':>8} {'Test MSE':>10} {'Test R²':>8}")
    print(f"  {'-'*15} {'-'*8} {'-'*10} {'-'*8}")
    for name, r in results.items():
        print(f"  {name:<15} {r['best_val_mse']:>8.4f} "
              f"{r['test_metrics']['mse']:>10.4f} {r['test_metrics']['r2']:>8.4f}")
    print(f"\n  Grasso baseline: MSE = 1.22")

    # Save
    out_path = RESULTS_DIR / 'nn_search_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == '__main__':
    main()
