#!/usr/bin/env python3
"""
Script 02: Random Forest Hyperparameter Search

RandomizedSearchCV for EACH of 4 feature types independently.
Default: 100 iterations, 5-fold CV. Use --quick for 30 iterations, 3-fold CV.

Grid:
  - n_estimators: [50, 75, 100, 150, 200, 300]
  - max_depth: [10, 15, 20, 25, 30, None]
  - min_samples_split: [2, 5, 10, 0.001, 0.01]
  - min_samples_leaf: [1, 2, 4, 0.0001, 0.001]
  - max_features: ['sqrt', 'log2', 0.5, 0.75, 1.0]

Same preprocessing: StandardScaler + log1p when applicable.
"""
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_loading import load_all_feature_types, preprocess, inverse_transform_target
from src.evaluation import compute_metrics, format_metrics

# ── Configuration ──────────────────────────────────────────────────────────
RANDOM_STATE = 42

parser = argparse.ArgumentParser()
parser.add_argument('--quick', action='store_true', help='30 iters, 3-fold CV (faster)')
_args, _ = parser.parse_known_args()

N_ITER = 30 if _args.quick else 100
CV_FOLDS = 3 if _args.quick else 5
RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results'

PARAM_DISTRIBUTIONS = {
    'n_estimators': [50, 75, 100, 150, 200, 300],
    'max_depth': [10, 15, 20, 25, 30, None],
    'min_samples_split': [2, 5, 10, 0.001, 0.01],
    'min_samples_leaf': [1, 2, 4, 0.0001, 0.001],
    'max_features': ['sqrt', 'log2', 0.5, 0.75, 1.0],
}


def search_one_feature_type(name, X_train, X_test, y_train, y_test):
    """Run RandomizedSearchCV for one feature type and evaluate on test set."""
    print(f"\n{'='*60}")
    print(f"  RF Search: {name}  ({X_train.shape[1]} features)")
    print(f"{'='*60}")

    # Preprocess
    X_tr, X_te, y_tr, y_te, prep_info = preprocess(
        X_train, X_test, y_train, y_test,
        scale_features=True,
        log_transform_target='auto',
    )
    print(f"  Log transform: {prep_info['log_transformed']}")

    # RandomizedSearchCV
    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)

    search = RandomizedSearchCV(
        rf,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=N_ITER,
        cv=CV_FOLDS,
        scoring='neg_mean_squared_error',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    t0 = time.time()
    search.fit(X_tr, y_tr)
    elapsed = time.time() - t0
    print(f"  Search completed in {elapsed:.1f}s")

    # Best params
    best = search.best_params_
    best_cv_mse = -search.best_score_
    print(f"  Best CV MSE: {best_cv_mse:.4f}")
    print(f"  Best params: {best}")

    # Extract all trial configs + CV metrics
    all_trials = []
    for i in range(len(search.cv_results_['mean_test_score'])):
        trial_params = {}
        for k in PARAM_DISTRIBUTIONS.keys():
            val = search.cv_results_[f'param_{k}'][i]
            if isinstance(val, (np.integer,)):
                val = int(val)
            elif isinstance(val, (np.floating,)):
                val = float(val)
            trial_params[k] = val
        all_trials.append({
            'config': trial_params,
            'cv_mse': float(-search.cv_results_['mean_test_score'][i]),
            'cv_mse_std': float(search.cv_results_['std_test_score'][i]),
            'rank': int(search.cv_results_['rank_test_score'][i]),
        })

    # Evaluate best model on test set
    y_pred_test_p = search.best_estimator_.predict(X_te)
    y_pred_test = inverse_transform_target(y_pred_test_p, prep_info['log_transformed'])

    test_metrics = compute_metrics(y_test, y_pred_test)
    print("\n" + format_metrics(test_metrics, prefix=f'{name} TEST'))

    # Also evaluate on train set
    y_pred_train_p = search.best_estimator_.predict(X_tr)
    y_pred_train = inverse_transform_target(y_pred_train_p, prep_info['log_transformed'])
    train_metrics = compute_metrics(y_train, y_pred_train)

    # Serialize best params (convert numpy types)
    best_serializable = {}
    for k, v in best.items():
        if isinstance(v, (np.integer,)):
            best_serializable[k] = int(v)
        elif isinstance(v, (np.floating,)):
            best_serializable[k] = float(v)
        else:
            best_serializable[k] = v

    return {
        'feature_type': name,
        'n_features': X_train.shape[1],
        'best_params': best_serializable,
        'best_cv_mse': float(best_cv_mse),
        'preprocessing': {
            'scaler': 'StandardScaler',
            'log_transform': prep_info['log_transformed'],
            'skewness': prep_info.get('skewness'),
        },
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'search_time_seconds': round(elapsed, 1),
        'n_iter': N_ITER,
        'cv_folds': CV_FOLDS,
        'all_trials': all_trials,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(RANDOM_STATE)

    # Load all feature types
    print("Loading all feature types...")
    all_data = load_all_feature_types()

    results = {}
    for name, (X_train, X_test, y_train, y_test, meta) in all_data.items():
        result = search_one_feature_type(name, X_train, X_test, y_train, y_test)
        results[name] = result

    # Summary
    print("\n" + "="*60)
    print("  SUMMARY: RF Hyperparameter Search Results")
    print("="*60)
    print(f"  {'Feature Type':<15} {'CV MSE':>8} {'Test MSE':>10} {'Test R²':>8}")
    print(f"  {'-'*15} {'-'*8} {'-'*10} {'-'*8}")
    for name, r in results.items():
        print(f"  {name:<15} {r['best_cv_mse']:>8.4f} {r['test_metrics']['mse']:>10.4f} {r['test_metrics']['r2']:>8.4f}")
    print(f"\n  Grasso baseline: MSE = 1.22")

    # Save
    out_path = RESULTS_DIR / 'rf_search_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == '__main__':
    main()
