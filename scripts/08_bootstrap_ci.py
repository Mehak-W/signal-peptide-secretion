#!/usr/bin/env python3
"""
Script 08: Bootstrap Confidence Intervals

Retrains each model (16 total: 5 RF + 4 NN scalar + 6 vector ensembles +
1 full-data optimized) with its best config, computes (y_true, y_pred) on the
test set, then bootstraps 10,000 times to produce 95% CIs for MSE, R², and
Spearman rho.

Note: Scalar NN models are retrained with a single random initialization (not
ensembled), so their point estimates may differ from those in Script 03, which
reports the best configuration selected from 40 candidates.

Outputs:
  - results/bootstrap_ci_results.json   (full CI data)
  - results/bootstrap_ci_summary.csv    (table-ready summary)
  - figures/bootstrap_forest_plot.png   (forest plot with CI whiskers)
"""
import sys
import json
import time
import csv
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_loading import (
    load_all_feature_types, load_physicochemical, load_plm_embeddings,
    load_plm_with_bins, preprocess, inverse_transform_target, BIN_COLUMNS,
)
from src.models import SignalPeptideRegressorNN, SignalPeptideVectorNN
from src.evaluation import compute_metrics

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results'
FIGURES_DIR = Path(__file__).resolve().parent.parent / 'figures'

# ── Bootstrap config ──────────────────────────────────────────────────────
N_BOOTSTRAP = 10_000
RANDOM_STATE = 42
CI_LEVEL = 0.95
ALPHA = (1 - CI_LEVEL) / 2  # 0.025

# ── Ensemble config ───────────────────────────────────────────────────────
SEEDS = [42, 123, 456, 789, 1024]
VAL_FRACTION = 0.2
BIN_CENTERS = np.arange(1, 11)

# ── Best RF configs (from rf_search_results.json) ─────────────────────────
# NOTE: Hyperparameters below are copied from Scripts 02/03 result JSONs.
# They are pinned here for exact reproducibility of the bootstrap analysis.
# To autoload from the search results instead, replace each config dict with:
#   json.load(open(RESULTS_DIR / 'rf_search_results.json'))['PhysChem']['best_params']
#   json.load(open(RESULTS_DIR / 'nn_search_results.json'))['PhysChem']['best_config']
RF_CONFIGS = {
    'RF PhysChem (baseline)': {
        'feature_type': 'PhysChem',
        'params': {
            'n_estimators': 75, 'max_depth': 25,
            'min_samples_split': 0.001, 'min_samples_leaf': 0.0001,
            'max_features': None,  # all features
        },
    },
    'RF PhysChem (tuned)': {
        'feature_type': 'PhysChem',
        'params': {
            'n_estimators': 75, 'max_depth': None,
            'min_samples_split': 5, 'min_samples_leaf': 2,
            'max_features': 0.5,
        },
    },
    'RF ESM2-650M': {
        'feature_type': 'ESM2-650M',
        'params': {
            'n_estimators': 300, 'max_depth': 25,
            'min_samples_split': 0.001, 'min_samples_leaf': 4,
            'max_features': 0.5,
        },
    },
    'RF ESM2-3B': {
        'feature_type': 'ESM2-3B',
        'params': {
            'n_estimators': 150, 'max_depth': 25,
            'min_samples_split': 2, 'min_samples_leaf': 2,
            'max_features': 0.5,
        },
    },
    'RF Ginkgo-AA0': {
        'feature_type': 'Ginkgo-AA0',
        'params': {
            'n_estimators': 300, 'max_depth': None,
            'min_samples_split': 10, 'min_samples_leaf': 4,
            'max_features': 0.5,
        },
    },
}

# ── Best NN scalar configs (from nn_search_results.json) ──────────────────
NN_CONFIGS = {
    'NN PhysChem': {
        'feature_type': 'PhysChem',
        'params': {
            'hidden_layers': (256, 128), 'dropout': 0.3,
            'l2_reg': 0.01, 'learning_rate': 0.001, 'batch_size': 64,
        },
    },
    'NN ESM2-650M': {
        'feature_type': 'ESM2-650M',
        'params': {
            'hidden_layers': (256, 128), 'dropout': 0.4,
            'l2_reg': 0.01, 'learning_rate': 0.001, 'batch_size': 64,
        },
    },
    'NN ESM2-3B': {
        'feature_type': 'ESM2-3B',
        'params': {
            'hidden_layers': (1024, 512), 'dropout': 0.4,
            'l2_reg': 0.01, 'learning_rate': 0.001, 'batch_size': 64,
        },
    },
    'NN Ginkgo-AA0': {
        'feature_type': 'Ginkgo-AA0',
        'params': {
            'hidden_layers': (256, 128), 'dropout': 0.3,
            'l2_reg': 0.0001, 'learning_rate': 0.0005, 'batch_size': 32,
        },
    },
}

# ── Vector regression configs ─────────────────────────────────────────────
EMBEDDING_MAP = {
    'ESM2-650M': 'esm2-650M',
    'ESM2-3B': 'esm2-3B',
    'Ginkgo-AA0': 'ginkgo-AA0-650M',
}

VECTOR_CONFIGS = {}
for emb_name in ['ESM2-650M', 'ESM2-3B', 'Ginkgo-AA0']:
    for loss in ['categorical_crossentropy', 'focal']:
        loss_label = 'focal' if loss == 'focal' else 'CE'
        key = f'Vec {emb_name} {loss_label}'
        VECTOR_CONFIGS[key] = {
            'embedding': emb_name,
            'model_name': EMBEDDING_MAP[emb_name],
            'loss': loss,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Bootstrap core
# ═══════════════════════════════════════════════════════════════════════════

def bootstrap_ci(y_true, y_pred, n_bootstrap=N_BOOTSTRAP, seed=RANDOM_STATE):
    """
    Compute point estimates and 95% bootstrap CIs for MSE, R², Spearman rho.

    Returns dict with keys: mse, r2, spearman, each containing
    'point', 'ci_lo', 'ci_hi'.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)

    mse_boot = np.empty(n_bootstrap)
    r2_boot = np.empty(n_bootstrap)
    spearman_boot = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]

        mse_boot[b] = np.mean((yt - yp) ** 2)
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2_boot[b] = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        spearman_boot[b] = sp_stats.spearmanr(yt, yp).statistic

    # Point estimates
    mse_point = float(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2_point = float(1 - ss_res / ss_tot)
    spearman_point = float(sp_stats.spearmanr(y_true, y_pred).statistic)

    return {
        'mse': {
            'point': mse_point,
            'ci_lo': float(np.percentile(mse_boot, 100 * ALPHA)),
            'ci_hi': float(np.percentile(mse_boot, 100 * (1 - ALPHA))),
        },
        'r2': {
            'point': r2_point,
            'ci_lo': float(np.percentile(r2_boot, 100 * ALPHA)),
            'ci_hi': float(np.percentile(r2_boot, 100 * (1 - ALPHA))),
        },
        'spearman': {
            'point': spearman_point,
            'ci_lo': float(np.percentile(spearman_boot, 100 * ALPHA)),
            'ci_hi': float(np.percentile(spearman_boot, 100 * (1 - ALPHA))),
        },
        'n_test': int(len(y_true)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Model training helpers
# ═══════════════════════════════════════════════════════════════════════════

def train_scalar_rf(X_tr, y_tr, X_te, params):
    """Train an RF with given params and return test predictions."""
    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
    rf.fit(X_tr, y_tr)
    return rf.predict(X_te)


def train_scalar_nn(X_tr, y_tr, X_te, params, seed=RANDOM_STATE):
    """Train an NN with 80/20 val split and return test predictions."""
    rng = np.random.default_rng(seed)
    n = len(y_tr)
    indices = rng.permutation(n)
    n_val = int(n * VAL_FRACTION)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    model = SignalPeptideRegressorNN(
        hidden_layers=params['hidden_layers'],
        dropout=params['dropout'],
        l2_reg=params['l2_reg'],
        learning_rate=params['learning_rate'],
        batch_size=params['batch_size'],
        epochs=200,
        random_state=seed,
    )
    model.fit(X_tr[train_idx], y_tr[train_idx],
              X_val=X_tr[val_idx], y_val=y_tr[val_idx], verbose=0)
    return model.predict(X_te)


def train_vector_ensemble(X_train_raw, y_train_bins, X_test_raw, y_test_wa, loss_type):
    """
    Train 5-seed vector ensemble and return (y_true, y_pred_wa) on test set.
    Scaler is fit on the full training set; per-seed split is 80/20.
    """
    scaler = StandardScaler()
    scaler.fit(X_train_raw)
    X_test_s = scaler.transform(X_test_raw)

    all_preds = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        n = len(y_train_bins)
        indices = rng.permutation(n)
        n_val = int(n * VAL_FRACTION)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        X_tr_s = scaler.transform(X_train_raw[train_idx])
        X_val_s = scaler.transform(X_train_raw[val_idx])

        model = SignalPeptideVectorNN(
            hidden_layers=(256, 256),
            dropout=0.2,
            learning_rate=5e-4,
            batch_size=32,
            epochs=200,
            loss=loss_type,
            random_state=seed,
        )
        model.fit(X_tr_s, y_train_bins[train_idx],
                  X_val=X_val_s, y_val_bins=y_train_bins[val_idx], verbose=0)
        all_preds.append(model.predict(X_test_s))

    avg_preds = np.mean(all_preds, axis=0)
    y_pred_wa = avg_preds @ BIN_CENTERS
    return y_test_wa, y_pred_wa


# ═══════════════════════════════════════════════════════════════════════════
# Forest plot
# ═══════════════════════════════════════════════════════════════════════════

def make_forest_plot(results, model_order):
    """Create horizontal forest plot of MSE with 95% CIs."""
    fig, ax = plt.subplots(figsize=(10, 8))

    labels = []
    points = []
    ci_los = []
    ci_his = []
    colors = []

    for name in model_order:
        r = results[name]
        labels.append(name)
        points.append(r['mse']['point'])
        ci_los.append(r['mse']['ci_lo'])
        ci_his.append(r['mse']['ci_hi'])

        if name.startswith('Vec'):
            colors.append('darkorange')
        else:
            colors.append('steelblue')

    y_pos = np.arange(len(labels))

    # Plot CIs as horizontal lines
    for i in range(len(labels)):
        ax.plot([ci_los[i], ci_his[i]], [y_pos[i], y_pos[i]],
                color=colors[i], linewidth=2, zorder=1)
        ax.plot(points[i], y_pos[i], 'o', color=colors[i],
                markersize=7, zorder=2)

    # Reference lines
    ax.axvline(x=1.22, color='red', linestyle='--', linewidth=1.0, alpha=0.8,
               label='Baseline (1.22)', zorder=0)
    ax.axvline(x=0.95, color='green', linestyle=':', linewidth=1.0, alpha=0.8,
               label='prior single-run (0.95, retracted)', zorder=0)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Test MSE')

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='steelblue', label='Scalar models',
               markersize=7, linewidth=2),
        Line2D([0], [0], marker='o', color='darkorange', label='Vector models',
               markersize=7, linewidth=2),
        Line2D([0], [0], color='red', linestyle='--', label='Baseline (1.22)'),
        Line2D([0], [0], color='green', linestyle=':', label='prior single-run (0.95, retracted)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    fig_path = FIGURES_DIR / 'bootstrap_forest_plot.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nForest plot saved to {fig_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    model_order = []

    t_total = time.time()

    # ── 1. Load all scalar feature types ──────────────────────────────────
    print("Loading all feature types...")
    all_data = load_all_feature_types()

    # ── 2. RF models ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RANDOM FOREST MODELS")
    print("=" * 60)

    for name, cfg in RF_CONFIGS.items():
        print(f"\n  {name}...")
        t0 = time.time()
        ft = cfg['feature_type']
        X_train, X_test, y_train, y_test, meta = all_data[ft]

        # Preprocess (StandardScaler, auto log — matches Script 02)
        X_tr, X_te, y_tr, y_te, prep_info = preprocess(
            X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy(),
            scale_features=True, log_transform_target='auto',
        )

        y_pred_p = train_scalar_rf(X_tr, y_tr, X_te, cfg['params'])
        y_pred = inverse_transform_target(y_pred_p, prep_info['log_transformed'])

        ci = bootstrap_ci(y_test, y_pred)
        ci['model_type'] = 'RF'
        ci['feature_type'] = ft
        results[name] = ci
        model_order.append(name)

        elapsed = time.time() - t0
        print(f"    MSE = {ci['mse']['point']:.4f} "
              f"[{ci['mse']['ci_lo']:.4f}, {ci['mse']['ci_hi']:.4f}]  "
              f"({elapsed:.1f}s)")

    # ── 3. NN scalar models (single best config, no ensemble) ─────────────
    print("\n" + "=" * 60)
    print("  NEURAL NETWORK SCALAR MODELS")
    print("=" * 60)

    for name, cfg in NN_CONFIGS.items():
        print(f"\n  {name}...")
        t0 = time.time()
        ft = cfg['feature_type']
        X_train, X_test, y_train, y_test, meta = all_data[ft]

        # Preprocess
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)

        y_pred = train_scalar_nn(X_tr_s, y_train, X_te_s, cfg['params'],
                                 seed=RANDOM_STATE)

        ci = bootstrap_ci(y_test, y_pred)
        ci['model_type'] = 'NN'
        ci['feature_type'] = ft
        results[name] = ci
        model_order.append(name)

        elapsed = time.time() - t0
        print(f"    MSE = {ci['mse']['point']:.4f} "
              f"[{ci['mse']['ci_lo']:.4f}, {ci['mse']['ci_hi']:.4f}]  "
              f"({elapsed:.1f}s)")

    # ── 4. Vector regression (5-seed ensembles) ───────────────────────────
    print("\n" + "=" * 60)
    print("  VECTOR REGRESSION MODELS (5-seed ensembles)")
    print("=" * 60)

    # Cache loaded data per embedding type
    vec_data_cache = {}

    for name, cfg in VECTOR_CONFIGS.items():
        print(f"\n  {name}...")
        t0 = time.time()
        emb_name = cfg['embedding']
        model_name = cfg['model_name']

        if emb_name not in vec_data_cache:
            X_train, X_test, y_train_wa, y_test_wa, y_train_bins, y_test_bins, vmeta = \
                load_plm_with_bins(model_name)
            vec_data_cache[emb_name] = (
                X_train, X_test, y_train_wa, y_test_wa, y_train_bins, y_test_bins
            )

        X_train, X_test, y_train_wa, y_test_wa, y_train_bins, y_test_bins = \
            vec_data_cache[emb_name]

        y_true, y_pred = train_vector_ensemble(
            X_train, y_train_bins, X_test, y_test_wa, cfg['loss']
        )

        ci = bootstrap_ci(y_true, y_pred)
        ci['model_type'] = 'Vector'
        ci['feature_type'] = emb_name
        ci['loss'] = cfg['loss']
        results[name] = ci
        model_order.append(name)

        elapsed = time.time() - t0
        print(f"    MSE = {ci['mse']['point']:.4f} "
              f"[{ci['mse']['ci_lo']:.4f}, {ci['mse']['ci_hi']:.4f}]  "
              f"({elapsed:.1f}s)")

    # ── 5. Summary ────────────────────────────────────────────────────────
    total_elapsed = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"  BOOTSTRAP CI SUMMARY  ({total_elapsed:.0f}s total)")
    print(f"{'=' * 60}")
    print(f"  {'Model':<28} {'MSE':>6} {'95% CI':>18} {'R²':>6} {'Spearman':>9}")
    print(f"  {'-'*28} {'-'*6} {'-'*18} {'-'*6} {'-'*9}")

    for name in model_order:
        r = results[name]
        mse = r['mse']
        r2 = r['r2']
        sp = r['spearman']
        print(f"  {name:<28} {mse['point']:>6.3f} "
              f"[{mse['ci_lo']:.3f}, {mse['ci_hi']:.3f}] "
              f"{r2['point']:>6.3f} {sp['point']:>9.3f}")

    # ── 6. Save JSON ──────────────────────────────────────────────────────
    out_json = RESULTS_DIR / 'bootstrap_ci_results.json'
    with open(out_json, 'w') as f:
        json.dump({'model_order': model_order, 'results': results}, f, indent=2)
    print(f"\nResults saved to {out_json}")

    # ── 7. Save CSV ───────────────────────────────────────────────────────
    out_csv = RESULTS_DIR / 'bootstrap_ci_summary.csv'
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Model', 'Type', 'Feature',
            'MSE', 'MSE_CI_lo', 'MSE_CI_hi',
            'R2', 'R2_CI_lo', 'R2_CI_hi',
            'Spearman', 'Spearman_CI_lo', 'Spearman_CI_hi',
            'N_test',
        ])
        for name in model_order:
            r = results[name]
            writer.writerow([
                name, r['model_type'], r.get('feature_type', ''),
                f"{r['mse']['point']:.4f}",
                f"{r['mse']['ci_lo']:.4f}",
                f"{r['mse']['ci_hi']:.4f}",
                f"{r['r2']['point']:.4f}",
                f"{r['r2']['ci_lo']:.4f}",
                f"{r['r2']['ci_hi']:.4f}",
                f"{r['spearman']['point']:.4f}",
                f"{r['spearman']['ci_lo']:.4f}",
                f"{r['spearman']['ci_hi']:.4f}",
                r['n_test'],
            ])
    print(f"CSV saved to {out_csv}")

    # ── 8. Forest plot ────────────────────────────────────────────────────
    make_forest_plot(results, model_order)

    return results


if __name__ == '__main__':
    main()
