#!/usr/bin/env python3
"""
Script 20: ESM2-650M Vector Regression with Best Architecture

Applies the best architecture found in Script 10 (Ginkgo-AA0) to ESM2-650M
embeddings to measure how close a fully open-weight model gets to the best
proprietary-embedding result.

Configurations run:
  1. dropout=0.35  (Script 10 best; full-data optimal)
  2. dropout=0.20  (validation-optimal baseline for comparison)

Both use: (256, 256, 128), focal loss, 5-seed ensemble, 300 epochs, no early
stopping, ReduceLROnPlateau on training loss.

All models trained on full data (NaN-filtered), evaluated on 1326 test samples.
Bootstrap CIs with 10,000 resamples.

Outputs:
  - results/esm2_optimization_results.json
  - figures/esm2_optimization.png
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
from src.data_loading import load_plm_with_bins, load_plm_embeddings
from src.models import FocalLoss

from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Paths -----------------------------------------------------------------
RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results'
FIGURES_DIR = Path(__file__).resolve().parent.parent / 'figures'

# Constants -------------------------------------------------------------
SEEDS = [42, 123, 456, 789, 1024]
BIN_CENTERS = np.arange(1, 11)
N_BOOTSTRAP = 10_000

# Reference values from prior scripts
GINKGO_BEST_MSE = 0.932       # Script 10 best (Ginkgo-AA0, drop=0.35)
GINKGO_BEST_CI = (0.823, 1.054)
NET4_REFERENCE_MSE = 0.953  # single-run reference value for the optimization sweeps


# ==========================================================================
# Model building (identical to Script 10)
# ==========================================================================

def build_model(input_dim, hidden_layers=(256, 256, 128), dropout=0.35,
                lr=5e-4, loss='focal', seed=42):
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

    loss_fn = FocalLoss(alpha=0.25, gamma=2.0) if loss == 'focal' else 'categorical_crossentropy'
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=loss_fn)
    return model


def train_full(X_train, y_bins, input_dim, cfg, seed=42, verbose=0):
    model = build_model(input_dim, hidden_layers=cfg['hidden_layers'],
                        dropout=cfg['dropout'], lr=cfg['lr'],
                        loss=cfg['loss'], seed=seed)

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor='loss', factor=0.5, patience=10, min_lr=1e-6, verbose=0),
    ]

    model.fit(X_train, y_bins, epochs=cfg['epochs'],
              batch_size=cfg.get('batch_size', 32),
              callbacks=callbacks, verbose=verbose)
    return model


def predict_wa(model, X_test):
    return model.predict(X_test, verbose=0) @ BIN_CENTERS


def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def train_predict_cleanup(X_tr, y_bins, dim, cfg, seed, X_te):
    model = train_full(X_tr, y_bins, dim, cfg, seed=seed)
    pred = predict_wa(model, X_te)
    del model
    keras.backend.clear_session()
    return pred


def bootstrap_ci(y_true, y_pred, n_boot=N_BOOTSTRAP, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    mse_b = np.empty(n_boot)
    r2_b = np.empty(n_boot)
    sp_b = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        mse_b[b] = np.mean((yt - yp) ** 2)
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2_b[b] = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        sp_b[b] = sp_stats.spearmanr(yt, yp).statistic

    point_mse = mse(y_true, y_pred)
    ss_res_all = np.sum((y_true - y_pred) ** 2)
    ss_tot_all = np.sum((y_true - y_true.mean()) ** 2)
    point_r2 = float(1 - ss_res_all / ss_tot_all) if ss_tot_all > 0 else 0.0
    point_sp = float(sp_stats.spearmanr(y_true, y_pred).statistic)

    return {
        'mse': {'point': point_mse,
                'ci_lo': float(np.percentile(mse_b, 2.5)),
                'ci_hi': float(np.percentile(mse_b, 97.5))},
        'r2': {'point': point_r2,
               'ci_lo': float(np.percentile(r2_b, 2.5)),
               'ci_hi': float(np.percentile(r2_b, 97.5))},
        'spearman': {'point': point_sp,
                     'ci_lo': float(np.percentile(sp_b, 2.5)),
                     'ci_hi': float(np.percentile(sp_b, 97.5))},
        'n_test': int(n),
    }


def run_ensemble(X_tr, y_bins, dim, cfg, seeds, X_te, y_test, label):
    """Train a seed ensemble and return (mse, y_pred_ensemble, per_seed_preds)."""
    preds = []
    for seed in seeds:
        t0 = time.time()
        pred = train_predict_cleanup(X_tr, y_bins, dim, cfg, seed, X_te)
        preds.append(pred)
        mse_i = mse(y_test, pred)
        print(f"    Seed {seed:>5}: MSE = {mse_i:.4f}  ({time.time()-t0:.1f}s)")

    y_ens = np.mean(preds, axis=0)
    mse_ens = mse(y_test, y_ens)
    print(f"  -> {label} ensemble MSE = {mse_ens:.4f}")
    return mse_ens, y_ens, preds


def save_results(results, filename='esm2_optimization_results.json'):
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(i) for i in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    with open(RESULTS_DIR / filename, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"  [saved to {filename}]")


# ==========================================================================
# Figure: Ginkgo-AA0 vs ESM2-650M comparison
# ==========================================================================

def _make_figure(results):
    """
    Side-by-side bar chart comparing Ginkgo-AA0 and ESM2-650M results,
    with bootstrap CI error bars.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), gridspec_kw={'width_ratios': [2, 1.2]})

    # --- Left panel: MSE bar chart with CIs ---
    ax = axes[0]

    labels = []
    mses = []
    ci_los = []
    ci_his = []
    colors = []

    # Ginkgo-AA0 reference (from Script 10)
    labels.append('Ginkgo-AA0\n(256,256,128)\ndrop=0.35')
    mses.append(GINKGO_BEST_MSE)
    ci_los.append(GINKGO_BEST_CI[0])
    ci_his.append(GINKGO_BEST_CI[1])
    colors.append('#2196F3')  # blue

    # ESM2-650M results
    for key, display_drop in [('dropout_0.35', '0.35'), ('dropout_0.20', '0.20')]:
        if key in results:
            r = results[key]
            ci = r['bootstrap_ci']
            labels.append(f'ESM2-650M\n(256,256,128)\ndrop={display_drop}')
            mses.append(ci['mse']['point'])
            ci_los.append(ci['mse']['ci_lo'])
            ci_his.append(ci['mse']['ci_hi'])
            colors.append('#4CAF50' if display_drop == '0.35' else '#FF9800')

    x = np.arange(len(labels))
    yerr_lo = [m - lo for m, lo in zip(mses, ci_los)]
    yerr_hi = [hi - m for m, hi in zip(mses, ci_his)]

    bars = ax.bar(x, mses, color=colors, alpha=0.85, edgecolor='none',
                  width=0.6)
    ax.errorbar(x, mses, yerr=[yerr_lo, yerr_hi], fmt='none', ecolor='black',
                capsize=6, capthick=1.5, linewidth=1.5)

    # Reference line
    ax.axhline(y=GINKGO_BEST_MSE, color='#2196F3', linestyle='--', linewidth=1.0,
               alpha=0.6, label=f'Ginkgo-AA0 best ({GINKGO_BEST_MSE})')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Test MSE (1326 samples)', fontsize=11)
    ax.legend(loc='upper right', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, max(ci_his) * 1.15)

    # --- Right panel: Summary table ---
    ax2 = axes[1]
    ax2.axis('off')

    table_data = [['Metric', 'Ginkgo-AA0', 'ESM2-650M\n(drop=0.35)']]

    if 'dropout_0.35' in results:
        esm_ci = results['dropout_0.35']['bootstrap_ci']
        ginkgo_mse_str = f'{GINKGO_BEST_MSE:.3f}\n[{GINKGO_BEST_CI[0]:.3f}, {GINKGO_BEST_CI[1]:.3f}]'
        esm_mse_str = (f"{esm_ci['mse']['point']:.3f}\n"
                       f"[{esm_ci['mse']['ci_lo']:.3f}, {esm_ci['mse']['ci_hi']:.3f}]")
        table_data.append(['MSE', ginkgo_mse_str, esm_mse_str])

        esm_r2_str = (f"{esm_ci['r2']['point']:.3f}\n"
                      f"[{esm_ci['r2']['ci_lo']:.3f}, {esm_ci['r2']['ci_hi']:.3f}]")
        table_data.append(['R^2', '--', esm_r2_str])

        esm_sp_str = (f"{esm_ci['spearman']['point']:.3f}\n"
                      f"[{esm_ci['spearman']['ci_lo']:.3f}, {esm_ci['spearman']['ci_hi']:.3f}]")
        table_data.append(['Spearman', '--', esm_sp_str])

        # Gap calculation
        gap = esm_ci['mse']['point'] - GINKGO_BEST_MSE
        pct_gap = 100 * gap / GINKGO_BEST_MSE
        sign = '+' if gap >= 0 else ''
        table_data.append(['Gap', '', f'{sign}{gap:.3f} ({sign}{pct_gap:.1f}%)'])

    table = ax2.table(cellText=[row for row in table_data],
                      cellLoc='center', loc='center',
                      colWidths=[0.3, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)

    # Style header row
    for j in range(3):
        table[0, j].set_facecolor('#E0E0E0')
        table[0, j].set_text_props(fontweight='bold')

    ax2.text(0.5, 0.98, 'Bootstrap 95\\% CIs (10,000 resamples)', fontsize=10,
             ha='center', va='top', transform=ax2.transAxes)

    plt.tight_layout()
    fig_path = FIGURES_DIR / 'esm2_optimization.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved to {fig_path}")


# ==========================================================================
# Main
# ==========================================================================

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    t_total = time.time()

    # Load ESM2-650M data ------------------------------------------------
    print("Loading ESM2-650M data (with bins)...")
    X_train_raw, _, _, _, y_train_bins, _, meta_bins = load_plm_with_bins('esm2-650M')
    _, X_test_raw, _, y_test, meta_full = load_plm_embeddings('esm2-650M')

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train_raw)
    X_te = scaler.transform(X_test_raw)
    dim = X_tr.shape[1]

    print(f"  Train: {meta_bins['n_train']} (dropped {meta_bins['n_train_dropped_nan_bins']} NaN-bin rows)")
    print(f"  Test:  {meta_full['n_test']}, Dim: {dim}")
    print(f"  Embedding model: ESM2-650M (open-weight)")

    all_results['data'] = {
        'embedding_model': 'ESM2-650M',
        'embedding_dim': dim,
        'n_train': meta_bins['n_train'],
        'n_train_dropped_nan_bins': meta_bins['n_train_dropped_nan_bins'],
        'n_test': meta_full['n_test'],
    }

    # ======================================================================
    # Config 1: dropout=0.35 (Script 10 best; full-data optimal)
    # ======================================================================
    print(f"\n{'='*60}")
    print("  CONFIG 1: dropout=0.35 (full-data optimal)")
    print(f"{'='*60}")

    cfg_035 = {
        'hidden_layers': (256, 256, 128),
        'dropout': 0.35,
        'lr': 5e-4,
        'loss': 'focal',
        'epochs': 300,
        'batch_size': 32,
    }

    mse_035, y_ens_035, preds_035 = run_ensemble(
        X_tr, y_train_bins, dim, cfg_035, SEEDS, X_te, y_test, 'ESM2-650M drop=0.35')

    print("\n  Computing bootstrap CIs (10,000 resamples)...")
    ci_035 = bootstrap_ci(y_test, y_ens_035)
    print(f"  MSE:      {ci_035['mse']['point']:.4f}  [{ci_035['mse']['ci_lo']:.4f}, {ci_035['mse']['ci_hi']:.4f}]")
    print(f"  R^2:      {ci_035['r2']['point']:.4f}  [{ci_035['r2']['ci_lo']:.4f}, {ci_035['r2']['ci_hi']:.4f}]")
    print(f"  Spearman: {ci_035['spearman']['point']:.4f}  [{ci_035['spearman']['ci_lo']:.4f}, {ci_035['spearman']['ci_hi']:.4f}]")

    all_results['dropout_0.35'] = {
        'config': {k: list(v) if isinstance(v, tuple) else v for k, v in cfg_035.items()},
        'ensemble_mse': mse_035,
        'seed_mses': [mse(y_test, p) for p in preds_035],
        'bootstrap_ci': ci_035,
    }
    save_results(all_results)

    # ======================================================================
    # Config 2: dropout=0.20 (validation-optimal baseline)
    # ======================================================================
    print(f"\n{'='*60}")
    print("  CONFIG 2: dropout=0.20 (validation-optimal)")
    print(f"{'='*60}")

    cfg_020 = {
        'hidden_layers': (256, 256, 128),
        'dropout': 0.20,
        'lr': 5e-4,
        'loss': 'focal',
        'epochs': 300,
        'batch_size': 32,
    }

    mse_020, y_ens_020, preds_020 = run_ensemble(
        X_tr, y_train_bins, dim, cfg_020, SEEDS, X_te, y_test, 'ESM2-650M drop=0.20')

    print("\n  Computing bootstrap CIs (10,000 resamples)...")
    ci_020 = bootstrap_ci(y_test, y_ens_020)
    print(f"  MSE:      {ci_020['mse']['point']:.4f}  [{ci_020['mse']['ci_lo']:.4f}, {ci_020['mse']['ci_hi']:.4f}]")
    print(f"  R^2:      {ci_020['r2']['point']:.4f}  [{ci_020['r2']['ci_lo']:.4f}, {ci_020['r2']['ci_hi']:.4f}]")
    print(f"  Spearman: {ci_020['spearman']['point']:.4f}  [{ci_020['spearman']['ci_lo']:.4f}, {ci_020['spearman']['ci_hi']:.4f}]")

    all_results['dropout_0.20'] = {
        'config': {k: list(v) if isinstance(v, tuple) else v for k, v in cfg_020.items()},
        'ensemble_mse': mse_020,
        'seed_mses': [mse(y_test, p) for p in preds_020],
        'bootstrap_ci': ci_020,
    }
    save_results(all_results)

    # ======================================================================
    # Final summary
    # ======================================================================
    total_min = (time.time() - t_total) / 60

    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY  ({total_min:.1f} min)")
    print(f"{'='*60}")
    print(f"  Embedding: ESM2-650M (open-weight, 650M params)")
    print(f"  Architecture: (256, 256, 128), focal loss, 5-seed ensemble")
    print()
    print(f"  {'Config':<22s} {'MSE':>8s}  {'95% CI':>20s}")
    print(f"  {'-'*52}")

    for key, label in [('dropout_0.35', 'drop=0.35 (best)'), ('dropout_0.20', 'drop=0.20 (val-opt)')]:
        r = all_results[key]
        ci = r['bootstrap_ci']
        print(f"  {label:<22s} {ci['mse']['point']:8.4f}  [{ci['mse']['ci_lo']:.4f}, {ci['mse']['ci_hi']:.4f}]")

    print()
    print(f"  Reference: Ginkgo-AA0 best = {GINKGO_BEST_MSE:.3f}  [{GINKGO_BEST_CI[0]:.3f}, {GINKGO_BEST_CI[1]:.3f}]")
    print(f"  Reference: net4 reference = {NET4_REFERENCE_MSE}")

    # Which ESM2 config was best?
    best_key = 'dropout_0.35' if mse_035 <= mse_020 else 'dropout_0.20'
    best_esm = all_results[best_key]['bootstrap_ci']['mse']['point']

    gap_ginkgo = best_esm - GINKGO_BEST_MSE
    pct_ginkgo = 100 * gap_ginkgo / GINKGO_BEST_MSE
    gap_prior = best_esm - NET4_REFERENCE_MSE
    pct_prior = 100 * gap_prior / NET4_REFERENCE_MSE

    print()
    if best_esm < NET4_REFERENCE_MSE:
        print(f"  ESM2-650M best ({best_key}): {best_esm:.4f}")
        print(f"    -> below net4 reference ({NET4_REFERENCE_MSE}) by {-gap_prior:.4f} ({-pct_prior:.1f}%)")
    else:
        print(f"  ESM2-650M best ({best_key}): {best_esm:.4f}")
        print(f"    -> Gap to net4 reference: +{gap_prior:.4f} ({pct_prior:+.1f}%)")

    sign = '+' if gap_ginkgo >= 0 else ''
    print(f"    -> Gap to Ginkgo-AA0: {sign}{gap_ginkgo:.4f} ({sign}{pct_ginkgo:.1f}%)")

    # Overlapping CIs?
    esm_ci = all_results[best_key]['bootstrap_ci']['mse']
    overlap = esm_ci['ci_lo'] <= GINKGO_BEST_CI[1] and esm_ci['ci_hi'] >= GINKGO_BEST_CI[0]
    print(f"    -> 95% CIs {'overlap' if overlap else 'do NOT overlap'} with Ginkgo-AA0")

    all_results['summary'] = {
        'best_esm2_config': best_key,
        'best_esm2_mse': best_esm,
        'ginkgo_best_mse': GINKGO_BEST_MSE,
        'ginkgo_best_ci': list(GINKGO_BEST_CI),
        'prior_benchmark_mse': NET4_REFERENCE_MSE,
        'gap_to_ginkgo': gap_ginkgo,
        'gap_to_ginkgo_pct': pct_ginkgo,
        'gap_to_prior': gap_prior,
        'gap_to_prior_pct': pct_prior,
        'cis_overlap': overlap,
        'total_time_min': total_min,
    }
    save_results(all_results)

    # Figure -------------------------------------------------------------
    _make_figure(all_results)

    return all_results


if __name__ == '__main__':
    main()
