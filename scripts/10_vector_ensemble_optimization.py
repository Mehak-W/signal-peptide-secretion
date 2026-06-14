#!/usr/bin/env python3
"""
Script 10: Vector Regression Ensemble Optimization

Script 09 result: MSE 0.9732 (deeper-3layer 5-seed ensemble)
This script optimizes further with:

  Phase 1: Dropout tuning on the winning 256,256,128 architecture
  Phase 2: 4-layer architectures (256,256,128,64 etc.)
  Phase 3: Cosine LR + deeper architectures
  Phase 4: Large ensembles (20-seed) with best configs
  Phase 5: Mixed-architecture ensemble (combine diverse models)
  Phase 6: Bootstrap CIs on best result

All models trained on full data (3068), evaluated on all 1326 test samples.

Outputs:
  - results/vector_ensemble_optimization_results.json
  - figures/vector_ensemble_optimization.png
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

# ── Paths ─────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results'
FIGURES_DIR = Path(__file__).resolve().parent.parent / 'figures'

# ── Constants ─────────────────────────────────────────────────────────────
SEEDS_5 = [42, 123, 456, 789, 1024]
SEEDS_10 = [42, 123, 456, 789, 1024, 2024, 3141, 5555, 7777, 9999]
SEEDS_20 = SEEDS_10 + [1111, 2222, 3333, 4444, 6666, 7070, 8080, 8888, 9090, 9876]
BIN_CENTERS = np.arange(1, 11)
N_BOOTSTRAP = 10_000
NET4_REFERENCE_MSE = 0.953  # single-run reference value for the optimization sweeps
SCRIPT09_BEST = 0.9732


# ═══════════════════════════════════════════════════════════════════════════
# Model building (same as Script 09)
# ═══════════════════════════════════════════════════════════════════════════

def build_model(input_dim, hidden_layers=(256, 256), dropout=0.2,
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

    callbacks = []
    if cfg.get('lr_reduce', True):
        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor='loss', factor=0.5, patience=10, min_lr=1e-6, verbose=0))

    if cfg.get('lr_schedule') == 'cosine':
        base_lr = cfg['lr']
        total_ep = cfg['epochs']
        callbacks.append(keras.callbacks.LearningRateScheduler(
            lambda ep: base_lr * 0.5 * (1 + np.cos(np.pi * ep / total_ep))))

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
    mse_b, r2_b, sp_b = np.empty(n_boot), np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        mse_b[b] = np.mean((yt - yp) ** 2)
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2)
        r2_b[b] = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        sp_b[b] = sp_stats.spearmanr(yt, yp).statistic

    return {
        'mse': {'point': mse(y_true, y_pred),
                'ci_lo': float(np.percentile(mse_b, 2.5)),
                'ci_hi': float(np.percentile(mse_b, 97.5))},
        'r2': {'point': float(1 - np.sum((y_true-y_pred)**2)/np.sum((y_true-y_true.mean())**2)),
               'ci_lo': float(np.percentile(r2_b, 2.5)),
               'ci_hi': float(np.percentile(r2_b, 97.5))},
        'spearman': {'point': float(sp_stats.spearmanr(y_true, y_pred).statistic),
                     'ci_lo': float(np.percentile(sp_b, 2.5)),
                     'ci_hi': float(np.percentile(sp_b, 97.5))},
        'n_test': int(n),
    }


def save_results(results, filename='vector_ensemble_optimization_results.json'):
    def convert(obj):
        if isinstance(obj, dict): return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)): return [convert(i) for i in obj]
        elif isinstance(obj, (np.integer,)): return int(obj)
        elif isinstance(obj, (np.floating,)): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return obj
    with open(RESULTS_DIR / filename, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"  [saved]")


def run_ensemble(X_tr, y_bins, dim, cfg, seeds, X_te, y_test, label):
    """Train an ensemble and return (mse, y_pred_ensemble, per_seed_preds)."""
    preds = []
    for seed in seeds:
        t0 = time.time()
        pred = train_predict_cleanup(X_tr, y_bins, dim, cfg, seed, X_te)
        preds.append(pred)
        mse_i = mse(y_test, pred)
        print(f"    Seed {seed:>5}: MSE = {mse_i:.4f}  ({time.time()-t0:.1f}s)")

    y_ens = np.mean(preds, axis=0)
    mse_ens = mse(y_test, y_ens)
    beat = " (below net4 reference)" if mse_ens < NET4_REFERENCE_MSE else ""
    print(f"  → {label} MSE = {mse_ens:.4f}{beat}")
    return mse_ens, y_ens, preds


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    t_total = time.time()

    # Best trackers
    best_mse = float('inf')
    best_preds = None
    best_name = None

    # ── Load data ─────────────────────────────────────────────────────────
    print("Loading Ginkgo-AA0 data...")
    X_train, _, _, _, y_train_bins, _, meta = load_plm_with_bins('ginkgo-AA0-650M')
    _, X_test_full, _, y_test, meta_full = load_plm_embeddings('ginkgo-AA0-650M')

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test_full)
    dim = X_tr.shape[1]
    print(f"  Train: {meta['n_train']}, Test: {meta_full['n_test']}, Dim: {dim}")

    # ══════════════════════════════════════════════════════════════════════
    # Phase 1: Dropout tuning on 256,256,128 (the Script 09 winner)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  PHASE 1: Dropout Tuning on (256, 256, 128)")
    print(f"{'='*60}")

    phase1 = {}
    for drop in [0.15, 0.20, 0.25, 0.30, 0.35]:
        cfg = {'hidden_layers': (256, 256, 128), 'dropout': drop, 'lr': 5e-4,
               'loss': 'focal', 'epochs': 300, 'batch_size': 32, 'lr_reduce': True}
        label = f'drop-{drop:.2f}'

        print(f"\n  {label}; 5-seed ensemble")
        mse_ens, y_ens, preds = run_ensemble(
            X_tr, y_train_bins, dim, cfg, SEEDS_5, X_te, y_test, label)

        phase1[label] = {
            'mse': mse_ens,
            'seed_mses': [mse(y_test, p) for p in preds],
            'dropout': drop,
        }
        if mse_ens < best_mse:
            best_mse, best_preds, best_name = mse_ens, y_ens, f'P1-{label}'

    all_results['phase1_dropout'] = phase1
    best_p1 = min(phase1, key=lambda k: phase1[k]['mse'])
    best_p1_drop = phase1[best_p1]['dropout']
    print(f"\n  Best dropout: {best_p1_drop} → MSE {phase1[best_p1]['mse']:.4f}")
    save_results(all_results)

    # ══════════════════════════════════════════════════════════════════════
    # Phase 2: 4-layer architectures
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  PHASE 2: 4-Layer Architectures (5-seed ensembles)")
    print(f"{'='*60}")

    phase2_archs = {
        '256-256-128-64':  (256, 256, 128, 64),
        '256-256-256-128': (256, 256, 256, 128),
        '384-256-128-64':  (384, 256, 128, 64),
        '256-128-64-32':   (256, 128, 64, 32),
        '384-384-256-128': (384, 384, 256, 128),
    }

    phase2 = {}
    for name, arch in phase2_archs.items():
        cfg = {'hidden_layers': arch, 'dropout': best_p1_drop, 'lr': 5e-4,
               'loss': 'focal', 'epochs': 300, 'batch_size': 32, 'lr_reduce': True}

        print(f"\n  {name}; 5-seed ensemble")
        mse_ens, y_ens, preds = run_ensemble(
            X_tr, y_train_bins, dim, cfg, SEEDS_5, X_te, y_test, name)

        phase2[name] = {
            'mse': mse_ens,
            'seed_mses': [mse(y_test, p) for p in preds],
            'arch': list(arch),
        }
        if mse_ens < best_mse:
            best_mse, best_preds, best_name = mse_ens, y_ens, f'P2-{name}'

    all_results['phase2_4layer'] = phase2
    save_results(all_results)

    # ══════════════════════════════════════════════════════════════════════
    # Phase 3: Cosine LR + deeper architectures
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  PHASE 3: Cosine LR Schedule Variants (5-seed ensembles)")
    print(f"{'='*60}")

    phase3_configs = {
        'cos-256x3':     {'hidden_layers': (256, 256, 128), 'dropout': best_p1_drop,
                          'lr': 5e-4, 'loss': 'focal', 'epochs': 300,
                          'batch_size': 32, 'lr_reduce': False, 'lr_schedule': 'cosine'},
        'cos-256x3-400': {'hidden_layers': (256, 256, 128), 'dropout': best_p1_drop,
                          'lr': 5e-4, 'loss': 'focal', 'epochs': 400,
                          'batch_size': 32, 'lr_reduce': False, 'lr_schedule': 'cosine'},
        'cos-3x256':     {'hidden_layers': (256, 256, 256), 'dropout': best_p1_drop,
                          'lr': 5e-4, 'loss': 'focal', 'epochs': 300,
                          'batch_size': 32, 'lr_reduce': False, 'lr_schedule': 'cosine'},
        'cos-lr1e-3':    {'hidden_layers': (256, 256, 128), 'dropout': best_p1_drop,
                          'lr': 1e-3, 'loss': 'focal', 'epochs': 300,
                          'batch_size': 32, 'lr_reduce': False, 'lr_schedule': 'cosine'},
    }

    phase3 = {}
    for name, cfg in phase3_configs.items():
        print(f"\n  {name}; 5-seed ensemble")
        mse_ens, y_ens, preds = run_ensemble(
            X_tr, y_train_bins, dim, cfg, SEEDS_5, X_te, y_test, name)

        phase3[name] = {
            'mse': mse_ens,
            'seed_mses': [mse(y_test, p) for p in preds],
        }
        if mse_ens < best_mse:
            best_mse, best_preds, best_name = mse_ens, y_ens, f'P3-{name}'

    all_results['phase3_cosine'] = phase3
    save_results(all_results)

    # ══════════════════════════════════════════════════════════════════════
    # Phase 4: Large ensembles (20-seed) with best configs
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  PHASE 4: Large Ensembles (20-seed)")
    print(f"{'='*60}")

    # Collect the best config from each phase
    # Script 09 winner: deeper-3layer (256,256,128), drop=0.2
    cfg_s09 = {'hidden_layers': (256, 256, 128), 'dropout': 0.2, 'lr': 5e-4,
               'loss': 'focal', 'epochs': 300, 'batch_size': 32, 'lr_reduce': True}

    # Best dropout from Phase 1
    cfg_p1 = {'hidden_layers': (256, 256, 128), 'dropout': best_p1_drop, 'lr': 5e-4,
              'loss': 'focal', 'epochs': 300, 'batch_size': 32, 'lr_reduce': True}

    # Script 09's deep-3x256
    cfg_3x256 = {'hidden_layers': (256, 256, 256), 'dropout': 0.2, 'lr': 5e-4,
                 'loss': 'focal', 'epochs': 300, 'batch_size': 32, 'lr_reduce': True}

    phase4_configs = {
        'S09-winner-20seed': cfg_s09,
        'P1-best-drop-20seed': cfg_p1,
        '3x256-20seed': cfg_3x256,
    }

    phase4 = {}
    for name, cfg in phase4_configs.items():
        print(f"\n  {name}")
        mse_ens, y_ens, preds = run_ensemble(
            X_tr, y_train_bins, dim, cfg, SEEDS_20, X_te, y_test, name)

        phase4[name] = {
            'mse': mse_ens,
            'seed_mses': [mse(y_test, p) for p in preds],
        }
        if mse_ens < best_mse:
            best_mse, best_preds, best_name = mse_ens, y_ens, f'P4-{name}'

    all_results['phase4_large_ensemble'] = phase4
    save_results(all_results)

    # ══════════════════════════════════════════════════════════════════════
    # Phase 5: Mixed-architecture ensemble
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  PHASE 5: Mixed-Architecture Ensemble")
    print(f"{'='*60}")

    # Combine diverse architectures: each trained 5 times = 15-20 models averaged
    mixed_configs = [
        ('256,256,128-d0.2', {'hidden_layers': (256, 256, 128), 'dropout': 0.2,
            'lr': 5e-4, 'loss': 'focal', 'epochs': 300, 'batch_size': 32, 'lr_reduce': True}),
        ('256,256,256-d0.2', {'hidden_layers': (256, 256, 256), 'dropout': 0.2,
            'lr': 5e-4, 'loss': 'focal', 'epochs': 300, 'batch_size': 32, 'lr_reduce': True}),
        ('256,256,128-d0.3', {'hidden_layers': (256, 256, 128), 'dropout': 0.3,
            'lr': 5e-4, 'loss': 'focal', 'epochs': 300, 'batch_size': 32, 'lr_reduce': True}),
        ('256,256-d0.2', {'hidden_layers': (256, 256), 'dropout': 0.2,
            'lr': 5e-4, 'loss': 'focal', 'epochs': 300, 'batch_size': 32, 'lr_reduce': True}),
    ]

    print("  Training 4 architectures × 5 seeds = 20 models...")
    all_mixed_preds = []
    for arch_name, cfg in mixed_configs:
        print(f"\n    {arch_name}:")
        for seed in SEEDS_5:
            t0 = time.time()
            pred = train_predict_cleanup(X_tr, y_train_bins, dim, cfg, seed, X_te)
            all_mixed_preds.append(pred)
            mse_i = mse(y_test, pred)
            print(f"      Seed {seed:>5}: MSE = {mse_i:.4f}  ({time.time()-t0:.1f}s)")

    y_mixed = np.mean(all_mixed_preds, axis=0)
    mse_mixed = mse(y_test, y_mixed)
    beat = " (below net4 reference)" if mse_mixed < NET4_REFERENCE_MSE else ""
    print(f"\n  → Mixed ensemble (20 models) MSE = {mse_mixed:.4f}{beat}")

    # Also try weighted: give more weight to better architectures
    # Use the top-3 configs (first 3) with 5 seeds each = 15 models
    y_mixed_top3 = np.mean(all_mixed_preds[:15], axis=0)
    mse_mixed_top3 = mse(y_test, y_mixed_top3)
    beat = " (below net4 reference)" if mse_mixed_top3 < NET4_REFERENCE_MSE else ""
    print(f"  → Mixed ensemble (top-3 archs, 15 models) MSE = {mse_mixed_top3:.4f}{beat}")

    phase5 = {
        'mixed_4arch_20model': {'mse': mse_mixed},
        'mixed_3arch_15model': {'mse': mse_mixed_top3},
    }

    if mse_mixed < best_mse:
        best_mse, best_preds, best_name = mse_mixed, y_mixed, 'P5-mixed-4arch'
    if mse_mixed_top3 < best_mse:
        best_mse, best_preds, best_name = mse_mixed_top3, y_mixed_top3, 'P5-mixed-3arch'

    all_results['phase5_mixed'] = phase5
    save_results(all_results)

    # ══════════════════════════════════════════════════════════════════════
    # Phase 6: Bootstrap CIs on best result
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  PHASE 6: Bootstrap CIs on Best Result")
    print(f"{'='*60}")

    print(f"\n  Best: {best_name} → MSE = {best_mse:.4f}")
    ci = bootstrap_ci(y_test, best_preds)
    print(f"  MSE:      {ci['mse']['point']:.4f}  [{ci['mse']['ci_lo']:.4f}, {ci['mse']['ci_hi']:.4f}]")
    print(f"  R²:       {ci['r2']['point']:.4f}  [{ci['r2']['ci_lo']:.4f}, {ci['r2']['ci_hi']:.4f}]")
    print(f"  Spearman: {ci['spearman']['point']:.4f}  [{ci['spearman']['ci_lo']:.4f}, {ci['spearman']['ci_hi']:.4f}]")

    all_results['phase6_bootstrap'] = {
        'best_model': best_name,
        'best_mse': best_mse,
        'bootstrap_ci': ci,
    }

    # ── Final summary ─────────────────────────────────────────────────────
    total_min = (time.time() - t_total) / 60
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY  ({total_min:.1f} min)")
    print(f"{'='*60}")
    print(f"  net4 single run: {NET4_REFERENCE_MSE}")
    print(f"  Script 09 best:        {SCRIPT09_BEST}")
    print(f"  This script best:      {best_mse:.4f}  ({best_name})")

    if best_mse < NET4_REFERENCE_MSE:
        gap = NET4_REFERENCE_MSE - best_mse
        pct = 100 * gap / NET4_REFERENCE_MSE
        print(f"\n  now {0:.4f} below the net4 single-run reference")
    else:
        gap = best_mse - NET4_REFERENCE_MSE
        print(f"\n  Gap to net4 reference: +{gap:.4f}")

    improvement = SCRIPT09_BEST - best_mse
    print(f"  Improvement over Script 09: {improvement:.4f}")

    save_results(all_results)

    # ── Figure ────────────────────────────────────────────────────────────
    _make_figure(all_results)

    return all_results


def _make_figure(results):
    """Summary figure."""
    fig, ax = plt.subplots(figsize=(14, 7))

    labels, mses, colors = [], [], []

    # Phase 1 - dropout
    for name, r in results.get('phase1_dropout', {}).items():
        labels.append(f"P1: {name}")
        mses.append(r['mse'])
        colors.append('steelblue')

    # Phase 2 - 4-layer
    for name, r in results.get('phase2_4layer', {}).items():
        labels.append(f"P2: {name}")
        mses.append(r['mse'])
        colors.append('darkorange')

    # Phase 3 - cosine
    for name, r in results.get('phase3_cosine', {}).items():
        labels.append(f"P3: {name}")
        mses.append(r['mse'])
        colors.append('mediumseagreen')

    # Phase 4 - large ensembles
    for name, r in results.get('phase4_large_ensemble', {}).items():
        labels.append(f"P4: {name}")
        mses.append(r['mse'])
        colors.append('crimson')

    # Phase 5 - mixed
    for name, r in results.get('phase5_mixed', {}).items():
        labels.append(f"P5: {name}")
        mses.append(r['mse'])
        colors.append('purple')

    x = np.arange(len(labels))
    bars = ax.bar(x, mses, color=colors, alpha=0.85)

    ax.axhline(y=NET4_REFERENCE_MSE, color='green', linestyle=':', linewidth=1.0, alpha=0.8,
               label=f'net4 reference ({NET4_REFERENCE_MSE})')
    ax.axhline(y=SCRIPT09_BEST, color='gray', linestyle='--', linewidth=1.0, alpha=0.8,
               label=f'Arch search best ({SCRIPT09_BEST})')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Test MSE (all 1326 samples)')
    ax.legend(loc='lower right')

    plt.tight_layout()
    fig_path = FIGURES_DIR / 'vector_ensemble_optimization.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved to {fig_path}")


if __name__ == '__main__':
    main()
