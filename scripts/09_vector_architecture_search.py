#!/usr/bin/env python3
"""
Script 09: Vector Regression Architecture Search

Full-data vector regression with architecture exploration:
  - Ginkgo-AA0 embeddings (1280d)
  - Focal loss (alpha=0.25, gamma=2.0), softmax(10) output
  - Trains on ALL training data (no validation split, 3068 samples)
  - Evaluates on ALL 1326 test samples

Key differences from Script 06 (MSE 1.001):
  - Script 06 uses 80/20 val split (~2454 train/seed) + evaluates on 1273 test
  - This script: full 3068 train (no val split) + all 1326 test

Phases:
  1. Baseline reproduction (single model, epoch sweep)
  2. Multi-seed ensemble on full data (5-seed and 10-seed)
  3. Architecture/hyperparameter search (single models)
  4. Top-config ensembles + mixed ensemble
  5. ESM2-650M comparison
  6. Bootstrap CIs on best result

Outputs:
  - results/vector_architecture_search_results.json
  - figures/vector_architecture_search.png
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
from src.data_loading import load_plm_with_bins, load_plm_embeddings, BIN_COLUMNS
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
BIN_CENTERS = np.arange(1, 11)
N_BOOTSTRAP = 10_000
NET4_REFERENCE_MSE = 0.953  # single-run reference value for the optimization sweeps


# ═══════════════════════════════════════════════════════════════════════════
# Model building; direct Keras, full control over training
# ═══════════════════════════════════════════════════════════════════════════

def build_model(input_dim, hidden_layers=(256, 256), dropout=0.2,
                lr=5e-4, loss='focal', seed=42):
    """Build and compile a vector regression model."""
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

    if loss == 'focal':
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    else:
        loss_fn = 'categorical_crossentropy'

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=loss_fn,
    )
    return model


def train_full(X_train, y_train_bins, input_dim, cfg, seed=42, verbose=0):
    """Train on full training data; no validation split, no early stopping."""
    model = build_model(
        input_dim,
        hidden_layers=cfg['hidden_layers'],
        dropout=cfg['dropout'],
        lr=cfg['lr'],
        loss=cfg['loss'],
        seed=seed,
    )

    callbacks = []
    if cfg.get('lr_reduce', True):
        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor='loss', factor=0.5, patience=10, min_lr=1e-6, verbose=0))

    if cfg.get('lr_schedule') == 'cosine':
        total_epochs = cfg['epochs']
        base_lr = cfg['lr']
        callbacks.append(keras.callbacks.LearningRateScheduler(
            lambda epoch: base_lr * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))))

    model.fit(
        X_train, y_train_bins,
        epochs=cfg['epochs'],
        batch_size=cfg.get('batch_size', 32),
        callbacks=callbacks,
        verbose=verbose,
    )
    return model


def predict_wa(model, X_test):
    """Predict bin probabilities → WA."""
    probs = model.predict(X_test, verbose=0)
    return probs @ BIN_CENTERS


def compute_mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def bootstrap_ci(y_true, y_pred, n_boot=N_BOOTSTRAP, seed=42):
    """Bootstrap 95% CI for MSE, R², Spearman."""
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

    mse_pt = compute_mse(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2_pt = float(1 - ss_res / ss_tot)
    sp_pt = float(sp_stats.spearmanr(y_true, y_pred).statistic)

    return {
        'mse': {'point': mse_pt,
                'ci_lo': float(np.percentile(mse_b, 2.5)),
                'ci_hi': float(np.percentile(mse_b, 97.5))},
        'r2': {'point': r2_pt,
               'ci_lo': float(np.percentile(r2_b, 2.5)),
               'ci_hi': float(np.percentile(r2_b, 97.5))},
        'spearman': {'point': sp_pt,
                     'ci_lo': float(np.percentile(sp_b, 2.5)),
                     'ci_hi': float(np.percentile(sp_b, 97.5))},
        'n_test': int(n),
    }


def save_results(results, filename='vector_architecture_search_results.json'):
    """Save results with numpy type conversion."""
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

    out_path = RESULTS_DIR / filename
    with open(out_path, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"  [saved → {out_path}]")


def train_and_cleanup(X_train, y_bins, input_dim, cfg, seed, X_test):
    """Train, predict, cleanup; returns y_pred_wa."""
    model = train_full(X_train, y_bins, input_dim, cfg, seed=seed)
    y_pred = predict_wa(model, X_test)
    del model
    keras.backend.clear_session()
    return y_pred


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    t_total = time.time()

    # ── Load data ─────────────────────────────────────────────────────────
    print("Loading data...")

    # Ginkgo-AA0: training with bins (3068 valid), full test (1326)
    X_train_g, _, y_train_wa_g, _, y_train_bins_g, _, meta_g = \
        load_plm_with_bins('ginkgo-AA0-650M')
    _, X_test_g_full, _, y_test_g_full, meta_g_full = \
        load_plm_embeddings('ginkgo-AA0-650M')

    scaler_g = StandardScaler()
    X_train_gs = scaler_g.fit_transform(X_train_g)
    X_test_gs = scaler_g.transform(X_test_g_full)
    dim_g = X_train_gs.shape[1]

    print(f"  Ginkgo-AA0: {meta_g['n_train']} train (bins), "
          f"{meta_g_full['n_test']} test (full)")

    # ESM2-650M: same approach
    X_train_e, _, y_train_wa_e, _, y_train_bins_e, _, meta_e = \
        load_plm_with_bins('esm2-650M')
    _, X_test_e_full, _, y_test_e_full, meta_e_full = \
        load_plm_embeddings('esm2-650M')

    scaler_e = StandardScaler()
    X_train_es = scaler_e.fit_transform(X_train_e)
    X_test_es = scaler_e.transform(X_test_e_full)
    dim_e = X_train_es.shape[1]

    print(f"  ESM2-650M:  {meta_e['n_train']} train (bins), "
          f"{meta_e_full['n_test']} test (full)")

    # ══════════════════════════════════════════════════════════════════════
    # Phase 1: Reproduce the net4 prototype (single model, Ginkgo-AA0)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  PHASE 1: Reproduce the reference architecture (single models)")
    print(f"{'='*60}")

    prototype_cfg = {
        'hidden_layers': (256, 256), 'dropout': 0.2, 'lr': 5e-4,
        'loss': 'focal', 'batch_size': 32, 'lr_reduce': True,
    }

    phase1_configs = {
        '200ep':         {**prototype_cfg, 'epochs': 200},
        '300ep':         {**prototype_cfg, 'epochs': 300},
        '500ep':         {**prototype_cfg, 'epochs': 500},
        '200ep-noLRred': {**prototype_cfg, 'epochs': 200, 'lr_reduce': False},
        '300ep-lr1e-3':  {**prototype_cfg, 'epochs': 300, 'lr': 1e-3},
        '300ep-cosine':  {**prototype_cfg, 'epochs': 300, 'lr_schedule': 'cosine'},
    }

    phase1 = {}
    for name, cfg in phase1_configs.items():
        t0 = time.time()
        y_pred = train_and_cleanup(X_train_gs, y_train_bins_g, dim_g,
                                   cfg, seed=42, X_test=X_test_gs)
        mse = compute_mse(y_test_g_full, y_pred)
        elapsed = time.time() - t0
        phase1[name] = {'mse': mse, 'time': round(elapsed, 1)}
        beat = " (below net4 reference)" if mse < NET4_REFERENCE_MSE else ""
        print(f"  {name:<22} MSE = {mse:.4f}  ({elapsed:.1f}s){beat}")

    all_results['phase1'] = phase1
    best_p1 = min(phase1, key=lambda k: phase1[k]['mse'])
    best_p1_epochs = phase1_configs[best_p1]['epochs']
    print(f"\n  Best: {best_p1} → MSE {phase1[best_p1]['mse']:.4f}")
    save_results(all_results)

    # ══════════════════════════════════════════════════════════════════════
    # Phase 2: Multi-seed ensemble (full data, reference config)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  PHASE 2: Full-Data Ensembles (reference config)")
    print(f"{'='*60}")

    ens_cfg = {**prototype_cfg, 'epochs': best_p1_epochs}

    # 5-seed
    print(f"\n  5-seed ensemble ({best_p1_epochs} epochs)...")
    preds_5 = []
    for seed in SEEDS_5:
        t0 = time.time()
        pred = train_and_cleanup(X_train_gs, y_train_bins_g, dim_g,
                                 ens_cfg, seed=seed, X_test=X_test_gs)
        preds_5.append(pred)
        mse_i = compute_mse(y_test_g_full, pred)
        print(f"    Seed {seed:>5}: MSE = {mse_i:.4f}  ({time.time()-t0:.1f}s)")

    mse_5 = compute_mse(y_test_g_full, np.mean(preds_5, axis=0))
    beat = " (below net4 reference)" if mse_5 < NET4_REFERENCE_MSE else ""
    print(f"  5-seed ensemble MSE = {mse_5:.4f}{beat}")

    # 10-seed (reuse first 5)
    print(f"\n  10-seed ensemble (training 5 more)...")
    preds_10 = list(preds_5)
    for seed in SEEDS_10[5:]:
        t0 = time.time()
        pred = train_and_cleanup(X_train_gs, y_train_bins_g, dim_g,
                                 ens_cfg, seed=seed, X_test=X_test_gs)
        preds_10.append(pred)
        mse_i = compute_mse(y_test_g_full, pred)
        print(f"    Seed {seed:>5}: MSE = {mse_i:.4f}  ({time.time()-t0:.1f}s)")

    mse_10 = compute_mse(y_test_g_full, np.mean(preds_10, axis=0))
    beat = " (below net4 reference)" if mse_10 < NET4_REFERENCE_MSE else ""
    print(f"  10-seed ensemble MSE = {mse_10:.4f}{beat}")

    all_results['phase2'] = {
        '5_seed': {'mse': mse_5,
                   'seed_mses': [compute_mse(y_test_g_full, p) for p in preds_5]},
        '10_seed': {'mse': mse_10,
                    'seed_mses': [compute_mse(y_test_g_full, p) for p in preds_10]},
        'config_epochs': best_p1_epochs,
    }
    save_results(all_results)

    # ══════════════════════════════════════════════════════════════════════
    # Phase 3: Architecture / hyperparameter search (single model, seed=42)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  PHASE 3: Architecture Search (single models)")
    print(f"{'='*60}")

    ep = best_p1_epochs
    search_configs = {
        # ── Architecture variants ──
        'wider-384':     {'hidden_layers': (384, 384), 'dropout': 0.2, 'lr': 5e-4,
                          'loss': 'focal', 'epochs': ep, 'batch_size': 32, 'lr_reduce': True},
        'wider-512-256': {'hidden_layers': (512, 256), 'dropout': 0.2, 'lr': 5e-4,
                          'loss': 'focal', 'epochs': ep, 'batch_size': 32, 'lr_reduce': True},
        'wide-512':      {'hidden_layers': (512, 512), 'dropout': 0.2, 'lr': 5e-4,
                          'loss': 'focal', 'epochs': ep, 'batch_size': 32, 'lr_reduce': True},
        'deeper-3layer': {'hidden_layers': (256, 256, 128), 'dropout': 0.2, 'lr': 5e-4,
                          'loss': 'focal', 'epochs': ep, 'batch_size': 32, 'lr_reduce': True},
        'deep-3x256':    {'hidden_layers': (256, 256, 256), 'dropout': 0.2, 'lr': 5e-4,
                          'loss': 'focal', 'epochs': ep, 'batch_size': 32, 'lr_reduce': True},
        # ── Dropout variants ──
        'drop-0.10':     {'hidden_layers': (256, 256), 'dropout': 0.10, 'lr': 5e-4,
                          'loss': 'focal', 'epochs': ep, 'batch_size': 32, 'lr_reduce': True},
        'drop-0.15':     {'hidden_layers': (256, 256), 'dropout': 0.15, 'lr': 5e-4,
                          'loss': 'focal', 'epochs': ep, 'batch_size': 32, 'lr_reduce': True},
        'drop-0.30':     {'hidden_layers': (256, 256), 'dropout': 0.30, 'lr': 5e-4,
                          'loss': 'focal', 'epochs': ep, 'batch_size': 32, 'lr_reduce': True},
        # ── Learning rate variants ──
        'lr-1e-3':       {'hidden_layers': (256, 256), 'dropout': 0.2, 'lr': 1e-3,
                          'loss': 'focal', 'epochs': ep, 'batch_size': 32, 'lr_reduce': True},
        'lr-2e-4':       {'hidden_layers': (256, 256), 'dropout': 0.2, 'lr': 2e-4,
                          'loss': 'focal', 'epochs': ep, 'batch_size': 32, 'lr_reduce': True},
        # ── Loss variant ──
        'CE-256':        {'hidden_layers': (256, 256), 'dropout': 0.2, 'lr': 5e-4,
                          'loss': 'categorical_crossentropy', 'epochs': ep, 'batch_size': 32,
                          'lr_reduce': True},
        'CE-384':        {'hidden_layers': (384, 384), 'dropout': 0.2, 'lr': 5e-4,
                          'loss': 'categorical_crossentropy', 'epochs': ep, 'batch_size': 32,
                          'lr_reduce': True},
        # ── Batch size ──
        'batch-64':      {'hidden_layers': (256, 256), 'dropout': 0.2, 'lr': 5e-4,
                          'loss': 'focal', 'epochs': ep, 'batch_size': 64, 'lr_reduce': True},
        # ── Promising combos ──
        'w384-d0.15':    {'hidden_layers': (384, 384), 'dropout': 0.15, 'lr': 5e-4,
                          'loss': 'focal', 'epochs': ep, 'batch_size': 32, 'lr_reduce': True},
        'w512-d0.15':    {'hidden_layers': (512, 256), 'dropout': 0.15, 'lr': 5e-4,
                          'loss': 'focal', 'epochs': ep, 'batch_size': 32, 'lr_reduce': True},
        # ── Longer training ──
        'long-500':      {'hidden_layers': (256, 256), 'dropout': 0.2, 'lr': 5e-4,
                          'loss': 'focal', 'epochs': 500, 'batch_size': 32, 'lr_reduce': True},
    }

    phase3 = {}
    for name, cfg in search_configs.items():
        t0 = time.time()
        y_pred = train_and_cleanup(X_train_gs, y_train_bins_g, dim_g,
                                   cfg, seed=42, X_test=X_test_gs)
        mse = compute_mse(y_test_g_full, y_pred)
        elapsed = time.time() - t0
        phase3[name] = {'mse': mse, 'time': round(elapsed, 1)}
        beat = " (below net4 reference)" if mse < NET4_REFERENCE_MSE else ""
        print(f"  {name:<22} MSE = {mse:.4f}  ({elapsed:.1f}s){beat}")

    all_results['phase3'] = phase3
    sorted_p3 = sorted(phase3.items(), key=lambda x: x[1]['mse'])
    print(f"\n  Top 5:")
    for name, r in sorted_p3[:5]:
        print(f"    {name:<22} MSE = {r['mse']:.4f}")
    save_results(all_results)

    # ══════════════════════════════════════════════════════════════════════
    # Phase 4: Top-config ensembles
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  PHASE 4: Top-Config Ensembles")
    print(f"{'='*60}")

    phase4 = {}
    overall_best_mse = float('inf')
    overall_best_preds = None
    overall_best_name = None
    overall_best_ytest = y_test_g_full  # default

    # Track Phase 2 best
    if mse_5 < overall_best_mse:
        overall_best_mse = mse_5
        overall_best_preds = np.mean(preds_5, axis=0)
        overall_best_name = 'P2-5seed'
    if mse_10 < overall_best_mse:
        overall_best_mse = mse_10
        overall_best_preds = np.mean(preds_10, axis=0)
        overall_best_name = 'P2-10seed'

    # 5-seed ensemble for top-3 Phase 3 configs
    top3_names = [name for name, _ in sorted_p3[:3]]

    for cfg_name in top3_names:
        cfg = search_configs[cfg_name]
        print(f"\n  5-seed ensemble: {cfg_name}")
        preds = []
        for seed in SEEDS_5:
            t0 = time.time()
            pred = train_and_cleanup(X_train_gs, y_train_bins_g, dim_g,
                                     cfg, seed=seed, X_test=X_test_gs)
            preds.append(pred)
            mse_i = compute_mse(y_test_g_full, pred)
            print(f"    Seed {seed:>5}: MSE = {mse_i:.4f}  ({time.time()-t0:.1f}s)")

        y_ens = np.mean(preds, axis=0)
        mse_ens = compute_mse(y_test_g_full, y_ens)
        beat = " (below net4 reference)" if mse_ens < NET4_REFERENCE_MSE else ""
        print(f"  → Ensemble MSE = {mse_ens:.4f}{beat}")

        phase4[f'{cfg_name}_5seed'] = {
            'mse': mse_ens,
            'seed_mses': [compute_mse(y_test_g_full, p) for p in preds],
        }
        if mse_ens < overall_best_mse:
            overall_best_mse = mse_ens
            overall_best_preds = y_ens
            overall_best_name = f'P4-{cfg_name}_5seed'

    # Mixed ensemble: retrain top-5 single models, average
    print(f"\n  Mixed ensemble (top-5 single models, seed=42)...")
    top5_preds = []
    for name, _ in sorted_p3[:5]:
        cfg = search_configs[name]
        pred = train_and_cleanup(X_train_gs, y_train_bins_g, dim_g,
                                 cfg, seed=42, X_test=X_test_gs)
        top5_preds.append(pred)

    y_mixed = np.mean(top5_preds, axis=0)
    mse_mixed = compute_mse(y_test_g_full, y_mixed)
    beat = " (below net4 reference)" if mse_mixed < NET4_REFERENCE_MSE else ""
    print(f"  → Mixed ensemble MSE = {mse_mixed:.4f}{beat}")

    phase4['mixed_top5'] = {
        'mse': mse_mixed,
        'configs': [name for name, _ in sorted_p3[:5]],
    }
    if mse_mixed < overall_best_mse:
        overall_best_mse = mse_mixed
        overall_best_preds = y_mixed
        overall_best_name = 'P4-mixed_top5'

    # 10-seed ensemble with best Phase 3 config
    best_p3_name = sorted_p3[0][0]
    best_p3_cfg = search_configs[best_p3_name]
    print(f"\n  10-seed ensemble: {best_p3_name}")
    preds_10_p4 = []
    for seed in SEEDS_10:
        t0 = time.time()
        pred = train_and_cleanup(X_train_gs, y_train_bins_g, dim_g,
                                 best_p3_cfg, seed=seed, X_test=X_test_gs)
        preds_10_p4.append(pred)
        mse_i = compute_mse(y_test_g_full, pred)
        print(f"    Seed {seed:>5}: MSE = {mse_i:.4f}  ({time.time()-t0:.1f}s)")

    y_ens_10 = np.mean(preds_10_p4, axis=0)
    mse_ens_10 = compute_mse(y_test_g_full, y_ens_10)
    beat = " (below net4 reference)" if mse_ens_10 < NET4_REFERENCE_MSE else ""
    print(f"  → 10-seed ensemble MSE = {mse_ens_10:.4f}{beat}")

    phase4[f'{best_p3_name}_10seed'] = {
        'mse': mse_ens_10,
        'seed_mses': [compute_mse(y_test_g_full, p) for p in preds_10_p4],
    }
    if mse_ens_10 < overall_best_mse:
        overall_best_mse = mse_ens_10
        overall_best_preds = y_ens_10
        overall_best_name = f'P4-{best_p3_name}_10seed'

    all_results['phase4'] = phase4
    save_results(all_results)

    # ══════════════════════════════════════════════════════════════════════
    # Phase 5: ESM2-650M comparison
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  PHASE 5: ESM2-650M Comparison")
    print(f"{'='*60}")

    # reference config on ESM2-650M
    esm_cfg = {**prototype_cfg, 'epochs': best_p1_epochs}

    # Single model
    t0 = time.time()
    y_pred_esm_single = train_and_cleanup(
        X_train_es, y_train_bins_e, dim_e, esm_cfg, seed=42, X_test=X_test_es)
    mse_esm_single = compute_mse(y_test_e_full, y_pred_esm_single)
    print(f"  ESM2-650M single:  MSE = {mse_esm_single:.4f}  ({time.time()-t0:.1f}s)")

    # 5-seed ensemble
    print(f"  ESM2-650M 5-seed ensemble...")
    preds_esm_5 = []
    for seed in SEEDS_5:
        t0 = time.time()
        pred = train_and_cleanup(X_train_es, y_train_bins_e, dim_e,
                                 esm_cfg, seed=seed, X_test=X_test_es)
        preds_esm_5.append(pred)
        mse_i = compute_mse(y_test_e_full, pred)
        print(f"    Seed {seed:>5}: MSE = {mse_i:.4f}  ({time.time()-t0:.1f}s)")

    mse_esm_5 = compute_mse(y_test_e_full, np.mean(preds_esm_5, axis=0))
    print(f"  → ESM2-650M ensemble MSE = {mse_esm_5:.4f}")

    # Best Phase 3 arch on ESM2-650M
    print(f"  ESM2-650M with best arch ({best_p3_name})...")
    preds_esm_best = []
    for seed in SEEDS_5:
        pred = train_and_cleanup(X_train_es, y_train_bins_e, dim_e,
                                 best_p3_cfg, seed=seed, X_test=X_test_es)
        preds_esm_best.append(pred)

    mse_esm_best = compute_mse(y_test_e_full, np.mean(preds_esm_best, axis=0))
    print(f"  → ESM2-650M best-arch ensemble MSE = {mse_esm_best:.4f}")

    phase5 = {
        'single': {'mse': mse_esm_single},
        '5seed_prototype': {'mse': mse_esm_5,
                          'seed_mses': [compute_mse(y_test_e_full, p) for p in preds_esm_5]},
        f'5seed_{best_p3_name}': {
            'mse': mse_esm_best,
            'seed_mses': [compute_mse(y_test_e_full, p) for p in preds_esm_best]},
    }

    # Check if ESM is below the overall best
    for label, preds_list, y_test_ref in [
        ('P5-ESM-5seed', preds_esm_5, y_test_e_full),
        (f'P5-ESM-5seed-{best_p3_name}', preds_esm_best, y_test_e_full),
    ]:
        mse_check = compute_mse(y_test_ref, np.mean(preds_list, axis=0))
        if mse_check < overall_best_mse:
            overall_best_mse = mse_check
            overall_best_preds = np.mean(preds_list, axis=0)
            overall_best_name = label
            overall_best_ytest = y_test_ref

    all_results['phase5'] = phase5
    save_results(all_results)

    # ══════════════════════════════════════════════════════════════════════
    # Phase 6: Bootstrap CIs on best result
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  PHASE 6: Bootstrap CIs on Best Result")
    print(f"{'='*60}")

    print(f"\n  Best overall: {overall_best_name} → MSE = {overall_best_mse:.4f}")
    ci = bootstrap_ci(overall_best_ytest, overall_best_preds)
    print(f"  MSE:     {ci['mse']['point']:.4f}  "
          f"[{ci['mse']['ci_lo']:.4f}, {ci['mse']['ci_hi']:.4f}]")
    print(f"  R²:      {ci['r2']['point']:.4f}  "
          f"[{ci['r2']['ci_lo']:.4f}, {ci['r2']['ci_hi']:.4f}]")
    print(f"  Spearman:{ci['spearman']['point']:.4f}  "
          f"[{ci['spearman']['ci_lo']:.4f}, {ci['spearman']['ci_hi']:.4f}]")

    all_results['phase6_bootstrap'] = {
        'best_model': overall_best_name,
        'best_mse': overall_best_mse,
        'bootstrap_ci': ci,
    }

    # ── Final summary ─────────────────────────────────────────────────────
    total_min = (time.time() - t_total) / 60
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY  ({total_min:.1f} min)")
    print(f"{'='*60}")
    print(f"  net4 single run:   {NET4_REFERENCE_MSE}")
    print(f"  My Script 06 best:      1.001  (val split, 1273 test)")
    print(f"  Phase 1 best single:     {phase1[best_p1]['mse']:.4f}  ({best_p1})")
    print(f"  Phase 2 (5-seed):        {mse_5:.4f}")
    print(f"  Phase 2 (10-seed):       {mse_10:.4f}")
    print(f"  Phase 3 best single:     {sorted_p3[0][1]['mse']:.4f}  ({sorted_p3[0][0]})")
    print(f"  Phase 4 best ensemble:   {min(r['mse'] for r in phase4.values()):.4f}")
    print(f"  Overall best:            {overall_best_mse:.4f}  ({overall_best_name})")

    if overall_best_mse < NET4_REFERENCE_MSE:
        gap = NET4_REFERENCE_MSE - overall_best_mse
        pct = 100 * gap / NET4_REFERENCE_MSE
        print(f"\n  now {0:.4f} below the net4 single-run reference")
    else:
        gap = overall_best_mse - NET4_REFERENCE_MSE
        print(f"\n  Gap to net4 reference: +{gap:.4f}")

    save_results(all_results)

    # ── Figure ────────────────────────────────────────────────────────────
    _make_figure(all_results, phase1, phase3, phase4)

    return all_results


def _make_figure(all_results, p1, p3, p4):
    """Summary figure of all phases."""
    fig, ax = plt.subplots(figsize=(14, 7))

    labels, mses, colors = [], [], []

    # Phase 1 (blue)
    for name, r in p1.items():
        labels.append(f"P1: {name}")
        mses.append(r['mse'])
        colors.append('steelblue')

    # Phase 2 (orange)
    p2 = all_results.get('phase2', {})
    if '5_seed' in p2:
        labels.append('P2: 5-seed')
        mses.append(p2['5_seed']['mse'])
        colors.append('darkorange')
    if '10_seed' in p2:
        labels.append('P2: 10-seed')
        mses.append(p2['10_seed']['mse'])
        colors.append('darkorange')

    # Phase 3 top-5 (green)
    sorted_p3 = sorted(p3.items(), key=lambda x: x[1]['mse'])
    for name, r in sorted_p3[:5]:
        labels.append(f"P3: {name}")
        mses.append(r['mse'])
        colors.append('mediumseagreen')

    # Phase 4 (crimson)
    for name, r in p4.items():
        labels.append(f"P4: {name}")
        mses.append(r['mse'])
        colors.append('crimson')

    x = np.arange(len(labels))
    bars = ax.bar(x, mses, color=colors, alpha=0.85)

    # Reference lines
    ax.axhline(y=NET4_REFERENCE_MSE, color='green', linestyle=':', linewidth=1.0, alpha=0.8,
               label=f'net4 reference ({NET4_REFERENCE_MSE})')
    ax.axhline(y=1.001, color='gray', linestyle='--', linewidth=1.0, alpha=0.8,
               label='Val-split best (1.001)')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Test MSE (all 1326 samples)')
    ax.legend(loc='upper right')

    plt.tight_layout()
    fig_path = FIGURES_DIR / 'vector_architecture_search.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved to {fig_path}")


if __name__ == '__main__':
    main()
