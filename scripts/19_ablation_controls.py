#!/usr/bin/env python3
"""
Script 19: Ablation Table + Control Experiments

Three analyses addressing reviewer concerns:

  Part 1; Ablation table: Assembles existing results to isolate the contributions
           of architecture depth, dropout, full-data training, and ensembling to
           the reproducible 0.957 +/- 0.009 result.

  Part 2; Random-split CV: Standard (non-gene-stratified) 5-fold CV on the same
           combined data as LOGO, to isolate gene-identity leakage from inherent
           CV variance.

  Part 3; Smaller architectures: Tests (128,), (128, 64), (64, 32) MLPs to
           determine if the (256, 256, 128) architecture is over-parameterized
           given the linear probe's strong performance.

Outputs:
  - results/ablation_controls_results.json
  - figures/ablation_controls.png
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
from sklearn.model_selection import KFold

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


def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def bootstrap_ci(y_true, y_pred, n_boot=N_BOOTSTRAP, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    mse_b = np.empty(n_boot)
    sp_b = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        mse_b[b] = np.mean((yt - yp) ** 2)
        sp_b[b] = sp_stats.spearmanr(yt, yp).statistic
    return {
        'mse': {'point': mse(y_true, y_pred),
                'ci_lo': float(np.percentile(mse_b, 2.5)),
                'ci_hi': float(np.percentile(mse_b, 97.5))},
        'spearman': {'point': float(sp_stats.spearmanr(y_true, y_pred).statistic),
                     'ci_lo': float(np.percentile(sp_b, 2.5)),
                     'ci_hi': float(np.percentile(sp_b, 97.5))},
    }


def build_and_train(X_train, y_bins, hidden_layers, dropout, seed,
                    epochs=300, batch_size=32, lr=5e-4):
    """Build, train, and return a vector regression model."""
    tf.random.set_seed(seed)
    np.random.seed(seed)

    inputs = keras.Input(shape=(X_train.shape[1],))
    x = inputs
    for units in hidden_layers:
        x = layers.Dense(units)(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss=FocalLoss(alpha=0.25, gamma=2.0))
    callbacks = [keras.callbacks.ReduceLROnPlateau(
        monitor='loss', factor=0.5, patience=10, min_lr=1e-6, verbose=0)]
    model.fit(X_train, y_bins, epochs=epochs, batch_size=batch_size,
              callbacks=callbacks, verbose=0)
    return model


# ═══════════════════════════════════════════════════════════════════════════
# Part 1: Ablation Table (assembled from existing results)
# ═══════════════════════════════════════════════════════════════════════════

def assemble_ablation_table():
    """Load existing result JSONs and extract ablation data points."""
    print("=" * 60)
    print("  PART 1: ABLATION TABLE (from existing results)")
    print("=" * 60)

    # Load result files
    with open(RESULTS_DIR / 'vector_regression_results.json') as f:
        s06 = json.load(f)
    with open(RESULTS_DIR / 'vector_ensemble_optimization_results.json') as f:
        s10 = json.load(f)
    with open(RESULTS_DIR / 'vector_architecture_search_results.json') as f:
        s09 = json.load(f)

    # Extract data points for the ablation
    # Starting point: val-split, (256,256), dropout=0.20, 5-seed (Script 06)
    s06_ginkgo = s06['Ginkgo-AA0_focal']
    s06_5seed_mse = s06_ginkgo['test_metrics']['mse']
    s06_seed_mses = s06_ginkgo['seed_mses']
    s06_mean_1seed = float(np.mean(s06_seed_mses))

    # Full-data, (256,256), dropout=0.20, 5-seed (Script 09 phase 2)
    s09_2layer = s09['phase2']['5_seed']
    s09_2layer_5seed = s09_2layer['mse']
    s09_2layer_seeds = s09_2layer['seed_mses']
    s09_2layer_mean1 = float(np.mean(s09_2layer_seeds))

    # Full-data, (256,256,128), dropout=0.20, 5-seed (Script 10 phase 1)
    s10_d020 = s10['phase1_dropout']['drop-0.20']
    s10_3layer_d020_5seed = s10_d020['mse']
    s10_3layer_d020_seeds = s10_d020['seed_mses']
    s10_3layer_d020_mean1 = float(np.mean(s10_3layer_d020_seeds))

    # Full-data, (256,256,128), dropout=0.35, 5-seed (Script 10 phase 1)
    s10_d035 = s10['phase1_dropout']['drop-0.35']
    s10_3layer_d035_5seed = s10_d035['mse']
    s10_3layer_d035_seeds = s10_d035['seed_mses']
    s10_3layer_d035_mean1 = float(np.mean(s10_3layer_d035_seeds))

    # Build the table: show cumulative effect of each change
    ablation = [
        {
            'step': 'Baseline',
            'description': 'Val-split, (256,256), drop=0.20, 5-seed',
            'training': 'val-split',
            'architecture': '(256, 256)',
            'dropout': 0.20,
            'seeds': 5,
            'ensemble_mse': round(s06_5seed_mse, 4),
            'mean_1seed_mse': round(s06_mean_1seed, 3),
            'test_n': 1273,
            'source': 'Script 06',
        },
        {
            'step': '+ Full-data training',
            'description': 'Full-data, (256,256), drop=0.20, 5-seed',
            'training': 'full-data',
            'architecture': '(256, 256)',
            'dropout': 0.20,
            'seeds': 5,
            'ensemble_mse': round(s09_2layer_5seed, 4),
            'mean_1seed_mse': round(s09_2layer_mean1, 3),
            'test_n': 1326,
            'source': 'Script 09',
        },
        {
            'step': '+ Deeper architecture',
            'description': 'Full-data, (256,256,128), drop=0.20, 5-seed',
            'training': 'full-data',
            'architecture': '(256, 256, 128)',
            'dropout': 0.20,
            'seeds': 5,
            'ensemble_mse': round(s10_3layer_d020_5seed, 4),
            'mean_1seed_mse': round(s10_3layer_d020_mean1, 3),
            'test_n': 1326,
            'source': 'Script 10',
        },
        {
            'step': '+ Higher dropout',
            'description': 'Full-data, (256,256,128), drop=0.35, 5-seed',
            'training': 'full-data',
            'architecture': '(256, 256, 128)',
            'dropout': 0.35,
            'seeds': 5,
            'ensemble_mse': round(s10_3layer_d035_5seed, 4),
            'mean_1seed_mse': round(s10_3layer_d035_mean1, 3),
            'test_n': 1326,
            'source': 'Script 10',
        },
    ]

    # Also add 1-seed vs 5-seed comparison
    ensemble_effect = {
        'config': '(256,256,128), drop=0.35, full-data',
        'mean_1seed_mse': round(s10_3layer_d035_mean1, 4),
        'ensemble_5seed_mse': round(s10_3layer_d035_5seed, 4),
        'ensemble_benefit_pct': round(
            (s10_3layer_d035_mean1 - s10_3layer_d035_5seed) / s10_3layer_d035_mean1 * 100, 1),
        'seed_mses': [round(s, 4) for s in s10_3layer_d035_seeds],
    }

    # Print summary
    print("\n  Cumulative ablation (each row adds one change):")
    print(f"  {'Step':<25} {'Ens MSE':>8} {'Mean 1-seed':>11} {'N_test':>6}")
    print("  " + "-" * 55)
    for row in ablation:
        print(f"  {row['step']:<25} {row['ensemble_mse']:>8.4f} "
              f"{row['mean_1seed_mse']:>11.3f} {row['test_n']:>6}")

    print(f"\n  Ensemble effect (drop=0.35, 3-layer, full-data):")
    print(f"    Mean 1-seed: {ensemble_effect['mean_1seed_mse']:.4f}")
    print(f"    5-seed ens:  {ensemble_effect['ensemble_5seed_mse']:.4f}")
    print(f"    Benefit:     {ensemble_effect['ensemble_benefit_pct']:.1f}%")

    return {'ablation_table': ablation, 'ensemble_effect': ensemble_effect}


# ═══════════════════════════════════════════════════════════════════════════
# Part 2: Random-Split 5-Fold CV
# ═══════════════════════════════════════════════════════════════════════════

def run_random_cv(X_all, y_bins_all, y_wa_all):
    """Random 5-fold CV on combined data (same data as LOGO, not gene-stratified)."""
    print(f"\n{'='*60}")
    print("  PART 2: RANDOM-SPLIT 5-FOLD CV")
    print(f"{'='*60}")

    t0 = time.time()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    all_preds = np.empty(len(y_wa_all))
    fold_mses = []

    for fold_i, (train_idx, test_idx) in enumerate(kf.split(X_all)):
        fold_t0 = time.time()
        X_tr, X_te = X_all[train_idx], X_all[test_idx]
        y_tr_bins = y_bins_all[train_idx]
        y_te_wa = y_wa_all[test_idx]

        # Fit scaler inside fold
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        model = build_and_train(X_tr_s, y_tr_bins, (256, 256, 128), 0.35, seed=42)
        pred = model.predict(X_te_s, verbose=0) @ BIN_CENTERS
        all_preds[test_idx] = pred

        fold_mse = mse(y_te_wa, pred)
        fold_mses.append(fold_mse)
        print(f"    Fold {fold_i+1}/5: n_test={len(test_idx)}, "
              f"MSE={fold_mse:.4f}  ({time.time()-fold_t0:.1f}s)")

        del model
        keras.backend.clear_session()

    overall_mse = mse(y_wa_all, all_preds)
    ci = bootstrap_ci(y_wa_all, all_preds)

    elapsed = time.time() - t0
    print(f"\n  Random 5-fold CV overall MSE: {overall_mse:.4f} "
          f"[{ci['mse']['ci_lo']:.3f}, {ci['mse']['ci_hi']:.3f}]")
    print(f"  Spearman: {ci['spearman']['point']:.4f}")
    print(f"  Per-fold MSEs: {[round(m, 4) for m in fold_mses]}")
    print(f"  ({elapsed:.1f}s total)")

    return {
        'overall_mse': overall_mse,
        'bootstrap_ci': ci,
        'fold_mses': fold_mses,
        'n_samples': len(y_wa_all),
        'n_folds': 5,
        'seed': 42,
        'elapsed_s': round(elapsed, 1),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Part 3: Smaller Architectures
# ═══════════════════════════════════════════════════════════════════════════

def run_smaller_architectures(X_tr, y_bins, X_te, y_test):
    """Test smaller MLP architectures to check over-parameterization."""
    print(f"\n{'='*60}")
    print("  PART 3: SMALLER ARCHITECTURES")
    print(f"{'='*60}")

    configs = [
        {'name': '(128,)', 'layers': (128,), 'dropout': 0.35, 'params_approx': '128*1280+128*10≈165K'},
        {'name': '(128, 64)', 'layers': (128, 64), 'dropout': 0.35, 'params_approx': '128*1280+64*128+64*10≈173K'},
        {'name': '(64, 32)', 'layers': (64, 32), 'dropout': 0.35, 'params_approx': '64*1280+32*64+32*10≈84K'},
        {'name': '(256, 128)', 'layers': (256, 128), 'dropout': 0.35, 'params_approx': '256*1280+128*256+128*10≈362K'},
    ]

    results = {}

    for cfg in configs:
        print(f"\n  Testing {cfg['name']} (drop={cfg['dropout']}, 5-seed)...")
        t0 = time.time()
        preds = []
        seed_mses = []

        for seed in SEEDS:
            model = build_and_train(X_tr, y_bins, cfg['layers'], cfg['dropout'], seed)
            pred = model.predict(X_te, verbose=0) @ BIN_CENTERS
            preds.append(pred)
            s_mse = mse(y_test, pred)
            seed_mses.append(s_mse)
            print(f"    Seed {seed:>5}: MSE = {s_mse:.4f}")
            del model
            keras.backend.clear_session()

        y_ens = np.mean(preds, axis=0)
        ens_mse = mse(y_test, y_ens)
        ci = bootstrap_ci(y_test, y_ens)
        elapsed = time.time() - t0

        print(f"  -> {cfg['name']} Ensemble MSE = {ens_mse:.4f} "
              f"[{ci['mse']['ci_lo']:.3f}, {ci['mse']['ci_hi']:.3f}]  ({elapsed:.1f}s)")

        results[cfg['name']] = {
            'hidden_layers': list(cfg['layers']),
            'dropout': cfg['dropout'],
            'ensemble_mse': ens_mse,
            'bootstrap_ci': ci,
            'seed_mses': seed_mses,
            'mean_1seed_mse': float(np.mean(seed_mses)),
            'elapsed_s': round(elapsed, 1),
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Figure
# ═══════════════════════════════════════════════════════════════════════════

def make_figure(ablation_data, cv_data, arch_data):
    """3-panel figure: ablation waterfall, CV comparison, architecture sweep."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- Panel A: Ablation waterfall ---
    ax = axes[0]
    steps = [r['step'].replace('+ ', '+\n') for r in ablation_data['ablation_table']]
    mses = [r['ensemble_mse'] for r in ablation_data['ablation_table']]
    colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#70AD47']
    bars = ax.bar(range(len(steps)), mses, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(steps, fontsize=8)
    ax.set_ylabel('Test MSE')
    ax.set_title('A. Ablation: Cumulative Changes', fontweight='bold', fontsize=11)
    for bar, val in zip(bars, mses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_ylim(0, max(mses) * 1.15)

    # --- Panel B: CV comparison (standard vs random-CV vs LOGO) ---
    ax = axes[1]
    # Load LOGO results
    with open(RESULTS_DIR / 'gene_stratified_evaluation_results.json') as f:
        logo_data = json.load(f)

    labels = ['Standard\n(1-seed)', 'Random\n5-fold CV', 'LOGO\n(gene-strat.)']
    vals = [
        logo_data['standard_comparison']['standard_test_mse_1seed'],
        cv_data['overall_mse'],
        logo_data['logo_cv']['overall_mse'],
    ]
    ci_lo = [
        None,  # no CI for standard 1-seed
        cv_data['bootstrap_ci']['mse']['ci_lo'],
        logo_data['logo_cv']['bootstrap_ci']['mse']['ci_lo'],
    ]
    ci_hi = [
        None,
        cv_data['bootstrap_ci']['mse']['ci_hi'],
        logo_data['logo_cv']['bootstrap_ci']['mse']['ci_hi'],
    ]
    colors_b = ['#70AD47', '#4472C4', '#C0504D']
    bars_b = ax.bar(range(3), vals, color=colors_b, edgecolor='black', linewidth=0.5)
    for i in range(3):
        if ci_lo[i] is not None:
            ax.errorbar(i, vals[i], yerr=[[vals[i]-ci_lo[i]], [ci_hi[i]-vals[i]]],
                       fmt='none', color='black', capsize=5, linewidth=1.5)
        ax.text(i, vals[i] + 0.05, f'{vals[i]:.3f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Test MSE')
    ax.set_title('B. Evaluation Protocol Comparison', fontweight='bold', fontsize=11)
    ax.set_ylim(0, max(vals) * 1.2)

    # --- Panel C: Architecture size sweep ---
    ax = axes[2]
    # Include linear probe from Script 18 and best NN
    with open(RESULTS_DIR / 'linear_baseline_results.json') as f:
        lb = json.load(f)

    arch_names = ['Linear\nprobe']
    arch_mses = [lb['linear_probe']['bootstrap_ci']['mse']['point']]
    arch_ci_lo = [lb['linear_probe']['bootstrap_ci']['mse']['ci_lo']]
    arch_ci_hi = [lb['linear_probe']['bootstrap_ci']['mse']['ci_hi']]

    for name in ['(64, 32)', '(128,)', '(128, 64)', '(256, 128)']:
        if name in arch_data:
            arch_names.append(name)
            arch_mses.append(arch_data[name]['ensemble_mse'])
            arch_ci_lo.append(arch_data[name]['bootstrap_ci']['mse']['ci_lo'])
            arch_ci_hi.append(arch_data[name]['bootstrap_ci']['mse']['ci_hi'])

    arch_names.append('(256,256,\n128)')
    arch_mses.append(0.957)  # reproducible value over retrains
    arch_ci_lo.append(0.823)
    arch_ci_hi.append(1.054)

    colors_c = plt.cm.Blues(np.linspace(0.3, 0.9, len(arch_names)))
    bars_c = ax.bar(range(len(arch_names)), arch_mses, color=colors_c,
                    edgecolor='black', linewidth=0.5)
    for i in range(len(arch_names)):
        ax.errorbar(i, arch_mses[i],
                    yerr=[[arch_mses[i]-arch_ci_lo[i]], [arch_ci_hi[i]-arch_mses[i]]],
                    fmt='none', color='black', capsize=4, linewidth=1.2)
        ax.text(i, arch_mses[i] + 0.005, f'{arch_mses[i]:.3f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_xticks(range(len(arch_names)))
    ax.set_xticklabels(arch_names, fontsize=8)
    ax.set_ylabel('Test MSE (5-seed ensemble)')
    ax.set_title('C. Architecture Size vs Performance', fontweight='bold', fontsize=11)
    ax.set_ylim(0.85, max(arch_mses) * 1.08)

    plt.tight_layout()
    fig_path = FIGURES_DIR / 'ablation_controls.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved to {fig_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    # ── Part 1: Ablation table ────────────────────────────────────────────
    ablation_data = assemble_ablation_table()

    # ── Load data for Parts 2 & 3 ────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Loading Ginkgo-AA0 data...")
    print(f"{'='*60}")

    X_train, X_test, y_train_wa, y_test_wa, y_train_bins, y_test_bins, meta = \
        load_plm_with_bins('ginkgo-AA0-650M')
    train_emb, train_bins, train_wa = X_train, y_train_bins, y_train_wa
    test_emb, test_bins, test_wa = X_test, y_test_bins, y_test_wa

    print(f"  Train: {train_emb.shape[0]}, Test: {test_emb.shape[0]}")

    # Combined data for CV (same as LOGO uses)
    X_all = np.vstack([train_emb, test_emb])
    y_bins_all = np.vstack([train_bins, test_bins])
    y_wa_all = np.concatenate([train_wa, test_wa])
    print(f"  Combined: {X_all.shape[0]} samples")

    # ── Part 2: Random CV ─────────────────────────────────────────────────
    cv_data = run_random_cv(X_all, y_bins_all, y_wa_all)

    # ── Part 3: Smaller architectures ─────────────────────────────────────
    # Scale for architecture tests (full training data, same as Script 10/18)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(train_emb)
    X_te_s = scaler.transform(test_emb)

    arch_data = run_smaller_architectures(X_tr_s, train_bins, X_te_s, test_wa)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")

    # Load LOGO for comparison
    with open(RESULTS_DIR / 'gene_stratified_evaluation_results.json') as f:
        logo = json.load(f)

    print(f"\n  Evaluation protocol comparison:")
    print(f"    Standard (1-seed):  {logo['standard_comparison']['standard_test_mse_1seed']:.4f}")
    print(f"    Random 5-fold CV:   {cv_data['overall_mse']:.4f}")
    print(f"    LOGO (gene-strat):  {logo['logo_cv']['overall_mse']:.4f}")

    ratio_rand = cv_data['overall_mse'] / logo['standard_comparison']['standard_test_mse_1seed']
    ratio_logo = logo['logo_cv']['overall_mse'] / logo['standard_comparison']['standard_test_mse_1seed']
    print(f"    Random/Standard:    {ratio_rand:.2f}x")
    print(f"    LOGO/Standard:      {ratio_logo:.2f}x")

    gene_leakage_pct = ((logo['logo_cv']['overall_mse'] - cv_data['overall_mse'])
                        / (logo['logo_cv']['overall_mse'] - logo['standard_comparison']['standard_test_mse_1seed'])
                        * 100)
    print(f"    Gene-identity portion of LOGO gap: {gene_leakage_pct:.1f}%")
    print(f"    CV variance portion of LOGO gap:   {100 - gene_leakage_pct:.1f}%")

    print(f"\n  Architecture scaling:")
    print(f"    {'Architecture':<20} {'Ens MSE':>8} {'Mean 1-seed':>11}")
    print(f"    {'-'*42}")
    print(f"    {'Linear probe':<20} {'1.037':>8} {'1.080':>11}")
    for name in ['(64, 32)', '(128,)', '(128, 64)', '(256, 128)']:
        if name in arch_data:
            d = arch_data[name]
            print(f"    {name:<20} {d['ensemble_mse']:>8.4f} {d['mean_1seed_mse']:>11.4f}")
    print(f"    {'(256, 256, 128)':<20} {'0.957':>8} {'0.979':>11}")

    elapsed = time.time() - t_start
    print(f"\n  Total elapsed: {elapsed/60:.1f} min")

    # ── Save results ──────────────────────────────────────────────────────
    all_results = {
        'ablation': ablation_data,
        'random_cv': cv_data,
        'smaller_architectures': arch_data,
        'comparisons': {
            'standard_1seed_mse': logo['standard_comparison']['standard_test_mse_1seed'],
            'random_cv_mse': cv_data['overall_mse'],
            'logo_mse': logo['logo_cv']['overall_mse'],
            'ratio_random_to_standard': round(ratio_rand, 3),
            'ratio_logo_to_standard': round(ratio_logo, 3),
            'gene_leakage_pct': round(gene_leakage_pct, 1),
        },
        'elapsed_minutes': round(elapsed / 60, 1),
    }

    results_path = RESULTS_DIR / 'ablation_controls_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    # ── Figure ────────────────────────────────────────────────────────────
    # Canonical Tufte figure generated by scripts/replot_figures.py (replot_19)


if __name__ == '__main__':
    main()
