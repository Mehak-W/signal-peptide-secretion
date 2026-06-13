#!/usr/bin/env python3
"""
Script 14: ReLU-Squared vs LeakyReLU Activation Comparison

Hypothesis: Signal peptide bin distributions are sparse — mass concentrates in
1-2 bins out of 10. ReLU²(x) = max(0,x)² produces sparser activations than
LeakyReLU because it zeros negatives AND compresses small positives toward zero.
This may help the network "commit" to dominant bins.

Reference: So et al. (2021) "Primer: Searching for Efficient Transformers."

Compares 5-seed ensembles (same config as Script 10 best) with:
  - LeakyReLU (current best)
  - ReLU² (proposed)

If ReLU² diverges at lr=5e-4, auto-fallback to lr=2e-4.

Metrics: MSE, R², Spearman with 10,000-resample bootstrap 95% CIs.
Sparsity analysis: fraction of zero activations + Hoyer sparsity per layer.

Outputs:
  - results/relu_squared_comparison_results.json
  - figures/relu_squared_comparison.png (2x2 Tufte-style)
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_loading import load_plm_with_bins, load_plm_embeddings
from src.models import FocalLoss, ReLUSquared

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
PRIOR_BENCHMARK_MSE = 0.953  # retracted single-draw, NOT a target (0.953 and 0.932 are seed noise; see docs/reproducibility_findings.md)
BEST_MSE = 0.9323  # RETRACTED single-draw (Script 10); reproducible ~0.957, see docs/reproducibility_findings.md


def build_model(input_dim, hidden_layers=(256, 256, 128), dropout=0.35,
                lr=5e-4, activation='leaky_relu', seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    inputs = keras.Input(shape=(input_dim,))
    x = inputs
    for units in hidden_layers:
        x = layers.Dense(units)(x)
        if activation == 'relu_squared':
            x = ReLUSquared()(x)
        else:
            x = layers.LeakyReLU()(x)
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=loss_fn)
    return model


def train_full(X_train, y_bins, cfg, seed=42):
    model = build_model(X_train.shape[1], hidden_layers=cfg['hidden_layers'],
                        dropout=cfg['dropout'], lr=cfg['lr'],
                        activation=cfg['activation'], seed=seed)
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor='loss', factor=0.5, patience=10, min_lr=1e-6, verbose=0),
    ]
    history = model.fit(X_train, y_bins, epochs=cfg['epochs'],
                        batch_size=cfg['batch_size'], callbacks=callbacks, verbose=0)
    return model, history


def predict_wa(model, X):
    return model.predict(X, verbose=0) @ BIN_CENTERS


def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


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

    mse_pt = mse(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return {
        'mse': {'point': mse_pt,
                'ci_lo': float(np.percentile(mse_b, 2.5)),
                'ci_hi': float(np.percentile(mse_b, 97.5))},
        'r2': {'point': float(1 - ss_res / ss_tot),
               'ci_lo': float(np.percentile(r2_b, 2.5)),
               'ci_hi': float(np.percentile(r2_b, 97.5))},
        'spearman': {'point': float(sp_stats.spearmanr(y_true, y_pred).statistic),
                     'ci_lo': float(np.percentile(sp_b, 2.5)),
                     'ci_hi': float(np.percentile(sp_b, 97.5))},
    }


def hoyer_sparsity(x):
    """Hoyer sparsity: (sqrt(n) - L1/L2) / (sqrt(n) - 1). Range [0,1], 1=maximally sparse."""
    n = x.size
    l1 = np.abs(x).sum()
    l2 = np.sqrt((x ** 2).sum())
    if l2 == 0:
        return 1.0
    sqn = np.sqrt(n)
    return float((sqn - l1 / l2) / (sqn - 1))


def analyze_sparsity(model, X_sample, activation_type):
    """Extract hidden layer activations and measure sparsity."""
    # Build sub-models for each hidden layer output
    layer_outputs = []
    for layer in model.layers:
        if isinstance(layer, (layers.LeakyReLU, ReLUSquared)):
            layer_outputs.append(layer.output)

    if not layer_outputs:
        return {'frac_zero': [], 'hoyer': []}

    extractor = keras.Model(inputs=model.input, outputs=layer_outputs)
    activations = extractor.predict(X_sample, verbose=0)
    if not isinstance(activations, list):
        activations = [activations]

    frac_zeros = []
    hoyers = []
    for act in activations:
        frac_zeros.append(float(np.mean(act == 0)))
        hoyers.append(hoyer_sparsity(act.ravel()))

    return {'frac_zero': frac_zeros, 'hoyer': hoyers}


def run_activation_experiment(X_tr, y_bins, dim, X_te, y_test, activation_name, lr=5e-4):
    """Train 5-seed ensemble, return results dict."""
    cfg = {
        'hidden_layers': (256, 256, 128),
        'dropout': 0.35,
        'lr': lr,
        'activation': activation_name,
        'epochs': 300,
        'batch_size': 32,
    }

    preds = []
    seed_mses = []
    models_for_sparsity = []
    diverged = False

    for seed in SEEDS:
        t0 = time.time()
        model, history = train_full(X_tr, y_bins, cfg, seed=seed)
        pred = predict_wa(model, X_te)

        # Check for divergence (NaN loss, high loss, or catastrophic test MSE)
        final_loss = history.history['loss'][-1]
        mse_i = mse(y_test, pred)
        if np.isnan(final_loss) or final_loss > 10 or mse_i > 3.0:
            diverged = True
            print(f"    Seed {seed:>5}: DIVERGED (loss={final_loss:.4f}, MSE={mse_i:.4f})")
            del model
            keras.backend.clear_session()
            break
        preds.append(pred)
        seed_mses.append(mse_i)
        if seed == SEEDS[0]:
            models_for_sparsity.append(model)
        else:
            del model
            keras.backend.clear_session()
        print(f"    Seed {seed:>5}: MSE = {mse_i:.4f}  ({time.time()-t0:.1f}s)")

    if diverged:
        # Cleanup any saved models
        for m in models_for_sparsity:
            del m
        keras.backend.clear_session()
        return None, True

    y_ens = np.mean(preds, axis=0)
    mse_ens = mse(y_test, y_ens)
    print(f"  -> {activation_name} ensemble MSE = {mse_ens:.4f}")

    # Sparsity analysis on first seed model
    sparsity = analyze_sparsity(models_for_sparsity[0], X_te[:500], activation_name)

    # Per-sample errors
    errors = y_test - y_ens

    # Bootstrap CIs
    ci = bootstrap_ci(y_test, y_ens)

    # Cleanup
    for m in models_for_sparsity:
        del m
    keras.backend.clear_session()

    return {
        'mse': mse_ens,
        'seed_mses': seed_mses,
        'bootstrap_ci': ci,
        'sparsity': sparsity,
        'errors': errors.tolist(),
        'lr_used': lr,
        'config': {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
    }, False


def make_figure(results):
    """2x2 Tufte-style comparison figure."""
    leaky = results['leaky_relu']
    relu2 = results['relu_squared']

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Colors
    c_leaky = '#4878A8'   # steel blue
    c_relu2 = '#D4652F'   # burnt orange

    # ── (A) MSE comparison — dot-and-whisker (log scale for large gap) ────
    ax = axes[0, 0]
    x = [0, 1]
    act_labels = ['LeakyReLU', 'ReLU\u00b2']
    mses = [leaky['bootstrap_ci']['mse']['point'], relu2['bootstrap_ci']['mse']['point']]
    ci_lo = [leaky['bootstrap_ci']['mse']['ci_lo'], relu2['bootstrap_ci']['mse']['ci_lo']]
    ci_hi = [leaky['bootstrap_ci']['mse']['ci_hi'], relu2['bootstrap_ci']['mse']['ci_hi']]

    for i, c in enumerate([c_leaky, c_relu2]):
        ax.plot(x[i], mses[i], 'o', color=c, markersize=8, zorder=3)
        ax.plot([x[i], x[i]], [ci_lo[i], ci_hi[i]], color=c, linewidth=2, zorder=2)
    ax.axhline(y=PRIOR_BENCHMARK_MSE, color='gray', linewidth=0.8, linestyle='--', alpha=0.7)
    ax.text(1.4, PRIOR_BENCHMARK_MSE, 'prior single-run (0.95, retracted)', fontsize=7, color='gray', va='center')
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(act_labels, fontsize=9)
    ax.set_ylabel('Test MSE (log scale)', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(-0.15, 1.05, 'A', transform=ax.transAxes, fontsize=12, fontweight='bold')

    # ── (B) Spearman comparison — dot-and-whisker, zoomed axis ────────────
    ax = axes[0, 1]
    sp_vals = [leaky['bootstrap_ci']['spearman']['point'], relu2['bootstrap_ci']['spearman']['point']]
    sp_lo = [leaky['bootstrap_ci']['spearman']['ci_lo'], relu2['bootstrap_ci']['spearman']['ci_lo']]
    sp_hi = [leaky['bootstrap_ci']['spearman']['ci_hi'], relu2['bootstrap_ci']['spearman']['ci_hi']]

    for i, c in enumerate([c_leaky, c_relu2]):
        ax.plot(x[i], sp_vals[i], 'o', color=c, markersize=8, zorder=3)
        ax.plot([x[i], x[i]], [sp_lo[i], sp_hi[i]], color=c, linewidth=2, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(act_labels, fontsize=9)
    ax.set_ylabel('Spearman \u03c1', fontsize=9)
    ymin = min(sp_lo) - 0.05
    ax.set_ylim(ymin, max(sp_hi) + 0.02)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(-0.15, 1.05, 'B', transform=ax.transAxes, fontsize=12, fontweight='bold')

    # ── (C) Hoyer sparsity per layer (both visible, unlike frac_zero) ────
    ax = axes[1, 0]
    n_layers = len(leaky['sparsity']['hoyer'])
    layer_x = np.arange(n_layers)
    w = 0.3

    ax.bar(layer_x - w/2, leaky['sparsity']['hoyer'], w,
           color=c_leaky, edgecolor='none', label='LeakyReLU')
    ax.bar(layer_x + w/2, relu2['sparsity']['hoyer'], w,
           color=c_relu2, edgecolor='none', label='ReLU\u00b2')
    ax.set_xticks(layer_x)
    ax.set_xticklabels([f'Layer {i+1}' for i in range(n_layers)], fontsize=9)
    ax.set_ylabel('Hoyer sparsity', fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(-0.15, 1.05, 'C', transform=ax.transAxes, fontsize=12, fontweight='bold')

    # ── (D) Per-sample error distributions — step histograms ──────────────
    ax = axes[1, 1]
    bins = np.linspace(-8, 8, 70)
    ax.hist(leaky['errors'], bins=bins, histtype='step', color=c_leaky,
            linewidth=1.5, label='LeakyReLU', density=True)
    ax.hist(relu2['errors'], bins=bins, histtype='step', color=c_relu2,
            linewidth=1.5, label='ReLU\u00b2', density=True)
    ax.set_xlabel('Prediction error (actual \u2212 predicted)', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(-0.15, 1.05, 'D', transform=ax.transAxes, fontsize=12, fontweight='bold')

    plt.tight_layout()
    fig_path = FIGURES_DIR / 'relu_squared_comparison.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved to {fig_path}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    # ── Load data ─────────────────────────────────────────────────────────
    print("Loading Ginkgo-AA0 data...")
    X_train, _, _, _, y_train_bins, _, meta = load_plm_with_bins('ginkgo-AA0-650M')
    _, X_test, _, y_test, _ = load_plm_embeddings('ginkgo-AA0-650M')

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    dim = X_tr.shape[1]
    print(f"  Train: {meta['n_train']}, Test: {len(y_test)}, Dim: {dim}")

    all_results = {}

    # ── LeakyReLU (baseline) ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  LeakyReLU (current best)")
    print(f"{'='*60}")
    leaky_result, _ = run_activation_experiment(X_tr, y_train_bins, dim, X_te, y_test, 'leaky_relu')
    all_results['leaky_relu'] = leaky_result

    # ── ReLU² ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  ReLU-Squared")
    print(f"{'='*60}")
    relu2_result, diverged = run_activation_experiment(X_tr, y_train_bins, dim, X_te, y_test, 'relu_squared')

    if diverged:
        print("  ReLU² diverged at lr=5e-4, retrying with lr=2e-4...")
        relu2_result, diverged = run_activation_experiment(
            X_tr, y_train_bins, dim, X_te, y_test, 'relu_squared', lr=2e-4)
        if diverged:
            print("  ReLU² still diverging. Aborting.")
            return

    all_results['relu_squared'] = relu2_result

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = (time.time() - t_total) / 60
    print(f"\n{'='*60}")
    print(f"  SUMMARY  ({elapsed:.1f} min)")
    print(f"{'='*60}")

    for name in ['leaky_relu', 'relu_squared']:
        r = all_results[name]
        ci = r['bootstrap_ci']
        print(f"\n  {name}:")
        print(f"    MSE:      {ci['mse']['point']:.4f}  [{ci['mse']['ci_lo']:.3f}, {ci['mse']['ci_hi']:.3f}]")
        print(f"    R²:       {ci['r2']['point']:.4f}  [{ci['r2']['ci_lo']:.3f}, {ci['r2']['ci_hi']:.3f}]")
        print(f"    Spearman: {ci['spearman']['point']:.4f}  [{ci['spearman']['ci_lo']:.3f}, {ci['spearman']['ci_hi']:.3f}]")
        print(f"    Sparsity: frac_zero={r['sparsity']['frac_zero']}, hoyer={[f'{h:.3f}' for h in r['sparsity']['hoyer']]}")
        print(f"    LR used:  {r['lr_used']}")

    delta_mse = all_results['leaky_relu']['mse'] - all_results['relu_squared']['mse']
    print(f"\n  MSE difference (LeakyReLU - ReLU²): {delta_mse:+.4f}")
    if delta_mse > 0:
        print("  -> ReLU² is better (lower MSE)")
    else:
        print("  -> LeakyReLU is better (lower MSE)")

    # Save results (exclude large error arrays from JSON for readability)
    save_results = {}
    for name, r in all_results.items():
        save_results[name] = {k: v for k, v in r.items() if k != 'errors'}
        save_results[name]['error_stats'] = {
            'mean': float(np.mean(r['errors'])),
            'std': float(np.std(r['errors'])),
            'median': float(np.median(r['errors'])),
        }
    with open(RESULTS_DIR / 'relu_squared_comparison_results.json', 'w') as f:
        json.dump(save_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"\n  Results saved.")

    # ── Figure ────────────────────────────────────────────────────────────
    make_figure(all_results)

    print(f"\nDone in {elapsed:.1f} min.")


if __name__ == '__main__':
    main()
