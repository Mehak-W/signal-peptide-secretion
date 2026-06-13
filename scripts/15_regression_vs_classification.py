#!/usr/bin/env python3
"""
Script 15: Regression vs Classification Comparison

Compares 5 output formulations on the same data/architecture to quantify
information loss as output dimensionality decreases, with a same-loss control:

  | Approach              | Output       | Loss | Bins→WA mapping         |
  |-----------------------|-------------|------|-------------------------|
  | Vector regression     | softmax(10) | focal| bins @ [1..10]          |
  | Vector regression     | softmax(10) | CCE  | bins @ [1..10]          |
  | 5-class classification| softmax(5)  | CCE  | argmax → class centroid |
  | 3-class classification| softmax(3)  | CCE  | argmax → class centroid |
  | Binary classification | sigmoid(1)  | BCE  | prob → low/high centroid|

The 10-bin CCE condition isolates the effect of output formulation from
the loss function (focal vs CCE confound flagged by reviewers).

All use same architecture (256,256,128, LeakyReLU, dropout=0.35) and 5-seed
ensembles. MSE is derived from classification outputs via class centroids to
enable fair comparison.

Class thresholds (applied to training WA):
  - Binary: WA >= median → high/low
  - 3-class: Low (<3.5), Medium (3.5-6.5), High (>=6.5)
  - 5-class: [1-2), [2-4), [4-6), [6-8), [8-10]

Outputs:
  - results/regression_vs_classification_results.json
  - figures/regression_vs_classification.png (2x2 Tufte-style)
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
from sklearn.metrics import f1_score, roc_auc_score

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
PRIOR_BENCHMARK_MSE = 0.953  # retracted single-draw, NOT a target (0.953 and 0.932 are seed noise; see docs/reproducibility_findings.md)

# Class thresholds
THREE_CLASS_BOUNDS = [3.5, 6.5]  # Low <3.5, Med 3.5-6.5, High >=6.5
FIVE_CLASS_BOUNDS = [2.0, 4.0, 6.0, 8.0]  # [1-2), [2-4), [4-6), [6-8), [8-10]


def discretize_wa(y, n_classes, train_median=None):
    """Convert continuous WA to class labels."""
    if n_classes == 2:
        return (y >= train_median).astype(int)
    elif n_classes == 3:
        labels = np.zeros(len(y), dtype=int)
        labels[y >= THREE_CLASS_BOUNDS[0]] = 1
        labels[y >= THREE_CLASS_BOUNDS[1]] = 2
        return labels
    elif n_classes == 5:
        labels = np.zeros(len(y), dtype=int)
        for i, bound in enumerate(FIVE_CLASS_BOUNDS):
            labels[y >= bound] = i + 1
        return labels


def class_centroids(n_classes, train_y, train_median=None):
    """Compute centroid WA for each class (mean of training WA in that class)."""
    labels = discretize_wa(train_y, n_classes, train_median)
    centroids = np.array([train_y[labels == c].mean() for c in range(n_classes)])
    return centroids


def build_classification_model(input_dim, n_classes, dropout=0.35, lr=5e-4, seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    inputs = keras.Input(shape=(input_dim,))
    x = inputs
    for units in (256, 256, 128):
        x = layers.Dense(units)(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(dropout)(x)

    if n_classes == 2:
        outputs = layers.Dense(1, activation='sigmoid')(x)
        loss = 'binary_crossentropy'
    else:
        outputs = layers.Dense(n_classes, activation='softmax')(x)
        loss = 'categorical_crossentropy'

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=loss)
    return model


def build_vector_model(input_dim, dropout=0.35, lr=5e-4, seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    inputs = keras.Input(shape=(input_dim,))
    x = inputs
    for units in (256, 256, 128):
        x = layers.Dense(units)(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss=FocalLoss(alpha=0.25, gamma=2.0))
    return model


def train_model(model, X_train, y_train, epochs=300, batch_size=32):
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor='loss', factor=0.5, patience=10, min_lr=1e-6, verbose=0),
    ]
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
              callbacks=callbacks, verbose=0)
    return model


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

    mse_pt = mse(y_true, y_pred)
    return {
        'mse': {'point': mse_pt,
                'ci_lo': float(np.percentile(mse_b, 2.5)),
                'ci_hi': float(np.percentile(mse_b, 97.5))},
        'spearman': {'point': float(sp_stats.spearmanr(y_true, y_pred).statistic),
                     'ci_lo': float(np.percentile(sp_b, 2.5)),
                     'ci_hi': float(np.percentile(sp_b, 97.5))},
    }


def run_vector_regression(X_tr, y_bins, X_te, y_test, loss='focal'):
    """Train 5-seed vector regression ensemble."""
    loss_label = 'focal loss' if loss == 'focal' else 'CCE'
    print(f"\n  Vector Regression (10-bin softmax, {loss_label})")
    preds = []
    for seed in SEEDS:
        t0 = time.time()
        if loss == 'focal':
            model = build_vector_model(X_tr.shape[1], seed=seed)
        else:
            # CCE variant: same architecture, different loss
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
                          loss='categorical_crossentropy')
        train_model(model, X_tr, y_bins)
        pred = model.predict(X_te, verbose=0) @ BIN_CENTERS
        preds.append(pred)
        mse_i = mse(y_test, pred)
        print(f"    Seed {seed:>5}: MSE = {mse_i:.4f}  ({time.time()-t0:.1f}s)")
        del model
        keras.backend.clear_session()

    y_ens = np.mean(preds, axis=0)
    mse_ens = mse(y_test, y_ens)
    ci = bootstrap_ci(y_test, y_ens)
    print(f"  -> Ensemble MSE = {mse_ens:.4f}")
    return y_ens, ci


def run_classification(X_tr, y_train_wa, X_te, y_test, n_classes, train_median=None):
    """Train 5-seed classification ensemble, return WA predictions via centroids."""
    labels_train = discretize_wa(y_train_wa, n_classes, train_median)
    labels_test = discretize_wa(y_test, n_classes, train_median)
    centroids = class_centroids(n_classes, y_train_wa, train_median)

    # One-hot for multi-class
    if n_classes > 2:
        y_train_oh = keras.utils.to_categorical(labels_train, n_classes)
    else:
        y_train_oh = labels_train.astype(np.float32)

    class_dist = {c: int((labels_train == c).sum()) for c in range(n_classes)}
    print(f"    Class distribution (train): {class_dist}")
    print(f"    Centroids: {[f'{c:.2f}' for c in centroids]}")

    all_probs = []
    all_wa_preds = []
    for seed in SEEDS:
        t0 = time.time()
        model = build_classification_model(X_tr.shape[1], n_classes, seed=seed)
        train_model(model, X_tr, y_train_oh)
        probs = model.predict(X_te, verbose=0)

        if n_classes == 2:
            probs = probs.ravel()
            pred_labels = (probs >= 0.5).astype(int)
            # WA from class centroids weighted by probability
            wa_pred = probs * centroids[1] + (1 - probs) * centroids[0]
        else:
            pred_labels = np.argmax(probs, axis=1)
            # WA from class centroids weighted by softmax probabilities
            wa_pred = probs @ centroids

        all_probs.append(probs)
        all_wa_preds.append(wa_pred)

        mse_i = mse(y_test, wa_pred)
        acc_i = float(np.mean(pred_labels == labels_test))
        print(f"    Seed {seed:>5}: MSE = {mse_i:.4f}, Acc = {acc_i:.3f}  ({time.time()-t0:.1f}s)")
        del model
        keras.backend.clear_session()

    # Ensemble: average WA predictions
    y_ens = np.mean(all_wa_preds, axis=0)
    mse_ens = mse(y_test, y_ens)

    # Ensemble: average probabilities for classification metrics
    avg_probs = np.mean(all_probs, axis=0)
    if n_classes == 2:
        avg_probs_flat = avg_probs
        ens_pred_labels = (avg_probs_flat >= 0.5).astype(int)
        acc = float(np.mean(ens_pred_labels == labels_test))
        f1 = float(f1_score(labels_test, ens_pred_labels))
        auc = float(roc_auc_score(labels_test, avg_probs_flat))
        class_metrics = {'accuracy': acc, 'f1': f1, 'auc_roc': auc}
    else:
        ens_pred_labels = np.argmax(avg_probs, axis=1)
        acc = float(np.mean(ens_pred_labels == labels_test))
        f1 = float(f1_score(labels_test, ens_pred_labels, average='macro'))
        class_metrics = {'accuracy': acc, 'macro_f1': f1}

    ci = bootstrap_ci(y_test, y_ens)
    print(f"  -> Ensemble MSE = {mse_ens:.4f}, Acc = {acc:.3f}")
    return y_ens, ci, class_metrics


def make_figure(results):
    """2x2 Tufte-style figure with 5 conditions."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    approaches = ['vector_10bin', 'vector_10bin_cce', '5class', '3class', 'binary']
    labels = ['10-bin\nfocal', '10-bin\nCCE', '5-class\nCCE', '3-class\nCCE', 'Binary\nBCE']
    colors = ['#2E6B8A', '#5B9BD5', '#4A9E6D', '#D4A843', '#C45B3E']

    # ── (A) MSE comparison ────────────────────────────────────────────────
    ax = axes[0, 0]
    mse_vals = [results[a]['bootstrap_ci']['mse']['point'] for a in approaches]
    ci_lo = [results[a]['bootstrap_ci']['mse']['ci_lo'] for a in approaches]
    ci_hi = [results[a]['bootstrap_ci']['mse']['ci_hi'] for a in approaches]
    yerr_lo = [m - lo for m, lo in zip(mse_vals, ci_lo)]
    yerr_hi = [hi - m for m, hi in zip(mse_vals, ci_hi)]

    x = np.arange(len(approaches))
    ax.bar(x, mse_vals, color=colors, width=0.6, edgecolor='none')
    ax.errorbar(x, mse_vals, yerr=[yerr_lo, yerr_hi], fmt='none', ecolor='black',
                capsize=4, capthick=1, linewidth=1)
    ax.axhline(y=PRIOR_BENCHMARK_MSE, color='gray', linewidth=0.8, linestyle='--', alpha=0.7)
    ax.text(4.4, PRIOR_BENCHMARK_MSE, 'prior single-run (0.95, retracted)', fontsize=7, color='gray', va='center')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('Test MSE', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(-0.15, 1.05, 'A', transform=ax.transAxes, fontsize=12, fontweight='bold')

    # ── (B) Spearman comparison — zoomed y-axis ───────────────────────────
    ax = axes[0, 1]
    sp_vals = [results[a]['bootstrap_ci']['spearman']['point'] for a in approaches]
    sp_lo = [results[a]['bootstrap_ci']['spearman']['ci_lo'] for a in approaches]
    sp_hi = [results[a]['bootstrap_ci']['spearman']['ci_hi'] for a in approaches]
    yerr_lo = [v - lo for v, lo in zip(sp_vals, sp_lo)]
    yerr_hi = [hi - v for v, hi in zip(sp_vals, sp_hi)]

    ax.bar(x, sp_vals, color=colors, width=0.6, edgecolor='none')
    ax.errorbar(x, sp_vals, yerr=[yerr_lo, yerr_hi], fmt='none', ecolor='black',
                capsize=4, capthick=1, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('Spearman \u03c1', fontsize=9)
    sp_min = min(sp_lo) - 0.02
    ax.set_ylim(sp_min, max(sp_hi) + 0.01)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(-0.15, 1.05, 'B', transform=ax.transAxes, fontsize=12, fontweight='bold')

    # ── (C) Pred vs actual — vector regression (continuous) ───────────────
    ax = axes[1, 0]
    y_pred = np.array(results['vector_10bin']['predictions'])
    y_true = np.array(results['y_test'])
    ax.scatter(y_true, y_pred, s=5, alpha=0.35, color=colors[0], edgecolors='none', rasterized=True)
    lims = [0.5, 10.5]
    ax.plot(lims, lims, color='gray', linewidth=0.7, linestyle='--', alpha=0.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Actual WA', fontsize=9)
    ax.set_ylabel('Predicted WA', fontsize=9)
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(-0.15, 1.05, 'C', transform=ax.transAxes, fontsize=12, fontweight='bold')
    ax.text(0.95, 0.05, '10-bin focal', transform=ax.transAxes, fontsize=8,
            ha='right', color=colors[0])

    # ── (D) Pred vs actual — 3-class (discrete staircase) ────────────────
    ax = axes[1, 1]
    y_pred_3c = np.array(results['3class']['predictions'])
    ax.scatter(y_true, y_pred_3c, s=5, alpha=0.35, color=colors[3], edgecolors='none', rasterized=True)
    ax.plot(lims, lims, color='gray', linewidth=0.7, linestyle='--', alpha=0.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Actual WA', fontsize=9)
    ax.set_ylabel('Predicted WA', fontsize=9)
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(-0.15, 1.05, 'D', transform=ax.transAxes, fontsize=12, fontweight='bold')
    ax.text(0.95, 0.05, '3-class', transform=ax.transAxes, fontsize=8,
            ha='right', color=colors[3])

    plt.tight_layout()
    fig_path = FIGURES_DIR / 'regression_vs_classification.png'
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
    _, X_test, _, y_test, _ = load_plm_embeddings('ginkgo-AA0-650M')

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    print(f"  Train: {meta['n_train']}, Test: {len(y_test)}, Dim: {X_tr.shape[1]}")

    train_median = float(np.median(y_train_wa))
    print(f"  Train WA median: {train_median:.2f}")

    all_results = {'y_test': y_test.tolist()}

    # ── 1a. Vector regression (10-bin, focal loss) ──────────────────────
    print(f"\n{'='*60}")
    print("  1a. Vector Regression (10-bin, focal loss)")
    print(f"{'='*60}")
    y_pred_vec, ci_vec = run_vector_regression(X_tr, y_train_bins, X_te, y_test, loss='focal')
    all_results['vector_10bin'] = {
        'bootstrap_ci': ci_vec,
        'predictions': y_pred_vec.tolist(),
        'output_dim': 10,
        'loss': 'focal',
    }

    # ── 1b. Vector regression (10-bin, CCE) ───────────────────────────────
    print(f"\n{'='*60}")
    print("  1b. Vector Regression (10-bin, CCE)")
    print(f"{'='*60}")
    y_pred_cce, ci_cce = run_vector_regression(X_tr, y_train_bins, X_te, y_test, loss='cce')
    all_results['vector_10bin_cce'] = {
        'bootstrap_ci': ci_cce,
        'predictions': y_pred_cce.tolist(),
        'output_dim': 10,
        'loss': 'categorical_crossentropy',
    }

    # ── 2. 5-class classification ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  2. 5-Class Classification")
    print(f"{'='*60}")
    y_pred_5c, ci_5c, metrics_5c = run_classification(
        X_tr, y_train_wa, X_te, y_test, n_classes=5)
    all_results['5class'] = {
        'bootstrap_ci': ci_5c,
        'predictions': y_pred_5c.tolist(),
        'class_metrics': metrics_5c,
        'output_dim': 5,
    }

    # ── 3. 3-class classification ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  3. 3-Class Classification")
    print(f"{'='*60}")
    y_pred_3c, ci_3c, metrics_3c = run_classification(
        X_tr, y_train_wa, X_te, y_test, n_classes=3)
    all_results['3class'] = {
        'bootstrap_ci': ci_3c,
        'predictions': y_pred_3c.tolist(),
        'class_metrics': metrics_3c,
        'output_dim': 3,
    }

    # ── 4. Binary classification ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  4. Binary Classification")
    print(f"{'='*60}")
    y_pred_bin, ci_bin, metrics_bin = run_classification(
        X_tr, y_train_wa, X_te, y_test, n_classes=2, train_median=train_median)
    all_results['binary'] = {
        'bootstrap_ci': ci_bin,
        'predictions': y_pred_bin.tolist(),
        'class_metrics': metrics_bin,
        'output_dim': 2,
        'threshold': train_median,
    }

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = (time.time() - t_total) / 60
    print(f"\n{'='*60}")
    print(f"  SUMMARY  ({elapsed:.1f} min)")
    print(f"{'='*60}")
    print(f"\n  {'Approach':<25} {'Dim':>4} {'Loss':<8} {'MSE':>8} {'MSE 95% CI':>20} {'Spearman':>10}")
    print(f"  {'-'*80}")
    for name, label, loss_name in [('vector_10bin', 'Vector (10-bin)', 'focal'),
                                    ('vector_10bin_cce', 'Vector (10-bin)', 'CCE'),
                                    ('5class', '5-class', 'CCE'),
                                    ('3class', '3-class', 'CCE'),
                                    ('binary', 'Binary', 'BCE')]:
        r = all_results[name]
        ci = r['bootstrap_ci']
        dim = r['output_dim']
        print(f"  {label:<25} {dim:>4} {loss_name:<8} {ci['mse']['point']:>8.4f} "
              f"[{ci['mse']['ci_lo']:.3f}, {ci['mse']['ci_hi']:.3f}] "
              f"{ci['spearman']['point']:>10.4f}")

    print(f"\n  Information loss as output dim decreases (vs 10-bin focal):")
    vec_mse = all_results['vector_10bin']['bootstrap_ci']['mse']['point']
    for name, label in [('vector_10bin_cce', '10bin-CCE→10bin-focal'),
                        ('5class', '5→10'), ('3class', '3→10'), ('binary', '2→10')]:
        cls_mse = all_results[name]['bootstrap_ci']['mse']['point']
        pct = 100 * (cls_mse - vec_mse) / vec_mse
        print(f"    {label}: {'+' if pct > 0 else ''}{pct:.1f}% MSE {'increase' if pct > 0 else 'decrease'}")

    # Save results (without large prediction arrays for JSON readability)
    save_data = {}
    for k, v in all_results.items():
        if k == 'y_test':
            continue
        save_data[k] = {kk: vv for kk, vv in v.items() if kk != 'predictions'}
    with open(RESULTS_DIR / 'regression_vs_classification_results.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved.")

    # ── Figure ────────────────────────────────────────────────────────────
    make_figure(all_results)

    print(f"\nDone in {elapsed:.1f} min.")


if __name__ == '__main__':
    main()
