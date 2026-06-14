#!/usr/bin/env python3
"""
Script 13: Cross-Dataset Fine-Tuning with Vector Model

Implements Professor Schrier's suggestion: pretrain the best vector regression
model on Grasso data, then fine-tune the output layer(s) on each external
dataset using 5-fold cross-validation.

Strategy:
  1. Pretrain 5-seed vector NN ensemble on full Grasso training data (ESM2-650M;
     the external datasets only ship ESM2-650M embeddings, so the frozen feature
     space must match; Ginkgo-AA0 embeddings are not available for these sets).
  2. For each external dataset:
     a. 5-fold CV (stratified for binary Wu dataset)
     b. Per fold: an inner validation split drives early stopping (I never
        restore weights on training loss); freeze the first two hidden blocks
        and fine-tune the last block + a fresh output head.
     c. For Wu (binary): sigmoid(1) head + BCE; continuous: linear(1) head + MSE.
  3. Aggregate by pooling the out-of-fold predictions into ONE vector and
     computing a single Spearman with a bootstrap 95% CI, the SAME estimator
     used for the zero-shot baseline (a mean of per-fold correlations is biased
     and high-variance on n~16-64 folds, and is not comparable to a full-N
     zero-shot rho).
  4. Compare fine-tuned vs zero-shot on equal footing; report whether the CI
     excludes zero rather than claiming an "improvement" that spans zero.

External datasets:
  - Wu:          81 SPs, binary (0/1), B. subtilis
  - Xue:        322 SPs, continuous (0–10437)
  - Zhang-P43:  114 SPs, continuous (0–193)
  - Zhang-PglVM: 114 SPs, continuous (0–327)
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.metrics import roc_auc_score
from scipy import stats

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import SignalPeptideVectorNN, FocalLoss
from src.data_loading import load_plm_with_bins

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Configuration ──────────────────────────────────────────────────────────
RANDOM_STATE = 42
SEEDS = [42, 123, 456, 789, 1024]
N_SEEDS = len(SEEDS)
N_FOLDS = 5
DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results'
FIGURES_DIR = Path(__file__).resolve().parent.parent / 'figures'

# Vector model config. Dropout 0.3 (a round, non-test-selected value; the old
# 0.35 was chosen on the test set). ESM2-650M embeddings (open/reproducible;
# the external datasets only ship ESM2, and AA-0 is discontinued anyway).
VECTOR_CONFIG = dict(
    hidden_layers=(256, 256, 128),
    dropout=0.3,
    learning_rate=5e-4,
    batch_size=32,
    epochs=200,
    loss='focal',
)

# Fine-tuning config
FT_EPOCHS = 100
FT_LR = 1e-4        # 5x lower than pretrain
FT_PATIENCE = 10     # stricter early stopping
FT_BATCH_SIZE = 32

# External datasets
EXTERNAL_DATASETS = {
    'Wu': {
        'file': 'wu_esm_embeddings.parquet',
        'is_binary': True,
        'description': '81 SPs, binary WA (functional/non-functional)',
    },
    'Xue': {
        'file': 'xue_esm_embeddings.parquet',
        'is_binary': False,
        'description': '322 SPs, WA 0–10437 (enzyme activity units)',
    },
    'Zhang-P43': {
        'file': 'zhang_p43_esm_embeddings.parquet',
        'is_binary': False,
        'description': '114 SPs, WA 0–193 (P43 promoter)',
    },
    'Zhang-PglVM': {
        'file': 'zhang_pglvm_esm_embeddings.parquet',
        'is_binary': False,
        'description': '114 SPs, WA 0–327 (PglVM promoter)',
    },
}

BIN_CENTERS = np.arange(1, 11, dtype=np.float64)  # [1, 2, ..., 10]


def load_external_dataset(filename):
    """Load an external ESM embedding parquet file."""
    path = DATA_DIR / filename
    df = pd.read_parquet(path)
    X = np.stack(df['embedding'].values)
    y = df['WA'].values.astype(np.float64)
    return X, y


def pretrain_vector_ensemble(X_train, y_train_bins, scaler):
    """Pretrain 5-seed vector NN ensemble on Grasso data."""
    X_train_scaled = scaler.transform(X_train)
    models = []

    for i, seed in enumerate(SEEDS):
        print(f"  Pretraining seed {seed} ({i+1}/{N_SEEDS})...", end=' ', flush=True)

        # 80/20 split for early stopping during pretrain
        rng = np.random.default_rng(seed)
        n = len(y_train_bins)
        indices = rng.permutation(n)
        n_val = int(n * 0.2)
        val_idx, train_idx = indices[:n_val], indices[n_val:]

        config = {**VECTOR_CONFIG, 'random_state': seed}
        model = SignalPeptideVectorNN(**config)
        model.fit(
            X_train_scaled[train_idx], y_train_bins[train_idx],
            X_val=X_train_scaled[val_idx], y_val_bins=y_train_bins[val_idx],
            verbose=0,
        )

        # Sanity: predict on val and compute WA MSE
        val_probs = model.predict(X_train_scaled[val_idx])
        val_wa_pred = val_probs @ BIN_CENTERS
        from src.data_loading import BIN_COLUMNS
        val_wa_true = y_train_bins[val_idx] @ BIN_CENTERS
        val_mse = float(np.mean((val_wa_true - val_wa_pred) ** 2))
        print(f"val MSE = {val_mse:.4f}")

        models.append(model)

    return models


def _build_ft_model(pretrained_model, output_activation, output_units, loss_fn,
                    metrics, seed):
    """
    Build a fine-tuning model by extracting pretrained feature layers and
    replacing the output head.

    Architecture of pretrained model (3 hidden blocks):
        layers[0]: Input
        layers[1]: Dense(256)   layers[2]: LeakyReLU  layers[3]: Dropout  (block 1)
        layers[4]: Dense(256)   layers[5]: LeakyReLU  layers[6]: Dropout  (block 2)
        layers[7]: Dense(128)   layers[8]: LeakyReLU  layers[9]: Dropout  (block 3)
        layers[10]: Dense(10, softmax)

    I freeze blocks 1-2 (layers 1-6) and keep block 3 + new output trainable.
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)

    src_model = pretrained_model.model_

    # Get the output of the last hidden block (before the softmax output layer)
    # This is the output of the last Dropout layer (layer index -2, i.e. second-to-last)
    feature_output = src_model.layers[-2].output  # last Dropout output

    # Build new model: pretrained features → new output head
    new_output = layers.Dense(output_units, activation=output_activation,
                              name='ft_output')(feature_output)
    ft_model = keras.Model(inputs=src_model.input, outputs=new_output)

    # Freeze first 2 hidden blocks (6 layers: Dense+LeakyReLU+Dropout × 2)
    # layers[0] = Input (skip), layers[1:7] = blocks 1-2 (freeze)
    # layers[7:10] = block 3 (trainable), layers[10] = new output (trainable)
    for layer in ft_model.layers[1:7]:
        layer.trainable = False
    for layer in ft_model.layers[7:]:
        layer.trainable = True

    ft_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=FT_LR),
        loss=loss_fn,
        metrics=metrics,
    )
    return ft_model


def clone_and_finetune_regression(pretrained_model, X_train, y_train,
                                  X_val, y_val, seed):
    """
    Fine-tune a pretrained vector model for continuous regression.

    Freezes first 2 hidden blocks, keeps last block trainable.
    Replaces the softmax(10) output with a linear(1) output.
    """
    ft_model = _build_ft_model(pretrained_model, 'linear', 1, 'mse', ['mae'], seed)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=FT_PATIENCE,
            restore_best_weights=True,
        ),
    ]

    fit_kwargs = dict(
        x=X_train, y=y_train,
        batch_size=FT_BATCH_SIZE,
        epochs=FT_EPOCHS,
        callbacks=callbacks,
        verbose=0,
    )
    if X_val is not None:
        fit_kwargs['validation_data'] = (X_val, y_val)

    ft_model.fit(**fit_kwargs)
    return ft_model


def clone_and_finetune_binary(pretrained_model, X_train, y_train,
                              X_val, y_val, seed):
    """
    Fine-tune a pretrained vector model for binary classification.

    Replaces softmax(10) output with sigmoid(1) for binary prediction.
    """
    ft_model = _build_ft_model(pretrained_model, 'sigmoid', 1,
                               'binary_crossentropy', ['accuracy'], seed)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=FT_PATIENCE,
            restore_best_weights=True,
        ),
    ]

    fit_kwargs = dict(
        x=X_train, y=y_train,
        batch_size=FT_BATCH_SIZE,
        epochs=FT_EPOCHS,
        callbacks=callbacks,
        verbose=0,
    )
    if X_val is not None:
        fit_kwargs['validation_data'] = (X_val, y_val)

    ft_model.fit(**fit_kwargs)
    return ft_model


def evaluate_fold(y_true, y_pred, is_binary):
    """Compute metrics for one fold."""
    result = {}

    sp_rho, sp_p = stats.spearmanr(y_true, y_pred)
    result['spearman_rho'] = float(sp_rho)
    result['spearman_p'] = float(sp_p)

    pe_r, pe_p = stats.pearsonr(y_true, y_pred)
    result['pearson_r'] = float(pe_r)
    result['pearson_p'] = float(pe_p)

    result['n_samples'] = len(y_true)

    if is_binary:
        try:
            auc = roc_auc_score(y_true, y_pred)
            result['auc_roc'] = float(auc)
        except ValueError:
            result['auc_roc'] = float('nan')

    return result


def spearman_with_ci(y_true, y_pred, is_binary=False, n_boot=5000, seed=42):
    """Spearman rho on the pooled predictions with a bootstrap 95% CI.

    This is the pooled aggregate estimator: ONE correlation over all N points,
    so zero-shot and fine-tuned (out-of-fold) predictions are compared on the
    same footing. For tiny folds, a mean of per-fold correlations is biased and
    high-variance; pooling fixes both.
    """
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    rho = float(stats.spearmanr(y_true, y_pred).statistic)
    pval = float(stats.spearmanr(y_true, y_pred).pvalue)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_pred[idx])) < 2 or len(np.unique(y_true[idx])) < 2:
            continue
        boots.append(stats.spearmanr(y_true[idx], y_pred[idx]).statistic)
    boots = np.array(boots, dtype=float)
    out = {
        'rho': rho, 'p_value': pval, 'n': int(n),
        'ci_lo': float(np.percentile(boots, 2.5)) if len(boots) else float('nan'),
        'ci_hi': float(np.percentile(boots, 97.5)) if len(boots) else float('nan'),
    }
    if is_binary:
        try:
            out['auc_roc'] = float(roc_auc_score(y_true, y_pred))
        except ValueError:
            out['auc_roc'] = float('nan')
    return out


def _inner_split(train_idx, y_ext, is_binary, fold_i, val_frac=0.15, min_val=6):
    """Carve a small inner validation set from a fold's training indices for
    early stopping (so I never restore weights on training loss)."""
    n = len(train_idx)
    n_val = max(min_val, int(round(val_frac * n)))
    if n_val >= n - min_val:        # fold too small for an inner split
        return train_idx, None
    strat = y_ext[train_idx] if is_binary else None
    try:
        inner_tr, inner_val = train_test_split(
            train_idx, test_size=n_val, random_state=RANDOM_STATE + fold_i,
            stratify=strat)
    except ValueError:              # stratification infeasible (rare class)
        inner_tr, inner_val = train_test_split(
            train_idx, test_size=n_val, random_state=RANDOM_STATE + fold_i)
    return inner_tr, inner_val


def run_finetuning_cv(pretrained_models, scaler_grasso, dataset_name, spec):
    """Run 5-fold CV fine-tuning for one external dataset."""
    X_ext, y_ext = load_external_dataset(spec['file'])
    is_binary = spec['is_binary']

    print(f"\n{'─'*60}")
    print(f"  {dataset_name}: {spec['description']}")
    print(f"  N={len(y_ext)}, binary={is_binary}")
    print(f"{'─'*60}")

    # --- Zero-shot baseline (no fine-tuning) ---
    X_ext_grasso_scaled = scaler_grasso.transform(X_ext)
    zs_preds = []
    for model in pretrained_models:
        probs = model.predict(X_ext_grasso_scaled)
        wa_pred = probs @ BIN_CENTERS
        zs_preds.append(wa_pred)
    zs_pred_avg = np.mean(zs_preds, axis=0)
    zs_metrics = evaluate_fold(y_ext, zs_pred_avg, is_binary)
    print(f"  Zero-shot Spearman: {zs_metrics['spearman_rho']:+.4f} "
          f"(p={zs_metrics['spearman_p']:.2e})")

    # --- Fine-tuned CV ---
    if is_binary:
        kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        splits = list(kf.split(X_ext, y_ext))
    else:
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        splits = list(kf.split(X_ext))

    # Out-of-fold predictions pooled into a single vector (primary estimator).
    oof_pred = np.full(len(y_ext), np.nan)
    fold_results = []

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        # Inner validation split for early stopping (never restore on
        # training loss). Scale with Grasso stats so the frozen pretrained
        # layers see the same input distribution as during pretraining.
        inner_tr, inner_val = _inner_split(train_idx, y_ext, is_binary, fold_i)
        X_itr_scaled = scaler_grasso.transform(X_ext[inner_tr])
        y_itr = y_ext[inner_tr]
        X_te_scaled = scaler_grasso.transform(X_ext[test_idx])
        y_te = y_ext[test_idx]
        if inner_val is not None:
            X_iv_scaled = scaler_grasso.transform(X_ext[inner_val])
            y_iv = y_ext[inner_val]
        else:
            X_iv_scaled = y_iv = None

        # Fine-tune each pretrained seed, average predictions on the held-out fold
        fold_preds = []
        for seed_i, pt_model in enumerate(pretrained_models):
            seed = SEEDS[seed_i]
            finetune = clone_and_finetune_binary if is_binary else clone_and_finetune_regression
            ft_model = finetune(pt_model, X_itr_scaled, y_itr,
                                X_iv_scaled, y_iv, seed)
            fold_preds.append(ft_model.predict(X_te_scaled, verbose=0).ravel())
            del ft_model
            keras.backend.clear_session()

        avg_pred = np.mean(fold_preds, axis=0)
        oof_pred[test_idx] = avg_pred

        fm = evaluate_fold(y_te, avg_pred, is_binary)
        fm.update(fold=fold_i, n_train=len(inner_tr), n_test=len(test_idx))
        fold_results.append(fm)
        extra = f"  AUC={fm.get('auc_roc', float('nan')):.3f}" if is_binary else ""
        print(f"  Fold {fold_i+1}/{N_FOLDS}: Spearman={fm['spearman_rho']:+.4f}  n_test={len(test_idx)}{extra}")

    # ── Primary aggregate: ONE Spearman over the pooled OOF predictions ──
    # (same estimator used for zero-shot, so the comparison is apples-to-apples)
    ft_pooled = spearman_with_ci(y_ext, oof_pred, is_binary=is_binary)
    zs_pooled = spearman_with_ci(y_ext, zs_pred_avg, is_binary=is_binary)

    # Per-fold mean kept only for reference (biased on tiny folds, not headline)
    agg = {}
    for key in ['spearman_rho', 'pearson_r']:
        vals = [f[key] for f in fold_results]
        agg[f'{key}_mean'] = float(np.mean(vals))
        agg[f'{key}_std'] = float(np.std(vals))

    print(f"\n  Zero-shot  Spearman (pooled): {zs_pooled['rho']:+.4f} "
          f"[{zs_pooled['ci_lo']:+.3f}, {zs_pooled['ci_hi']:+.3f}]")
    print(f"  Fine-tuned Spearman (pooled): {ft_pooled['rho']:+.4f} "
          f"[{ft_pooled['ci_lo']:+.3f}, {ft_pooled['ci_hi']:+.3f}]  p={ft_pooled['p_value']:.3g}")
    sig = "significant (CI excludes 0)" if (ft_pooled['ci_lo'] > 0 or ft_pooled['ci_hi'] < 0) \
          else "NOT significant (CI spans 0)"
    print(f"  → fine-tuned rho is {sig}")

    return {
        'dataset': dataset_name,
        'description': spec['description'],
        'is_binary': is_binary,
        'n_samples': len(y_ext),
        'n_folds': N_FOLDS,
        'n_seeds': N_SEEDS,
        'zero_shot': zs_metrics,
        'zero_shot_pooled': zs_pooled,
        'fine_tuned_pooled': ft_pooled,
        'fine_tuned_folds': fold_results,
        'fine_tuned_fold_mean': agg,
    }


def make_figure(all_results, save_path):
    """Generate comparison figure: zero-shot vs fine-tuned."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    datasets = [r['dataset'] for r in all_results]
    zs = [r['zero_shot_pooled'] for r in all_results]
    ft = [r['fine_tuned_pooled'] for r in all_results]
    zs_spearman = [z['rho'] for z in zs]
    ft_spearman = [f['rho'] for f in ft]
    # asymmetric bootstrap-CI error bars
    zs_err = [[z['rho'] - z['ci_lo'] for z in zs], [z['ci_hi'] - z['rho'] for z in zs]]
    ft_err = [[f['rho'] - f['ci_lo'] for f in ft], [f['ci_hi'] - f['rho'] for f in ft]]

    # Panel A: Side-by-side bar chart with 95% bootstrap CIs (pooled OOF)
    ax = axes[0]
    x = np.arange(len(datasets))
    width = 0.35

    bars_zs = ax.bar(x - width/2, zs_spearman, width, yerr=zs_err, label='Zero-shot',
                     color='steelblue', alpha=0.85, capsize=4, edgecolor='none')
    bars_ft = ax.bar(x + width/2, ft_spearman, width, yerr=ft_err,
                     label='Fine-tuned (pooled 5-fold OOF)', color='forestgreen',
                     alpha=0.85, capsize=4, edgecolor='none')

    ax.set_ylabel('Spearman ρ (95% bootstrap CI)')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha='right')
    ax.legend(fontsize=9)
    ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='--')

    # Panel B: Improvement
    ax = axes[1]
    improvements = [f - z for f, z in zip(ft_spearman, zs_spearman)]
    colors = ['forestgreen' if imp > 0 else 'firebrick' for imp in improvements]
    bars = ax.bar(datasets, improvements, color=colors, alpha=0.85,
                  edgecolor='none')

    ax.set_ylabel('Spearman ρ Improvement')
    ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='--')

    ax.set_xticklabels(datasets, rotation=15, ha='right')

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved to {save_path}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(RANDOM_STATE)

    # ── 1. Load Grasso training data with bins ────────────────────────────
    print("Loading Grasso ESM2-650M data with bin probabilities...")
    (X_train, X_test, y_train_wa, y_test_wa,
     y_train_bins, y_test_bins, meta) = load_plm_with_bins('esm2-650M')
    print(f"  Train: {meta['n_train']} samples ({meta['n_train_dropped_nan_bins']} dropped for NaN bins)")
    print(f"  Test:  {meta['n_test']} samples")

    # ── 2. Fit scaler on Grasso training data ─────────────────────────────
    scaler_grasso = StandardScaler()
    scaler_grasso.fit(X_train)

    # ── 3. Pretrain vector ensemble on Grasso ─────────────────────────────
    print(f"\nPretraining {N_SEEDS}-seed vector ensemble on Grasso...")
    pretrained_models = pretrain_vector_ensemble(X_train, y_train_bins, scaler_grasso)

    # Sanity check on Grasso test set
    X_test_scaled = scaler_grasso.transform(X_test)
    test_preds = []
    for m in pretrained_models:
        probs = m.predict(X_test_scaled)
        wa_pred = probs @ BIN_CENTERS
        test_preds.append(wa_pred)
    test_pred_avg = np.mean(test_preds, axis=0)
    test_mse = float(np.mean((y_test_wa - test_pred_avg) ** 2))
    test_sp, _ = stats.spearmanr(y_test_wa, test_pred_avg)
    print(f"\n  Grasso test MSE: {test_mse:.4f}, Spearman: {test_sp:.4f}")

    # ── 4. Fine-tune on each external dataset ─────────────────────────────
    print(f"\n{'='*60}")
    print("  Cross-Dataset Fine-Tuning (5-Fold CV)")
    print(f"{'='*60}")

    all_results = []
    for name, spec in EXTERNAL_DATASETS.items():
        result = run_finetuning_cv(pretrained_models, scaler_grasso, name, spec)
        all_results.append(result)

    # ── 5. Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Dataset':<15} {'N':>4}  {'Zero-Shot ρ':>12}  {'Fine-Tuned ρ [95% CI]':>26}  {'sig?':>6}")
    print(f"  {'-'*15} {'-'*4}  {'-'*12}  {'-'*26}  {'-'*6}")
    for r in all_results:
        zs = r['zero_shot_pooled']['rho']
        ftp = r['fine_tuned_pooled']
        sig = 'yes' if (ftp['ci_lo'] > 0 or ftp['ci_hi'] < 0) else 'no'
        print(f"  {r['dataset']:<15} {r['n_samples']:>4}  {zs:>+12.4f}  "
              f"{ftp['rho']:>+.3f} [{ftp['ci_lo']:+.3f}, {ftp['ci_hi']:+.3f}]  {sig:>6}")
    print("\n  (Spearman computed once over pooled out-of-fold predictions, "
          "not as a mean of per-fold correlations.)")

    # ── 6. Figure ─────────────────────────────────────────────────────────
    fig_path = FIGURES_DIR / 'cross_dataset_finetuning.png'
    make_figure(all_results, fig_path)

    # ── 7. Save results ───────────────────────────────────────────────────
    results_out = {
        'config': {
            'vector_config': {k: list(v) if isinstance(v, tuple) else v
                              for k, v in VECTOR_CONFIG.items()},
            'finetune_epochs': FT_EPOCHS,
            'finetune_lr': FT_LR,
            'finetune_patience': FT_PATIENCE,
            'n_folds': N_FOLDS,
            'n_seeds': N_SEEDS,
            'seeds': SEEDS,
        },
        'grasso_pretrain': {
            'test_mse': test_mse,
            'test_spearman': float(test_sp),
        },
        'datasets': {r['dataset']: r for r in all_results},
    }

    out_path = RESULTS_DIR / 'cross_dataset_finetuning_results.json'
    with open(out_path, 'w') as f:
        json.dump(results_out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
