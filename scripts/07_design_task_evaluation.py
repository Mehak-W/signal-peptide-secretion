#!/usr/bin/env python3
"""
Script 07: Design Task Evaluation

Evaluates whether models trained on characterized signal peptide variants can
predict relative performance of novel designed variants.

Feature types evaluated:
  - PhysChem (156d): available for all design variants from xlsx
  - ESM2-650M (1280d): generated via Script 00

Models evaluated per feature type:
  - RF with best hyperparameters from Script 02 (feature-specific)
  - NN regression 5-seed ensemble from Script 03 (feature-specific)

Design variants:
  - From xlsx Library_w_Bins_and_WA sheet (Set == NaN, WA not NaN)
  - 5,229 variants across 135 genes
  - Train/test sequences explicitly excluded (no leakage)
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy import stats

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_loading import (
    load_physicochemical, load_plm_embeddings, load_design_embeddings,
    GRASSO_FEATURES,
    MIN_SP_LENGTH, MAX_SP_LENGTH, MIN_WA, MAX_WA,
)
from src.models import SignalPeptideRegressorNN
from src.evaluation import compute_metrics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Configuration ──────────────────────────────────────────────────────────
SEEDS = [42, 123, 456, 789, 1024]
N_SEEDS = len(SEEDS)
RANDOM_STATE = 42
VAL_FRACTION = 0.2
MIN_GENE_VARIANTS = 5  # skip genes with fewer design variants

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results'
FIGURES_DIR = Path(__file__).resolve().parent.parent / 'figures'

# Best PhysChem RF hyperparameters (from Script 02)
BEST_PC_RF_PARAMS = dict(
    n_estimators=75,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features=0.5,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# Best ESM2-650M RF hyperparameters (from Script 02)
BEST_ESM_RF_PARAMS = dict(
    n_estimators=300,
    max_depth=25,
    min_samples_split=0.001,
    min_samples_leaf=4,
    max_features=0.5,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# Best PhysChem NN hyperparameters (from Script 03)
BEST_NN_PARAMS = dict(
    hidden_layers=(256, 128),
    dropout=0.3,
    l2_reg=0.01,
    learning_rate=1e-3,
    batch_size=64,
    epochs=200,
)

# Best ESM2-650M NN hyperparameters (from Script 03)
BEST_ESM_NN_PARAMS = dict(
    hidden_layers=(256, 128),
    dropout=0.4,
    l2_reg=0.01,
    learning_rate=1e-3,
    batch_size=64,
    epochs=200,
)


def evaluate_per_gene(df, pred_col, wt_wa_map):
    """
    Evaluate predictions per gene.

    Returns dict of per-gene metrics and overall metrics.
    """
    gene_results = {}
    all_actual = []
    all_pred = []

    for gene, group in df.groupby('gene'):
        if len(group) < MIN_GENE_VARIANTS:
            continue

        actual = group['WA'].values
        predicted = group[pred_col].values
        all_actual.extend(actual.tolist())
        all_pred.extend(predicted.tolist())

        # Spearman rank correlation
        if len(np.unique(actual)) > 1 and len(np.unique(predicted)) > 1:
            sp_rho, sp_p = stats.spearmanr(actual, predicted)
        else:
            sp_rho, sp_p = 0.0, 1.0

        # Pearson correlation
        if len(np.unique(actual)) > 1 and len(np.unique(predicted)) > 1:
            pe_r, pe_p = stats.pearsonr(actual, predicted)
        else:
            pe_r, pe_p = 0.0, 1.0

        # R²
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Classification accuracy: better/worse than WT
        wt_wa = wt_wa_map.get(gene)
        if wt_wa is not None and len(actual) > 0:
            actual_better = actual > wt_wa
            pred_better = predicted > wt_wa
            class_acc = float(np.mean(actual_better == pred_better))
        else:
            class_acc = None

        gene_results[gene] = {
            'n_variants': len(group),
            'spearman_rho': float(sp_rho),
            'spearman_p': float(sp_p),
            'pearson_r': float(pe_r),
            'pearson_p': float(pe_p),
            'r2': float(r2),
            'classification_accuracy': class_acc,
            'wt_wa': float(wt_wa) if wt_wa is not None else None,
        }

    # Overall metrics
    all_actual = np.array(all_actual)
    all_pred = np.array(all_pred)
    overall_sp, overall_sp_p = stats.spearmanr(all_actual, all_pred)
    overall_pe, overall_pe_p = stats.pearsonr(all_actual, all_pred)
    overall_mse = float(np.mean((all_actual - all_pred) ** 2))

    # Mean per-gene classification accuracy
    accs = [v['classification_accuracy'] for v in gene_results.values()
            if v['classification_accuracy'] is not None]
    mean_class_acc = float(np.mean(accs)) if accs else None

    overall = {
        'spearman_rho': float(overall_sp),
        'spearman_p': float(overall_sp_p),
        'pearson_r': float(overall_pe),
        'pearson_p': float(overall_pe_p),
        'mse': overall_mse,
        'n_variants': len(all_actual),
        'n_genes': len(gene_results),
        'mean_classification_accuracy': mean_class_acc,
    }

    return gene_results, overall


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(RANDOM_STATE)

    # ── 1. Load full xlsx ──────────────────────────────────────────────────
    print("Loading Library_w_Bins_and_WA from xlsx...")
    xlsx_path = DATA_DIR / 'sb2c00328_si_011.xlsx'
    df_lib = pd.read_excel(xlsx_path, sheet_name='Library_w_Bins_and_WA')
    print(f"  Total rows: {len(df_lib)}")

    # ── 2. Extract train/test sets (physicochemical features) ──────────────
    print("\nPreparing train/test data (physicochemical features)...")
    X_train_pc, X_test_pc, y_train, y_test, meta_pc = load_physicochemical()
    print(f"  Train: {meta_pc['n_train']} samples, {meta_pc['n_features']} features")
    print(f"  Test:  {meta_pc['n_test']} samples")

    # ── 2b. Load ESM2-650M train/test data ──────────────────────────────────
    print("\nPreparing train/test data (ESM2-650M)...")
    X_train_esm, X_test_esm, y_train_esm, y_test_esm, meta_esm = load_plm_embeddings('esm2-650M')
    print(f"  Train: {meta_esm['n_train']} samples, {meta_esm['n_features']} features")
    print(f"  Test:  {meta_esm['n_test']} samples")

    # Get train/test sequences for leakage check
    train_test_mask = df_lib['Set'].notna()
    qc_mask = (
        df_lib['WA'].notna() &
        df_lib['SP_aa'].notna() &
        (df_lib['SP_aa'].str.len() >= MIN_SP_LENGTH) &
        (df_lib['SP_aa'].str.len() <= MAX_SP_LENGTH) &
        (df_lib['WA'] >= MIN_WA) &
        (df_lib['WA'] <= MAX_WA)
    )
    train_test_seqs = set(df_lib[train_test_mask & qc_mask]['SP_aa'].values)
    print(f"  Train/test unique sequences: {len(train_test_seqs)}")

    # ── 3. Extract design variants ─────────────────────────────────────────
    print("\nExtracting design variants...")
    design_mask = df_lib['Set'].isna() & df_lib['WA'].notna() & df_lib['SP_aa'].notna()
    df_design = df_lib[design_mask].copy()
    print(f"  Design variants (Set=NaN, WA not NaN): {len(df_design)}")

    # Exclude any that overlap with train/test sequences
    leakage_mask = df_design['SP_aa'].isin(train_test_seqs)
    n_leaked = leakage_mask.sum()
    df_design = df_design[~leakage_mask].copy()
    print(f"  Excluded (in train/test): {n_leaked}")
    print(f"  Remaining design variants: {len(df_design)}")
    print(f"  Genes represented: {df_design['gene'].nunique()}")

    # Extract PhysChem features for design variants
    available_features = [f for f in GRASSO_FEATURES if f in df_design.columns]
    X_design_pc = df_design[available_features].fillna(0).values.astype(np.float64)
    y_design = df_design['WA'].values.astype(np.float64)
    print(f"  PhysChem design feature matrix: {X_design_pc.shape}")

    # ── 3b. Load ESM2-650M design embeddings ───────────────────────────────
    print("\nLoading ESM2-650M design embeddings...")
    df_design_emb = load_design_embeddings('esm2-650M')
    print(f"  Design embeddings loaded: {len(df_design_emb)} unique sequences")

    # Map embeddings to df_design rows by matching SP_aa to sequence
    emb_map = {}
    for _, row in df_design_emb.iterrows():
        emb_map[row['sequence']] = np.array(row['embedding'])

    design_esm_mask = df_design['SP_aa'].isin(emb_map)
    df_design_esm = df_design[design_esm_mask].copy()
    X_design_esm = np.stack([emb_map[seq] for seq in df_design_esm['SP_aa'].values])
    print(f"  Design variants with ESM embeddings: {len(df_design_esm)}")
    print(f"  ESM design feature matrix: {X_design_esm.shape}")

    # ── 4. Compute WT WA values ────────────────────────────────────────────
    # Match WT SP_aa sequences from the WT sheet to Library rows to get WA
    print("\nComputing WT WA values...")
    df_wt_sheet = pd.read_excel(xlsx_path, sheet_name='WT sequences')
    wt_wa_map = {}
    for _, wt_row in df_wt_sheet.iterrows():
        gene = wt_row.get('gene')
        sp_aa = wt_row.get('SP_aa')
        if pd.isna(gene) or pd.isna(sp_aa):
            continue
        # Find Library rows with this exact WT sequence that have WA
        matching = df_lib[(df_lib['SP_aa'] == sp_aa) & df_lib['WA'].notna()]
        if len(matching) > 0:
            wt_wa_map[gene] = float(matching['WA'].mean())
    print(f"  WT genes with WA: {len(wt_wa_map)}")
    for gene, wa in sorted(wt_wa_map.items())[:10]:
        print(f"    {gene}: WA = {wa:.2f}")
    if len(wt_wa_map) > 10:
        print(f"    ... ({len(wt_wa_map) - 10} more)")

    # Per-gene design variant counts
    print(f"\n  Per-gene design variant counts (top 20):")
    gene_counts = df_design['gene'].value_counts()
    for gene, count in gene_counts.head(20).items():
        skip = " (SKIP: <5)" if count < MIN_GENE_VARIANTS else ""
        wt_tag = f" [WT={wt_wa_map[gene]:.1f}]" if gene in wt_wa_map else ""
        print(f"    {gene}: {count}{wt_tag}{skip}")
    if len(gene_counts) > 20:
        print(f"    ... ({len(gene_counts) - 20} more genes)")

    # ── 5. Scale features ──────────────────────────────────────────────────
    scaler_pc = StandardScaler()
    X_train_pc_scaled = scaler_pc.fit_transform(X_train_pc)
    X_test_pc_scaled = scaler_pc.transform(X_test_pc)
    X_design_pc_scaled = scaler_pc.transform(X_design_pc)

    scaler_esm = StandardScaler()
    X_train_esm_scaled = scaler_esm.fit_transform(X_train_esm)
    X_test_esm_scaled = scaler_esm.transform(X_test_esm)
    X_design_esm_scaled = scaler_esm.transform(X_design_esm)

    results = {
        'feature_types': {
            'PhysChem': {
                'n_features': len(available_features),
                'n_design': len(df_design),
                'models': {},
            },
            'ESM2-650M': {
                'n_features': 1280,
                'n_design': len(df_design_esm),
                'models': {},
            },
        },
        'n_excluded_traintest': int(n_leaked),
        'n_genes': int(df_design['gene'].nunique()),
        'wt_wa_map': {k: float(v) for k, v in wt_wa_map.items()},
    }

    # ── 6. Train and predict: PhysChem RF ──────────────────────────────────
    print(f"\n{'='*60}")
    print("  Training RF (PhysChem)...")
    print(f"{'='*60}")

    rf_pc = RandomForestRegressor(**BEST_PC_RF_PARAMS)
    rf_pc.fit(X_train_pc_scaled, y_train)

    y_pred_rf_test = rf_pc.predict(X_test_pc_scaled)
    rf_test_mse = float(np.mean((y_test - y_pred_rf_test) ** 2))
    print(f"  RF Grasso test MSE: {rf_test_mse:.4f} (expected ~1.21)")

    df_design['pred_rf'] = rf_pc.predict(X_design_pc_scaled)

    # ── 7. Train and predict: PhysChem NN ensemble ─────────────────────────
    print(f"\n{'='*60}")
    print(f"  Training {N_SEEDS}-seed NN ensemble (PhysChem)...")
    print(f"{'='*60}")

    nn_preds = []
    for i, seed in enumerate(SEEDS):
        print(f"  Seed {seed} ({i+1}/{N_SEEDS})...", end=' ')
        rng = np.random.default_rng(seed)
        n = len(y_train)
        indices = rng.permutation(n)
        n_val = int(n * VAL_FRACTION)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        nn = SignalPeptideRegressorNN(
            **BEST_NN_PARAMS,
            random_state=seed,
        )
        nn.fit(
            X_train_pc_scaled[train_idx], y_train[train_idx],
            X_val=X_train_pc_scaled[val_idx], y_val=y_train[val_idx],
            verbose=0,
        )

        pred = nn.predict(X_design_pc_scaled)
        nn_preds.append(pred)

        test_pred = nn.predict(X_test_pc_scaled)
        seed_mse = float(np.mean((y_test - test_pred) ** 2))
        print(f"test MSE = {seed_mse:.4f}")

    df_design['pred_nn'] = np.mean(nn_preds, axis=0)

    # ── 8. Train and predict: ESM2-650M RF ─────────────────────────────────
    print(f"\n{'='*60}")
    print("  Training RF (ESM2-650M)...")
    print(f"{'='*60}")

    rf_esm = RandomForestRegressor(**BEST_ESM_RF_PARAMS)
    rf_esm.fit(X_train_esm_scaled, y_train_esm)

    y_pred_rf_esm_test = rf_esm.predict(X_test_esm_scaled)
    rf_esm_test_mse = float(np.mean((y_test_esm - y_pred_rf_esm_test) ** 2))
    print(f"  RF ESM Grasso test MSE: {rf_esm_test_mse:.4f} (expected ~1.19)")

    df_design_esm['pred_rf_esm'] = rf_esm.predict(X_design_esm_scaled)

    # ── 9. Train and predict: ESM2-650M NN ensemble ────────────────────────
    print(f"\n{'='*60}")
    print(f"  Training {N_SEEDS}-seed NN ensemble (ESM2-650M)...")
    print(f"{'='*60}")

    nn_esm_preds = []
    for i, seed in enumerate(SEEDS):
        print(f"  Seed {seed} ({i+1}/{N_SEEDS})...", end=' ')
        rng = np.random.default_rng(seed)
        n = len(y_train_esm)
        indices = rng.permutation(n)
        n_val = int(n * VAL_FRACTION)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        nn = SignalPeptideRegressorNN(
            **BEST_ESM_NN_PARAMS,
            random_state=seed,
        )
        nn.fit(
            X_train_esm_scaled[train_idx], y_train_esm[train_idx],
            X_val=X_train_esm_scaled[val_idx], y_val=y_train_esm[val_idx],
            verbose=0,
        )

        pred = nn.predict(X_design_esm_scaled)
        nn_esm_preds.append(pred)

        test_pred = nn.predict(X_test_esm_scaled)
        seed_mse = float(np.mean((y_test_esm - test_pred) ** 2))
        print(f"test MSE = {seed_mse:.4f}")

    df_design_esm['pred_nn_esm'] = np.mean(nn_esm_preds, axis=0)

    # ── 10. Evaluate per-gene and overall ──────────────────────────────────
    print(f"\n{'='*60}")
    print("  Design Task Evaluation Results")
    print(f"{'='*60}")

    model_configs = [
        ('PhysChem RF',  'pred_rf',      df_design,     'PhysChem'),
        ('PhysChem NN',  'pred_nn',      df_design,     'PhysChem'),
        ('ESM2-650M RF', 'pred_rf_esm',  df_design_esm, 'ESM2-650M'),
        ('ESM2-650M NN', 'pred_nn_esm',  df_design_esm, 'ESM2-650M'),
    ]

    for model_name, pred_col, df_eval, feat_type in model_configs:
        gene_results, overall = evaluate_per_gene(df_eval, pred_col, wt_wa_map)

        print(f"\n  --- {model_name} ---")
        print(f"  Overall: Spearman={overall['spearman_rho']:.4f} (p={overall['spearman_p']:.2e}), "
              f"Pearson={overall['pearson_r']:.4f}, MSE={overall['mse']:.4f}, "
              f"n={overall['n_variants']}")
        if overall['mean_classification_accuracy'] is not None:
            print(f"  Mean per-gene classification accuracy: {overall['mean_classification_accuracy']:.4f}")

        print(f"\n  Per-gene results (genes with >={MIN_GENE_VARIANTS} variants):")
        print(f"  {'Gene':<15} {'N':>4} {'Spearman':>10} {'Pearson':>10} {'ClassAcc':>10}")
        print(f"  {'-'*15} {'-'*4} {'-'*10} {'-'*10} {'-'*10}")
        for gene, gm in sorted(gene_results.items(), key=lambda x: -x[1]['n_variants']):
            ca = f"{gm['classification_accuracy']:.4f}" if gm['classification_accuracy'] is not None else "N/A"
            print(f"  {gene:<15} {gm['n_variants']:>4} {gm['spearman_rho']:>+10.4f} "
                  f"{gm['pearson_r']:>+10.4f} {ca:>10}")

        results['feature_types'][feat_type]['models'][model_name] = {
            'per_gene': gene_results,
            'overall': overall,
        }

    # ── 11. Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Model':<16} {'Features':<12} {'Spearman':>10} {'Pearson':>10} {'MSE':>10} {'ClassAcc':>10} {'N':>6}")
    print(f"  {'-'*16} {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*6}")
    for model_name, pred_col, df_eval, feat_type in model_configs:
        o = results['feature_types'][feat_type]['models'][model_name]['overall']
        ca = f"{o['mean_classification_accuracy']:.4f}" if o['mean_classification_accuracy'] else "N/A"
        print(f"  {model_name:<16} {feat_type:<12} {o['spearman_rho']:>+10.4f} {o['pearson_r']:>+10.4f} "
              f"{o['mse']:>10.4f} {ca:>10} {o['n_variants']:>6}")

    # ── 12. Figure ─────────────────────────────────────────────────────────
    _make_figure(df_design, df_design_esm, results, wt_wa_map)

    # ── 13. Save JSON ──────────────────────────────────────────────────────
    out_path = RESULTS_DIR / 'design_task_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


def _make_figure(df_design, df_design_esm, results, wt_wa_map):
    """Generate 2x3 multi-panel design task evaluation figure."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row configs: (row_idx, feature_type, df, pred_cols, model_colors, model_labels, row_label)
    row_configs = [
        (0, 'PhysChem', df_design,
         {'PhysChem RF': 'pred_rf', 'PhysChem NN': 'pred_nn'},
         {'PhysChem RF': 'steelblue', 'PhysChem NN': 'darkorange'},
         {'PhysChem RF': 'RF', 'PhysChem NN': 'NN'}),
        (1, 'ESM2-650M', df_design_esm,
         {'ESM2-650M RF': 'pred_rf_esm', 'ESM2-650M NN': 'pred_nn_esm'},
         {'ESM2-650M RF': 'forestgreen', 'ESM2-650M NN': 'firebrick'},
         {'ESM2-650M RF': 'RF', 'ESM2-650M NN': 'NN'}),
    ]

    for row_idx, feat_type, df_eval, pred_cols, colors, labels in row_configs:
        # ── Panel A: Predicted vs Actual WA scatter ────────────────────────
        ax = axes[row_idx, 0]
        for model_name, pred_col in pred_cols.items():
            ax.scatter(df_eval['WA'], df_eval[pred_col],
                       alpha=0.2, s=10, color=colors[model_name],
                       label=labels[model_name])

        wa_min = min(df_eval['WA'].min(), 1)
        wa_max = max(df_eval['WA'].max(), 10)
        ax.plot([wa_min, wa_max], [wa_min, wa_max], 'k--', lw=0.8, alpha=0.5)
        ax.set_xlabel('Actual WA')
        ax.set_ylabel('Predicted WA')
        panel_letter = 'A' if row_idx == 0 else 'D'
        ax.set_title(f'({panel_letter}) Predicted vs Actual WA')
        ax.legend(fontsize=8)

        # ── Panel B: Per-gene Spearman bar chart (top 15) ──────────────────
        ax = axes[row_idx, 1]
        model_keys = list(pred_cols.keys())
        all_genes = sorted(set.intersection(*[
            set(results['feature_types'][feat_type]['models'][m]['per_gene'].keys())
            for m in model_keys
        ]))
        genes_by_n = sorted(all_genes,
                            key=lambda g: -results['feature_types'][feat_type]['models'][model_keys[0]]['per_gene'][g]['n_variants'])
        genes = genes_by_n[:15]

        if genes:
            x = np.arange(len(genes))
            width = 0.35

            for i, model_name in enumerate(model_keys):
                gene_rhos = [results['feature_types'][feat_type]['models'][model_name]['per_gene'][g]['spearman_rho']
                             for g in genes]
                ax.bar(x + i * width, gene_rhos, width,
                       label=labels[model_name], color=colors[model_name], alpha=0.85)

            ax.set_xticks(x + width / 2)
            ax.set_xticklabels(genes, rotation=45, ha='right', fontsize=7)
            ax.set_ylabel('Spearman Rank Correlation')
            panel_letter = 'B' if row_idx == 0 else 'E'
            ax.set_title(f'({panel_letter}) Per-Gene Ranking (top 15)')
            ax.legend(fontsize=8)
            ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='--')

        # ── Panel C: Per-gene classification accuracy ──────────────────────
        ax = axes[row_idx, 2]
        genes_with_wt = [g for g in genes
                         if results['feature_types'][feat_type]['models'][model_keys[0]]['per_gene'][g]['classification_accuracy'] is not None]

        if genes_with_wt:
            x = np.arange(len(genes_with_wt))
            width = 0.35

            for i, model_name in enumerate(model_keys):
                accs = [results['feature_types'][feat_type]['models'][model_name]['per_gene'][g]['classification_accuracy']
                        for g in genes_with_wt]
                ax.bar(x + i * width, accs, width,
                       label=labels[model_name], color=colors[model_name], alpha=0.85)

            ax.set_xticks(x + width / 2)
            ax.set_xticklabels(genes_with_wt, rotation=45, ha='right', fontsize=7)
            ax.set_ylabel('Classification Accuracy')
            panel_letter = 'C' if row_idx == 0 else 'F'
            ax.set_title(f'({panel_letter}) Better/Worse than WT')
            ax.legend(fontsize=8)
            ax.set_ylim(0, 1.05)
        else:
            panel_letter = 'C' if row_idx == 0 else 'F'
            ax.text(0.5, 0.5, 'No genes with WT WA\navailable for classification',
                    ha='center', va='center', transform=ax.transAxes, fontsize=11)
            ax.set_title(f'({panel_letter}) Better/Worse than WT')

    plt.tight_layout()

    fig_path = FIGURES_DIR / 'design_task_evaluation.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to {fig_path}")


if __name__ == '__main__':
    main()
