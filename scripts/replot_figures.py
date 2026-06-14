#!/usr/bin/env python3
"""Regenerate figures from saved JSON results (no retraining), applying the
shared Tufte style (src/plotstyle.py): single-column, no chartjunk, muted gray,
captions carry context.

Figures use the Grasso RF baseline (1.22) as the reference line.
"""
import sys, json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from src.plotstyle import apply_tufte, tufte_ax, GRAY, LIGHT_GRAY, PALETTE
RESULTS_DIR = BASE / 'results'
FIGURES_DIR = BASE / 'figures'
GRASSO = 1.22


def replot_06():
    """vector_regression_comparison.png; CE vs focal per embedding."""
    results = json.load(open(RESULTS_DIR / 'vector_regression_results.json'))
    emb = ['ESM2-650M', 'ESM2-3B', 'Ginkgo-AA0']
    ce = [results[f"{e}_categorical_crossentropy"]['test_metrics']['mse'] for e in emb]
    fo = [results[f"{e}_focal"]['test_metrics']['mse'] for e in emb]
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    x = np.arange(len(emb)); w = 0.36
    ax.bar(x - w/2, ce, w, label='cross-entropy', color=LIGHT_GRAY)
    ax.bar(x + w/2, fo, w, label='focal', color=GRAY)
    ax.set_xticks(x); ax.set_xticklabels(emb); ax.set_ylabel('Test MSE')
    ax.set_ylim(0, max(ce + fo) * 1.12)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
    tufte_ax(ax); fig.savefig(FIGURES_DIR / 'vector_regression_comparison.png', bbox_inches='tight'); plt.close(fig)
    print("  vector_regression_comparison.png")


def replot_08():
    """bootstrap_forest_plot.png; point + 95% CI per model."""
    data = json.load(open(RESULTS_DIR / 'bootstrap_ci_results.json'))
    results, order = data['results'], data['model_order']
    fig, ax = plt.subplots(figsize=(3.6, 4.2))
    y = np.arange(len(order))
    for i, name in enumerate(order):
        r = results[name]['mse']
        c = GRAY if name.startswith('Vec') else LIGHT_GRAY
        ax.plot([r['ci_lo'], r['ci_hi']], [y[i], y[i]], color=c, lw=1.4, zorder=1)
        ax.plot(r['point'], y[i], 'o', color=c, ms=4, zorder=2)
    ax.axvline(GRASSO, color='#999999', ls=':', lw=0.8, label='Grasso baseline (1.22)')
    ax.set_yticks(y); ax.set_yticklabels(order, fontsize=6.5); ax.invert_yaxis()
    ax.set_xlabel('Test MSE (point + 95% CI)')
    ax.legend(handles=[
        Line2D([0], [0], marker='o', color=GRAY, label='vector models', ms=4, lw=1.4),
        Line2D([0], [0], marker='o', color=LIGHT_GRAY, label='scalar models', ms=4, lw=1.4),
        Line2D([0], [0], color='#999999', ls=':', label='Grasso baseline (1.22)')],
        loc='lower center', bbox_to_anchor=(0.5, 1.005), ncol=2, frameon=False, fontsize=7.5)
    tufte_ax(ax); fig.savefig(FIGURES_DIR / 'bootstrap_forest_plot.png', bbox_inches='tight'); plt.close(fig)
    print("  bootstrap_forest_plot.png")


def replot_13():
    """cross_dataset_finetuning.png; pooled OOF zero-shot vs fine-tuned + 95% CI."""
    d = json.load(open(RESULTS_DIR / 'cross_dataset_finetuning_results.json'))['datasets']
    names = list(d.keys())
    def vals(key): return [d[n][key]['rho'] for n in names]
    def err(key): return [[d[n][key]['rho'] - d[n][key]['ci_lo'] for n in names],
                          [d[n][key]['ci_hi'] - d[n][key]['rho'] for n in names]]
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    x = np.arange(len(names)); w = 0.36
    ax.bar(x - w/2, vals('zero_shot_pooled'), w, yerr=err('zero_shot_pooled'), color=LIGHT_GRAY,
           label='zero-shot', error_kw=dict(ecolor=GRAY, lw=0.8, capsize=2))
    ax.bar(x + w/2, vals('fine_tuned_pooled'), w, yerr=err('fine_tuned_pooled'), color=GRAY,
           label='fine-tuned', error_kw=dict(ecolor=GRAY, lw=0.8, capsize=2))
    ax.axhline(0, color='#999999', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylabel(r'Spearman $\rho$ (pooled OOF, 95% CI)'); ax.legend()
    tufte_ax(ax); fig.savefig(FIGURES_DIR / 'cross_dataset_finetuning.png'); plt.close(fig)
    print("  cross_dataset_finetuning.png")


def replot_01():
    """grasso_reproduction.png; train vs test MSE (shows RF overfit) + baseline."""
    d = json.load(open(RESULTS_DIR / 'grasso_reproduction.json'))
    tr, te = d['train_metrics']['mse'], d['test_metrics']['mse']
    fig, ax = plt.subplots(figsize=(3.0, 2.7))
    ax.bar([0, 1], [tr, te], color=[LIGHT_GRAY, GRAY], width=0.6)
    ax.axhline(GRASSO, color='#999999', ls=':', lw=0.8)
    ax.text(1.45, GRASSO, ' Grasso 1.22', color='#777777', fontsize=7, va='center')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['train', 'test'])
    ax.set_ylabel('MSE'); ax.set_ylim(0, max(te, GRASSO) * 1.15)
    for xi, v in zip([0, 1], [tr, te]):
        ax.text(xi, v + 0.03, f'{v:.2f}', ha='center', fontsize=7, color=GRAY)
    tufte_ax(ax); fig.savefig(FIGURES_DIR / 'grasso_reproduction.png'); plt.close(fig)
    print("  grasso_reproduction.png  (train<<test = RF overfit)")


def replot_11():
    """dropout_validation.png; validation MSE vs dropout (all indistinguishable)."""
    r = json.load(open(RESULTS_DIR / 'dropout_validation_results.json'))['results']
    drops = sorted(float(k.split('-')[1]) for k in r)
    means = [r[f'drop-{d:.2f}']['val_mse_mean'] for d in drops]
    stds = [r[f'drop-{d:.2f}']['val_mse_std'] for d in drops]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.errorbar(drops, means, yerr=stds, marker='o', ms=4, color=GRAY, capsize=2, lw=1.2)
    ax.set_xlabel('Dropout rate'); ax.set_ylabel('Validation MSE')
    ax.set_xticks(drops)
    tufte_ax(ax); fig.savefig(FIGURES_DIR / 'dropout_validation.png'); plt.close(fig)
    print("  dropout_validation.png  (flat -> dropouts statistically indistinguishable)")


def replot_14():
    """relu_squared_comparison.png; LeakyReLU vs ReLU-squared test MSE."""
    d = json.load(open(RESULTS_DIR / 'relu_squared_comparison_results.json'))
    keys = ['leaky_relu', 'relu_squared']; labels = ['LeakyReLU', 'ReLU$^2$']
    pts = [d[k]['bootstrap_ci']['mse']['point'] for k in keys]
    err = [[d[k]['bootstrap_ci']['mse']['point'] - d[k]['bootstrap_ci']['mse']['ci_lo'] for k in keys],
           [d[k]['bootstrap_ci']['mse']['ci_hi'] - d[k]['bootstrap_ci']['mse']['point'] for k in keys]]
    fig, ax = plt.subplots(figsize=(3.0, 2.7))
    ax.bar([0, 1], pts, yerr=err, color=[GRAY, LIGHT_GRAY], width=0.6,
           error_kw=dict(ecolor=GRAY, lw=0.8, capsize=2))
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_ylabel('Test MSE'); ax.set_ylim(0, max(pts) * 1.12)
    tufte_ax(ax); fig.savefig(FIGURES_DIR / 'relu_squared_comparison.png'); plt.close(fig)
    print("  relu_squared_comparison.png")


def replot_18():
    """linear_baseline.png; linear/Ridge/XGBoost/NN test MSE on embeddings."""
    d = json.load(open(RESULTS_DIR / 'linear_baseline_results.json'))
    keys = ['linear_probe', 'ridge', 'xgboost', 'best_nn']
    labels = ['linear\nprobe', 'Ridge', 'XGBoost', 'vector\nNN']
    keys = [k for k in keys if k in d]
    pts = [d[k]['bootstrap_ci']['mse']['point'] for k in keys]
    fig, ax = plt.subplots(figsize=(3.4, 2.7)); x = np.arange(len(keys))
    ax.bar(x, pts, color=GRAY, width=0.6)
    ax.axhline(GRASSO, color='#999999', ls=':', lw=0.8); ax.text(len(keys)-0.4, GRASSO, ' Grasso', color='#777777', fontsize=7, va='center')
    ax.set_xticks(x); ax.set_xticklabels([labels[i] for i in range(len(keys))])
    ax.set_ylabel('Test MSE'); ax.set_ylim(0, max(pts + [GRASSO]) * 1.1)
    tufte_ax(ax); fig.savefig(FIGURES_DIR / 'linear_baseline.png'); plt.close(fig)
    print("  linear_baseline.png")


def replot_05():
    """cross_dataset_generalization.png; zero-shot Spearman per external dataset."""
    d = json.load(open(RESULTS_DIR / 'cross_dataset_results.json'))['external_datasets']
    names = list(d.keys())
    rf = [d[n]['rf_metrics']['spearman_rho'] for n in names]
    nn = [d[n]['nn_metrics']['spearman_rho'] for n in names]
    fig, ax = plt.subplots(figsize=(4.0, 2.7)); x = np.arange(len(names)); w = 0.36
    ax.bar(x - w/2, rf, w, color=LIGHT_GRAY, label='RF')
    ax.bar(x + w/2, nn, w, color=GRAY, label='NN')
    ax.axhline(0, color='#999999', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylabel(r'Zero-shot Spearman $\rho$')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
    tufte_ax(ax); fig.savefig(FIGURES_DIR / 'cross_dataset_generalization.png'); plt.close(fig)
    print("  cross_dataset_generalization.png")


def replot_15():
    """regression_vs_classification.png; test MSE per output formulation."""
    d = json.load(open(RESULTS_DIR / 'regression_vs_classification_results.json'))
    labels = {'vector_10bin': '10-bin focal', 'vector_10bin_cce': '10-bin CE',
              '5class': '5-class', '3class': '3-class', 'binary': 'binary'}
    keys = [k for k in labels if k in d]
    def mse(v): return v.get('bootstrap_ci', {}).get('mse', {}).get('point', v.get('mse'))
    vals = [mse(d[k]) for k in keys]
    fig, ax = plt.subplots(figsize=(4.0, 2.7)); x = np.arange(len(keys))
    ax.bar(x, vals, color=GRAY)
    ax.set_xticks(x); ax.set_xticklabels([labels[k] for k in keys], rotation=15, ha='right')
    ax.set_ylabel('Test MSE'); ax.set_ylim(0, max(vals) * 1.12)
    tufte_ax(ax); fig.savefig(FIGURES_DIR / 'regression_vs_classification.png'); plt.close(fig)
    print("  regression_vs_classification.png")


def replot_17():
    """gene_stratified_evaluation.png; standard vs leave-one-gene-out MSE."""
    d = json.load(open(RESULTS_DIR / 'gene_stratified_evaluation_results.json'))
    # use the fair single-seed standard (0.964)
    std = d['standard_comparison']['standard_test_mse_1seed']
    logo = d['logo_cv']['overall_mse']
    fig, ax = plt.subplots(figsize=(3.2, 2.7))
    ax.bar([0, 1], [std, logo], color=[LIGHT_GRAY, GRAY], width=0.6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['standard\n(gene overlap)', 'leave-one-\ngene-out'])
    ax.set_ylabel('Test MSE'); ax.set_ylim(0, logo * 1.15)
    for xi, v in zip([0, 1], [std, logo]):
        ax.text(xi, v + 0.03, f'{v:.2f}', ha='center', fontsize=7, color=GRAY)
    tufte_ax(ax); fig.savefig(FIGURES_DIR / 'gene_stratified_evaluation.png'); plt.close(fig)
    print("  gene_stratified_evaluation.png")


def replot_19():
    """ablation_controls.png; cumulative ablation decomposition."""
    tbl = json.load(open(RESULTS_DIR / 'ablation_controls_results.json'))['ablation']['ablation_table']
    steps = [r['step'] for r in tbl]; vals = [r['ensemble_mse'] for r in tbl]
    fig, ax = plt.subplots(figsize=(4.4, 2.8)); x = np.arange(len(steps))
    ax.bar(x, vals, color=GRAY)
    ax.set_xticks(x); ax.set_xticklabels(steps, rotation=20, ha='right', fontsize=6)
    ax.set_ylabel('Ensemble test MSE'); ax.set_ylim(0, max(vals) * 1.1)
    tufte_ax(ax); fig.savefig(FIGURES_DIR / 'ablation_controls.png'); plt.close(fig)
    print("  ablation_controls.png")


def _sweep_fig(json_name, phase_keys, out_name):
    """Generic clean bar chart for a config sweep (exploration figures)."""
    res = json.load(open(RESULTS_DIR / json_name))
    labels, mses = [], []
    for pk in phase_keys:
        for name, r in res.get(pk, {}).items():
            if isinstance(r, dict) and 'mse' in r:
                labels.append(name); mses.append(r['mse'])
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    x = np.arange(len(labels))
    ax.bar(x, mses, color=GRAY)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=6)
    ax.set_ylabel('Test MSE'); ax.set_ylim(0, max(mses) * 1.12)
    tufte_ax(ax); fig.savefig(FIGURES_DIR / out_name); plt.close(fig)
    print(f"  {out_name}")


if __name__ == '__main__':
    apply_tufte()
    print("Regenerating figures (Tufte style) from saved results...")
    replot_01()
    replot_05()
    replot_06()
    replot_08()
    replot_11()
    replot_13()
    replot_14()
    replot_15()
    replot_17()
    replot_18()
    replot_19()
    # exploration sweeps (note: these depict the test-set hyperparameter search
    # discussed as a limitation in the paper; shown for transparency)
    _sweep_fig('vector_architecture_search_results.json',
               ['phase1', 'phase3', 'phase4'], 'vector_architecture_search.png')
    _sweep_fig('vector_ensemble_optimization_results.json',
               ['phase1_dropout', 'phase2_4layer', 'phase3_cosine',
                'phase4_large_ensemble', 'phase5_mixed'], 'vector_ensemble_optimization.png')
    print("Done.")
