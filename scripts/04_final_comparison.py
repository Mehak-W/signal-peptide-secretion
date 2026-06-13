#!/usr/bin/env python3
"""
Script 04: Final Comparison

Load results from Scripts 01-03 and produce a grouped bar chart comparing
all models across all feature types on the same test set.

Outputs:
  - results/final_comparison.csv
  - figures/final_comparison.png
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.plotstyle import apply_tufte, tufte_ax, GRAY, LIGHT_GRAY, PALETTE
apply_tufte()

RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results'
FIGURES_DIR = Path(__file__).resolve().parent.parent / 'figures'
GRASSO_BASELINE_MSE = 1.22


def load_results():
    """Load all result JSON files."""
    grasso_path = RESULTS_DIR / 'grasso_reproduction.json'
    rf_path = RESULTS_DIR / 'rf_search_results.json'
    nn_path = RESULTS_DIR / 'nn_search_results.json'

    results = {}

    # Grasso reproduction (Script 01)
    if grasso_path.exists():
        with open(grasso_path) as f:
            results['grasso'] = json.load(f)
    else:
        print(f"[WARN] Missing {grasso_path}")

    # RF search (Script 02)
    if rf_path.exists():
        with open(rf_path) as f:
            results['rf_search'] = json.load(f)
    else:
        print(f"[WARN] Missing {rf_path}")

    # NN search (Script 03)
    if nn_path.exists():
        with open(nn_path) as f:
            results['nn_search'] = json.load(f)
    else:
        print(f"[WARN] Missing {nn_path}")

    return results


def build_comparison_table(results):
    """Build a comparison DataFrame from all results."""
    rows = []

    feature_types = ['PhysChem', 'ESM2-650M', 'ESM2-3B', 'Ginkgo-AA0']

    # Grasso reproduction (PhysChem only)
    if 'grasso' in results:
        g = results['grasso']
        rows.append({
            'Feature Type': 'PhysChem',
            'Model': 'Grasso RF (exact)',
            'Test MSE': g['test_metrics']['mse'],
            'Test RMSE': g['test_metrics']['rmse'],
            'Test MAE': g['test_metrics']['mae'],
            'Test R²': g['test_metrics']['r2'],
            'Spearman ρ': g['test_metrics']['spearman_rho'],
            'Pearson r': g['test_metrics']['pearson_r'],
            'N Test': g['test_metrics']['n_samples'],
        })

    # RF search results (all feature types)
    if 'rf_search' in results:
        for ft in feature_types:
            if ft in results['rf_search']:
                r = results['rf_search'][ft]
                rows.append({
                    'Feature Type': ft,
                    'Model': 'RF (tuned)',
                    'Test MSE': r['test_metrics']['mse'],
                    'Test RMSE': r['test_metrics']['rmse'],
                    'Test MAE': r['test_metrics']['mae'],
                    'Test R²': r['test_metrics']['r2'],
                    'Spearman ρ': r['test_metrics']['spearman_rho'],
                    'Pearson r': r['test_metrics']['pearson_r'],
                    'N Test': r['test_metrics']['n_samples'],
                })

    # NN search results (all feature types)
    if 'nn_search' in results:
        for ft in feature_types:
            if ft in results['nn_search']:
                r = results['nn_search'][ft]
                rows.append({
                    'Feature Type': ft,
                    'Model': 'NN (regression)',
                    'Test MSE': r['test_metrics']['mse'],
                    'Test RMSE': r['test_metrics']['rmse'],
                    'Test MAE': r['test_metrics']['mae'],
                    'Test R²': r['test_metrics']['r2'],
                    'Spearman ρ': r['test_metrics']['spearman_rho'],
                    'Pearson r': r['test_metrics']['pearson_r'],
                    'N Test': r['test_metrics']['n_samples'],
                })

    return pd.DataFrame(rows)


def plot_grouped_bar(df, save_path):
    """
    Grouped bar chart: feature type (x) x model type (groups), MSE on y-axis.
    Horizontal line at Grasso baseline 1.22.
    """
    feature_types = ['PhysChem', 'ESM2-650M', 'ESM2-3B', 'Ginkgo-AA0']
    models = df['Model'].unique().tolist()
    n_models = len(models)

    # Muted palette (Tufte): encode model by shade of gray, not saturated hue
    colors = ['#2b2b2b', '#6e6e6e', '#9e9e9e', '#bdbdbd', '#dadada']

    fig, ax = plt.subplots(figsize=(4.2, 2.8))

    bar_width = 0.8 / n_models
    x = np.arange(len(feature_types))

    for i, model in enumerate(models):
        mse_values = []
        for ft in feature_types:
            subset = df[(df['Feature Type'] == ft) & (df['Model'] == model)]
            if len(subset) > 0:
                mse_values.append(subset['Test MSE'].values[0])
            else:
                mse_values.append(0)

        offset = (i - n_models / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, mse_values, bar_width * 0.9,
                      label=model, color=colors[i % len(colors)], edgecolor='none')

    # Grasso baseline line
    ax.axhline(y=GRASSO_BASELINE_MSE, color='#999999', linestyle=':', linewidth=0.8,
               label=f'Grasso baseline ({GRASSO_BASELINE_MSE})')

    ax.set_xlabel('Feature Type')
    ax.set_ylabel('Test MSE')
    ax.set_xticks(x)
    ax.set_xticklabels(feature_types)
    # Legend placed above the axes so it never overlaps the bars.
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=2,
              frameon=False, fontsize=7, handlelength=1.2, columnspacing=1.2)
    # Bar charts must start the y-axis at 0 so bar heights are proportional to
    # the values (a truncated axis visually exaggerates small MSE differences).
    ax.set_ylim(0, max(df['Test MSE'].max() * 1.08, GRASSO_BASELINE_MSE * 1.08))
    tufte_ax(ax)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to {save_path}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading results from Scripts 01-03...")
    results = load_results()

    print("\nBuilding comparison table...")
    df = build_comparison_table(results)

    if df.empty:
        print("[ERROR] No results found. Run Scripts 01-03 first.")
        return

    # Print comparison table
    print("\n" + "="*80)
    print("  FINAL COMPARISON")
    print("="*80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    print(df.to_string(index=False, float_format='%.4f'))
    print(f"\n  Grasso baseline MSE: {GRASSO_BASELINE_MSE}")

    # Save CSV
    csv_path = RESULTS_DIR / 'final_comparison.csv'
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\nCSV saved to {csv_path}")

    # Plot
    fig_path = FIGURES_DIR / 'final_comparison.png'
    plot_grouped_bar(df, fig_path)

    return df


if __name__ == '__main__':
    main()
