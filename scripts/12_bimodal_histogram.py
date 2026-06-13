#!/usr/bin/env python3
"""
Script 12: Bimodal Bin Distribution Figure

Generates a motivating figure showing examples where the weighted average (WA)
metric falls between two peaks in the bin probability distribution. This
illustrates why predicting full 10-bin distributions (vector regression) is
more informative than predicting scalar WA alone.

Selection criterion: bimodal distributions where WA falls between two
substantial peaks, with probability mass spread across multiple bins
(not just two extreme bars). Ranked by number of bins with >5% mass,
then by peak separation.
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Configuration ──────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
FIGURES_DIR = Path(__file__).resolve().parent.parent / 'figures'
RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results'

BIN_COLS = [f'Perc_unambiguousReads_BIN{i:02d}_bin' for i in range(1, 11)]


def find_bimodal_examples(df, n_examples=2):
    """
    Find sequences with spread bimodal bin distributions where WA falls
    between peaks. Prefers examples with mass across many bins (not just
    two extreme bars).

    Returns:
        list of dicts sorted by (n_substantial_bins, peak_separation) descending
    """
    examples = []

    for idx, row in df.iterrows():
        wa = row['WA']
        if pd.isna(wa):
            continue

        probs = row[BIN_COLS].values.astype(float)
        if np.any(np.isnan(probs)):
            continue

        # Require mass spread across at least 4 bins
        n_substantial = int(np.sum(probs > 0.05))
        if n_substantial < 4:
            continue

        # Find peaks: local maxima (or edges) with >= 10% probability
        peaks = []
        for i in range(10):
            p = probs[i]
            if p < 0.10:
                continue
            left = probs[i - 1] if i > 0 else 0
            right = probs[i + 1] if i < 9 else 0
            if p >= left and p >= right:
                peaks.append((i + 1, p))  # 1-indexed

        if len(peaks) < 2:
            continue

        peaks.sort(key=lambda x: x[1], reverse=True)
        p1_bin, p1_prob = peaks[0]
        p2_bin, p2_prob = peaks[1]
        low, high = sorted([p1_bin, p2_bin])
        separation = high - low

        if separation < 3:
            continue

        # WA must fall between the two peaks
        if not (low < wa < high):
            continue

        # Valley between peaks should be clearly lower than both peaks
        valley_probs = probs[low:high - 1]  # bins between peaks (0-indexed)
        if len(valley_probs) == 0:
            continue
        valley_max = valley_probs.max()
        peak_min = min(p1_prob, p2_prob)
        if valley_max >= peak_min * 0.7:
            continue

        gene = row.get('gene', 'unknown')
        sp_aa = row.get('SP_aa', 'unknown')

        examples.append({
            'idx': idx,
            'gene': str(gene),
            'sp_aa': str(sp_aa),
            'wa': float(wa),
            'wa_prob': float(probs[max(0, min(9, int(round(wa)) - 1))]),
            'peak1_bin': int(p1_bin),
            'peak1_prob': float(p1_prob),
            'peak2_bin': int(p2_bin),
            'peak2_prob': float(p2_prob),
            'separation': separation,
            'n_substantial': n_substantial,
            'probs': probs.tolist(),
        })

    # Prefer spread distributions, then larger separation
    examples.sort(key=lambda x: (x['n_substantial'], x['separation']),
                  reverse=True)
    return examples[:n_examples]


def make_figure(examples, save_path):
    """Generate the bimodal distribution figure (single panel, Tufte style)."""
    ex = examples[0]
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    bins = np.arange(1, 11)
    probs = np.array(ex['probs'])
    wa = ex['wa']

    # Uniform gray bars, no edges
    ax.bar(bins, probs, color='0.55', edgecolor='none')

    # WA vertical dashed line
    ax.axvline(wa, color='firebrick', linestyle='--', linewidth=1.5, zorder=5)

    # Gene label (upper left)
    ax.text(0.05, 0.92, ex['gene'], transform=ax.transAxes,
            fontsize=10, fontstyle='italic', va='top')

    # WA label next to dashed line, lifted above the tallest bar
    ax.text(wa + 0.15, 0.275, f'WA = {wa:.2f}', fontsize=9, color='firebrick')

    ax.set_xlabel('Bin Number', fontsize=11)
    ax.set_ylabel('Fraction of Reads', fontsize=11)
    ax.set_xticks(bins)
    ax.set_ylim(0, 0.30)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.plotstyle import tufte_ax
    tufte_ax(ax)  # remove top/right box spines for consistency with all other figures

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to {save_path}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data from xlsx
    xlsx_path = DATA_DIR / 'sb2c00328_si_011.xlsx'
    print(f"Loading data from {xlsx_path}...")
    df = pd.read_excel(xlsx_path, sheet_name='Library_w_Bins_and_WA')
    print(f"  Loaded {len(df)} rows")

    # Find bimodal examples
    print("\nSearching for bimodal examples...")
    examples = find_bimodal_examples(df, n_examples=1)
    print(f"  Found {len(examples)} bimodal examples:")

    for i, ex in enumerate(examples):
        print(f"\n  Example {i+1}:")
        print(f"    Gene:       {ex['gene']}")
        print(f"    Sequence:   {ex['sp_aa'][:40]}{'...' if len(ex['sp_aa']) > 40 else ''}")
        print(f"    WA:         {ex['wa']:.2f}")
        print(f"    Peak 1:     Bin {ex['peak1_bin']} ({ex['peak1_prob']:.1%})")
        print(f"    Peak 2:     Bin {ex['peak2_bin']} ({ex['peak2_prob']:.1%})")
        print(f"    Separation: {ex['separation']} bins")
        print(f"    Row index:  {ex['idx']}")

    # Generate figure
    fig_path = FIGURES_DIR / 'bimodal_bin_distributions.png'
    make_figure(examples, fig_path)

    # Save metadata for reference
    meta = {
        'description': 'Bimodal bin distribution examples where WA falls between peaks',
        'selection_criterion': 'Largest peak separation, then highest combined peak probability',
        'min_peak_probability': 0.15,
        'min_separation': 2,
        'examples': [{k: v for k, v in ex.items() if k != 'probs'} for ex in examples],
    }
    meta_path = RESULTS_DIR / 'bimodal_examples.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata saved to {meta_path}")


if __name__ == '__main__':
    main()
