#!/usr/bin/env python3
"""Script 21: Reproducible open-weight headline; open-PLM ensemble.

The best single embedding (Ginkgo-AA-0) is API-only, proprietary, and the API is
discontinued, so it cannot anchor a reproducible result. This script establishes
the reproducible headline using OPEN-weight PLMs (regenerable by anyone):

  1. (prerequisite) generate ProtT5 and ProtBERT embeddings with
     scripts/gen_open_embeddings.py  (open weights, HuggingFace; no API key).
  2. train the vector NN (256,256,128 / PReLU / dropout 0.3 / focal, 5-seed) on
     each embedding, evaluate solo and as ensembles, mean +/- std over R retrains.

Result (results/eval_prott5.json): the ProtT5+ESM2-650M+ProtBERT ensemble reaches
~0.98, recovering the discontinued AA-0 (~0.96) to within ~0.02 and improving ~20%
over the Grasso RF baseline (1.22); fully reproducibly.

Run with --recompute to retrain (needs TF GPU + the generated parquets); by default
it just (re)builds the figure from the cached results JSON.
"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from src.plotstyle import apply_tufte, tufte_ax, GRAY
RESULTS = BASE / 'results' / 'eval_prott5.json'
FIG = BASE / 'figures' / 'open_embedding_ensemble.png'
GRASSO = 1.22

LABELS = {
    'prott5+esm2+protbert': 'ProtT5 + ESM2 + ProtBERT\n(open ensemble)',
    'prott5+protbert': 'ProtT5 + ProtBERT',
    'prott5+esm2': 'ProtT5 + ESM2',
    'prott5': 'ProtT5',
    'esm2-650M': 'ESM2-650M',
    'protbert': 'ProtBERT',
    'ginkgo-AA0-650M': 'Ginkgo AA-0  (discontinued)',
}

def make_figure(d):
    apply_tufte()
    order = ['ginkgo-AA0-650M', 'esm2-650M', 'protbert', 'prott5',
             'prott5+esm2', 'prott5+protbert', 'prott5+esm2+protbert']
    order = [k for k in order if k in d]
    means = [d[k]['mean'] for k in order]
    errs = [d[k]['std'] for k in order]
    # open = gray; discontinued AA-0 = lighter/outlined to mark "not reproducible"
    colors = [GRAY if d[k]['open'] else '#c0a0a0' for k in order]
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    y = np.arange(len(order))
    ax.barh(y, means, xerr=errs, color=colors, height=0.7,
            error_kw=dict(ecolor=GRAY, lw=0.8, capsize=2))
    ax.axvline(GRASSO, color='#999999', lw=0.8, ls=':')
    ax.text(GRASSO, len(order) - 0.3, ' Grasso 1.22', color='#777777', fontsize=7, va='top')
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS.get(k, k) for k in order], fontsize=7)
    ax.set_xlabel('Test MSE (mean ± std over retrains)')
    ax.set_xlim(0, max(means) * 1.18)
    tufte_ax(ax)
    fig.savefig(FIG)
    plt.close(fig)
    print(f"saved {FIG}")

def main():
    if '--recompute' in sys.argv:
        import subprocess
        subprocess.run([sys.executable, str(BASE / 'scripts' / 'eval_open_ensemble.py')], check=True)
    d = json.load(open(RESULTS))
    best = min((k for k in d if d[k]['open']), key=lambda k: d[k]['mean'])
    print(f"Reproducible (open) headline: {best} = {d[best]['mean']:.4f} ± {d[best]['std']:.4f}")
    print(f"  vs Grasso 1.22: {100*(GRASSO-d[best]['mean'])/GRASSO:.1f}% improvement")
    if 'ginkgo-AA0-650M' in d:
        print(f"  vs discontinued AA-0 {d['ginkgo-AA0-650M']['mean']:.4f}: gap "
              f"{d[best]['mean']-d['ginkgo-AA0-650M']['mean']:+.4f}")
    make_figure(d)

if __name__ == '__main__':
    main()
