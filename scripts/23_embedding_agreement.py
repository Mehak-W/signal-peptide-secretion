#!/usr/bin/env python3
"""Script 23: Embedding agreement (linear-probe predictions).

Quantifies how closely a linear probe on Ginkgo AA-0 and a linear probe on
ESM2-650M agree, supporting the claim that the open ESM2 embedding recovers
AA-0's task-relevant signal rather than merely matching its score. For each
embedding I train the same linear probe used in Script 18 (Input -> softmax(10),
focal loss, 5-seed ensemble), predict WA on the shared test set, then report the
Pearson/Spearman correlation between the two prediction vectors and each probe's
rank correlation with the measured WA.

Outputs:
  - results/embedding_agreement.json
"""
import os, sys, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import FocalLoss

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'data'
OUT = BASE / 'results' / 'embedding_agreement.json'
BIN = [f'Perc_unambiguousReads_BIN{i:02d}_bin' for i in range(1, 11)]
C = np.arange(1, 11)
SEEDS = [42, 123, 456, 789, 1024]


def load(nm):
    tr = pd.read_parquet(DATA / f'trainAA_{nm}.parquet')
    te = pd.read_parquet(DATA / f'testAA_{nm}.parquet')
    m = tr[BIN].notna().all(axis=1).values
    Xtr = np.stack(tr['embedding'].values)[m]
    bins = tr[BIN].values[m].astype('float64')
    Xte = np.stack(te['embedding'].values)
    yte = te['WA'].values.astype('float64')
    return Xtr, bins, Xte, yte


def probe_preds(nm):
    """Linear probe (Input -> softmax(10), focal, 5-seed ensemble); test WA preds + truth."""
    Xtr, bins, Xte, yte = load(nm)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    preds = []
    for s in SEEDS:
        tf.random.set_seed(s); np.random.seed(s)
        inp = keras.Input(shape=(Xtr.shape[1],))
        out = layers.Dense(10, activation='softmax')(inp)
        m = keras.Model(inp, out)
        m.compile(optimizer=keras.optimizers.Adam(5e-4), loss=FocalLoss(0.25, 2.0))
        m.fit(Xtr, bins, epochs=300, batch_size=32,
              callbacks=[keras.callbacks.ReduceLROnPlateau(
                  monitor='loss', factor=0.5, patience=10, min_lr=1e-6)],
              verbose=0)
        preds.append(m.predict(Xte, verbose=0) @ C)
        keras.backend.clear_session()
        print(f"  {nm} seed {s} done", flush=True)
    return np.mean(preds, 0), yte


p_aa0, y_aa0 = probe_preds('ginkgo-AA0-650M')
p_esm, y_esm = probe_preds('esm2-650M')
assert len(y_aa0) == len(y_esm) and np.allclose(y_aa0, y_esm), \
    "AA-0 and ESM2 test sets are not row-aligned; correlation would be invalid"
yte = y_aa0

res = {
    'description': ('Linear probe (Input->softmax(10), focal, 5-seed ensemble) WA '
                    'predictions on the shared test set; correlation between the AA-0 '
                    'and ESM2-650M prediction vectors.'),
    'n_test': int(len(yte)),
    'pearson_aa0_vs_esm2': float(pearsonr(p_aa0, p_esm)[0]),
    'spearman_aa0_vs_esm2': float(spearmanr(p_aa0, p_esm).statistic),
    'spearman_aa0_vs_truth': float(spearmanr(p_aa0, yte).statistic),
    'spearman_esm2_vs_truth': float(spearmanr(p_esm, yte).statistic),
}
json.dump(res, open(OUT, 'w'), indent=2)
print(json.dumps(res, indent=2))
print("EMBEDDING_AGREEMENT_DONE")
