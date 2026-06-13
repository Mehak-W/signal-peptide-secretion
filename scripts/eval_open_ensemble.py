#!/usr/bin/env python3
"""Decisive reproducible-headline test: vector NN on ProtT5 (open) vs AA-0 vs
ESM2, and open-model ensembles. Base (256,256,128)/PReLU/dropout0.3/focal,
5-seed, mean over R retrains."""
import os, sys, json, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np, pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import FocalLoss
BIN = [f'Perc_unambiguousReads_BIN{i:02d}_bin' for i in range(1, 11)]; C = np.arange(1, 11)
SEEDS = [42, 123, 456, 789, 1024]; R = 2
DATA = Path(__file__).resolve().parent.parent / 'data'
OUT = Path(__file__).resolve().parent.parent / 'results' / 'eval_prott5.json'
def log(m): print(m, flush=True)
def load(nm):
    tr = pd.read_parquet(DATA/f'trainAA_{nm}.parquet'); te = pd.read_parquet(DATA/f'testAA_{nm}.parquet')
    m = tr[BIN].notna().all(axis=1).values
    return np.stack(tr['embedding'].values)[m], tr[BIN].values[m].astype('float64'), np.stack(te['embedding'].values), te['WA'].values.astype('float64')
def train_pred(Xtr, bins, Xte, seed):
    tf.random.set_seed(seed); np.random.seed(seed)
    inp = keras.Input(shape=(Xtr.shape[1],)); x = inp
    for u in (256, 256, 128):
        x = layers.Dense(u)(x); x = layers.PReLU()(x); x = layers.Dropout(0.3)(x)
    out = layers.Dense(10, activation='softmax')(x); m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(5e-4), loss=FocalLoss(0.25, 2.0))
    m.fit(Xtr, bins, epochs=300, batch_size=32,
          callbacks=[keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, min_lr=1e-6)], verbose=0)
    p = m.predict(Xte, verbose=0) @ C; keras.backend.clear_session(); return p
def mse(y, p): return float(np.mean((y - p) ** 2))

EMB = ['prott5', 'esm2-650M', 'protbert', 'ginkgo-AA0-650M']
data = {nm: load(nm) for nm in EMB}
yte = data['prott5'][3]
log(f"test {len(yte)}  dims: " + ", ".join(f"{k}={v[0].shape[1]}" for k, v in data.items()))
pool = {nm: [] for nm in EMB}     # pool[nm][r] = list of 5 seed preds
for r in range(R):
    for nm in EMB:
        Xtr, bins, Xte, _ = data[nm]; sc = StandardScaler().fit(Xtr)
        preds = [train_pred(sc.transform(Xtr), bins, sc.transform(Xte), s) for s in SEEDS]
        pool[nm].append(preds); log(f"  pool{r} {nm}: {mse(yte, np.mean(preds,0)):.4f}")
RES = {}
def ens(names, r): return mse(yte, np.mean([p for nm in names for p in pool[nm][r]], 0))
combos = {nm: [nm] for nm in EMB}
combos['prott5+esm2'] = ['prott5', 'esm2-650M']
combos['prott5+protbert'] = ['prott5', 'protbert']
combos['prott5+esm2+protbert'] = ['prott5', 'esm2-650M', 'protbert']   # all OPEN
combos['prott5+esm2+protbert+ginkgo'] = EMB
for name, names in combos.items():
    vals = [ens(names, r) for r in range(R)]
    RES[name] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)),
                 'open': all(n != 'ginkgo-AA0-650M' for n in names)}
    json.dump(RES, open(OUT, 'w'), indent=2)
log("\n=== NN results (reproducible = open only) ===")
for k, v in sorted(RES.items(), key=lambda kv: kv[1]['mean']):
    log(f"  {k:30s}: {v['mean']:.4f} ± {v['std']:.4f}  {'[OPEN/reproducible]' if v['open'] else '[uses AA-0]'}")
log("EVAL_PROTT5_DONE")
