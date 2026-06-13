#!/usr/bin/env python3
"""Design-task evaluation vs Grasso et al. Figure 4e.

Grasso Fig 4e: 11 of 15 hand-picked, independently (NLR-) validated designs fell
within +/-1 WA of prediction (73%). We evaluate our models on the FULL design
library (predicted vs library-measured WA) and report the comparable +/-1-WA
fraction, plus a parity plot (RF = open marker, NN = filled; PhysChem = circle,
ESM2 = square), with the y=x line and Grasso's 73% reference.

Note: not a strict apples-to-apples comparison (his 15 were curated and
independently re-assayed; ours is the full, unselected, noisier library) — the
honest, lower-bias evaluation.
"""
import os, sys, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from src.plotstyle import apply_tufte, tufte_ax, GRAY
from src.data_loading import GRASSO_FEATURES, load_physicochemical, load_plm_embeddings, load_design_embeddings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
RES = BASE / 'results' / 'design_parity_results.json'
FIG = BASE / 'figures' / 'design_parity.png'
GRASSO_HIT = 0.733  # 11/15

def nn_predict(Xtr, ytr, Xte, seed=42):
    tf.random.set_seed(seed); np.random.seed(seed)
    inp = keras.Input(shape=(Xtr.shape[1],)); x = inp
    for u in (256, 128):
        x = layers.Dense(u)(x); x = layers.PReLU()(x); x = layers.Dropout(0.3)(x)
    out = layers.Dense(1)(x); m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')
    m.fit(Xtr, ytr, epochs=150, batch_size=32, verbose=0,
          callbacks=[keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=8, min_lr=1e-6)])
    p = m.predict(Xte, verbose=0).ravel(); keras.backend.clear_session(); return p

def hit1(a, p): return float(np.mean(np.abs(a - p) <= 1.0))

# ---- PhysChem (train on Grasso, predict design) ----
xl = pd.read_excel(BASE / 'data' / 'sb2c00328_si_011.xlsx', sheet_name='Library_w_Bins_and_WA')
feats = [f for f in GRASSO_FEATURES if f in xl.columns]
tr = xl[xl.Set == 'Train']; des = xl[xl.Set.isna() & xl.WA.notna()]
Xtr_pc = tr[feats].fillna(0).values.astype(float); ytr = tr.WA.values.astype(float)
Xde_pc = des[feats].fillna(0).values.astype(float); yde = des.WA.values.astype(float)
sc = StandardScaler().fit(Xtr_pc)
rf = RandomForestRegressor(n_estimators=300, max_depth=25, min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=4)
rf.fit(sc.transform(Xtr_pc), ytr)
preds = {('PhysChem', 'RF'): rf.predict(sc.transform(Xde_pc))}
preds[('PhysChem', 'NN')] = nn_predict(sc.transform(Xtr_pc), ytr, sc.transform(Xde_pc))

# ---- ESM2-650M (train on Grasso, predict design) ----
gXtr, _, gytr, _, _ = load_plm_embeddings('esm2-650M')
demb = load_design_embeddings('esm2-650M')
Xde_e = np.stack(demb['embedding'].values); yde_e = demb['WA'].values.astype(float)
m = ~np.isnan(yde_e); Xde_e, yde_e = Xde_e[m], yde_e[m]
sce = StandardScaler().fit(gXtr)
rfe = RandomForestRegressor(n_estimators=300, max_depth=25, min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=4)
rfe.fit(sce.transform(gXtr), gytr)
preds[('ESM2', 'RF')] = rfe.predict(sce.transform(Xde_e))
preds[('ESM2', 'NN')] = nn_predict(sce.transform(gXtr), gytr, sce.transform(Xde_e))

# ---- metrics ----
actuals = {'PhysChem': yde, 'ESM2': yde_e}
out = {'grasso_fig4e_within1WA': GRASSO_HIT}
for (feat, mod), p in preds.items():
    a = actuals[feat]
    out[f'{feat}_{mod}'] = {'within1WA': hit1(a, p), 'spearman': float(spearmanr(a, p).statistic),
                            'mae': float(np.mean(np.abs(a - p))), 'n': int(len(a))}
    print(f"  {feat:8s} {mod}: within±1WA={100*hit1(a,p):.1f}%  Spearman={spearmanr(a,p).statistic:.3f}  (Grasso Fig4e: 73%)")
json.dump(out, open(RES, 'w'), indent=2)

# ---- parity figure (subsample for legibility) ----
apply_tufte()
fig, ax = plt.subplots(figsize=(4.6, 3.8))
styles = {('PhysChem', 'RF'): ('o', 'none'), ('PhysChem', 'NN'): ('o', GRAY),
          ('ESM2', 'RF'): ('s', 'none'), ('ESM2', 'NN'): ('s', GRAY)}
rng = np.random.default_rng(0)
for (feat, mod), p in preds.items():
    a = actuals[feat]; idx = rng.choice(len(a), size=min(400, len(a)), replace=False)
    mk, fc = styles[(feat, mod)]
    ax.scatter(a[idx], p[idx], marker=mk, s=10, facecolors=fc, edgecolors=GRAY,
               linewidths=0.5, alpha=0.5, label=f'{feat} {mod}')
lim = [1, 10]; xs = np.array(lim)
ax.plot(xs, xs, color='#777777', lw=0.9)                  # y = x
ax.plot(xs, xs + 1, color='#bbbbbb', lw=0.7, ls='--')     # +/-1 WA band
ax.plot(xs, xs - 1, color='#bbbbbb', lw=0.7, ls='--')
mean_hit = 100 * np.mean([hit1(actuals[f], p) for (f, _), p in preds.items()])
ax.text(1.3, 9.5, f'within ±1 WA: ~{mean_hit:.0f}%\n(Grasso Fig 4e: 73%)', fontsize=6, color=GRAY, va='top')
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
ax.set_xlabel('Measured WA'); ax.set_ylabel('Predicted WA')
# Legend outside the plotting area so it never sits on top of the points.
ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.02, 1.0),
          markerscale=1.6, frameon=False, borderaxespad=0)
tufte_ax(ax); fig.savefig(FIG, bbox_inches='tight'); plt.close(fig)
print(f"saved {FIG}")
print("DESIGN_PARITY_DONE")
