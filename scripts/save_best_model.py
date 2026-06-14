#!/usr/bin/env python3
"""
Save Best Model Weights

Trains the best 5-seed vector NN ensemble (Script 10 config) and persists:
  - models/vector_nn_seed_{42,123,456,789,1024}.keras
  - models/scaler.joblib
  - models/config.json

This is infrastructure; no figures generated. After saving, validates
round-trip by reloading and checking predictions match.
"""
import sys
import json
import time
import numpy as np
from pathlib import Path

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_loading import load_plm_with_bins, load_plm_embeddings
from src.models import FocalLoss, load_ensemble, predict_ensemble_wa

# ── Paths ─────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE / 'models'

# ── Best config from Script 10 ───────────────────────────────────────────
BEST_CONFIG = {
    'hidden_layers': (256, 256, 128),
    'dropout': 0.35,
    'lr': 5e-4,
    'loss': 'focal',
    'epochs': 300,
    'batch_size': 32,
}
SEEDS = [42, 123, 456, 789, 1024]
BIN_CENTERS = np.arange(1, 11)


def build_model(input_dim, cfg, seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    inputs = keras.Input(shape=(input_dim,))
    x = inputs
    for units in cfg['hidden_layers']:
        x = layers.Dense(units)(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(cfg['dropout'])(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    loss_fn = FocalLoss(alpha=0.25, gamma=2.0) if cfg['loss'] == 'focal' else 'categorical_crossentropy'
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg['lr']), loss=loss_fn)
    return model


def train_full(X_train, y_bins, cfg, seed=42):
    model = build_model(X_train.shape[1], cfg, seed=seed)
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor='loss', factor=0.5, patience=10, min_lr=1e-6, verbose=0),
    ]
    model.fit(X_train, y_bins, epochs=cfg['epochs'],
              batch_size=cfg['batch_size'], callbacks=callbacks, verbose=0)
    return model


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    # ── Load data ─────────────────────────────────────────────────────────
    print("Loading Ginkgo-AA0 data...")
    X_train, _, _, _, y_train_bins, _, meta = load_plm_with_bins('ginkgo-AA0-650M')
    _, X_test, _, y_test, _ = load_plm_embeddings('ginkgo-AA0-650M')

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    print(f"  Train: {meta['n_train']}, Test: {len(y_test)}, Dim: {X_tr.shape[1]}")

    # ── Train & save each seed ────────────────────────────────────────────
    print(f"\nTraining 5-seed ensemble (config: {BEST_CONFIG['hidden_layers']}, "
          f"drop={BEST_CONFIG['dropout']}, focal loss, {BEST_CONFIG['epochs']} epochs)...")

    models = []
    for seed in SEEDS:
        t0 = time.time()
        model = train_full(X_tr, y_train_bins, BEST_CONFIG, seed=seed)
        save_path = MODELS_DIR / f'vector_nn_seed_{seed}.keras'
        model.save(save_path)

        pred = model.predict(X_te, verbose=0) @ BIN_CENTERS
        mse_i = float(np.mean((y_test - pred) ** 2))
        print(f"  Seed {seed:>5}: MSE = {mse_i:.4f}  saved → {save_path.name}  ({time.time()-t0:.1f}s)")
        models.append(model)

    # Ensemble prediction
    preds = [m.predict(X_te, verbose=0) @ BIN_CENTERS for m in models]
    y_ens = np.mean(preds, axis=0)
    mse_ens = float(np.mean((y_test - y_ens) ** 2))
    print(f"\n  Ensemble MSE = {mse_ens:.4f}")

    # ── Save scaler ───────────────────────────────────────────────────────
    joblib.dump(scaler, MODELS_DIR / 'scaler.joblib')
    print(f"  Saved scaler.joblib")

    # ── Save config ───────────────────────────────────────────────────────
    config = {
        'hidden_layers': list(BEST_CONFIG['hidden_layers']),
        'dropout': BEST_CONFIG['dropout'],
        'learning_rate': BEST_CONFIG['lr'],
        'loss': BEST_CONFIG['loss'],
        'epochs': BEST_CONFIG['epochs'],
        'batch_size': BEST_CONFIG['batch_size'],
        'seeds': SEEDS,
        'embedding_model': 'ginkgo-AA0-650M',
        'embedding_dim': int(X_tr.shape[1]),
        'n_train': int(meta['n_train']),
        'n_test': int(len(y_test)),
        'ensemble_mse': mse_ens,
    }
    with open(MODELS_DIR / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  Saved config.json")

    # ── Round-trip validation ─────────────────────────────────────────────
    print(f"\nValidating round-trip load...")
    loaded_models, loaded_scaler, loaded_config = load_ensemble(MODELS_DIR)
    X_te_reloaded = loaded_scaler.transform(X_test)
    preds_reloaded = [m.predict(X_te_reloaded, verbose=0) @ BIN_CENTERS for m in loaded_models]
    y_ens_reloaded = np.mean(preds_reloaded, axis=0)
    mse_reloaded = float(np.mean((y_test - y_ens_reloaded) ** 2))

    max_diff = float(np.max(np.abs(y_ens - y_ens_reloaded)))
    print(f"  Original MSE:  {mse_ens:.6f}")
    print(f"  Reloaded MSE:  {mse_reloaded:.6f}")
    print(f"  Max pred diff: {max_diff:.2e}")

    if max_diff < 1e-5:
        print("  Round-trip validation PASSED")
    else:
        print(f"  WARNING: predictions differ by {max_diff:.2e}")

    # Also test predict_ensemble_wa convenience function
    y_convenience = predict_ensemble_wa(loaded_models, X_test, loaded_scaler)
    conv_diff = float(np.max(np.abs(y_ens_reloaded - y_convenience)))
    print(f"  predict_ensemble_wa diff: {conv_diff:.2e}")

    elapsed = time.time() - t_total
    print(f"\nDone in {elapsed/60:.1f} min. Models saved to {MODELS_DIR}")


if __name__ == '__main__':
    main()
