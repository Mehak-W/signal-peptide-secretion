"""
Model definitions for signal peptide study.

SignalPeptideRegressorNN: Dense regression network.
  - Dense(units, relu) -> BatchNorm -> Dropout per hidden layer
  - Output: Dense(1, linear)
  - Loss: MSE

SignalPeptideVectorNN: Vector regression network for bin probability prediction.
  - Dense(units, LeakyReLU) -> Dropout per hidden layer
  - Output: Dense(10, softmax)
  - Loss: FocalLoss or categorical_crossentropy

ReLUSquared: Custom activation layer — ReLU²(x) = max(0,x)².
  Ref: So et al. (2021) "Primer: Searching for Efficient Transformers."
"""
import json
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


class SignalPeptideRegressorNN:
    """
    Regression neural network for signal peptide efficiency prediction.

    Architecture:
        For each hidden layer: Dense(units, relu) -> BatchNorm -> Dropout
        Output: Dense(1, linear)

    Loss: MSE (mean squared error)
    """

    def __init__(self, hidden_layers=(128, 64), dropout=0.3, l2_reg=1e-3,
                 learning_rate=1e-3, batch_size=32, epochs=200, random_state=42):
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state
        self.model_ = None
        self.history_ = None

    def _build_model(self, input_dim):
        tf.random.set_seed(self.random_state)

        inputs = keras.Input(shape=(input_dim,))
        x = inputs

        for units in self.hidden_layers:
            x = layers.Dense(
                units,
                activation='relu',
                kernel_regularizer=regularizers.l2(self.l2_reg),
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.dropout)(x)

        # Output: single neuron, linear activation for regression
        outputs = layers.Dense(1, activation='linear')(x)

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae'],
        )
        return model

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=0):
        """
        Train the model.

        Args:
            X_train, y_train: training data
            X_val, y_val: optional validation data (used for early stopping)
            verbose: Keras verbosity
        """
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)

        self.model_ = self._build_model(X_train.shape[1])

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=15,
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=7,
                min_lr=1e-6,
            ),
        ]

        fit_kwargs = dict(
            x=X_train,
            y=y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=verbose,
        )

        if X_val is not None and y_val is not None:
            fit_kwargs['validation_data'] = (X_val, y_val)

        self.history_ = self.model_.fit(**fit_kwargs)
        return self

    def predict(self, X):
        """Predict continuous WA values."""
        return self.model_.predict(X, verbose=0).ravel()

    def get_params(self):
        """Return model hyperparameters as a dict."""
        return {
            'hidden_layers': list(self.hidden_layers),
            'dropout': self.dropout,
            'l2_reg': self.l2_reg,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
        }


class FocalLoss(keras.losses.Loss):
    """
    Focal loss (Lin et al. 2017) for bin probability prediction.

    Down-weights well-classified examples to focus on hard cases.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha=0.25, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        weight = self.alpha * tf.pow(1.0 - y_pred, self.gamma)
        focal = weight * ce
        return tf.reduce_sum(focal, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({'alpha': self.alpha, 'gamma': self.gamma})
        return config


class SignalPeptideVectorNN:
    """
    Vector regression network for 10-dim bin probability prediction.

    Architecture (matching Schrier's Wolfram network):
        For each hidden layer: Dense(units) -> LeakyReLU -> Dropout
        Output: Dense(10, softmax)

    No BatchNorm. LeakyReLU instead of ReLU.
    """

    def __init__(self, hidden_layers=(256, 256), dropout=0.2,
                 learning_rate=5e-4, batch_size=32, epochs=200,
                 loss='focal', random_state=42):
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss = loss
        self.random_state = random_state
        self.model_ = None
        self.history_ = None

    def _build_model(self, input_dim):
        tf.random.set_seed(self.random_state)

        inputs = keras.Input(shape=(input_dim,))
        x = inputs

        for units in self.hidden_layers:
            x = layers.Dense(units)(x)
            x = layers.LeakyReLU()(x)
            x = layers.Dropout(self.dropout)(x)

        outputs = layers.Dense(10, activation='softmax')(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        if self.loss == 'focal':
            loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        else:
            loss_fn = 'categorical_crossentropy'

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=loss_fn,
        )
        return model

    def fit(self, X_train, y_train_bins, X_val=None, y_val_bins=None, verbose=0):
        """
        Train the model.

        Args:
            X_train: input features
            y_train_bins: 10-dim bin probability targets (N, 10)
            X_val, y_val_bins: optional validation data
            verbose: Keras verbosity
        """
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)

        self.model_ = self._build_model(X_train.shape[1])

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=15,
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=7,
                min_lr=1e-6,
            ),
        ]

        fit_kwargs = dict(
            x=X_train,
            y=y_train_bins,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=verbose,
        )

        if X_val is not None and y_val_bins is not None:
            fit_kwargs['validation_data'] = (X_val, y_val_bins)

        self.history_ = self.model_.fit(**fit_kwargs)
        return self

    def predict(self, X):
        """Predict 10-dim bin probability distributions."""
        return self.model_.predict(X, verbose=0)

    def get_params(self):
        """Return model hyperparameters as a dict."""
        return {
            'hidden_layers': list(self.hidden_layers),
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'loss': self.loss,
        }


# ---------------------------------------------------------------------------
# ReLU² activation layer
# ---------------------------------------------------------------------------

@keras.utils.register_keras_serializable(package='signal_peptide')
class ReLUSquared(layers.Layer):
    """ReLU²(x) = max(0, x)². Produces sparser activations than LeakyReLU."""

    def call(self, x):
        return tf.square(tf.nn.relu(x))

    def get_config(self):
        return super().get_config()


# ---------------------------------------------------------------------------
# Ensemble persistence
# ---------------------------------------------------------------------------
BIN_CENTERS = np.arange(1, 11)


def load_ensemble(models_dir):
    """
    Load a saved 5-seed ensemble from disk.

    Args:
        models_dir: path to directory containing .keras files, scaler.joblib, config.json

    Returns:
        (models_list, scaler, config_dict)
    """
    from pathlib import Path
    import joblib

    models_dir = Path(models_dir)
    with open(models_dir / 'config.json') as f:
        config = json.load(f)

    scaler = joblib.load(models_dir / 'scaler.joblib')

    custom_objects = {
        'FocalLoss': FocalLoss,
        'ReLUSquared': ReLUSquared,
    }
    models = []
    for seed in config['seeds']:
        path = models_dir / f'vector_nn_seed_{seed}.keras'
        model = keras.models.load_model(path, custom_objects=custom_objects)
        models.append(model)

    return models, scaler, config


def predict_ensemble_wa(models, X, scaler=None):
    """
    Predict weighted-average WA from an ensemble of vector NN models.

    Args:
        models: list of Keras models (softmax(10) output)
        X: raw feature matrix (unscaled) or pre-scaled if scaler is None
        scaler: StandardScaler to apply, or None if X is pre-scaled

    Returns:
        1-D array of predicted WA values
    """
    if scaler is not None:
        X = scaler.transform(X)

    preds = [m.predict(X, verbose=0) @ BIN_CENTERS for m in models]
    return np.mean(preds, axis=0)
