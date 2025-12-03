"""
auto_encoder.py

Enhanced Autoencoder anomaly detection module with:
- Global BEST_AE_PARAMS
- Fully parameterized model + training
- Hyperparameter tuning function (K-Fold CV)
- Baseline + tuned scoring outputs
- Backward compatibility with your pipeline

This mirrors the structure used in pca.py and isolation_forest.py.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras import layers, models, callbacks
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


# ============================================================
# GLOBAL BEST PARAMETERS (updated by tuning)
# ============================================================

BEST_AE_PARAMS = {
    "latent_dim": 3,
    "hidden_units": 32,
    "activation": "relu",       # "relu" | "leaky_relu"
    "dropout": 0.0,
    "epochs": 20,
    "batch_size": 32,
    "scaler": "standard",       # "standard" | "minmax" | "robust"
    "learning_rate": 1e-3,
    "seed": 42,
}


# ============================================================
# Utility: seeds + scalers + activations
# ============================================================

def set_seeds(seed: int = 42):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_scaler(name: str):
    if name == "standard":
        return StandardScaler()
    if name == "minmax":
        return MinMaxScaler()
    if name == "robust":
        return RobustScaler()
    raise ValueError(f"Unknown scaler: {name}")


def get_activation_layer(activation: str):
    if activation == "relu":
        return activation  # Keras accepts string
    if activation == "leaky_relu":
        return layers.LeakyReLU(alpha=0.01)
    raise ValueError(f"Unknown activation: {activation}")


# ============================================================
# Data handling
# ============================================================

def load_data(path: str = "../../preprocess_and_filter/preprocessed_refinitiv.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def select_esg_features(df: pd.DataFrame):
    feature_cols = [
        "environment", "social", "governance", "human_rights", "community",
        "workforce", "product_responsibility", "shareholders", "management",
        "controversy", "esg_combined",
    ]
    df_clean = df.dropna(subset=feature_cols)
    X = df_clean[feature_cols].copy()
    tickers = df_clean["ticker"].values
    return X, tickers, feature_cols


def scale_features(X: pd.DataFrame, scaler_name: str = None):
    scaler_name = scaler_name or BEST_AE_PARAMS["scaler"]
    scaler = get_scaler(scaler_name)
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


# ============================================================
# Autoencoder model
# ============================================================

def build_autoencoder(
    input_dim: int,
    latent_dim: int,
    hidden_units: int,
    activation: str = "relu",
    dropout: float = 0.0,
    learning_rate: float = 1e-3,
    seed: int = 42,
):
    set_seeds(seed)

    act = get_activation_layer(activation)

    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(hidden_units),
        act if isinstance(act, layers.Layer) else layers.Activation(act),
        layers.Dropout(dropout),
        layers.Dense(latent_dim),
        act if isinstance(act, layers.Layer) else layers.Activation(act),
        layers.Dense(hidden_units),
        act if isinstance(act, layers.Layer) else layers.Activation(act),
        layers.Dropout(dropout),
        layers.Dense(input_dim, activation="linear"),
    ])

    optimizer = keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse")
    return model


# ============================================================
# Core scoring
# ============================================================

def compute_autoencoder_scores(
    X_scaled: np.ndarray,
    latent_dim: int | None = None,
    hidden_units: int | None = None,
    activation: str | None = None,
    dropout: float | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    seed: int | None = None,
):
    """Train AE on X_scaled and return per-sample reconstruction errors and the model."""
    params = BEST_AE_PARAMS.copy()
    if latent_dim is not None: params["latent_dim"] = latent_dim
    if hidden_units is not None: params["hidden_units"] = hidden_units
    if activation is not None: params["activation"] = activation
    if dropout is not None: params["dropout"] = dropout
    if epochs is not None: params["epochs"] = epochs
    if batch_size is not None: params["batch_size"] = batch_size
    if learning_rate is not None: params["learning_rate"] = learning_rate
    if seed is not None: params["seed"] = seed

    set_seeds(params["seed"])

    input_dim = X_scaled.shape[1]
    ae = build_autoencoder(
        input_dim=input_dim,
        latent_dim=params["latent_dim"],
        hidden_units=params["hidden_units"],
        activation=params["activation"],
        dropout=params["dropout"],
        learning_rate=params["learning_rate"],
        seed=params["seed"],
    )

    es = callbacks.EarlyStopping(monitor="loss", patience=8, restore_best_weights=True)
    ae.fit(X_scaled, X_scaled, epochs=params["epochs"], batch_size=params["batch_size"], verbose=0, callbacks=[es])

    reconstructed = ae.predict(X_scaled, verbose=0)
    errors = np.mean((X_scaled - reconstructed) ** 2, axis=1)
    return errors, ae


# ============================================================
# External API (defaults from BEST_AE_PARAMS, overridable)
# ============================================================

def run_autoencoder(csv_path: str = "../../preprocess_and_filter/preprocessed_refinitiv.csv", **override_params):
    df = load_data(csv_path)
    X, tickers, _ = select_esg_features(df)
    scaler_name = override_params.get("scaler", BEST_AE_PARAMS["scaler"])
    X_scaled, _ = scale_features(X, scaler_name)

    errors, ae_model = compute_autoencoder_scores(X_scaled, **override_params)
    return {tickers[i]: float(errors[i]) for i in range(len(tickers))}


# ============================================================
# Hyperparameter tuning (K-Fold CV on validation error)
# ============================================================

def tune_autoencoder(
    X,
    latent_dims=None,
    hidden_units_list=None,
    dropout_values=None,
    activation_list=None,
    epochs_list=None,
    batch_sizes=None,
    scaler_options=None,
    learning_rates=None,
    n_splits=5,
    update_global=True,
    seed=None,
    fast_mode=True,
):
    """Grid-search hyperparameter tuning minimizing validation reconstruction error."""

    if fast_mode:
        latent_dims = [3]
        hidden_units_list = [32]
        dropout_values = [0.0]
        activation_list = ["relu"]
        epochs_list = [20]
        batch_sizes = [32]
        scaler_options = ["standard"]
        learning_rates = [1e-3]
    else:
        latent_dims = latent_dims or [3, 5]
        hidden_units_list = hidden_units_list or [32, 64]
        dropout_values = dropout_values or [0.0, 0.1]
        activation_list = activation_list or ["relu", "leaky_relu"]
        epochs_list = epochs_list or [20, 40]
        batch_sizes = batch_sizes or [16, 32]
        scaler_options = scaler_options or ["standard", "minmax"]
        learning_rates = learning_rates or [1e-3, 5e-4]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []

    import itertools
    combos = list(itertools.product(
        latent_dims, hidden_units_list, dropout_values, activation_list,
        epochs_list, batch_sizes, scaler_options, learning_rates
    ))

    for (ld, hu, dr, act, ep, bs, scaler_name, lr) in combos:
        X_scaled, _ = scale_features(X, scaler_name)
        fold_losses = []

        for tr_idx, va_idx in kf.split(X_scaled):
            X_tr, X_va = X_scaled[tr_idx], X_scaled[va_idx]

            model = build_autoencoder(
                input_dim=X_tr.shape[1],
                latent_dim=ld,
                hidden_units=hu,
                activation=act,
                dropout=dr,
                learning_rate=lr,
                seed=BEST_AE_PARAMS["seed"] if seed is None else seed,
            )

            es = callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
            model.fit(X_tr, X_tr, epochs=ep, batch_size=bs, verbose=0, validation_data=(X_va, X_va), callbacks=[es])

            recon = model.predict(X_va, verbose=0)
            fold_losses.append(float(np.mean((X_va - recon) ** 2)))

        results.append({
            "latent_dim": ld,
            "hidden_units": hu,
            "dropout": dr,
            "activation": act,
            "epochs": ep,
            "batch_size": bs,
            "scaler": scaler_name,
            "learning_rate": lr,
            "val_error_mean": float(np.mean(fold_losses)),
        })

    results_df = pd.DataFrame(results)
    best_row = results_df.loc[results_df["val_error_mean"].idxmin()]

    best_params = {
        "latent_dim": int(best_row["latent_dim"]),
        "hidden_units": int(best_row["hidden_units"]),
        "activation": str(best_row["activation"]),
        "dropout": float(best_row["dropout"]),
        "epochs": int(best_row["epochs"]),
        "batch_size": int(best_row["batch_size"]),
        "scaler": str(best_row["scaler"]),
        "learning_rate": float(best_row["learning_rate"]),
    }

    if update_global:
        BEST_AE_PARAMS.update(best_params)

    return results_df, best_params


# ============================================================
# Visualization
# ============================================================

def save_autoencoder_plots(errors: np.ndarray, output_dir: str = "./outputs/"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=40, color="purple", alpha=0.8)
    plt.title("Autoencoder Reconstruction Error Distribution")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "autoencoder_error_distribution.png"))
    plt.close()


# ============================================================
# Direct execution
# ============================================================

if __name__ == "__main__":
    print("Running Autoencoder anomaly detection + generating plot…")

    df = load_data()
    X, tickers, _ = select_esg_features(df)
    X_scaled, _ = scale_features(X, BEST_AE_PARAMS["scaler"])

    errors, model = compute_autoencoder_scores(X_scaled)

    save_autoencoder_plots(errors, output_dir="./outputs/")
    print("✔ Plots saved to ./outputs/")

    print("Sample AE anomaly scores (first 10):")
    for t, s in list({tickers[i]: float(errors[i]) for i in range(len(tickers))}.items())[:10]:
        print(f"{t}: {s:.6f}")
