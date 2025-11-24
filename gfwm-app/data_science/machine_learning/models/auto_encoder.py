"""
autoencoder.py

Feedforward Autoencoder anomaly detection on ESG features.
Returns: { ticker: ae_anomaly_score } (higher = more anomalous)

When run directly:
- Trains/validates autoencoder
- Computes reconstruction-error anomaly scores
- Saves:
    - Model (autoencoder.keras)
    - Scaler (scaler.pkl)
    - Training curves: ae_loss_curve.png
    - Error distribution: ae_recon_error_distribution.png
    - Top anomalies: ae_top_anomalies.csv
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, callbacks, regularizers


# -------------------------------------------------------
# Config
# -------------------------------------------------------
CSV_PATH = "../../preprocess_and_filter/preprocessed_refinitiv.csv"
OUTPUT_DIR = "./outputs/"
MODEL_PATH = "./outputs/autoencoder.keras"
SCALER_PATH = "./outputs/scaler.pkl"

FEATURE_COLS = [
    "environment",
    "social",
    "governance",
    "human_rights",
    "community",
    "workforce",
    "product_responsibility",
    "shareholders",
    "management",
    "controversy",
    "esg_combined",
]

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)


# -------------------------------------------------------
# Data helpers
# -------------------------------------------------------
def load_data(path: str = CSV_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def select_esg_features(df: pd.DataFrame):
    df_clean = df.dropna(subset=FEATURE_COLS).copy()
    X = df_clean[FEATURE_COLS].astype(float).values
    tickers = df_clean["ticker"].values
    return X, tickers, df_clean


def scale_features(X: np.ndarray, scaler: StandardScaler | None = None):
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return X_scaled, scaler


# -------------------------------------------------------
# Model factory
# -------------------------------------------------------
def build_autoencoder(input_dim: int, latent_dim: int = 4, l2_reg: float = 1e-4):
    """
    Simple symmetric AE: input -> 8 -> latent -> 8 -> output
    Uses L2 regularization + dropout for stability.
    """
    encoder = models.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(8, activation="relu",
                         kernel_regularizer=regularizers.l2(l2_reg)),
            layers.Dropout(0.05),
            layers.Dense(latent_dim, activation="relu",
                         kernel_regularizer=regularizers.l2(l2_reg)),
        ],
        name="encoder",
    )

    decoder = models.Sequential(
        [
            layers.Input(shape=(latent_dim,)),
            layers.Dense(8, activation="relu",
                         kernel_regularizer=regularizers.l2(l2_reg)),
            layers.Dropout(0.05),
            layers.Dense(input_dim, activation="linear"),
        ],
        name="decoder",
    )

    inputs = layers.Input(shape=(input_dim,))
    z = encoder(inputs)
    outputs = decoder(z)
    ae = models.Model(inputs, outputs, name="autoencoder")
    ae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
               loss="mse")
    return ae, encoder, decoder


# -------------------------------------------------------
# Training
# -------------------------------------------------------
def train_autoencoder(
    X_scaled: np.ndarray,
    val_split: float = 0.15,
    epochs: int = 200,
    batch_size: int = 32,
):
    X_train, X_val = train_test_split(
        X_scaled, test_size=val_split, random_state=SEED, shuffle=True
    )

    ae, encoder, decoder = build_autoencoder(input_dim=X_scaled.shape[1])

    es = callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True
    )
    rlrop = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=8, min_lr=1e-5
    )

    history = ae.fit(
        X_train,
        X_train,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=[es, rlrop],
        verbose=0,
    )
    return ae, history


# -------------------------------------------------------
# Scoring
# -------------------------------------------------------
def compute_reconstruction_errors(ae: tf.keras.Model, X_scaled: np.ndarray) -> np.ndarray:
    X_hat = ae.predict(X_scaled, verbose=0)
    # Mean squared error per row = anomaly score
    errors = np.mean((X_scaled - X_hat) ** 2, axis=1)
    return errors


# -------------------------------------------------------
# Public entrypoint (for pipeline)
# -------------------------------------------------------
def run_autoencoder(csv_path: str = CSV_PATH, save_outputs=True) -> dict[str, float]:
    # 1) Load/select/scale
    df = load_data(csv_path)
    X, tickers, _ = select_esg_features(df)
    X_scaled, scaler = scale_features(X)

    # 2) Train
    ae, history = train_autoencoder(X_scaled)

    # 3) Score
    errors = compute_reconstruction_errors(ae, X_scaled)
    anomaly_dict = {tickers[i]: float(errors[i]) for i in range(len(tickers))}

    # Only save if explicitly requested
    if save_outputs:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        ae.save(MODEL_PATH)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(scaler, f)

        save_training_plot(history)
        save_error_distribution(errors)

        pd.DataFrame({
            "ticker": tickers,
            "ae_recon_error": errors
        }).sort_values("ae_recon_error", ascending=False).to_csv(
            os.path.join(OUTPUT_DIR, "ae_top_anomalies.csv"), index=False
        )


    return anomaly_dict


# -------------------------------------------------------
# Visualization helpers
# -------------------------------------------------------
def save_training_plot(history, out_path=os.path.join(OUTPUT_DIR, "ae_loss_curve.png")):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Autoencoder Training Curve (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_error_distribution(errors, out_path=os.path.join(OUTPUT_DIR, "ae_recon_error_distribution.png")):
    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=40, alpha=0.8)
    plt.title("Autoencoder Reconstruction Error Distribution\n(Higher = More Anomalous)")
    plt.xlabel("Reconstruction Error (MSE)")
    plt.ylabel("Frequency")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# -------------------------------------------------------
# Direct execution
# -------------------------------------------------------
if __name__ == "__main__":
    print("Training Autoencoder and generating outputs...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Data
    df = load_data(CSV_PATH)
    X, tickers, df_clean = select_esg_features(df)
    X_scaled, scaler = scale_features(X)

    # Train
    ae, history = train_autoencoder(X_scaled)

    # Score
    errors = compute_reconstruction_errors(ae, X_scaled)
    anomaly_dict = {tickers[i]: float(errors[i]) for i in range(len(tickers))}

    # Persist artifacts
    ae.save(MODEL_PATH)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    # Outputs
    save_training_plot(history)
    save_error_distribution(errors)
    pd.DataFrame({"ticker": tickers, "ae_recon_error": errors}).sort_values(
        "ae_recon_error", ascending=False
    ).to_csv(os.path.join(OUTPUT_DIR, "ae_top_anomalies.csv"), index=False)

    print("✔ Saved:")
    print(f"  - Model:   {MODEL_PATH}")
    print(f"  - Scaler:  {SCALER_PATH}")
    print(f"  - Curves:  {os.path.join(OUTPUT_DIR, 'ae_loss_curve.png')}")
    print(f"  - Errors:  {os.path.join(OUTPUT_DIR, 'ae_recon_error_distribution.png')}")
    print(f"  - CSV:     {os.path.join(OUTPUT_DIR, 'ae_top_anomalies.csv')}")

    # Preview
    print("\nSample anomalies (top 10):")
    for t, s in list(
        dict(sorted(anomaly_dict.items(), key=lambda kv: kv[1], reverse=True)).items()
    )[:10]:
        print(f"{t}: {s:.6f}")
