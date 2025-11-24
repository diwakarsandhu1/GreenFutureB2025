"""
pca.py

Runs PCA-based anomaly detection on ESG features.
Returns a dictionary: { ticker: pca_anomaly_score }

Also generates and saves visualization plots when run directly:
- Explained Variance Plot
- Scatter Plot of First Two PCA Components
- Distribution of Reconstruction Errors
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



# ---------------------------------------------
# Load data
# ---------------------------------------------
def load_data(path="../../preprocess_and_filter/preprocessed_refinitiv.csv"):
    df = pd.read_csv(path)
    return df


# ---------------------------------------------
# Select ESG features to feed into PCA
# ---------------------------------------------
def select_pca_features(df: pd.DataFrame):
    feature_cols = [
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

    df_clean = df.dropna(subset=feature_cols)
    X = df_clean[feature_cols].copy()
    tickers = df_clean["ticker"].values

    return X, tickers


# ---------------------------------------------
# Scale features
# ---------------------------------------------
def scale_features(X: pd.DataFrame):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


# ---------------------------------------------
# PCA anomaly scoring
# ---------------------------------------------
def compute_pca_scores(X_scaled, n_components=5):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    X_reconstructed = pca.inverse_transform(X_pca)

    reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)

    return pca, X_pca, reconstruction_error


# ---------------------------------------------
# Main PCA pipeline for external use
# ---------------------------------------------
def run_pca_anomaly_detection(csv_path="../../preprocess_and_filter/preprocessed_refinitiv.csv"):
    df = load_data(csv_path)
    X, tickers = select_pca_features(df)
    X_scaled, scaler = scale_features(X)

    pca, X_pca, scores = compute_pca_scores(X_scaled, n_components=5)

    anomaly_dict = {tickers[i]: float(scores[i]) for i in range(len(tickers))}
    return anomaly_dict



# ---------------------------------------------
# Visualization Helpers
# ---------------------------------------------
def save_plots(pca, X_pca, reconstruction_errors, output_path):
    os.makedirs(output_path, exist_ok=True)

    # ----- 1. Explained Variance Plot -----
    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker="o")
    plt.title("PCA Explained Variance (Cumulative)")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "pca_explained_variance.png"))
    plt.close()


    # ----- 2. Scatter Plot of First Two PCA Components -----
    plt.figure(figsize=(7, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], s=10, alpha=0.6)
    plt.title("PCA Component Scatter Plot")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "pca_scatter_pc1_pc2.png"))
    plt.close()


    # ----- 3. Reconstruction Error Distribution -----
    plt.figure(figsize=(8, 5))
    plt.hist(reconstruction_errors, bins=40, alpha=0.75, color="steelblue")
    plt.title("PCA Reconstruction Error Distribution")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "pca_reconstruction_error_distribution.png"))
    plt.close()



# ---------------------------------------------
# Direct Execution for Debugging + Visualization
# ---------------------------------------------
if __name__ == "__main__":
    print("Running PCA anomaly detection and generating plots...")

    df = load_data()
    X, tickers = select_pca_features(df)
    X_scaled, scaler = scale_features(X)

    pca, X_pca, reconstruction_errors = compute_pca_scores(X_scaled, n_components=5)

    output_dir = "./outputs/"
    save_plots(pca, X_pca, reconstruction_errors, output_dir)

    print(f"Plots saved to: {output_dir}")

    # Print sample anomaly outputs
    anomaly_dict = {tickers[i]: float(reconstruction_errors[i]) for i in range(len(tickers))}
    print("Sample anomaly scores:")
    for t, s in list(anomaly_dict.items())[:10]:
        print(f"{t}: {s:.6f}")
