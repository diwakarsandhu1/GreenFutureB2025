"""
isolation_forest.py

Isolation Forest anomaly detection on ESG features.
Returns: { ticker: iso_anomaly_score }

When run directly:
- Trains model
- Computes anomaly scores
- Generates plots:
    1. Histogram of anomaly scores
    2. Score_samples distribution
    3. Scatter plot of 1st two ESG principal dimensions (just for visual spread)
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# -------------------------------------------------------
# Load data
# -------------------------------------------------------
def load_data(path="../../preprocess_and_filter/preprocessed_refinitiv.csv"):
    df = pd.read_csv(path)
    return df


# -------------------------------------------------------
# Select ESG features
# -------------------------------------------------------
def select_esg_features(df: pd.DataFrame):
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

    return X, tickers, feature_cols


# -------------------------------------------------------
# Scale features
# -------------------------------------------------------
def scale_features(X: pd.DataFrame):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


# -------------------------------------------------------
# Isolation Forest scoring
# -------------------------------------------------------
def compute_iso_forest_scores(X_scaled, contamination=0.05, random_state=42):
    iso = IsolationForest(
        contamination=contamination,
        n_estimators=300,
        max_samples="auto",
        random_state=random_state,
    )

    iso.fit(X_scaled)

    raw_scores = iso.score_samples(X_scaled)     # higher = more normal
    anomaly_scores = -raw_scores                # invert → higher = more anomalous

    return anomaly_scores, raw_scores, iso


# -------------------------------------------------------
# Main pipeline call
# -------------------------------------------------------
def run_isolation_forest(csv_path="../../preprocess_and_filter/preprocessed_refinitiv.csv"):
    df = load_data(csv_path)
    X, tickers, feature_cols = select_esg_features(df)
    X_scaled, scaler = scale_features(X)

    anomaly_scores, raw_scores, iso_model = compute_iso_forest_scores(X_scaled)

    anomaly_dict = {tickers[i]: float(anomaly_scores[i]) for i in range(len(tickers))}
    return anomaly_dict


# -------------------------------------------------------
# Visualization / Debug Execution
# -------------------------------------------------------
def save_iso_plots(anomaly_scores, raw_scores, X_scaled, output_dir="./outputs/"):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Histogram of anomaly scores
    plt.figure(figsize=(8, 5))
    plt.hist(anomaly_scores, bins=40, alpha=0.8, color="steelblue")
    plt.title("Isolation Forest Anomaly Score Distribution\n(Higher = More Anomalous)")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Frequency")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "iso_anomaly_scores.png"))
    plt.close()

    # 2. Histogram of raw score_samples values
    plt.figure(figsize=(8, 5))
    plt.hist(raw_scores, bins=40, alpha=0.8, color="darkorange")
    plt.title("Isolation Forest score_samples Distribution\n(Higher = More Normal)")
    plt.xlabel("score_samples")
    plt.ylabel("Frequency")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "iso_score_samples_distribution.png"))
    plt.close()

    # 3. Scatter plot of two standardized feature dimensions
    plt.figure(figsize=(7, 6))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], s=12, alpha=0.6)
    plt.title("ESG Feature Space (Standardized)\nFirst Two Dimensions")
    plt.xlabel("Feature 1 (scaled)")
    plt.ylabel("Feature 2 (scaled)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "iso_feature_scatter.png"))
    plt.close()


# -------------------------------------------------------
# Direct execution
# -------------------------------------------------------
if __name__ == "__main__":
    print("Running Isolation Forest anomaly detection + generating plots...")

    df = load_data()
    X, tickers, feature_cols = select_esg_features(df)
    X_scaled, scaler = scale_features(X)

    anomaly_scores, raw_scores, iso = compute_iso_forest_scores(X_scaled)

    # Save plots
    save_iso_plots(anomaly_scores, raw_scores, X_scaled, output_dir="./outputs/")

    print("✔ Plots saved to ./outputs/")
    print("\nSample anomaly scores:")
    for t, s in list({tickers[i]: float(anomaly_scores[i]) for i in range(len(tickers))}.items())[:10]:
        print(f"{t}: {s:.6f}")
