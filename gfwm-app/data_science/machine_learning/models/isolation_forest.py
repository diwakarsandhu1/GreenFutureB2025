"""
isolation_forest.py

Enhanced Isolation Forest anomaly detection module with:
- Global BEST_ISO_PARAMS
- Fully parameterized model call
- Hyperparameter tuning function
- Baseline + tuned scoring consistency
- Backwards compatibility with your existing pipeline

This mirrors the structure used in pca.py.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import KFold


# ============================================================
# GLOBAL BEST PARAMETERS (updated by tuning)
# ============================================================

BEST_ISO_PARAMS = {
    "contamination": 0.03,
    "n_estimators": 300,
    "max_samples": 1.0,
    "max_features": 0.5,
    "bootstrap": True,
    "scaler": "standard",
    "random_state": 42,
}


# ============================================================
# Scaler helper
# ============================================================

def get_scaler(name: str):
    if name == "standard":
        return StandardScaler()
    elif name == "minmax":
        return MinMaxScaler()
    elif name == "robust":
        return RobustScaler()
    else:
        raise ValueError(f"Unknown scaler: {name}")


# ============================================================
# DATA HANDLING
# ============================================================

def load_data(path="../../preprocess_and_filter/preprocessed_refinitiv.csv"):
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


def scale_features(X: pd.DataFrame, scaler_name="standard"):
    scaler = get_scaler(scaler_name)
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


# ============================================================
# CORE ISOLATION FOREST MODEL (fully parameterized)
# ============================================================

def compute_iso_forest_scores(
    X_scaled,
    contamination=None,
    n_estimators=None,
    max_samples=None,
    max_features=None,
    bootstrap=None,
    random_state=None,
):
    """
    Compute Isolation Forest scores, with defaults drawn from BEST_ISO_PARAMS.
    """

    contamination = contamination if contamination is not None else BEST_ISO_PARAMS["contamination"]
    n_estimators = n_estimators or BEST_ISO_PARAMS["n_estimators"]
    max_samples = max_samples or BEST_ISO_PARAMS["max_samples"]
    max_features = max_features or BEST_ISO_PARAMS["max_features"]
    bootstrap = bootstrap if bootstrap is not None else BEST_ISO_PARAMS["bootstrap"]
    random_state = random_state if random_state is not None else BEST_ISO_PARAMS["random_state"]

    iso = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        bootstrap=bootstrap,
        random_state=random_state,
    )

    iso.fit(X_scaled)

    raw_scores = iso.score_samples(X_scaled)   # higher = more normal
    anomaly_scores = -raw_scores               # invert → higher = more anomalous

    return anomaly_scores, raw_scores, iso


# ============================================================
# EXTERNAL API (uses BEST_ISO_PARAMS by default)
# ============================================================

def run_isolation_forest(
    csv_path="../../preprocess_and_filter/preprocessed_refinitiv.csv",
    **override_params,
):
    """
    Run Isolation Forest with either default BEST_ISO_PARAMS or overridden values.
    """
    df = load_data(csv_path)
    X, tickers, _ = select_esg_features(df)

    scaler_name = override_params.get("scaler", BEST_ISO_PARAMS["scaler"])
    X_scaled, _ = scale_features(X, scaler_name)

    anomaly_scores, raw_scores, iso_model = compute_iso_forest_scores(
        X_scaled,
        **override_params,
    )

    return {tickers[i]: float(anomaly_scores[i]) for i in range(len(tickers))}


# ============================================================
# HYPERPARAMETER TUNING
# ============================================================

def tune_isolation_forest(
    X,
    contamination_values=None,
    n_estimator_values=None,
    max_samples_values=None,
    max_features_values=None,
    bootstrap_options=None,
    scaler_options=None,
    n_splits=5,
    update_global=True,
    top_k=25,
):
    """
    Hyperparameter tuning using cross-validation + stability (Jaccard similarity).
    """

    # Default search grids
    contamination_values = contamination_values or [0.03, 0.05, 0.10]
    n_estimator_values = n_estimator_values or [100, 250, 300]
    max_samples_values = max_samples_values or ["auto", 0.75, 1.0]
    max_features_values = max_features_values or [0.5, 0.75, 1.0]
    bootstrap_options = bootstrap_options or [False, True]
    scaler_options = scaler_options or ["standard", "minmax", "robust"]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []

    import itertools
    param_grid = list(itertools.product(
        contamination_values,
        n_estimator_values,
        max_samples_values,
        max_features_values,
        bootstrap_options,
        scaler_options,
    ))

    for cont, est, ms, mf, boot, scaler_name in param_grid:
        X_scaled, _ = scale_features(X, scaler_name)
        overlaps = []

        for train_idx, val_idx in kf.split(X_scaled):
            X_train = X_scaled[train_idx]
            X_val = X_scaled[val_idx]

            iso = IsolationForest(
                contamination=cont,
                n_estimators=est,
                max_samples=ms,
                max_features=mf,
                bootstrap=boot,
                random_state=42,
            )

            iso.fit(X_train)
            scores = -iso.score_samples(X_val)
            top_idx = np.argsort(scores)[-top_k:]
            overlaps.append(set(top_idx))

        # Compute stability via Jaccard index
        jaccard_vals = []
        for i in range(len(overlaps)):
            for j in range(i + 1, len(overlaps)):
                A, B = overlaps[i], overlaps[j]
                jacc = len(A & B) / len(A | B) if len(A | B) else 0
                jaccard_vals.append(jacc)

        results.append({
            "contamination": cont,
            "n_estimators": est,
            "max_samples": ms,
            "max_features": mf,
            "bootstrap": boot,
            "scaler": scaler_name,
            "Mean_Jaccard": np.mean(jaccard_vals),
            "Std_Jaccard": np.std(jaccard_vals),
        })

    results_df = pd.DataFrame(results)
    best_row = results_df.loc[results_df["Mean_Jaccard"].idxmax()]

    best_params = {
        "contamination": best_row["contamination"],
        "n_estimators": int(best_row["n_estimators"]),
        "max_samples": best_row["max_samples"],
        "max_features": float(best_row["max_features"]),
        "bootstrap": bool(best_row["bootstrap"]),
        "scaler": best_row["scaler"],
    }

    if update_global:
        BEST_ISO_PARAMS.update(best_params)

    return results_df, best_params


# ============================================================
# VISUALIZATION
# ============================================================

def save_iso_plots(anomaly_scores, raw_scores, X_scaled, output_dir="./outputs/"):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Histogram of anomaly scores
    plt.figure(figsize=(8, 5))
    plt.hist(anomaly_scores, bins=40, alpha=0.8, color="steelblue")
    plt.title("Isolation Forest Anomaly Score Distribution (Higher = More Anomalous)")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Frequency")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "iso_anomaly_scores.png"))
    plt.close()

    # 2. score_samples distribution
    plt.figure(figsize=(8, 5))
    plt.hist(raw_scores, bins=40, alpha=0.8, color="darkorange")
    plt.title("score_samples Distribution (Higher = More Normal)")
    plt.xlabel("score_samples")
    plt.ylabel("Frequency")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "iso_score_samples_distribution.png"))
    plt.close()

    # 3. Feature scatter
    plt.figure(figsize=(7, 6))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], s=12, alpha=0.6)
    plt.title("ESG Feature Space (Standardized) — First Two Dimensions")
    plt.xlabel("Feature 1 (scaled)")
    plt.ylabel("Feature 2 (scaled)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "iso_feature_scatter.png"))
    plt.close()


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("Running Isolation Forest anomaly detection + plots...")

    df = load_data()
    X, tickers, _ = select_esg_features(df)
    X_scaled, _ = scale_features(X, BEST_ISO_PARAMS["scaler"])

    anomaly_scores, raw_scores, iso = compute_iso_forest_scores(X_scaled)

    save_iso_plots(anomaly_scores, raw_scores, X_scaled, output_dir="./outputs/")
    print("✔ Plots saved to ./outputs/")

    print("Sample anomaly scores (first 10):")
    for t, s in list({tickers[i]: float(anomaly_scores[i]) for i in range(len(tickers))}.items())[:10]:
        print(f"{t}: {s:.6f}")
