"""
pca.py

Runs PCA-based anomaly detection on ESG features.
Returns a dictionary: { ticker: pca_anomaly_score }

Also includes hyperparameter tuning functionality.
Best parameters are stored in a global dict BEST_PCA_PARAMS and
used as defaults in the main PCA call.

Also generates and saves visualization plots when run directly.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold



# ============================================================
# GLOBAL BEST PARAMETERS (UPDATED BY TUNING)
# ============================================================

BEST_PCA_PARAMS = {
    "n_components": 5,
    "whiten": False,
    "scaler": "standard"   # "standard", "minmax", "robust"
}



# ============================================================
# Scaler Helper
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
# DATA LOADING / FEATURE SELECTION
# ============================================================

def load_data(path="../../preprocess_and_filter/preprocessed_refinitiv.csv"):
    df = pd.read_csv(path)
    return df


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



# ============================================================
# SCALING FUNCTION — uses selected scaler
# ============================================================

def scale_features(X: pd.DataFrame, scaler_name="standard"):
    scaler = get_scaler(scaler_name)
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler



# ============================================================
# CORE PCA FUNCTION (CONFIGURABLE + DEFAULTS FROM GLOBAL)
# ============================================================

def compute_pca_scores(
    X_scaled,
    n_components=None,
    whiten=None
):
    """
    Computes PCA reconstruction error.
    Parameters default to BEST_PCA_PARAMS if None.
    """

    n_components = n_components or BEST_PCA_PARAMS["n_components"]
    whiten       = whiten if whiten is not None else BEST_PCA_PARAMS["whiten"]

    pca = PCA(n_components=n_components, whiten=whiten, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    X_reconstructed = pca.inverse_transform(X_pca)

    reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
    return pca, X_pca, reconstruction_error



# ============================================================
# FULL PIPELINE USED EXTERNALLY
# ============================================================

def run_pca_anomaly_detection(
    csv_path="../../preprocess_and_filter/preprocessed_refinitiv.csv",
    n_components=None,
    whiten=None,
    scaler_name=None
):
    """
    External API call — uses BEST_PCA_PARAMS defaults unless overridden.
    """

    # Use global defaults if parameters unspecified
    n_components = n_components or BEST_PCA_PARAMS["n_components"]
    whiten       = whiten if whiten is not None else BEST_PCA_PARAMS["whiten"]
    scaler_name  = scaler_name or BEST_PCA_PARAMS["scaler"]

    df = load_data(csv_path)
    X, tickers = select_pca_features(df)
    X_scaled, scaler = scale_features(X, scaler_name=scaler_name)

    pca, X_pca, scores = compute_pca_scores(
        X_scaled,
        n_components=n_components,
        whiten=whiten
    )

    anomaly_dict = {tickers[i]: float(scores[i]) for i in range(len(tickers))}
    return anomaly_dict



# ============================================================
# HYPERPARAMETER TUNING FUNCTION
# ============================================================

def tune_pca_hyperparameters(
    X,
    k_values=None,
    whiten_options=None,
    scaler_options=None,
    n_splits=5,
    update_global=True
):
    """
    Performs PCA hyperparameter tuning with cross-validation.
    Updates BEST_PCA_PARAMS automatically (unless update_global=False).
    """

    # Default search spaces
    if k_values is None:
        k_values = list(range(2, 16))

    if whiten_options is None:
        whiten_options = [False, True]

    if scaler_options is None:
        scaler_options = ["standard", "minmax", "robust"]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = []

    for scaler_name in scaler_options:
        X_scaled, _ = scale_features(X, scaler_name)

        for whiten_flag in whiten_options:
            for k in k_values:

                fold_errors = []
                fold_evr = []

                for train_idx, val_idx in kf.split(X_scaled):
                    X_train = X_scaled[train_idx]
                    X_val   = X_scaled[val_idx]

                    pca = PCA(n_components=k, whiten=whiten_flag)
                    pca.fit(X_train)

                    X_val_recon = pca.inverse_transform(pca.transform(X_val))
                    recon_error = np.mean((X_val - X_val_recon) ** 2)
                    fold_errors.append(recon_error)

                    fold_evr.append(np.sum(pca.explained_variance_ratio_))

                results.append({
                    "scaler": scaler_name,
                    "whiten": whiten_flag,
                    "n_components": k,
                    "Mean_Val_Error": np.mean(fold_errors),
                    "Std_Val_Error": np.std(fold_errors),
                    "Mean_EVR": np.mean(fold_evr),
                })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Best by lowest validation error
    best_row = results_df.loc[results_df["Mean_Val_Error"].idxmin()]

    best_params = {
        "scaler": best_row["scaler"],
        "whiten": bool(best_row["whiten"]),
        "n_components": int(best_row["n_components"]),
    }

    if update_global:
        BEST_PCA_PARAMS.update(best_params)

    return results_df, best_params



# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def save_plots(pca, X_pca, reconstruction_errors, output_path):
    os.makedirs(output_path, exist_ok=True)

    # ----- 1. Explained Variance Plot -----
    plt.figure(figsize=(8, 5))
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    components = np.arange(1, len(cum_var) + 1)
    plt.plot(components, cum_var, marker="o")
    plt.title("PCA Explained Variance (Cumulative)")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.xticks(components)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "pca_explained_variance.png"))
    plt.close()

    # ----- 2. PC1-PC2 Scatter -----
    plt.figure(figsize=(7, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], s=10, alpha=0.6)
    plt.title("PCA Component Scatter Plot")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "pca_scatter_pc1_pc2.png"))
    plt.close()

    # ----- 3. Reconstruction Error Histogram -----
    plt.figure(figsize=(8, 5))
    plt.hist(reconstruction_errors, bins=40, alpha=0.75, color="steelblue")
    plt.title("PCA Reconstruction Error Distribution")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "pca_reconstruction_error_distribution.png"))
    plt.close()



# ============================================================
# DIRECT EXECUTION BLOCK
# ============================================================

if __name__ == "__main__":
    print("Running PCA anomaly detection and generating plots...")

    df = load_data()
    X, tickers = select_pca_features(df)
    X_scaled, scaler = scale_features(X, scaler_name=BEST_PCA_PARAMS["scaler"])

    pca, X_pca, reconstruction_errors = compute_pca_scores(
        X_scaled,
        BEST_PCA_PARAMS["n_components"],
        BEST_PCA_PARAMS["whiten"]
    )

    output_dir = "./outputs/"
    save_plots(pca, X_pca, reconstruction_errors, output_dir)

    print(f"Plots saved to: {output_dir}")

    anomaly_dict = {tickers[i]: float(reconstruction_errors[i]) for i in range(len(tickers))}
    print("Sample anomaly scores:")
    for t, s in list(anomaly_dict.items())[:10]:
        print(f"{t}: {s:.6f}")
