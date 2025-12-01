"""
hypertuning_validation.py

Hyperparameter tuning + baseline comparison for:
- PCA
- Isolation Forest
- Autoencoder

Runs baselines, tunes models, compares baseline vs tuned, and (optionally) saves outputs.
"""

import os
import sys
import traceback
import pandas as pd
import numpy as np

# Ensure local modules (pca.py, isolation_forest.py, auto_encoder.py) are importable
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import pca
import isolation_forest
# NOTE: autoencoder is lazy-imported inside functions to avoid TF import issues

# ------------- Baseline runners -------------

def run_baseline_pca(csv_path: str) -> pd.DataFrame:
    df = pca.load_data(csv_path)
    X, tickers = pca.select_pca_features(df)
    X_scaled, _ = pca.scale_features(X, scaler_name=pca.BEST_PCA_PARAMS["scaler"])

    _, _, scores = pca.compute_pca_scores(
        X_scaled,
        n_components=pca.BEST_PCA_PARAMS["n_components"],
        whiten=pca.BEST_PCA_PARAMS["whiten"],
    )
    out = pd.DataFrame({"ticker": tickers, "baseline_pca": scores})
    return out

def run_baseline_if(csv_path: str) -> pd.DataFrame:
    df = isolation_forest.load_data(csv_path)
    X, tickers, _ = isolation_forest.select_esg_features(df)
    X_scaled, _ = isolation_forest.scale_features(X, scaler_name=isolation_forest.BEST_ISO_PARAMS["scaler"])

    scores, raw, _ = isolation_forest.compute_iso_forest_scores(X_scaled)
    out = pd.DataFrame({"ticker": tickers, "baseline_if": scores})
    return out

def run_baseline_ae(csv_path: str) -> pd.DataFrame:
    # Lazy import to avoid TF import unless needed
    import auto_encoder
    df = auto_encoder.load_data(csv_path)
    X, tickers, _ = auto_encoder.select_esg_features(df)
    X_scaled, _ = auto_encoder.scale_features(X, scaler_name=auto_encoder.BEST_AE_PARAMS["scaler"])

    errors, _ = auto_encoder.compute_autoencoder_scores(X_scaled)
    out = pd.DataFrame({"ticker": tickers, "baseline_ae": errors})
    return out

# ------------- Tuning runners -------------

def run_pca_tuning(csv_path: str, update_global: bool = False):
    df = pca.load_data(csv_path)
    X, tickers = pca.select_pca_features(df)

    tuning_results, best_params = pca.tune_pca_hyperparameters(
        X,
        k_values=None,
        whiten_options=None,
        scaler_options=None,
        n_splits=5,
        update_global=update_global,
    )

    X_scaled, _ = pca.scale_features(X, scaler_name=best_params["scaler"])
    _, _, tuned_scores = pca.compute_pca_scores(
        X_scaled,
        n_components=best_params["n_components"],
        whiten=best_params["whiten"],
    )

    tuned_df = pd.DataFrame({"ticker": tickers, "tuned_pca": tuned_scores})
    return tuning_results, best_params, tuned_df

def run_if_tuning(csv_path: str, update_global: bool = False):
    df = isolation_forest.load_data(csv_path)
    X, tickers, _ = isolation_forest.select_esg_features(df)

    tuning_results, best_params = isolation_forest.tune_isolation_forest(
        X,
        update_global=update_global,
    )

    X_scaled, _ = isolation_forest.scale_features(X, scaler_name=best_params["scaler"])
    tuned_scores, raw, _ = isolation_forest.compute_iso_forest_scores(
        X_scaled,
        contamination=best_params["contamination"],
        n_estimators=best_params["n_estimators"],
        max_samples=best_params["max_samples"],
        max_features=best_params["max_features"],
        bootstrap=best_params["bootstrap"],
    )

    tuned_df = pd.DataFrame({"ticker": tickers, "tuned_if": tuned_scores})
    return tuning_results, best_params, tuned_df

def run_ae_tuning(csv_path: str, update_global: bool = False):
    import auto_encoder
    df = auto_encoder.load_data(csv_path)
    X, tickers, _ = auto_encoder.select_esg_features(df)

    tuning_results, best_params = auto_encoder.tune_autoencoder(
        X,
        update_global=update_global,
    )

    X_scaled, _ = auto_encoder.scale_features(X, scaler_name=best_params["scaler"])
    tuned_scores, model = auto_encoder.compute_autoencoder_scores(
        X_scaled,
        latent_dim=best_params["latent_dim"],
        hidden_units=best_params["hidden_units"],
        activation=best_params["activation"],
        dropout=best_params["dropout"],
        epochs=best_params["epochs"],
        batch_size=best_params["batch_size"],
        learning_rate=best_params["learning_rate"],
    )

    tuned_df = pd.DataFrame({"ticker": tickers, "tuned_ae": tuned_scores})
    return tuning_results, best_params, tuned_df

# ------------- Comparison helpers -------------

def merge_all_results(b_pca: pd.DataFrame, t_pca: pd.DataFrame,
                      b_if: pd.DataFrame, t_if: pd.DataFrame,
                      b_ae: pd.DataFrame, t_ae: pd.DataFrame) -> pd.DataFrame:
    df = b_pca.merge(t_pca, on="ticker", how="inner")
    df = df.merge(b_if, on="ticker", how="inner").merge(t_if, on="ticker", how="inner")
    df = df.merge(b_ae, on="ticker", how="inner").merge(t_ae, on="ticker", how="inner")
    df["diff_pca"] = df["tuned_pca"] - df["baseline_pca"]
    df["diff_if"]  = df["tuned_if"]  - df["baseline_if"]
    df["diff_ae"]  = df["tuned_ae"]  - df["baseline_ae"]
    return df

# ------------- Full pipeline -------------

def run_full_validation(csv_path: str, save_outputs: bool = False, out_dir: str = "./hyper_outputs/"):
    print("=========================")
    print("Running FULL validation…")
    print("=========================\n")

    try:
        print("[1/9] Running baseline PCA…")
        baseline_pca = run_baseline_pca(csv_path)
        print(f"✔ Baseline PCA complete. rows={len(baseline_pca)}\n")
    except Exception as e:
        print("✖ Baseline PCA failed!")
        traceback.print_exc()
        raise

    try:
        print("[2/9] Running baseline Isolation Forest…")
        baseline_if = run_baseline_if(csv_path)
        print(f"✔ Baseline IF complete. rows={len(baseline_if)}\n")
    except Exception as e:
        print("✖ Baseline IF failed!")
        traceback.print_exc()
        raise

    try:
        print("[3/9] Running baseline Autoencoder…")
        baseline_ae = run_baseline_ae(csv_path)
        print(f"✔ Baseline AE complete. rows={len(baseline_ae)}\n")
    except Exception as e:
        print("✖ Baseline AE failed!")
        traceback.print_exc()
        print("Hint: Ensure TensorFlow is installed and `auto_encoder.py` loads.")
        raise

    try:
        print("[4/9] Tuning PCA hyperparameters…")
        pca_tune_results, pca_best, tuned_pca = run_pca_tuning(csv_path)
        print("✔ PCA tuning complete. Best params:", pca_best, "\n")
    except Exception:
        print("✖ PCA tuning failed!")
        traceback.print_exc()
        raise

    try:
        print("[5/9] Tuning Isolation Forest hyperparameters…")
        if_tune_results, if_best, tuned_if = run_if_tuning(csv_path)
        print("✔ IF tuning complete. Best params:", if_best, "\n")
    except Exception:
        print("✖ Isolation Forest tuning failed!")
        traceback.print_exc()
        raise

    try:
        print("[6/9] Tuning Autoencoder hyperparameters…")
        ae_tune_results, ae_best, tuned_ae = run_ae_tuning(csv_path)
        print("✔ AE tuning complete. Best params:", ae_best, "\n")
    except Exception:
        print("✖ Autoencoder tuning failed!")
        traceback.print_exc()
        print("Hint: Ensure TensorFlow is installed and GPU/CPU supports it.")
        raise

    try:
        print("[7/9] Merging baseline vs tuned comparisons…")
        merged = merge_all_results(baseline_pca, tuned_pca, baseline_if, tuned_if, baseline_ae, tuned_ae)
        print(f"✔ Merge complete. rows={len(merged)}\n")
    except Exception:
        print("✖ Merge failed! Check that 'ticker' exists in all frames and overlaps.")
        traceback.print_exc()
        raise

    if save_outputs:
        try:
            print("[8/9] Saving outputs…")
            os.makedirs(out_dir, exist_ok=True)
            baseline_pca.to_csv(os.path.join(out_dir, "baseline_pca.csv"), index=False)
            baseline_if.to_csv(os.path.join(out_dir, "baseline_if.csv"), index=False)
            baseline_ae.to_csv(os.path.join(out_dir, "baseline_ae.csv"), index=False)
            pca_tune_results.to_csv(os.path.join(out_dir, "pca_tuning_results.csv"), index=False)
            if_tune_results.to_csv(os.path.join(out_dir, "if_tuning_results.csv"), index=False)
            ae_tune_results.to_csv(os.path.join(out_dir, "ae_tuning_results.csv"), index=False)
            tuned_pca.to_csv(os.path.join(out_dir, "tuned_pca.csv"), index=False)
            tuned_if.to_csv(os.path.join(out_dir, "tuned_if.csv"), index=False)
            tuned_ae.to_csv(os.path.join(out_dir, "tuned_ae.csv"), index=False)
            merged.to_csv(os.path.join(out_dir, "merged_comparison_all_models.csv"), index=False)
            print("✔ Outputs saved.\n")
        except Exception:
            print("✖ Saving outputs failed!")
            traceback.print_exc()
            raise
    else:
        print("[8/9] Output saving skipped.\n")

    print("[9/9] Validation complete! Returning results…")
    return {
        "baseline_pca": baseline_pca,
        "baseline_if": baseline_if,
        "baseline_ae": baseline_ae,
        "tuning_pca": pca_tune_results,
        "tuning_if": if_tune_results,
        "tuning_ae": ae_tune_results,
        "best_pca_params": pca_best,
        "best_if_params": if_best,
        "best_ae_params": ae_best,
        "tuned_pca": tuned_pca,
        "tuned_if": tuned_if,
        "tuned_ae": tuned_ae,
        "merged_results": merged,
    }

# ------------- Direct execution -------------

if __name__ == "__main__":
    results = run_full_validation(
        csv_path="../../preprocess_and_filter/preprocessed_refinitiv.csv",
        save_outputs=True,
        out_dir="./hyper_outputs/"
    )
    print("Full validation pipeline finished!")