import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------
# Helpers
# ---------------------------------------------

def diagnostics_cov(name: str, S: pd.DataFrame):
    """Print basic eigen diagnostics for a covariance matrix."""
    vals = np.linalg.eigvalsh(S.values)
    min_eig = vals.min()
    max_eig = vals.max()
    cond = max_eig / max(min_eig, 1e-18)  # avoid divide-by-zero
    print(f"[Diagnostics: {name}]")
    print(f"  min eigenvalue   : {min_eig:.4e}")
    print(f"  max eigenvalue   : {max_eig:.4e}")
    print(f"  condition number : {cond:.4e}")

def fraction_within_tol(delta: np.ndarray, atol: float = 1e-6) -> float:
    """Return fraction of entries with |delta| <= atol."""
    return np.mean(np.abs(delta) <= atol)


# NEW: eigen-spectrum plotting helper
def plot_eigen_spectrum(name: str,
                        S_old: pd.DataFrame,
                        S_new: pd.DataFrame,
                        out_dir: str):
    """
    Plot eigenvalue spectrum of old vs new covariance matrices.

    - Sort eigenvalues in descending order
    - Plot on semilog-y axis (index vs eigenvalue)
    - Save as PNG in out_dir
    """
    os.makedirs(out_dir, exist_ok=True)

    vals_old = np.linalg.eigvalsh(S_old.values)
    vals_new = np.linalg.eigvalsh(S_new.values)

    # Sort descending just for nicer visualization (largest to smallest)
    vals_old_sorted = np.sort(vals_old)[::-1]
    vals_new_sorted = np.sort(vals_new)[::-1]

    plt.figure(figsize=(8, 5))
    plt.semilogy(range(1, len(vals_old_sorted) + 1),
                 vals_old_sorted,
                 label="old")
    plt.semilogy(range(1, len(vals_new_sorted) + 1),
                 vals_new_sorted,
                 label="new")

    plt.xlabel("Eigenvalue index (sorted)")
    plt.ylabel("Eigenvalue")
    plt.title(f"Eigenvalue spectrum: {name}")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"{name}_eigs.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[PLOT] Saved eigen spectrum for {name} to: {out_path}")


# ---------------------------------------------
# Main comparison logic
# ---------------------------------------------

if __name__ == "__main__":
    # Base paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    old_dir = base_dir                          # old files live directly here
    new_dir = os.path.join(base_dir, "generated_data")

    print(f"[INFO] Base directory: {base_dir}")
    print(f"[INFO] Old dir       : {old_dir}")
    print(f"[INFO] New dir       : {new_dir}")

    # ---------- SP500 returns timeseries ----------
    old_ts_path = os.path.join(old_dir, "sp500_timeseries_13-24.csv")
    new_ts_path = os.path.join(new_dir, "sp500_timeseries_13-24.csv")

    old_ts = pd.read_csv(old_ts_path, index_col=0)
    new_ts = pd.read_csv(new_ts_path, index_col=0)

    print(f"[INFO] Loaded old sp500_timeseries from {old_ts_path} with shape {old_ts.shape}")
    print(f"[INFO] Loaded new sp500_timeseries from {new_ts_path} with shape {new_ts.shape}")

    print("\n===== Comparing SP500 returns timeseries =====")
    print(f"Old shape: {old_ts.shape}, New shape: {new_ts.shape}")

    # Try label-based comparison first
    common_index = old_ts.index.intersection(new_ts.index)
    common_cols = old_ts.columns.intersection(new_ts.columns)

    if len(common_index) == 0 or len(common_cols) == 0:
        print("[WARN] SP500 returns timeseries: no overlapping index/columns to compare by labels.")
        # position-based comparison
        min_rows = min(old_ts.shape[0], new_ts.shape[0])
        min_cols = min(old_ts.shape[1], new_ts.shape[1])
        old_vals = old_ts.iloc[:min_rows, :min_cols].to_numpy()
        new_vals = new_ts.iloc[:min_rows, :min_cols].to_numpy()
        delta = new_vals - old_vals
        frac = fraction_within_tol(delta, atol=1e-12)
        print("SP500 returns timeseries (position-based):")
        print(f"  shape             : {old_vals.shape}")
        print(f"  max abs diff      : {np.nanmax(np.abs(delta)) if np.isfinite(delta).any() else np.nan: .6e}")
        print(f"  mean abs diff     : {np.nanmean(np.abs(delta)) if np.isfinite(delta).any() else np.nan: .6e}")
        print(f"  fraction |Δ|<=atol: {frac: .3f}")
    else:
        old_sub = old_ts.loc[common_index, common_cols]
        new_sub = new_ts.loc[common_index, common_cols]
        delta = new_sub.values - old_sub.values
        frac = fraction_within_tol(delta, atol=1e-12)
        print("SP500 returns timeseries (label-based):")
        print(f"  common index size : {len(common_index)}")
        print(f"  common columns    : {len(common_cols)}")
        print(f"  max abs diff      : {np.nanmax(np.abs(delta)): .6e}")
        print(f"  mean abs diff     : {np.nanmean(np.abs(delta)): .6e}")
        print(f"  fraction |Δ|<=atol: {frac: .3f}")

    # ---------- Performance summaries ----------
    old_perf_path = os.path.join(old_dir, "sp500_performance_summaries.csv")
    new_perf_path = os.path.join(new_dir, "sp500_performance_summaries.csv")

    old_perf = pd.read_csv(old_perf_path, index_col=0)
    new_perf = pd.read_csv(new_perf_path, index_col=0)

    print(f"\n[INFO] Loaded old performance_summaries from {old_perf_path} with shape {old_perf.shape}")
    print(f"[INFO] Loaded new performance_summaries from {new_perf_path} with shape {new_perf.shape}")

    # Old perf is in (2, n) shape, new is (n, 2); transpose old for comparison
    if old_perf.shape[0] == 2:
        old_perf = old_perf.T
        print(f"[INFO] Transposed old performance_summaries for comparison. New shape: {old_perf.shape}")

    print("\n===== Comparing Performance summaries =====")
    print(f"Old shape: {old_perf.shape}, New shape: {new_perf.shape}")

    common_index = old_perf.index.intersection(new_perf.index)
    common_cols = old_perf.columns.intersection(new_perf.columns)

    if len(common_index) == 0 or len(common_cols) == 0:
        print("[WARN] Performance summaries: no overlapping index/columns to compare by labels.")
        # position-based
        min_rows = min(old_perf.shape[0], new_perf.shape[0])
        min_cols = min(old_perf.shape[1], new_perf.shape[1])
        old_vals = old_perf.iloc[:min_rows, :min_cols].to_numpy()
        new_vals = new_perf.iloc[:min_rows, :min_cols].to_numpy()
        delta = new_vals - old_vals
        frac = fraction_within_tol(delta, atol=1e-12)
        print("Performance summaries (position-based):")
        print(f"  shape             : {old_vals.shape}")
        print(f"  max abs diff      : {np.nanmax(np.abs(delta)): .6e}")
        print(f"  mean abs diff     : {np.nanmean(np.abs(delta)): .6e}")
        print(f"  fraction |Δ|<=atol: {frac: .3f}")
    else:
        old_sub = old_perf.loc[common_index, common_cols]
        new_sub = new_perf.loc[common_index, common_cols]
        delta = new_sub.values - old_sub.values
        frac = fraction_within_tol(delta, atol=1e-12)
        print("Performance summaries (label-based):")
        print(f"  common index size : {len(common_index)}")
        print(f"  common columns    : {len(common_cols)}")
        print(f"  max abs diff      : {np.nanmax(np.abs(delta)): .6e}")
        print(f"  mean abs diff     : {np.nanmean(np.abs(delta)): .6e}")
        print(f"  fraction |Δ|<=atol: {frac: .3f}")

    # ---------- Raw covariance ----------
    old_raw_path = os.path.join(old_dir, "sp500_raw_cov_matrix.csv")
    new_raw_path = os.path.join(new_dir, "sp500_raw_cov_matrix.csv")

    old_raw = pd.read_csv(old_raw_path, index_col=0)
    new_raw = pd.read_csv(new_raw_path, index_col=0)

    print(f"\n[INFO] Loaded old raw covariance from {old_raw_path} with shape {old_raw.shape}")
    print(f"[INFO] Loaded new raw covariance from {new_raw_path} with shape {new_raw.shape}")

    print("\n===== Comparing Raw covariance =====")
    print(f"Old shape: {old_raw.shape}, New shape: {new_raw.shape}")

    # label-based intersection
    common_index = old_raw.index.intersection(new_raw.index)
    common_cols = old_raw.columns.intersection(new_raw.columns)

    if len(common_index) > 0 and len(common_cols) > 0:
        old_sub = old_raw.loc[common_index, common_cols]
        new_sub = new_raw.loc[common_index, common_cols]
        delta = new_sub.values - old_sub.values
        frac = fraction_within_tol(delta, atol=1e-6)
        print("Raw covariance (label-based):")
        print(f"  common index size : {len(common_index)}")
        print(f"  common columns    : {len(common_cols)}")
        print(f"  max abs diff      : {np.max(np.abs(delta)): .6e}")
        print(f"  mean abs diff     : {np.mean(np.abs(delta)): .6e}")
        print(f"  fraction |Δ|<=atol: {frac: .3f}")
    else:
        print("[WARN] Raw covariance: no overlapping index/columns to compare.")

    # Diagnostics for raw cov
    diagnostics_cov("Σ_raw (old)", old_raw)
    diagnostics_cov("Σ_raw (new)", new_raw)

    # NEW: plot eigen spectrum for raw cov
    diag_dir = os.path.join(new_dir, "diagnostics")
    plot_eigen_spectrum("Sigma_raw", old_raw, new_raw, diag_dir)

    # ---------- Adjusted covariance ----------
    old_adj_path = os.path.join(old_dir, "sp500_adjusted_cov_matrix.csv")
    new_adj_path = os.path.join(new_dir, "sp500_adjusted_cov_matrix.csv")

    old_adj = pd.read_csv(old_adj_path, index_col=0)
    new_adj = pd.read_csv(new_adj_path, index_col=0)

    print(f"\n[INFO] Loaded old adjusted covariance from {old_adj_path} with shape {old_adj.shape}")
    print(f"[INFO] Loaded new adjusted covariance from {new_adj_path} with shape {new_adj.shape}")

    print("\n===== Comparing Adjusted covariance =====")
    print(f"Old shape: {old_adj.shape}, New shape: {new_adj.shape}")

    # label-based intersection
    common_index = old_adj.index.intersection(new_adj.index)
    common_cols = old_adj.columns.intersection(new_adj.columns)

    if len(common_index) > 0 and len(common_cols) > 0:
        old_sub = old_adj.loc[common_index, common_cols]
        new_sub = new_adj.loc[common_index, common_cols]
        delta = new_sub.values - old_sub.values
        frac = fraction_within_tol(delta, atol=1e-6)
        print("Adjusted covariance (label-based):")
        print(f"  common index size : {len(common_index)}")
        print(f"  common columns    : {len(common_cols)}")
        print(f"  max abs diff      : {np.max(np.abs(delta)): .6e}")
        print(f"  mean abs diff     : {np.mean(np.abs(delta)): .6e}")
        print(f"  fraction |Δ|<=atol: {frac: .3f}")
    else:
        print("[WARN] Adjusted covariance: no overlapping index/columns to compare.")

    # Diagnostics for adjusted cov
    diagnostics_cov("Σ_adjusted (old)", old_adj)
    diagnostics_cov("Σ_adjusted (new)", new_adj)

    # NEW: plot eigen spectrum for adjusted cov
    plot_eigen_spectrum("Sigma_adjusted", old_adj, new_adj, diag_dir)

    # ---------- SPY returns ----------
    old_spy_path = os.path.join(old_dir, "spy_timeseries_13-24.csv")
    new_spy_path = os.path.join(new_dir, "spy_timeseries_13-24.csv")

    old_spy = pd.read_csv(old_spy_path, index_col=0)
    new_spy = pd.read_csv(new_spy_path, index_col=0)

    print(f"\n[INFO] Loaded old SPY timeseries from {old_spy_path} with shape {old_spy.shape}")
    print(f"[INFO] Loaded new SPY timeseries from {new_spy_path} with shape {new_spy.shape}")

    print("\n===== Comparing SPY returns timeseries =====")
    print(f"Old shape: {old_spy.shape}, New shape: {new_spy.shape}")

    common_index = old_spy.index.intersection(new_spy.index)
    common_cols = old_spy.columns.intersection(new_spy.columns)

    if len(common_index) == 0 or len(common_cols) == 0:
        print("[WARN] SPY returns timeseries: no overlapping index/columns to compare by labels.")
        min_rows = min(old_spy.shape[0], new_spy.shape[0])
        min_cols = min(old_spy.shape[1], new_spy.shape[1])
        old_vals = old_spy.iloc[:min_rows, :min_cols].to_numpy()
        new_vals = new_spy.iloc[:min_rows, :min_cols].to_numpy()
        delta = new_vals - old_vals
        frac = fraction_within_tol(delta, atol=1e-12)
        print("SPY returns timeseries (position-based):")
        print(f"  shape             : {old_vals.shape}")
        print(f"  max abs diff      : {np.nanmax(np.abs(delta)): .6e}")
        print(f"  mean abs diff     : {np.nanmean(np.abs(delta)): .6e}")
        print(f"  fraction |Δ|<=atol: {frac: .3f}")
    else:
        old_sub = old_spy.loc[common_index, common_cols]
        new_sub = new_spy.loc[common_index, common_cols]
        delta = new_sub.values - old_sub.values
        frac = fraction_within_tol(delta, atol=1e-12)
        print("SPY returns timeseries (label-based):")
        print(f"  common index size : {len(common_index)}")
        print(f"  common columns    : {len(common_cols)}")
        print(f"  max abs diff      : {np.nanmax(np.abs(delta)): .6e}")
        print(f"  mean abs diff     : {np.nanmean(np.abs(delta)): .6e}")
        print(f"  fraction |Δ|<=atol: {frac: .3f}")

    print("\n[INFO] Comparison complete.")
