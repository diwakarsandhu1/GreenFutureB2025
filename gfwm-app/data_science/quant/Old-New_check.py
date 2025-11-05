import numpy as np
import pandas as pd
from pathlib import Path
import numpy.linalg as LA


# ---------- Diagnostics helper ----------

def covariance_diagnostics(name: str, cov: pd.DataFrame) -> None:
    """Print basic diagnostics for a covariance matrix."""
    vals = LA.eigvalsh(cov.values)
    min_eig = float(vals.min())
    max_eig = float(vals.max())
    cond = max_eig / max(min_eig, 1e-18)
    print(f"[Diagnostics: {name}]")
    print(f"  min eigenvalue   : {min_eig: .4e}")
    print(f"  max eigenvalue   : {max_eig: .4e}")
    print(f"  condition number : {cond: .4e}")
    print()


# ---------- Comparison helpers ----------

def compare_by_position(name: str, old_df: pd.DataFrame, new_df: pd.DataFrame,
                        atol: float = 1e-8) -> None:
    """Compare two DataFrames by position (ignoring labels)."""
    if old_df.shape != new_df.shape:
        print(f"[{name}] Position compare: shapes differ, cannot compare.")
        return

    diff = old_df.values - new_df.values
    abs_diff = np.abs(diff)

    max_abs = float(abs_diff.max())
    mean_abs = float(abs_diff.mean())
    frac_within = float((abs_diff <= atol).sum() / abs_diff.size)

    print(f"{name} (position-based):")
    print(f"  shape             : {old_df.shape}")
    print(f"  max abs diff      : {max_abs: .6e}")
    print(f"  mean abs diff     : {mean_abs: .6e}")
    print(f"  fraction |Δ|<=atol: {frac_within: .3f}")
    print()


def compare_by_labels_or_position(name: str, old_df: pd.DataFrame, new_df: pd.DataFrame,
                                  atol: float = 1e-8) -> None:
    """
    Try label-based comparison first (index & columns intersection).
    If no overlap but shapes match, fall back to position-based comparison.
    """
    print(f"===== Comparing {name} =====")
    print(f"Old shape: {old_df.shape}, New shape: {new_df.shape}")

    # Label-based comparison
    common_index = old_df.index.intersection(new_df.index)
    common_cols = old_df.columns.intersection(new_df.columns)

    if len(common_index) > 0 and len(common_cols) > 0:
        old_sub = old_df.loc[common_index, common_cols].sort_index().sort_index(axis=1)
        new_sub = new_df.loc[common_index, common_cols].sort_index().sort_index(axis=1)

        diff = old_sub.values - new_sub.values
        abs_diff = np.abs(diff)

        max_abs = float(abs_diff.max())
        mean_abs = float(abs_diff.mean())
        frac_within = float((abs_diff <= atol).sum() / abs_diff.size)

        print(f"{name} (label-based):")
        print(f"  common index size : {len(common_index)}")
        print(f"  common columns    : {len(common_cols)}")
        print(f"  max abs diff      : {max_abs: .6e}")
        print(f"  mean abs diff     : {mean_abs: .6e}")
        print(f"  fraction |Δ|<=atol: {frac_within: .3f}")
        print()

    else:
        print(f"[WARN] {name}: no overlapping index/columns to compare by labels.")

    # Position-based fallback
    if old_df.shape == new_df.shape:
        compare_by_position(name, old_df, new_df, atol=atol)
    else:
        print(f"[WARN] {name}: shapes differ, skipping position-based comparison.\n")


# ---------- Main script ----------

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    old_dir = base_dir  # old files in quant/
    new_dir = base_dir / "generated_data"  # new files in quant/generated_data

    print(f"[INFO] Base directory: {base_dir}")
    print(f"[INFO] Old dir       : {old_dir}")
    print(f"[INFO] New dir       : {new_dir}")

    # ---------- 1) SP500 returns timeseries ----------
    old_ts_path = old_dir / "sp500_timeseries_13-24.csv"
    new_ts_path = new_dir / "sp500_timeseries_13-24.csv"

    old_ts = pd.read_csv(old_ts_path, index_col=0)
    new_ts = pd.read_csv(new_ts_path, index_col=0)

    print(f"[INFO] Loaded old sp500_timeseries from {old_ts_path} with shape {old_ts.shape}")
    print(f"[INFO] Loaded new sp500_timeseries from {new_ts_path} with shape {new_ts.shape}\n")

    compare_by_labels_or_position("SP500 returns timeseries", old_ts, new_ts, atol=1e-8)

    # ---------- 2) Performance summaries ----------
    old_perf_path = old_dir / "sp500_performance_summaries.csv"
    new_perf_path = new_dir / "sp500_performance_summaries.csv"

    old_perf = pd.read_csv(old_perf_path, index_col=0)
    new_perf = pd.read_csv(new_perf_path, index_col=0)

    print(f"[INFO] Loaded old performance_summaries from {old_perf_path} with shape {old_perf.shape}")
    print(f"[INFO] Loaded new performance_summaries from {new_perf_path} with shape {new_perf.shape}")

    # Old file was originally saved transposed (2 x N); new is (N x 2).
    if old_perf.shape[0] <= 5 and old_perf.shape[1] > old_perf.shape[0]:
        old_perf = old_perf.T
        print(f"[INFO] Transposed old performance_summaries for comparison. New shape: {old_perf.shape}\n")
    else:
        print()

    compare_by_labels_or_position("Performance summaries", old_perf, new_perf, atol=1e-10)

    # ---------- 3) Raw covariance ----------
    old_raw_path = old_dir / "sp500_raw_cov_matrix.csv"
    new_raw_path = new_dir / "sp500_raw_cov_matrix.csv"

    old_raw = pd.read_csv(old_raw_path, index_col=0)
    new_raw = pd.read_csv(new_raw_path, index_col=0)

    print(f"[INFO] Loaded old raw covariance from {old_raw_path} with shape {old_raw.shape}")
    print(f"[INFO] Loaded new raw covariance from {new_raw_path} with shape {new_raw.shape}\n")

    compare_by_labels_or_position("Raw covariance", old_raw, new_raw, atol=1e-8)

    covariance_diagnostics("Σ_raw (old)", old_raw)
    covariance_diagnostics("Σ_raw (new)", new_raw)

    # ---------- 4) Adjusted covariance ----------
    old_adj_path = old_dir / "sp500_adjusted_cov_matrix.csv"
    new_adj_path = new_dir / "sp500_adjusted_cov_matrix.csv"

    old_adj = pd.read_csv(old_adj_path, index_col=0)
    new_adj = pd.read_csv(new_adj_path, index_col=0)

    print(f"[INFO] Loaded old adjusted covariance from {old_adj_path} with shape {old_adj.shape}")
    print(f"[INFO] Loaded new adjusted covariance from {new_adj_path} with shape {new_adj.shape}\n")

    compare_by_labels_or_position("Adjusted covariance", old_adj, new_adj, atol=1e-8)

    covariance_diagnostics("Σ_adjusted (old)", old_adj)
    covariance_diagnostics("Σ_adjusted (new)", new_adj)

    # ---------- 5) SPY returns timeseries ----------
    old_spy_path = old_dir / "spy_timeseries_13-24.csv"
    new_spy_path = new_dir / "spy_timeseries_13-24.csv"

    old_spy = pd.read_csv(old_spy_path, index_col=0)
    new_spy = pd.read_csv(new_spy_path, index_col=0)

    print(f"[INFO] Loaded old SPY timeseries from {old_spy_path} with shape {old_spy.shape}")
    print(f"[INFO] Loaded new SPY timeseries from {new_spy_path} with shape {new_spy.shape}\n")

    compare_by_labels_or_position("SPY returns timeseries", old_spy, new_spy, atol=1e-10)

    print("[INFO] Comparison complete.")
