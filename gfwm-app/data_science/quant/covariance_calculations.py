"""
covariance_calculations.py

Risk model layer:
    - builds raw (sample) covariance
    - builds shrunk + PSD covariance via Ledoit–Wolf
    - OPTIONAL: can apply downside-weighting to returns before covariance
    - saves matrices into generated_data/ for Markowitz, Michaud, MC, ESG

Assumptions:
    - inputs are DAILY log-returns (not annualized)
    - covariances are also in daily units
    - annualization (×252, ×sqrt(252)) is done downstream (Markowitz/reporting)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any

from sklearn.covariance import LedoitWolf
import warnings

# All generated artifacts live here (folder already exists)
GENERATED_DIR = Path(__file__).resolve().parent / "generated_data"

# Default input returns file produced by preprocessing.py
DEFAULT_RETURNS_PATH = GENERATED_DIR / "sp500_timeseries_13-24.csv"


# -----------------------------------------------------------------------------
# OPTIONAL: downside-weighting of returns (non-standard, experimental)
# -----------------------------------------------------------------------------
def apply_downside_weighting(
    returns_df: pd.DataFrame,
    down_weight: float = 1.1,
    up_weight: float = 0.9,
) -> pd.DataFrame:
    """
    Optional pre-processing step that emphasizes downside moves.

    For each return r:
        if r <= 0: r' = r * down_weight
        else:      r' = r * up_weight

    This is NOT industry standard. It's a behavioral/experimental tweak that
    makes negative returns count more in the covariance and positive returns
    count slightly less. We keep it available for experiments, but DO NOT
    enable it by default in the main pipeline.
    """
    scaled = np.where(returns_df <= 0, returns_df * down_weight, returns_df * up_weight)
    return pd.DataFrame(scaled, index=returns_df.index, columns=returns_df.columns)


# -----------------------------------------------------------------------------
# Core estimators
# -----------------------------------------------------------------------------
def estimate_sample_covariance(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate the sample covariance matrix Σ_raw from daily log returns.

    This is your "realized risk" estimate: no shrinkage, no clipping.
    It is often too noisy for portfolio optimization in large universes,
    but it's excellent for diagnostics and for comparing against the shrunk
    covariance.
    """
    cov = returns_df.cov()  # pairwise complete observations, ddof=1
    return cov


def estimate_shrunk_covariance_lw(
    returns_df: pd.DataFrame,
    epsilon: float = 1e-8,
    min_obs: int = 250,
) -> pd.DataFrame:
    """
    Estimate a shrunk, PSD covariance matrix Σ_LW+PSD using Ledoit–Wolf.

    Steps:
        1. Drop any rows with NaNs to enforce a common data window.
        2. Fit Ledoit–Wolf on the cleaned returns (rows = days, cols = tickers).
        3. Extract Σ_shrunk from the estimator.
        4. Eigen-decompose Σ_shrunk = V Λ Vᵀ.
        5. Clip small/negative eigenvalues at epsilon: λ_i := max(λ_i, epsilon).
        6. Reconstruct Σ_psd = V diag(λ_clipped) Vᵀ.

    The result is numerically stable and positive semidefinite, which is
    exactly what Markowitz, Michaud, and MC simulations need.
    """
    # 1) Drop rows with any NaNs — common window across assets for LW
    n_rows_before = returns_df.shape[0]
    clean_df = returns_df.dropna(axis=0, how="any")
    n_rows_after = clean_df.shape[0]

    if n_rows_after < n_rows_before:
        frac_dropped = (n_rows_before - n_rows_after) / max(n_rows_before, 1)
        warnings.warn(
            f"[LW] Dropped {n_rows_before - n_rows_after} rows with NaNs "
            f"({frac_dropped:.1%} of observations) before Ledoit–Wolf."
        )

    if n_rows_after < min_obs:
        warnings.warn(
            f"[LW] Only {n_rows_after} clean observations left after dropna; "
            f"this is below min_obs={min_obs}. Covariance estimates may be noisy."
        )

    if n_rows_after == 0:
        raise ValueError(
            "[LW] No observations left after dropping NaN rows. "
            "Check preprocessing / data coverage."
        )

    X = clean_df.values  # shape (n_samples, n_assets)

    # 2) Ledoit–Wolf shrinkage
    lw = LedoitWolf().fit(X)
    sigma_shrunk = lw.covariance_  # numpy array, shape (n_assets, n_assets)

    # 3–6) Eigen-decomposition and PSD repair
    eigvals, eigvecs = np.linalg.eigh(sigma_shrunk)
    eigvals_clipped = np.clip(eigvals, epsilon, None)
    sigma_psd = (eigvecs * eigvals_clipped) @ eigvecs.T

    tickers = list(returns_df.columns)
    sigma_psd_df = pd.DataFrame(sigma_psd, index=tickers, columns=tickers)

    # Sanity check: no NaNs or infs
    if not np.isfinite(sigma_psd_df.values).all():
        raise ValueError("[LW] Non-finite values found in shrunk covariance matrix.")

    return sigma_psd_df


def covariance_diagnostics(
    cov_matrix: pd.DataFrame,
    name: str = "covariance",
) -> Dict[str, Any]:
    """
    Compute simple diagnostics for a covariance matrix:
        - min eigenvalue
        - max eigenvalue
        - condition number (max / min)

    Returns a dict of metrics and prints them. This is extremely helpful
    for debugging and for ensuring Σ is well-conditioned before optimization.
    """
    S = cov_matrix.values
    eigvals = np.linalg.eigvalsh(S)

    min_eig = float(eigvals.min())
    max_eig = float(eigvals.max())
    cond_num = float(max_eig / max(min_eig, 1e-16))  # avoid div-by-zero

    info = {
        "name": name,
        "min_eigenvalue": min_eig,
        "max_eigenvalue": max_eig,
        "condition_number": cond_num,
    }

    print(f"[Diagnostics: {name}]")
    print(f"  min eigenvalue    : {min_eig:.4e}")
    print(f"  max eigenvalue    : {max_eig:.4e}")
    print(f"  condition number  : {cond_num:.4e}")
    print()

    return info


# -----------------------------------------------------------------------------
# Build-and-save wrapper
# -----------------------------------------------------------------------------
def build_and_save_covariance_matrices(
    returns_df: pd.DataFrame,
    out_dir: Path | str = GENERATED_DIR,
    use_downside_weighting: bool = False,
    epsilon: float = 1e-8,
    min_obs: int = 250,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    High-level ETL wrapper for the risk model layer.

    Inputs:
        returns_df : daily log return panel from preprocessing.py
                     (index = dates, columns = tickers)
        out_dir    : where to save the resulting CSVs
        use_downside_weighting :
                     if True, first apply apply_downside_weighting() to returns
                     before any covariance estimation. By default this is False
                     to match industry-standard practice.
        epsilon    : eigenvalue floor for PSD clipping in Σ_LW+PSD.
        min_obs    : minimum number of clean observations (rows) after dropna
                     before LW; below this, a warning is issued.

    Outputs (also saved to CSV in `out_dir`):
        sp500_raw_cov_matrix.csv     : sample covariance Σ_raw
        sp500_adjusted_cov_matrix.csv: shrunk + PSD covariance Σ_LW+PSD

    Returns:
        (cov_raw_df, cov_psd_df)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optional: downside-weighting experiment
    if use_downside_weighting:
        print("[INFO] Applying downside weighting to returns before covariance.")
        returns_for_cov = apply_downside_weighting(returns_df)
    else:
        returns_for_cov = returns_df

    # 1) Sample covariance (raw)
    cov_raw_df = estimate_sample_covariance(returns_for_cov)
    covariance_diagnostics(cov_raw_df, name="Σ_raw (sample)")

    if not np.isfinite(cov_raw_df.values).all():
        raise ValueError("Non-finite values in sample covariance matrix Σ_raw.")

    raw_path = out_dir / "sp500_raw_cov_matrix.csv"
    cov_raw_df.to_csv(raw_path, index=True)
    print(f"[INFO] Saved raw covariance to: {raw_path}")

    # 2) Shrunk + PSD covariance via Ledoit–Wolf
    cov_psd_df = estimate_shrunk_covariance_lw(
        returns_for_cov,
        epsilon=epsilon,
        min_obs=min_obs,
    )
    covariance_diagnostics(cov_psd_df, name="Σ_LW+PSD (shrunk)")

    adj_path = out_dir / "sp500_adjusted_cov_matrix.csv"
    cov_psd_df.to_csv(adj_path, index=True)
    print(f"[INFO] Saved adjusted (shrunk + PSD) covariance to: {adj_path}")

    return cov_raw_df, cov_psd_df


# -----------------------------------------------------------------------------
# Loader helpers so other modules don't hard-code file paths
# -----------------------------------------------------------------------------
def load_returns_panel(
    filepath: Path | str = DEFAULT_RETURNS_PATH,
) -> pd.DataFrame:
    """
    Load the returns panel produced by preprocessing.py.

    This is the canonical daily log returns DataFrame that all covariances,
    optimizations, and simulations should be based on.
    """
    filepath = Path(filepath)
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return df


def load_raw_covariance(
    filepath: Path | str = GENERATED_DIR / "sp500_raw_cov_matrix.csv",
) -> pd.DataFrame:
    """
    Load the sample covariance matrix Σ_raw from CSV.
    """
    filepath = Path(filepath)
    df = pd.read_csv(filepath, index_col=0)
    return df


def load_adjusted_covariance(
    filepath: Path | str = GENERATED_DIR / "sp500_adjusted_cov_matrix.csv",
) -> pd.DataFrame:
    """
    Load the shrunk + PSD covariance matrix Σ_LW+PSD from CSV.
    This is the matrix you should use as the risk model for Markowitz,
    Michaud, Monte Carlo, and ESG-aware optimizations.
    """
    filepath = Path(filepath)
    df = pd.read_csv(filepath, index_col=0)
    return df


# -----------------------------------------------------------------------------
# CLI / script entrypoint (optional)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    """
    If you run this file directly, it will:
        1. Load the canonical returns panel from generated_data/sp500_timeseries_13-24.csv
        2. Build Σ_raw and Σ_LW+PSD
        3. Save both into generated_data/
    """
    print("[INFO] Loading returns panel from default path...")
    returns = load_returns_panel(DEFAULT_RETURNS_PATH)
    print(f"[INFO] Loaded returns panel with shape: {returns.shape}")

    build_and_save_covariance_matrices(
        returns_df=returns,
        out_dir=GENERATED_DIR,
        use_downside_weighting=False,  # toggle to True ONLY for experiments
        epsilon=1e-8,
        min_obs=250,
    )
    print("[INFO] Covariance matrices built and saved.")
