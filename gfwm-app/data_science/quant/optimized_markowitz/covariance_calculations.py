"""
Optimized Markowitz Covariance Builder

Reads universal data (returns, tickers, summaries) from:
    data_science/data/universal_data/

Writes model-specific covariance matrices to:
    data_science/data/covariance_models/optimized_markowitz/

Used to generate Σ_raw and Σ_LW+PSD for the optimized Markowitz model.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
from sklearn.covariance import LedoitWolf
import warnings

# -----------------------------------------------------------------------------
# Paths / constants
# -----------------------------------------------------------------------------

# This file lives in: data_science/quant/optimized_markowitz/
THIS_DIR = Path(__file__).resolve().parent
DATA_SCIENCE_DIR = THIS_DIR.parents[1]                      # .../data_science
UNIVERSAL_DIR = DATA_SCIENCE_DIR / "data" / "universal_data"
MODEL_DIR = DATA_SCIENCE_DIR / "data" / "covariance_models" / "optimized_markowitz"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_RETURNS_PATH = UNIVERSAL_DIR / "sp500_timeseries_13-24.csv"
DEFAULT_RAW_COV_PATH = MODEL_DIR / "sp500_raw_cov_matrix.csv"
DEFAULT_ADJ_COV_PATH = MODEL_DIR / "sp500_adjusted_cov_matrix.csv"

USE_DOWNSIDE_WEIGHTING = False

# -----------------------------------------------------------------------------
# Loading helpers
# -----------------------------------------------------------------------------

def load_returns_panel(path: Optional[Path] = None) -> pd.DataFrame:
    """Load daily log returns (dates × tickers) from universal_data."""
    path = Path(path or DEFAULT_RETURNS_PATH)
    print(f"[INFO] Loading returns panel from: {path}")
    returns = pd.read_csv(path, index_col=0, parse_dates=True)
    returns = returns.sort_index().apply(pd.to_numeric, errors="coerce")
    print(f"[INFO] Loaded returns panel with shape: {returns.shape}")
    return returns

# -----------------------------------------------------------------------------
# Optional downside weighting
# -----------------------------------------------------------------------------

def get_downside_weighted_returns(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Heuristic tilt: scale down up days, scale up down days."""
    weighted = returns_df.copy()
    weighted[weighted <= 0] *= 1.1
    weighted[weighted > 0] *= 0.9
    return weighted

# -----------------------------------------------------------------------------
# Cleaning for covariance estimation
# -----------------------------------------------------------------------------

def prepare_returns_for_covariance(
    returns_df: pd.DataFrame,
    min_ticker_coverage: float = 0.80,
    min_day_coverage: float = 0.95,
    fill_limit: int = 5,
) -> pd.DataFrame:
    """Hybrid cleaning to produce a dense matrix for covariance estimation."""
    returns = returns_df.copy()

    # Drop sparse tickers
    good_tickers = returns.notna().mean(axis=0)
    good_tickers = good_tickers[good_tickers >= min_ticker_coverage].index
    if len(good_tickers) < returns.shape[1]:
        print(f"[CLEAN] Dropped {returns.shape[1] - len(good_tickers)} sparse tickers")
    returns = returns[good_tickers]

    # Fill short gaps
    if fill_limit and fill_limit > 0:
        returns = returns.ffill(limit=fill_limit).bfill(limit=fill_limit)

    # Drop sparse days
    good_days = returns.notna().mean(axis=1)
    good_days = good_days[good_days >= min_day_coverage].index
    returns = returns.loc[good_days]

    # Drop remaining NaNs
    returns = returns.dropna(axis=0, how="any")

    print(f"[CLEAN] Final matrix: {returns.shape[0]} days × {returns.shape[1]} tickers.")
    return returns

# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------

def covariance_diagnostics(cov: pd.DataFrame | np.ndarray, name: str = "Σ"):
    """Print min/max eigenvalue and condition number."""
    mat = cov.values if isinstance(cov, pd.DataFrame) else np.asarray(cov)
    eigvals = np.linalg.eigvalsh(mat)
    min_eig, max_eig = float(eigvals.min()), float(eigvals.max())
    cond = np.inf if min_eig == 0 else max_eig / min_eig
    print(f"[Diagnostics: {name}]")
    print(f"  min eigenvalue   : {min_eig: .4e}")
    print(f"  max eigenvalue   : {max_eig: .4e}")
    print(f"  condition number : {cond: .4e}")

# -----------------------------------------------------------------------------
# Covariance estimation
# -----------------------------------------------------------------------------

def estimate_sample_covariance(
    returns_df: pd.DataFrame,
    use_downside_weighting: bool = False,
) -> pd.DataFrame:
    """Estimate Σ_raw on cleaned returns."""
    returns = get_downside_weighted_returns(returns_df) if use_downside_weighting else returns_df
    returns_clean = prepare_returns_for_covariance(returns)
    return returns_clean.cov()

def estimate_shrunk_covariance_lw(
    returns_df: pd.DataFrame,
    epsilon: float = 1e-8,
    use_downside_weighting: bool = False,
) -> pd.DataFrame:
    """Estimate Ledoit–Wolf shrunk covariance with PSD clipping."""
    returns = get_downside_weighted_returns(returns_df) if use_downside_weighting else returns_df
    returns_clean = prepare_returns_for_covariance(returns)
    lw = LedoitWolf().fit(returns_clean.values)
    Sigma_shrunk = lw.covariance_
    eigvals, eigvecs = np.linalg.eigh(Sigma_shrunk)
    eigvals_clipped = np.clip(eigvals, epsilon, None)
    Sigma_psd = (eigvecs * eigvals_clipped) @ eigvecs.T
    cov_df = pd.DataFrame(Sigma_psd, index=returns_clean.columns, columns=returns_clean.columns)
    return cov_df

# -----------------------------------------------------------------------------
# Build & Save Covariance Matrices
# -----------------------------------------------------------------------------

def build_and_save_covariance_matrices(
    returns_path: Optional[Path] = None,
    out_raw_path: Optional[Path] = None,
    out_adj_path: Optional[Path] = None,
    use_downside_weighting: bool = USE_DOWNSIDE_WEIGHTING,
) -> None:
    """Build and save Σ_raw and Σ_LW+PSD matrices."""

    returns_path = Path(returns_path or DEFAULT_RETURNS_PATH)
    out_raw_path = Path(out_raw_path or DEFAULT_RAW_COV_PATH)
    out_adj_path = Path(out_adj_path or DEFAULT_ADJ_COV_PATH)

    # 1) Load returns
    returns_df = load_returns_panel(returns_path)

    # 2) Raw covariance
    Sigma_raw = estimate_sample_covariance(returns_df, use_downside_weighting)
    covariance_diagnostics(Sigma_raw, name="Σ_raw (sample)")
    Sigma_raw.to_csv(out_raw_path, index=True)
    print(f"[INFO] Saved raw covariance to: {out_raw_path}")

    # 3) Shrunk + PSD covariance
    Sigma_adj = estimate_shrunk_covariance_lw(returns_df, use_downside_weighting=use_downside_weighting)
    covariance_diagnostics(Sigma_adj, name="Σ_LW+PSD (shrunk)")
    Sigma_adj.to_csv(out_adj_path, index=True)
    print(f"[INFO] Saved adjusted covariance to: {out_adj_path}")

    print("[INFO] ✅ Covariance matrices built and saved.")

# -----------------------------------------------------------------------------
# Ensure & CLI
# -----------------------------------------------------------------------------

def ensure_optimized_covariance_artifacts():
    """Ensure optimized Markowitz covariances exist (depends on universal data)."""
    raw_csv, adj_csv = DEFAULT_RAW_COV_PATH, DEFAULT_ADJ_COV_PATH
    if not Path(raw_csv).exists() or not Path(adj_csv).exists():
        print("[covariance] Missing matrices — rebuilding...")
        build_and_save_covariance_matrices()
    else:
        print("[covariance] Covariance matrices already present.")

if __name__ == "__main__":
    build_and_save_covariance_matrices()
