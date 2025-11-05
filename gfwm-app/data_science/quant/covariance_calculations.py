import os
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


# -----------------------------------------------------------------------------
# Paths / constants
# -----------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_RETURNS_PATH = os.path.join(
    BASE_DIR, "generated_data", "sp500_timeseries_13-24.csv"
)
DEFAULT_RAW_COV_PATH = os.path.join(
    BASE_DIR, "generated_data", "sp500_raw_cov_matrix.csv"
)
DEFAULT_ADJ_COV_PATH = os.path.join(
    BASE_DIR, "generated_data", "sp500_adjusted_cov_matrix.csv"
)

# Toggle if you want downside weighting baked into Σ
USE_DOWNSIDE_WEIGHTING = False


# -----------------------------------------------------------------------------
# Loading helpers
# -----------------------------------------------------------------------------

def load_returns_panel(path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the canonical returns panel (daily log returns, dates × tickers).

    By default, reads from generated_data/sp500_timeseries_13-24.csv as
    produced by preprocessing.py::build_and_save_core_data.
    """
    if path is None:
        path = DEFAULT_RETURNS_PATH

    print(f"[INFO] Loading returns panel from: {path}")
    returns = pd.read_csv(path, index_col=0, parse_dates=True)

    # Ensure sorted by date
    returns = returns.sort_index()

    # Make sure everything is numeric (coerce any stray strings to NaN)
    returns = returns.apply(pd.to_numeric, errors="coerce")

    print(f"[INFO] Loaded returns panel with shape: {returns.shape}")
    return returns


# -----------------------------------------------------------------------------
# Optional downside weighting
# -----------------------------------------------------------------------------

def get_downside_weighted_returns(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a simple heuristic tilt:
      - Down days (r <= 0) are scaled up by 1.1
      - Up days (r > 0) are scaled down by 0.9

    This is not industry standard, but matches your previous logic and
    can be toggled with USE_DOWNSIDE_WEIGHTING.
    """
    weighted = returns_df.copy()

    # r <= 0  →  1.1 * r
    mask_down = weighted <= 0
    weighted[mask_down] = weighted[mask_down] * 1.1

    # r > 0   →  0.9 * r
    mask_up = weighted > 0
    weighted[mask_up] = weighted[mask_up] * 0.9

    return weighted


# -----------------------------------------------------------------------------
# Hybrid NaN cleaning for covariance
# -----------------------------------------------------------------------------

def prepare_returns_for_covariance(
    returns_df: pd.DataFrame,
    min_ticker_coverage: float = 0.80,
    min_day_coverage: float = 0.95,
    fill_limit: int = 5,
) -> pd.DataFrame:
    """
    Hybrid cleaning for covariance estimation:

    1) Drop sparse tickers (low coverage across time).
    2) Lightly fill short gaps forward/backward per ticker.
    3) Drop sparse days (low cross-sectional coverage).
    4) Drop any remaining rows with NaNs.

    This avoids Ledoit–Wolf dropping the majority of rows while still
    ensuring a dense, well-behaved matrix.

    Parameters
    ----------
    returns_df : DataFrame
        Daily log returns (dates × tickers).
    min_ticker_coverage : float
        Minimum fraction of days a ticker must have non-NaN data to be kept.
    min_day_coverage : float
        Minimum fraction of tickers a given day must have non-NaN data to be kept.
    fill_limit : int
        Max number of consecutive NaNs to forward/backward fill. Longer gaps remain NaN.

    Returns
    -------
    DataFrame
        Cleaned returns matrix suitable for covariance estimation.
    """
    returns = returns_df.copy()

    # 1) Drop tickers with too little data
    ticker_cov = returns.notna().mean(axis=0)  # fraction of valid days per ticker
    good_tickers = ticker_cov[ticker_cov >= min_ticker_coverage].index
    n_dropped_tickers = returns.shape[1] - len(good_tickers)
    if n_dropped_tickers > 0:
        print(
            f"[CLEAN] Dropped {n_dropped_tickers} tickers "
            f"with coverage < {min_ticker_coverage:.0%}"
        )
    returns = returns[good_tickers]

    # 2) Lightly fill short gaps in time (forward then backward)
    if fill_limit is not None and fill_limit > 0:
        returns = returns.ffill(limit=fill_limit).bfill(limit=fill_limit)

    # 3) Drop days where too many tickers are missing
    day_cov = returns.notna().mean(axis=1)  # fraction of valid tickers per day
    good_days = day_cov[day_cov >= min_day_coverage].index
    n_dropped_days = returns.shape[0] - len(good_days)
    if n_dropped_days > 0:
        print(
            f"[CLEAN] Dropped {n_dropped_days} days "
            f"with cross-sectional coverage < {min_day_coverage:.0%}"
        )
    returns = returns.loc[good_days]

    # 4) Final drop of any residual NaN rows
    before_rows = returns.shape[0]
    returns = returns.dropna(axis=0, how="any")
    n_dropped_final = before_rows - returns.shape[0]
    if n_dropped_final > 0:
        print(f"[CLEAN] Dropped {n_dropped_final} rows with remaining NaNs.")

    print(
        f"[CLEAN] Final returns matrix for covariance: "
        f"{returns.shape[0]} days × {returns.shape[1]} tickers."
    )

    return returns


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------

def covariance_diagnostics(
    cov: pd.DataFrame | np.ndarray,
    name: str = "Σ",
) -> Tuple[float, float, float]:
    """
    Print simple diagnostics (min eigenvalue, max eigenvalue, condition number)
    for a covariance matrix.
    """
    if isinstance(cov, pd.DataFrame):
        mat = cov.values
    else:
        mat = np.asarray(cov)

    eigvals = np.linalg.eigvalsh(mat)
    min_eig = float(eigvals.min())
    max_eig = float(eigvals.max())
    if min_eig == 0:
        cond = np.inf
    else:
        cond = max_eig / min_eig

    print(f"[Diagnostics: {name}]")
    print(f"  min eigenvalue   : {min_eig: .4e}")
    print(f"  max eigenvalue   : {max_eig: .4e}")
    print(f"  condition number : {cond: .4e}")
    return min_eig, max_eig, cond


# -----------------------------------------------------------------------------
# Covariance estimation
# -----------------------------------------------------------------------------

def estimate_sample_covariance(
    returns_df: pd.DataFrame,
    use_downside_weighting: bool = False,
    min_ticker_coverage: float = 0.80,
    min_day_coverage: float = 0.95,
    fill_limit: int = 5,
) -> pd.DataFrame:
    """
    Estimate sample covariance on a hybrid-cleaned returns matrix.
    Optionally apply downside weighting before cleaning.
    """
    returns = returns_df.copy()

    if use_downside_weighting:
        print("[INFO] Applying downside weighting to returns before sample covariance.")
        returns = get_downside_weighted_returns(returns)

    # Hybrid NaN cleaning
    returns_clean = prepare_returns_for_covariance(
        returns,
        min_ticker_coverage=min_ticker_coverage,
        min_day_coverage=min_day_coverage,
        fill_limit=fill_limit,
    )

    cov = returns_clean.cov()
    return cov


def estimate_shrunk_covariance_lw(
    returns_df: pd.DataFrame,
    epsilon: float = 1e-8,
    use_downside_weighting: bool = False,
    min_ticker_coverage: float = 0.80,
    min_day_coverage: float = 0.95,
    fill_limit: int = 5,
) -> pd.DataFrame:
    """
    Estimate Ledoit–Wolf shrunk covariance with PSD eigenvalue clipping.

    Steps:
      1. Optional downside weighting.
      2. Hybrid NaN cleaning for a dense returns matrix.
      3. Ledoit–Wolf shrinkage on the cleaned matrix.
      4. Eigen-decomposition and eigenvalue clipping to enforce PSD.
      5. Reconstruct a DataFrame with ticker labels.
    """
    returns = returns_df.copy()

    if use_downside_weighting:
        print("[INFO] Applying downside weighting to returns before covariance.")
        returns = get_downside_weighted_returns(returns)

    # Hybrid NaN cleaning
    returns_clean = prepare_returns_for_covariance(
        returns,
        min_ticker_coverage=min_ticker_coverage,
        min_day_coverage=min_day_coverage,
        fill_limit=fill_limit,
    )

    # Ledoit–Wolf on cleaned data
    lw = LedoitWolf().fit(returns_clean.values)
    Sigma_shrunk = lw.covariance_
    tickers = returns_clean.columns.to_list()

    # Eigen-decomposition + clipping for PSD
    eigvals, eigvecs = np.linalg.eigh(Sigma_shrunk)
    min_eig_before = float(eigvals.min())
    if min_eig_before < 0:
        print(
            f"[INFO] LW covariance had negative eigenvalues "
            f"(min = {min_eig_before:.3e}); clipping to {epsilon:.1e}."
        )
    eigvals_clipped = np.clip(eigvals, epsilon, None)
    Sigma_psd = (eigvecs * eigvals_clipped) @ eigvecs.T

    # Wrap back into a DataFrame with proper labels
    cov_df = pd.DataFrame(Sigma_psd, index=tickers, columns=tickers)
    return cov_df


# -----------------------------------------------------------------------------
# High-level ETL: build & save covariance matrices
# -----------------------------------------------------------------------------

def build_and_save_covariance_matrices(
    returns_path: Optional[str] = None,
    out_raw_path: Optional[str] = None,
    out_adj_path: Optional[str] = None,
    use_downside_weighting: bool = USE_DOWNSIDE_WEIGHTING,
) -> None:
    """
    High-level routine:

      1. Load returns panel.
      2. Build sample covariance Σ_raw (hybrid-cleaned).
      3. Build shrunk + PSD covariance Σ_LW+PSD.
      4. Run diagnostics on both.
      5. Save to CSV in generated_data/.
    """
    # Paths
    if returns_path is None:
        returns_path = DEFAULT_RETURNS_PATH
    if out_raw_path is None:
        out_raw_path = DEFAULT_RAW_COV_PATH
    if out_adj_path is None:
        out_adj_path = DEFAULT_ADJ_COV_PATH

    # 1) Load returns
    returns_df = load_returns_panel(returns_path)

    # 2) Sample covariance (raw)
    Sigma_raw = estimate_sample_covariance(
        returns_df,
        use_downside_weighting=use_downside_weighting,
    )
    covariance_diagnostics(Sigma_raw, name="Σ_raw (sample)")

    # Save raw
    Sigma_raw.to_csv(out_raw_path, index=True)
    print(f"[INFO] Saved raw covariance to: {out_raw_path}")

    # 3) Ledoit–Wolf + PSD covariance (adjusted)
    Sigma_adj = estimate_shrunk_covariance_lw(
        returns_df,
        use_downside_weighting=use_downside_weighting,
    )
    covariance_diagnostics(Sigma_adj, name="Σ_LW+PSD (shrunk)")

    # Save adjusted
    Sigma_adj.to_csv(out_adj_path, index=True)
    print(f"[INFO] Saved adjusted (shrunk + PSD) covariance to: {out_adj_path}")

    print("[INFO] Covariance matrices built and saved.")


# -----------------------------------------------------------------------------
# CLI entrypoint
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    build_and_save_covariance_matrices()
