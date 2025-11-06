import pandas as pd
import numpy as np
from pathlib import Path

# -----------------------------------------------------------------------------
# Paths / constants
# -----------------------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
DATA_SCIENCE_DIR = THIS_DIR.parents[1]
UNIVERSAL_DIR = DATA_SCIENCE_DIR / "data" / "universal_data"
MODEL_DIR = DATA_SCIENCE_DIR / "data" / "covariance_models" / "baseline_markowitz"
DEFAULT_RAW_COV_PATH = MODEL_DIR / "sp500_raw_cov_matrix.csv"
DEFAULT_ADJ_COV_PATH = MODEL_DIR / "sp500_adjusted_cov_matrix.csv"

def build_and_save_covariance_matrices():
    """
    Baseline Markowitz covariance builder.
    Logic identical to the original notebook version.
    """

    # ------------------------------------------------------------------
    # Load universal data
    # ------------------------------------------------------------------
    print("[baseline] Loading universal data...")

    esg_tickers = pd.read_csv(UNIVERSAL_DIR / "tickers_to_keep.csv")["ticker"]
    tickers = esg_tickers.sort_values(ignore_index=True)

    # Load timeseries
    timeseries = pd.read_csv(UNIVERSAL_DIR / "sp500_timeseries_13-24.csv", index_col=0, parse_dates=True)
    timeseries = timeseries.filter(items=tickers)

    # Load SPY (for reference only)
    spy_timeseries = pd.read_csv(UNIVERSAL_DIR / "spy_timeseries_13-24.csv", index_col=0, parse_dates=True)
    spy_mean_log_return = spy_timeseries.mean() * 252
    spy_annual_volatility = spy_timeseries.std() * np.sqrt(252)
    print(f"[baseline] SPY mean={spy_mean_log_return.iloc[0]:.4f}, vol={spy_annual_volatility.iloc[0]:.4f}")

    # ------------------------------------------------------------------
    # Covariance calculations
    # ------------------------------------------------------------------
    first_valid_dates = [timeseries[t].first_valid_index() for t in tickers]

    def covariances(x, y):
        n = len(x)

        x_bar = np.mean(x)
        y_bar = np.mean(y)    
        
        raw_covariance = np.sum((x-x_bar) * (y-y_bar)) / (n-1)
        
        
        # weight losers more heavily than winners
        x_prime = np.where(x <= 0, x * 1.1, x * 0.9)
        y_prime = np.where(y <= 0, y * 1.1, y * 0.9)
        
        x_prime_bar = np.mean(x_prime)
        y_prime_bar = np.mean(y_prime)
        
        weighted_covariance = np.sum((x_prime - x_prime_bar) * 
                                    (y_prime - y_prime_bar)) / (n-1)
        #print(raw_covariance, weighted_covariance, weighted_covariance/raw_covariance)
        return raw_covariance, weighted_covariance

    print("[baseline] Calculating covariance matrices...")
    raw_cov_matrix = pd.DataFrame(np.nan, index=tickers, columns=tickers)
    adjusted_cov_matrix = pd.DataFrame(np.nan, index=tickers, columns=tickers)

    for i in range(len(tickers)):
        for j in range(i, len(tickers)):
            # start at the later date after which both tickers have data
            start_date = pd.to_datetime(max(first_valid_dates[i], first_valid_dates[j]))
            
            ticker_i_slice = timeseries.loc[timeseries.index >= start_date, tickers[i]]
            ticker_j_slice = timeseries.loc[timeseries.index >= start_date, tickers[j]]
            
            # calculate and write both raw and adjusted (weighted) covariances
            raw, adj = covariances(ticker_i_slice, ticker_j_slice)
            
            raw_cov_matrix.iloc[i, j] = raw_cov_matrix.iloc[j, i] = raw
            adjusted_cov_matrix.iloc[i, j] = adjusted_cov_matrix.iloc[j, i] = adj


    std_devs = np.sqrt(np.diagonal(adjusted_cov_matrix.values))
    # Create a matrix of standard deviations (outer product of std_devs with itself)
    # Compute correlation matrix
    corr_matrix = adjusted_cov_matrix / np.outer(std_devs, std_devs)
    nan_count = adjusted_cov_matrix.isna().sum().sum()

    n = len(tickers)

    # diagonal matrix of volatilities
    vol_matrix = np.diag(np.diag(adjusted_cov_matrix.values))

    # matrix of average correlations, except for diagonal of 1s
    rho = np.mean(corr_matrix)
    average_corr_matrix = np.full((n, n), rho)
    np.fill_diagonal(average_corr_matrix, 1)

    # F = V C V
    structured_cov_matrix = vol_matrix @ average_corr_matrix @ vol_matrix

    delta = 0.5

    lw_adjusted_cov_matrix = (1-delta) * adjusted_cov_matrix + delta * structured_cov_matrix

    eigenvalues, eigenvectors = np.linalg.eigh(lw_adjusted_cov_matrix)

    # Clip eigenvalues smaller than epsilon
    eigenvalues = np.clip(eigenvalues, 1e-6, None)
    lw_adjusted_cov_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    #write dataframes to csv
    adjusted_cov_matrix_df = pd.DataFrame(lw_adjusted_cov_matrix, columns=tickers)
    adjusted_cov_matrix_df.set_index(tickers, inplace=True)
    adjusted_cov_matrix_df.rename_axis('ticker', inplace=True)

    adjusted_cov_matrix_df.to_csv(DEFAULT_ADJ_COV_PATH, index=True)
    raw_cov_matrix.set_index(tickers, inplace=True)
    raw_cov_matrix.rename_axis('ticker', inplace=True)

    raw_cov_matrix.to_csv(DEFAULT_RAW_COV_PATH, index = True)


def ensure_baseline_covariance_artifacts():
    """
    Ensures baseline Markowitz covariance matrices exist. Builds if missing.
    """
    data_root = Path(__file__).resolve().parents[1] / "data"
    model_dir = data_root / "covariance_models" / "baseline_markowitz"
    raw_csv = model_dir / "sp500_raw_cov_matrix.csv"
    adj_csv = model_dir / "sp500_adjusted_cov_matrix.csv"

    if not raw_csv.exists() or not adj_csv.exists():
        print("[baseline] Covariance matrices missing — rebuilding...")
        build_and_save_covariance_matrices()
    else:
        print("[baseline] Covariance matrices already present.")


if __name__ == "__main__":
    build_and_save_covariance_matrices()
