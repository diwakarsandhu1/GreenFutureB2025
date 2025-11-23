from pathlib import Path
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------

# This file lives in: data_science/preprocess_and_filter/
THIS_DIR = Path(__file__).resolve().parent
DATA_SCIENCE_DIR = THIS_DIR.parent

DATA_DIR = DATA_SCIENCE_DIR / "data" / "raw_data"
OUTPUT_DIR = DATA_SCIENCE_DIR / "data" / "universal_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRADING_DAYS_PER_YEAR = 252
EXCLUDED_TICKERS = {"PEAK", "PXD", "WRK", "CDAY", "FLT", "GOOG", "FOX", "NWS"}

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_raw_panel(filepath: Path) -> pd.DataFrame:
    """Load a raw SP500-style timeseries CSV and return a cleaned DataFrame (dates × tickers)."""
    df = pd.read_csv(filepath).dropna(axis=1, how="all")
    df = df.set_index(df.columns[0]).T
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def compute_log_returns(price_panel: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns."""
    return np.log(price_panel / price_panel.shift(1)).iloc[1:]


# ---------------------------------------------------------------------
# Build universal datasets
# ---------------------------------------------------------------------

def build_universal_data():
    """
    Builds:
      - tickers_to_keep.csv
      - sp500_timeseries_13-24.csv
      - spy_timeseries_13-24.csv
      - sp500_performance_summaries.csv
    Outputs go to data/universal_data/
    """
    print("[universal] Building universal data...")

    # Input files
    esg_path = DATA_DIR / "Refinitiv ESG Final Data for Analysis.csv"
    company_path = DATA_DIR / "Company_data.csv"
    sp500_raw_13_18 = DATA_DIR / "SP500_raw_timeseries_1-1-13--12-31-18.csv"
    sp500_raw_19_24 = DATA_DIR / "SP500_raw_timeseries_1-1-19--11-1-24.csv"
    spy_raw = DATA_DIR / "Spy_raw_timeseries_13-24.csv"

    # --- Build ticker universe ---
    esg_df = pd.read_csv(esg_path)
    companies = pd.read_csv(company_path)

    tickers_esg = esg_df["Symbol"].astype(str).str.upper()
    companies["ticker"] = companies["ticker"].astype(str).str.upper()
    seasoned = companies.loc[companies["days_since_ipo"] > 180, "ticker"]

    universe = sorted((set(tickers_esg) & set(seasoned)) - EXCLUDED_TICKERS)
    pd.Series(universe, name="ticker").to_csv(OUTPUT_DIR / "tickers_to_keep.csv", index=False)
    print(f"[universal] Saved tickers_to_keep.csv ({len(universe)} tickers)")

    # --- Build SP500 returns ---
    p1 = load_raw_panel(sp500_raw_13_18)
    p2 = load_raw_panel(sp500_raw_19_24)
    prices = pd.concat([p1, p2]).sort_index().drop_duplicates()
    prices = prices.filter(items=universe)
    sp500_rets = compute_log_returns(prices)
    sp500_rets.reset_index().rename(columns={"index": "date"}).to_csv(
        OUTPUT_DIR / "sp500_timeseries_13-24.csv", index=False
    )

    print(f"[universal] Saved sp500_timeseries_13-24.csv {sp500_rets.shape}")

    # --- Build SPY returns ---
    spy_prices = load_raw_panel(spy_raw)
    if spy_prices.shape[1] != 1:
        spy_prices = spy_prices.rename(columns={spy_prices.columns[0]: "SPY"})[["SPY"]]
    spy_rets = compute_log_returns(spy_prices)
    spy_rets.to_csv(OUTPUT_DIR / "spy_timeseries_13-24.csv", index=True)
    print(f"[universal] Saved spy_timeseries_13-24.csv {spy_rets.shape}")

    # --- Performance summaries ---
    mean_d = sp500_rets.mean(axis=0)
    vol_d = sp500_rets.std(axis=0)
    perf = pd.DataFrame({
        "mean_log_return": mean_d * TRADING_DAYS_PER_YEAR,
        "standard_deviation": vol_d * np.sqrt(TRADING_DAYS_PER_YEAR),
    })
    perf.index.name = "ticker"
    perf.T.to_csv(OUTPUT_DIR / "sp500_performance_summaries.csv", index=True)
    print(f"[universal] Saved sp500_performance_summaries.csv ({len(perf)} rows)")

    print("[universal] ✅ Universal data build complete.")


def ensure_universal_data():
    """Only rebuild if files are missing."""
    required = [
        OUTPUT_DIR / "tickers_to_keep.csv",
        OUTPUT_DIR / "sp500_timeseries_13-24.csv",
        OUTPUT_DIR / "spy_timeseries_13-24.csv",
        OUTPUT_DIR / "sp500_performance_summaries.csv",
    ]
    missing = [f for f in required if not f.exists()]
    if missing:
        print(f"[universal] Missing: {[p.name for p in missing]}")
        build_universal_data()
    else:
        print("[universal] All universal data present.")
