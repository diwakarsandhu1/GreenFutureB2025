import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------
# Configuration / constants
# ---------------------------------------------------------------------

EXCLUDED_TICKERS = [
    "PEAK",  # dropped / name changes
    "PXD",   # acquired by XOM
    "WRK",   # foreign listing
    "CDAY",  # renamed to DAY
    "FLT",   # foreign listing
    "GOOG",  # use GOOGL instead
    "FOX",   # dual class with FOXA
    "NWS",   # dual class with NWSA
]

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------
# 1) Universe construction
# ---------------------------------------------------------------------

def build_investable_universe(
    esg_path: str,
    company_path: str,
    min_days_since_ipo: int = 180,
    excluded_tickers=None,
) -> pd.Index:
    """
    Build the canonical investable universe of tickers.

    - Reads ESG ticker list.
    - Reads company_data with days_since_ipo.
    - Filters by IPO seasoning.
    - Applies manual exclusions (dual classes, weird listings, M&A, etc.).
    - Intersects ESG tickers with those that have sufficient price / company data.
    """
    if excluded_tickers is None:
        excluded_tickers = EXCLUDED_TICKERS

    esg_df = pd.read_csv(esg_path)
    esg_tickers = esg_df["Symbol"].astype(str).str.upper()

    company_data = pd.read_csv(company_path)
    company_data["ticker"] = company_data["ticker"].astype(str).str.upper()

    # Tickers with enough trading history
    seasoned = company_data.loc[
        company_data["days_since_ipo"] > min_days_since_ipo, "ticker"
    ]

    # Apply manual exclusions to ESG list
    esg_tickers = esg_tickers[~esg_tickers.isin(excluded_tickers)]

    # Final universe = intersection of ESG + seasoned tickers
    universe = pd.Index(sorted(set(esg_tickers) & set(seasoned)), name="ticker")

    return universe


# ---------------------------------------------------------------------
# 2) Raw price panels (per era)
# ---------------------------------------------------------------------

def load_raw_price_panel(filepath: str) -> pd.DataFrame:
    """
    Load a raw SP500 timeseries file and clean it.

    Expected raw file layout:
    - rows: tickers (with a 'ticker' column)
    - columns: dates (plus the 'ticker' column)

    Cleaning steps:
    - drop all-NaN date columns (holidays/no trading)
    - set index to ticker, transpose to dates x tickers
    - sort dates ascending
    """
    df = pd.read_csv(filepath)

    # drop completely empty date columns
    df = df.dropna(axis=1, how="all")

    # set index to ticker and transpose -> dates x tickers
    df = df.set_index("ticker").T
    df.index.name = "date"

    # ensure chronological order
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    return df


# ---------------------------------------------------------------------
# 3) Stitch eras into a single price panel
# ---------------------------------------------------------------------

def build_price_panel(filepaths, tickers: pd.Index) -> pd.DataFrame:
    """
    Build the canonical price panel for the given universe.

    - Loads each era using load_raw_price_panel.
    - Concatenates along the date axis.
    - Deduplicates and sorts dates.
    - Filters columns to the investable universe.
    """
    panels = [load_raw_price_panel(path) for path in filepaths]

    price_panel = pd.concat(panels, axis=0)

    # Deduplicate + sort dates to avoid misalignment
    price_panel = price_panel[~price_panel.index.duplicated(keep="first")]
    price_panel = price_panel.sort_index()

    # Only keep tickers in our universe
    price_panel = price_panel.filter(items=tickers)

    return price_panel


# ---------------------------------------------------------------------
# 4) Returns & performance summaries
# ---------------------------------------------------------------------

def compute_log_returns(price_panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns from a price panel.

    r_t = log(P_t / P_{t-1}) for each ticker.
    """
    simple_returns = price_panel / price_panel.shift(1)
    log_returns = np.log(simple_returns).iloc[1:]
    return log_returns


def compute_performance_summaries(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    From daily log returns, compute:
    - annualized mean log return
    - annualized volatility
    """
    mean_daily = returns_df.mean(axis=0)
    std_daily = returns_df.std(axis=0)

    annualized_mean = mean_daily * TRADING_DAYS_PER_YEAR
    annualized_vol = std_daily * np.sqrt(TRADING_DAYS_PER_YEAR)

    perf = pd.DataFrame(
        {
            "mean_log_return": annualized_mean,
            "standard_deviation": annualized_vol,
        }
    )
    perf.index.name = "ticker"
    return perf


# ---------------------------------------------------------------------
# 5) SPY benchmark returns
# ---------------------------------------------------------------------

def build_spy_returns(spy_filepath: str):
    """
    Build SPY log returns and basic annualized stats.

    Returns:
    - spy_returns: DataFrame with one column 'SPY' of daily log returns
    - spy_annual_mean: float
    - spy_annual_vol: float
    """
    spy_raw = pd.read_csv(spy_filepath).dropna(axis=1, how="all")

    # Expect same layout: tickers as rows, dates as columns
    spy_raw = spy_raw.set_index(spy_raw.columns[0]).T
    spy_raw.index.name = "date"
    spy_raw.index = pd.to_datetime(spy_raw.index)
    spy_raw = spy_raw.sort_index()

    spy_raw.rename(columns={spy_raw.columns[0]: "SPY"}, inplace=True)

    simple_returns = spy_raw / spy_raw.shift(1)
    spy_returns = np.log(simple_returns).iloc[1:]

    spy_annual_mean = spy_returns.mean().iloc[0] * TRADING_DAYS_PER_YEAR
    spy_annual_vol = spy_returns.std().iloc[0] * np.sqrt(TRADING_DAYS_PER_YEAR)

    return spy_returns, float(spy_annual_mean), float(spy_annual_vol)


# ---------------------------------------------------------------------
# 6) High-level ETL to build & save core data
# ---------------------------------------------------------------------

def build_and_save_core_data(
    esg_path: str = "gfwm/data_science/raw_data/Refinitiv ESG Final Data for Analysis.csv",
    company_path: str = "gfwm/data_science/raw_data/company_data.csv",
    price_filepaths=None,
    spy_filepath: str | None = "spy_raw_timeseries_13-24.csv",
    out_dir: str = "generated_data",
):
    """
    High-level pipeline that:
    - builds the investable universe
    - builds the price panel
    - computes daily log returns
    - computes performance summaries
    - optionally builds SPY returns
    - saves all results to ./generated_data/
    """
    if price_filepaths is None:
        price_filepaths = [
            "SP500_raw_timeseries_1-1-13--12-31-18.csv",
            "SP500_raw_timeseries_1-1-19--11-1-24.csv",
        ]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)  # harmless if already exists

    # 1) Universe
    tickers = build_investable_universe(esg_path, company_path)

    # 2) Price panel for that universe
    price_panel = build_price_panel(price_filepaths, tickers)

    # 3) Daily log returns
    returns_df = compute_log_returns(price_panel)

    # 4) Performance summaries
    perf_df = compute_performance_summaries(returns_df)

    # 5) SPY (optional)
    spy_returns = None
    spy_mean = None
    spy_vol = None
    if spy_filepath is not None:
        spy_returns, spy_mean, spy_vol = build_spy_returns(spy_filepath)
        spy_returns.to_csv(out_dir / "spy_timeseries_13-24.csv", index=True)

    # 6) Save main outputs into generated_data/
    returns_df.to_csv(out_dir / "sp500_timeseries_13-24.csv", index=True)
    perf_df.to_csv(out_dir / "sp500_performance_summaries.csv", index=True)
    tickers.to_series().to_csv(out_dir / "tickers_to_keep.csv", index=False)

    return tickers, returns_df, perf_df, spy_returns, spy_mean, spy_vol


# ---------------------------------------------------------------------
# 7) Loader helpers
# ---------------------------------------------------------------------

def load_returns_panel(path: str = "generated_data/sp500_timeseries_13-24.csv") -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


def load_performance_summaries(path: str = "generated_data/sp500_performance_summaries.csv") -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def load_tickers(path: str = "generated_data/tickers_to_keep.csv") -> pd.Index:
    df = pd.read_csv(path)
    # assume single column of tickers
    col = df.columns[0]
    return pd.Index(df[col].astype(str).values, name="ticker")

def ensure_preprocessing_artifacts():
    """
    Check for core data outputs (timeseries, performance summaries, etc.).
    If any are missing, rebuild them using local raw_data files.

    Safe to call from anywhere — does nothing if all files already exist.
    """
    this_dir = Path(__file__).resolve().parent           # .../data_science/quant
    data_dir = this_dir.parent                           # .../data_science
    raw_data = data_dir / "raw_data"                     # <-- updated line
    generated = this_dir / "generated_data"
    generated.mkdir(parents=True, exist_ok=True)

    required = [
        generated / "sp500_timeseries_13-24.csv",
        generated / "sp500_performance_summaries.csv",
        generated / "tickers_to_keep.csv",
        generated / "spy_timeseries_13-24.csv",
    ]

    missing = [f for f in required if not f.exists()]
    if missing:
        print(f"[preprocessing] Missing core data: {[f.name for f in missing]}")
        # ✅ Now point to raw_data folder for input CSVs
        esg_path = raw_data / "Refinitiv ESG Final Data for Analysis.csv"
        company_path = raw_data / "Company_data.csv"
        price_filepaths = [
            raw_data / "SP500_raw_timeseries_1-1-13--12-31-18.csv",
            raw_data / "SP500_raw_timeseries_1-1-19--11-1-24.csv",
        ]
        spy_filepath = raw_data / "Spy_raw_timeseries_13-24.csv"

        build_and_save_core_data(
            esg_path=str(esg_path),
            company_path=str(company_path),
            price_filepaths=[str(p) for p in price_filepaths],
            spy_filepath=str(spy_filepath),
            out_dir=str(generated),
        )
        print("[preprocessing] Core data built.")
    else:
        print("[preprocessing] Core data already present.")

# ---------------------------------------------------------------------
# Script entrypoint for local testing
# ---------------------------------------------------------------------

if __name__ == "__main__":
    build_and_save_core_data()
