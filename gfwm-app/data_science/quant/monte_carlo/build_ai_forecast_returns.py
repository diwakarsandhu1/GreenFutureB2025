import pandas as pd
from pathlib import Path
import re
from rapidfuzz import process, fuzz

# =============================================================================
# PATH SETUP
# =============================================================================

BASE_DIR = Path(__file__).resolve().parents[2]   # data_science/
AI_DATA_DIR = BASE_DIR / "data" / "monte_carlo_ai_forecasting"
RAW_DATA_DIR = BASE_DIR / "data" / "raw_data"

FILE_1 = AI_DATA_DIR / "OLS 1.csv"
FILE_2 = AI_DATA_DIR / "OLS 2.csv"
FILE_3 = AI_DATA_DIR / "OLS 3.csv"

REFINITIV_DATA = RAW_DATA_DIR / "Refinitiv ESG Final Data For Analysis.csv"
OUTPUT = AI_DATA_DIR / "ai_forward_returns.csv"


# =============================================================================
# HELPERS
# =============================================================================

def normalize(s):
    if not isinstance(s, str):
        return ""
    s = s.upper()
    s = re.sub(r"[^A-Z0-9 ]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def to_forward_returns(df, forecast_cols, mode="compounded"):
    years = sorted([int(c.split("_")[1]) for c in forecast_cols])
    ordered_cols = [f"forecast_{y}" for y in years]

    df = df.copy()

    for idx, row in df.iterrows():
        prev_cum = 0.0

        for col in ordered_cols:
            C_t = max(float(row[col]), -0.999)       # Prevent <= -1 values

            if mode == "compounded":
                g_prev = max(1.0 + prev_cum, 1e-6)   # Avoid division by zero
                g_now  = max(1.0 + C_t, 1e-6)
                fwd = g_now / g_prev - 1.0
            else:
                fwd = C_t - prev_cum

            df.at[idx, col] = fwd
            prev_cum = C_t

    df["forecast_average"] = df[ordered_cols].mean(axis=1)
    return df


def load_ols(file_path):
    df = pd.read_csv(file_path)
    forecast_cols = [c for c in df.columns if c.startswith("forecast_")]
    if not forecast_cols:
        raise ValueError(f"No forecast columns in {file_path}")
    return df[["Ticker"] + forecast_cols], forecast_cols


# =============================================================================
# MAP OLS NAME → REFINITIV SYMBOL
# =============================================================================

def map_tickers(df):
    ref = pd.read_csv(REFINITIV_DATA)

    ref["clean_name"] = ref["Name"].apply(normalize)
    df["clean_name"] = df["Ticker"].apply(normalize)

    # Deduplicate Refinitiv symbols
    ref = ref.drop_duplicates(subset=["Symbol"])

    ref_names = ref["clean_name"].tolist()

    tickers = []
    for name in df["clean_name"]:
        match, score, idx = process.extractOne(
            name,
            ref_names,
            scorer=fuzz.WRatio
        )
        if score >= 85:
            tickers.append(ref.iloc[idx]["Symbol"])
        else:
            tickers.append(None)

    df["ticker"] = tickers

    missing = df[df["ticker"].isna()]
    print(f"\nUnmapped companies: {len(missing)}")
    if len(missing) > 0:
        print(missing["Ticker"].tolist())

    df = df.dropna(subset=["ticker"])
    return df.drop(columns=["clean_name"])


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def build_ai_forward_return_file():

    # -------------------------------------------------------------------------
    # LOAD + CONVERT EACH OLS RUN INTO FORWARD RETURNS (before averaging)
    # -------------------------------------------------------------------------
    df1, cols = load_ols(FILE_1)
    df2, _    = load_ols(FILE_2)
    df3, _    = load_ols(FILE_3)

    print("Converting OLS1 to forward returns...")
    df1 = to_forward_returns(df1, cols)

    print("Converting OLS2 to forward returns...")
    df2 = to_forward_returns(df2, cols)

    print("Converting OLS3 to forward returns...")
    df3 = to_forward_returns(df3, cols)

    # -------------------------------------------------------------------------
    # MERGE FORWARD RETURNS BY AVERAGING
    # -------------------------------------------------------------------------
    merged = df1.copy()
    for col in cols:
        merged[col] = (df1[col] + df2[col] + df3[col]) / 3

    merged["forecast_average"] = (
        df1["forecast_average"] + df2["forecast_average"] + df3["forecast_average"]
    ) / 3

    # -------------------------------------------------------------------------
    # MAP TICKER NAMES → REAL SYMBOLS
    # -------------------------------------------------------------------------
    merged = map_tickers(merged)

    # -------------------------------------------------------------------------
    # ORDER + SAVE
    # -------------------------------------------------------------------------
    final_cols = ["ticker"] + [c for c in merged.columns if c not in ["ticker", "Ticker"]]
    merged = merged[final_cols]

    merged.to_csv(OUTPUT, index=False)

    print(f"\nAI forward-return forecast saved to:\n  {OUTPUT}")
    print(f"Total tickers: {len(merged)}")


if __name__ == "__main__":
    build_ai_forward_return_file()
