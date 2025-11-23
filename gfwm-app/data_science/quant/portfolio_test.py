import os
import sys
import numpy as np
import pandas as pd

# -------------------------------------------------------------------
# Ensure Python can see the project root (gfwm-app) as a package root
# -------------------------------------------------------------------

THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))  # data_science/quant -> data_science -> gfwm-app

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Now imports work the same way as inside your main app.
from data_science.quant.portfolio_calculator import calculate_portfolio


ANNUAL_RISK_FREE_RATE = 0.0153


def run_single_test(label, use_baseline_markowitz, use_optimized_markowitz,
                    cash_percent=0.2, n_tickers=100):
    print(f"\n================ {label.upper()} TEST ================")

    # Load performance summaries to grab some valid tickers
    perf_path = os.path.join(PROJECT_ROOT, "data_science", "data", "universal_data", "sp500_performance_summaries.csv")
    perf = pd.read_csv(perf_path)

    # All columns from the performance summary file
    all_cols = perf.columns.tolist()

    # Filter out junk columns like 'Unnamed: 0', "metric", "row", etc.
    candidate_cols = [
        c for c in all_cols
        if not c.lower().startswith("unnamed")
        and c.lower() not in ("metric", "row", "index")
    ]

    tickers = candidate_cols[:n_tickers]
    print(f"[INFO] Using {len(tickers)} tickers: {tickers}")

    ticker_compatibility_df = pd.DataFrame({"ticker": tickers})

    # Run the selected Markowitz engine
    result = calculate_portfolio(
        ticker_compatibility_df=ticker_compatibility_df,
        cash_percent=cash_percent,
        use_baseline_markowitz=use_baseline_markowitz,
        use_optimized_markowitz=use_optimized_markowitz,
        return_summary_statistics=True,
    )

    weights_list, expected_return, expected_vol, sharpe = result
    weights = np.array(weights_list, dtype=float)
    n = len(tickers)

    print(f"[RESULT] #weights        : {len(weights)}")
    print(f"[RESULT] Sum(weights)    : {weights.sum():.6f}")
    print(f"[RESULT] Cash percent    : {cash_percent:.4f}")
    print(f"[RESULT] Expected return : {expected_return:.6f}")
    print(f"[RESULT] Expected vol    : {expected_vol:.6f}")
    print(f"[RESULT] Sharpe          : {sharpe:.6f}")

    # --------------------- sanity checks ---------------------
    assert len(weights) == n, f"Expected {n} weights, got {len(weights)}"
    assert np.all(np.isfinite(weights)), "Non-finite weights (NaN/inf)"
    assert np.all(weights >= -1e-6), "Negative weight found (check constraints)"

    target_sum = 1.0 - cash_percent
    if not np.isclose(weights.sum(), target_sum, atol=1e-3):
        print(f"[WARN] Sum(weights)={weights.sum():.6f} != target={target_sum:.6f} (within 1e-3)")

    assert np.isfinite(expected_return), "Expected return is NaN/inf"
    assert np.isfinite(expected_vol), "Expected volatility is NaN/inf"
    assert expected_vol >= 0.0, "Expected volatility is negative"

    print("[OK] Basic sanity checks passed.")


def main():
    # Baseline Markowitz (existing pipeline)


    # Optimized Markowitz with Michaud resampling (your new code)
    run_single_test(
        label="optimized_michaud",
        use_baseline_markowitz=False,
        use_optimized_markowitz=True,
        cash_percent=0.2,
        n_tickers=15,
    )
    run_single_test(
        label="baseline",
        use_baseline_markowitz=True,
        use_optimized_markowitz=False,
        cash_percent=0.2,
        n_tickers=15,
    )


if __name__ == "__main__":
    main()
