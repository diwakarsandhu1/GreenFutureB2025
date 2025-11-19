import numpy as np
import pandas as pd
from typing import Dict

ANNUAL_RISK_FREE_RATE = 0.0153
TRADING_DAYS = 252

def run_monte_carlo_simulation(
    portfolio: Dict[str, float],
    cash_percent: float,
    horizon: int,
    num_paths: int,
    portfolio_weighing_scheme: str = "Equal Weights"
):
    tickers = list(portfolio.keys())
    weights = np.array(list(portfolio.values())).reshape(-1, 1)
    weights = weights / weights.sum()

    # -------------------------
    # COVARIANCE MATRIX
    # -------------------------
    if portfolio_weighing_scheme == "Equal Weights":
        portfolio_weighing_scheme = "Baseline Markowitz"

    scheme = portfolio_weighing_scheme.lower().replace(" ", "_")
    base_path = f"data_science/data/covariance_models/{scheme}"
    adj_cov = pd.read_csv(f"{base_path}/sp500_adjusted_cov_matrix.csv")
    adj_cov.set_index("ticker", inplace=True)
    cov_matrix = adj_cov.loc[tickers, tickers].to_numpy()

    # -------------------------
    # EXPECTED RETURNS
    # -------------------------
    perf = pd.read_csv("data_science/data/universal_data/sp500_performance_summaries.csv", index_col=0)

    mu_annual_log_vec = perf.loc["mean_log_return", tickers].values.reshape(-1, 1)
    mu_monthly_log_vec = mu_annual_log_vec / 12

    port_mu_monthly_log = float(weights.T @ mu_monthly_log_vec)
    port_mu_annual_log = float(port_mu_monthly_log * 12)

    port_sigma_annual = np.sqrt(weights.T @ cov_matrix @ weights) * np.sqrt(252)
    port_sigma_monthly = port_sigma_annual / np.sqrt(12)

    # -------------------------
    # LIGHT FAT-TAIL FIX (NO DRIFT CHANGE)
    # -------------------------
    # Adds rare but realistic tail events without altering average return.
    CRASH_PROB = 0.015       # ~1.5% chance per month
    CRASH_SHIFT = -2.0       # -2σ downward shift during crash
    CRASH_MULTIPLIER = 2.0   # increase in magnitude of tail shock

    z = np.random.normal(size=(horizon, num_paths))
    crashes = np.random.rand(horizon, num_paths) < CRASH_PROB
    z[crashes] = CRASH_MULTIPLIER * z[crashes] + CRASH_SHIFT

    # unchanged drift
    port_rand = port_mu_monthly_log + port_sigma_monthly * z
    stock_growth = np.exp(np.cumsum(port_rand, axis=0))

    # -------------------------
    # CASH GROWTH
    # -------------------------
    rf_monthly = np.log1p(ANNUAL_RISK_FREE_RATE) / 12
    cash_rand = np.full((horizon, num_paths), rf_monthly)
    cash_growth = np.exp(np.cumsum(cash_rand, axis=0))

    port_growth = (1 - cash_percent) * stock_growth + cash_percent * cash_growth

    pct = lambda q: np.percentile(port_growth, q, axis=1).tolist()

    port_mean_annual = np.expm1(port_mu_annual_log)
    port_vol_annual = port_sigma_monthly * np.sqrt(12)
    port_sharpe = (port_mean_annual - ANNUAL_RISK_FREE_RATE) / (port_vol_annual + 1e-8)

    port_p50 = pct(50)
    port_runmax = np.maximum.accumulate(port_p50)
    port_max_dd = float(np.min((np.array(port_p50) - port_runmax) / port_runmax))

    # -------------------------
    # SP500 (same fat-tail logic)
    # -------------------------
    spy_df = pd.read_csv("data_science/data/universal_data/spy_timeseries_13-24.csv")
    spy_daily_returns = spy_df["SPY"]
    spy_log = np.log1p(spy_daily_returns)

    spy_mu_daily = spy_log.mean()
    spy_sigma_daily = spy_log.std()

    spy_mu_monthly_log = spy_mu_daily * TRADING_DAYS / 12
    spy_sigma_monthly = spy_sigma_daily * np.sqrt(TRADING_DAYS / 12)

    z_spy = np.random.normal(size=(horizon, num_paths))
    crashes_spy = np.random.rand(horizon, num_paths) < CRASH_PROB
    z_spy[crashes_spy] = CRASH_MULTIPLIER * z_spy[crashes_spy] + CRASH_SHIFT

    spy_rand = spy_mu_monthly_log + spy_sigma_monthly * z_spy
    spy_growth = np.exp(np.cumsum(spy_rand, axis=0))

    pct_spy = lambda q: np.percentile(spy_growth, q, axis=1).tolist()

    spy_mean_annual = np.expm1(spy_mu_monthly_log * 12)
    spy_vol_annual = spy_sigma_monthly * np.sqrt(12)
    spy_sharpe = (spy_mean_annual - ANNUAL_RISK_FREE_RATE) / (spy_vol_annual + 1e-8)

    spy_p50 = pct_spy(50)
    spy_runmax = np.maximum.accumulate(spy_p50)
    spy_max_dd = float(np.min((np.array(spy_p50) - spy_runmax) / spy_runmax))

    # -------------------------
    # VAR CALCULATION FIX
    # -------------------------

    # 1-Month VaR (unchanged logic)
    port_monthly_returns = port_growth[0] - 1
    spy_monthly_returns = spy_growth[0] - 1

    port_VaR_95 = np.percentile(port_monthly_returns, 5)
    port_VaR_99 = np.percentile(port_monthly_returns, 1)

    spy_VaR_95 = np.percentile(spy_monthly_returns, 5)
    spy_VaR_99 = np.percentile(spy_monthly_returns, 1)

    # Horizon VaR FIX:
    # 🔥 Use *annualized returns* instead of total terminal return.
    years = horizon / 12.0

    port_final_annual = port_growth[-1]**(1.0 / years) - 1.0
    spy_final_annual  = spy_growth[-1]**(1.0 / years) - 1.0

    port_VaR_95_horizon = np.percentile(port_final_annual, 5)
    port_VaR_99_horizon = np.percentile(port_final_annual, 1)

    spy_VaR_95_horizon = np.percentile(spy_final_annual, 5)
    spy_VaR_99_horizon = np.percentile(spy_final_annual, 1)

    # -------------------------
    # DATE RANGE
    # -------------------------
    spy_df = pd.read_csv("data_science/data/universal_data/sp500_timeseries_13-24.csv", parse_dates=['date'])
    last_date = spy_df["date"].iloc[-1]
    dates = pd.date_range(start=last_date, periods=horizon+1, freq="M")[1:].strftime("%Y-%m-%d").tolist()

    # -------------------------
    # RETURN API STRUCTURE
    # -------------------------
    return {
        "times": dates,

        "portfolio": {
            "pctBands": {
                "p5": pct(5),
                "p25": pct(25),
                "p50": pct(50),
                "p75": pct(75),
                "p95": pct(95),
            },
            "stats": {
                "mean_return": float(port_mean_annual),
                "volatility": float(port_vol_annual),
                "max_drawdown": float(port_max_dd),
                "sharpe_ratio": float(port_sharpe),
                "method": portfolio_weighing_scheme,
                "VaR": {
                    "1m" :{
                        "var95": float(port_VaR_95),
                        "var99": float(port_VaR_99)
                    },
                    "horizon": {
                        "var95": float(port_VaR_95_horizon),
                        "var99": float(port_VaR_99_horizon)
                    }
                }
            }
        },

        "sp500": {
            "pctBands": {
                "p5": pct_spy(5),
                "p25": pct_spy(25),
                "p50": pct_spy(50),
                "p75": pct_spy(75),
                "p95": pct_spy(95),
            },
            "stats": {
                "mean_return": float(spy_mean_annual),
                "volatility": float(spy_vol_annual),
                "max_drawdown": float(spy_max_dd),
                "sharpe_ratio": float(spy_sharpe),
                "VaR": {
                    "1m" :{
                        "var95": float(spy_VaR_95),
                        "var99": float(spy_VaR_99)
                    },
                    "horizon": {
                        "var95": float(spy_VaR_95_horizon),
                        "var99": float(spy_VaR_99_horizon)
                    }
                }
            }
        }
    }
