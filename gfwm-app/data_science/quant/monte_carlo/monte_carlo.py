import numpy as np
import pandas as pd
from typing import Dict, Optional
from functools import lru_cache

ANNUAL_RISK_FREE_RATE = 0.0407  # based off current 2025 RFR
TRADING_DAYS = 252

# ------------------------------------------------------------
# Helper: load AI forward returns once (cached)
# CSV schema expected:
#   ticker, forecast_2023, ..., forecast_2033, forecast_average
# values are annual arithmetic returns (e.g., 0.08 for 8%)
# ------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_ai_forward_returns(path: str = "data_science/data/monte_carlo_ai_forecasting/ai_forward_returns.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "ticker" not in df.columns:
        raise ValueError("ai_forward_returns.csv must include a 'ticker' column.")
    df["ticker"] = df["ticker"].str.upper().str.strip()
    return df.set_index("ticker")


def run_monte_carlo_simulation(
    portfolio: Dict[str, float],
    cash_percent: float,
    horizon: int,
    num_paths: int,
    advisor_fee: float,
    portfolio_weighing_scheme: str = "Equal Weights",
    rebalancing_rule: str = "Annually",
    forecast_mode: str = "CAPM",
    ai_forward_returns_path: Optional[str] = "data_science/data/monte_carlo_ai_forecasting/ai_forward_returns.csv",
):
    tickers = [t.upper() for t in portfolio.keys()]
    weights = np.array(list(portfolio.values())).reshape(-1, 1)
    weights = weights / max(weights.sum(), 1e-12)
    n_assets = len(tickers)

    # ------------------------------------------
    # LOAD RETURN DATA
    # ------------------------------------------
    spy_returns_df = pd.read_csv(
        "data_science/data/universal_data/spy_timeseries_13-24.csv",
        index_col=0,
        parse_dates=True
    )
    returns_df = pd.read_csv(
        "data_science/data/universal_data/sp500_timeseries_13-24.csv",
        index_col=0,
        parse_dates=True
    )

    # Align and prep
    combined = returns_df.join(spy_returns_df[["SPY"]], how="inner")
    daily_cov = returns_df.cov() * TRADING_DAYS
    cov_matrix_full = daily_cov.loc[tickers, tickers].to_numpy()

    std = np.sqrt(np.diag(cov_matrix_full))
    std = np.where(std == 0, 1e-12, std)
    corr = cov_matrix_full / np.outer(std, std)

    # ------------------------------------------
    # VOLATILITY SMOOTHING (TARGET VOLATILITY)
    # ------------------------------------------
    target_vol = 0.14  # 14% annual portfolio-level volatility
    idiosyncratic = 0.4
    systemic = 0.6

    cov_matrix = (
        systemic * corr * (target_vol ** 2) +
        idiosyncratic * np.eye(n_assets) * (target_vol ** 2)
    )
    cov_monthly = cov_matrix / 12.0

    # --- Ensure Positive Semidefinite Covariance ---
    eigvals, eigvecs = np.linalg.eigh(cov_monthly)
    eigvals_clipped = np.clip(eigvals, 1e-8, None)
    cov_monthly_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    L = np.linalg.cholesky(cov_monthly_psd)

    # ------------------------------------------
    # FORWARD-LOOKING EXPECTED RETURNS (CAPM baseline)
    # (kept EXACTLY as your original logic)
    # ------------------------------------------
    monthly = combined.resample("M").mean()
    equity_risk_premium = 0.055
    rf = ANNUAL_RISK_FREE_RATE

    capm_mu_annual_log = []
    for t in tickers:
        stock_monthly = monthly[t]
        market_monthly = monthly["SPY"]
        beta = stock_monthly.cov(market_monthly) / (market_monthly.var() + 1e-12)
        er = rf + beta * equity_risk_premium  # annual arithmetic return
        capm_mu_annual_log.append(np.log1p(er))
    capm_mu_annual_log_vec = np.array(capm_mu_annual_log).reshape(-1, 1)

    # For debugging: store decisions about forecasts by ticker + year
    forecast_logs = {tk: [] for tk in tickers}

    # CAPM shrinkage → anchor (unchanged)
    shrink_lambda = 0.40
    anchor_return = np.log1p(0.07)  # 7% nominal long-run return
    capm_mu_annual_log_vec = (
        shrink_lambda * capm_mu_annual_log_vec +
        (1 - shrink_lambda) * anchor_return
    )
    capm_mu_monthly_log_vec = capm_mu_annual_log_vec / 12.0

    # ------------------------------------------
    # AI EXPECTED RETURNS (per-year through 2033, then long-term average)
    # Only used when forecast_mode == "AI". Otherwise we use CAPM.
    # - AI values are annual arithmetic returns (not log)
    # - No shrinkage applied to AI numbers (per your latest instruction)
    # - Advisor fee applied monthly in both modes
    # ------------------------------------------
    # Build a (horizon x n_assets) drift matrix of monthly log returns.
    mu_monthly_log_timeline = np.zeros((horizon, n_assets))

    # Determine simulation start year from historical last date
    last_date = returns_df.index[-1]  # monthly dates later depend on this, see bottom
    # First simulated month is the month after last_date
    start_year = (last_date + pd.offsets.MonthEnd(1)).year

    ai_df = None
    if forecast_mode.lower() == "ai":
        # Try to load AI forward returns; if missing, we will silently fall back to CAPM per-ticker/year
        try:
            ai_df = _load_ai_forward_returns(ai_forward_returns_path)
        except Exception:
            ai_df = None  # complete fallback to CAPM below

    # Precompute monthly CAPM drift vector for quick reuse
    capm_mu_monthly_log_vec_flat = capm_mu_monthly_log_vec.flatten()

    # Fill timeline month by month
    for t_step in range(horizon):
        current_year = start_year + (t_step // 12)

        if ai_df is None or forecast_mode.lower() != "ai":
            # Use CAPM monthly log drift for all assets
            mu_monthly_log_timeline[t_step, :] = capm_mu_monthly_log_vec_flat
            continue

        # In AI mode: per-asset selection with fallback to CAPM
        row_exists = ai_df.index.intersection(tickers).shape[0] > 0
        # choose the column for the current year
        if current_year <= 2033:
            col = f"forecast_{current_year}".lower()
        else:
            col = "forecast_average"

        for j, tk in enumerate(tickers):
            # Default = CAPM monthly drift
            drift_source = "CAPM"
            annual_used = np.expm1(capm_mu_annual_log_vec[j, 0])  # convert annual log → annual arithmetic
            drift_log_month = capm_mu_monthly_log_vec_flat[j]

            if row_exists and tk in ai_df.index:
                ai_val = None

                # Per-year forecasting through 2033
                if col in ai_df.columns:
                    ai_val = ai_df.at[tk, col]

                # Long-run average after 2033
                elif current_year > 2033 and "forecast_average" in ai_df.columns:
                    ai_val = ai_df.at[tk, "forecast_average"]

                # Use AI value when valid
                if ai_val is not None and pd.notnull(ai_val):
                    drift_source = "AI"
                    annual_used = float(ai_val)  # annual arithmetic
                    drift_log_month = np.log1p(annual_used) / 12.0

            # Final monthly drift (before fee)
            mu_monthly_log_timeline[t_step, j] = drift_log_month

            # Log decision
            forecast_logs[tk].append({
                "simulation_month": t_step,
                "simulation_year": current_year,
                "source": drift_source,
                "annual_return_used": float(annual_used),
                "monthly_log_return_used": float(drift_log_month),
                "column_used": col if drift_source == "AI" else "CAPM"
            })


    # Apply advisor fee monthly to the drift timeline (both modes)
    fee_monthly_log = np.log1p(-advisor_fee / 12.0)
    mu_monthly_log_timeline = mu_monthly_log_timeline + fee_monthly_log

    # ------------------------------------------
    # FAT-TAIL SHOCKS
    # ------------------------------------------
    CRASH_PROB = 0.01
    CRASH_SHIFT = -0.35
    CRASH_MULTIPLIER = 1.0

    epsilon = np.random.normal(size=(horizon, num_paths, n_assets))
    crashes = (np.random.rand(horizon, num_paths, 1) < CRASH_PROB)
    crashes = np.repeat(crashes, n_assets, axis=2)
    epsilon[crashes] = CRASH_MULTIPLIER * epsilon[crashes] + CRASH_SHIFT

    shocks = epsilon @ L.T  # (horizon, num_paths, n_assets)

    # Compose asset log-returns per step: drift (per month, per asset) + correlated shock
    # Broadcast mu_monthly_log_timeline to (horizon, num_paths, n_assets)
    stock_log_returns = mu_monthly_log_timeline[:, None, :] + shocks
    stock_multipliers = np.exp(stock_log_returns)

    # ------------------------------------------
    # REBALANCING
    # ------------------------------------------
    rule = "".join(c for c in rebalancing_rule.lower() if c.isalpha())
    if rule.startswith("quarter"):
        rebalance_interval = 3
    elif rule.startswith("semi"):
        rebalance_interval = 6
    else:
        rebalance_interval = 12

    w_vec = weights.flatten()
    holdings = w_vec[:, None] * np.ones((n_assets, num_paths))
    portfolio_values = np.zeros((horizon, num_paths))

    for t in range(horizon):
        holdings = holdings * stock_multipliers[t].T
        portfolio_values[t] = holdings.sum(axis=0)
        if (t + 1) % rebalance_interval == 0:
            holdings = portfolio_values[t] * w_vec[:, None]

    stock_growth = portfolio_values

    # ------------------------------------------
    # CASH GROWTH
    # ------------------------------------------
    rf_monthly = np.log1p(ANNUAL_RISK_FREE_RATE) / 12.0
    cash_growth = np.exp(np.cumsum(np.full((horizon, num_paths), rf_monthly), axis=0))

    # combine portfolio stocks + cash
    port_growth = (1 - cash_percent) * stock_growth + cash_percent * cash_growth

    pct = lambda q: np.percentile(port_growth, q, axis=1).tolist()

    # ------------------------------------------
    # MONTHLY RETURNS → STATS
    # ------------------------------------------
    port_monthly_returns = np.zeros((horizon, num_paths))
    port_monthly_returns[0] = port_growth[0] - 1.0
    for t in range(1, horizon):
        port_monthly_returns[t] = port_growth[t] / port_growth[t - 1] - 1.0

    years = horizon / 12.0
    port_annualized_paths = port_growth[-1] ** (1.0 / years) - 1.0

    port_mean_annual = float(np.mean(port_annualized_paths))
    monthly_vol = np.std(port_monthly_returns.flatten())
    port_vol_annual = float(monthly_vol * np.sqrt(12.0))

    p50 = pct(50)
    runmax = np.maximum.accumulate(p50)
    port_max_dd = float(np.min((np.array(p50) - runmax) / runmax))

    port_sharpe = (port_mean_annual - ANNUAL_RISK_FREE_RATE) / (port_vol_annual + 1e-12)

    # -------------------------
    # SP500 SIMULATION
    # -------------------------
    spy_sigma_monthly = np.sqrt(cov_monthly_psd.mean())

    # CAPM expected annual return for SPY (beta ≈ 1), then shrinkage and anchor
    spy_mu_annual = rf + equity_risk_premium
    spy_mu_annual_log = np.log1p(spy_mu_annual)
    shrink_lambda_spy = 0.40
    anchor_return_spy = np.log1p(0.07)
    spy_mu_annual_log = (
        shrink_lambda_spy * spy_mu_annual_log +
        (1 - shrink_lambda_spy) * anchor_return_spy
    )
    spy_mu_monthly_log = spy_mu_annual_log / 12.0

    z_spy = np.random.normal(size=(horizon, num_paths))
    crashes_spy = np.random.rand(horizon, num_paths) < CRASH_PROB
    z_spy[crashes_spy] = CRASH_MULTIPLIER * z_spy[crashes_spy] + CRASH_SHIFT

    spy_rand = spy_mu_monthly_log + spy_sigma_monthly * z_spy
    spy_growth = np.exp(np.cumsum(spy_rand, axis=0))
    spy_growth_adjusted = (1 - cash_percent) * spy_growth + cash_percent * cash_growth

    pct_spy = lambda q: np.percentile(spy_growth_adjusted, q, axis=1).tolist()

    spy_monthly_returns = np.zeros((horizon, num_paths))
    spy_monthly_returns[0] = spy_growth_adjusted[0] - 1.0
    for t in range(1, horizon):
        spy_monthly_returns[t] = spy_growth_adjusted[t] / spy_growth_adjusted[t-1] - 1.0

    spy_annualized_paths = spy_growth_adjusted[-1]**(1.0 / years) - 1.0

    spy_mean_annual = float(np.mean(spy_annualized_paths))
    spy_monthly_vol = np.std(spy_monthly_returns.flatten())
    spy_vol_annual = float(spy_monthly_vol * np.sqrt(12))
    spy_sharpe = (spy_mean_annual - ANNUAL_RISK_FREE_RATE) / (spy_vol_annual + 1e-12)

    spy_p50 = pct_spy(50)
    spy_runmax = np.maximum.accumulate(spy_p50)
    spy_max_dd = float(np.min((np.array(spy_p50) - spy_runmax) / spy_runmax))

    # ------------------------------------------
    # VAR
    # ------------------------------------------
    port_VaR_95 = np.percentile(port_monthly_returns[0], 5)
    port_VaR_99 = np.percentile(port_monthly_returns[0], 1)

    spy_VaR_95 = np.percentile(spy_monthly_returns[0], 5)
    spy_VaR_99 = np.percentile(spy_monthly_returns[0], 1)

    port_VaR_95_horizon = np.percentile(port_annualized_paths, 5)
    port_VaR_99_horizon = np.percentile(port_annualized_paths, 1)

    spy_VaR_95_horizon = np.percentile(spy_annualized_paths, 5)
    spy_VaR_99_horizon = np.percentile(spy_annualized_paths, 1)

    # ------------------------------------------
    # DATE RANGE
    # ------------------------------------------
    last_date_for_dates = returns_df.index[-1]
    dates = pd.date_range(
        start=last_date_for_dates,
        periods=horizon + 1,
        freq="M"
    )[1:].strftime("%Y-%m-%d").tolist()

    # Attach forecast logs for debugging
    debug_log = forecast_logs

    # ------------------------------------------
    # RETURN STRUCTURE
    # ------------------------------------------
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
                "mean_return": port_mean_annual,
                "volatility": port_vol_annual,
                "max_drawdown": port_max_dd,
                "sharpe_ratio": port_sharpe,
                "method": portfolio_weighing_scheme,
                "VaR": {
                    "1m": {
                        "var95": float(port_VaR_95),
                        "var99": float(port_VaR_99),
                    },
                    "horizon": {
                        "var95": float(port_VaR_95_horizon),
                        "var99": float(port_VaR_99_horizon),
                    },
                },
            },
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
                "mean_return": spy_mean_annual,
                "volatility": spy_vol_annual,
                "max_drawdown": spy_max_dd,
                "sharpe_ratio": spy_sharpe,
                "VaR": {
                    "1m": {
                        "var95": float(spy_VaR_95),
                        "var99": float(spy_VaR_99),
                    },
                    "horizon": {
                        "var95": float(spy_VaR_95_horizon),
                        "var99": float(spy_VaR_99_horizon),
                    },
                },
            },
        },

        "debug_forecast_log": debug_log,
    }
