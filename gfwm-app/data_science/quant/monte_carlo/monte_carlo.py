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
    advisor_fee: float,
    portfolio_weighing_scheme: str = "Equal Weights",
    rebalancing_rule: str = "Annually",
):
    tickers = list(portfolio.keys())
    weights = np.array(list(portfolio.values())).reshape(-1, 1)
    weights = weights / weights.sum()
    n_assets = len(tickers)

    # ------------------------------------------
    # LOAD RETURN DATA
    # ------------------------------------------
    spy_returns_df = pd.read_csv(
        "data_science/data/universal_data/spy_timeseries_13-24.csv",
        index_col=0,
        parse_dates=True
    )

    # ------------------------------------------
    # HISTORICAL COVARIANCE → CORRELATION MATRIX
    # ------------------------------------------
    returns_df = pd.read_csv(
        "data_science/data/universal_data/sp500_timeseries_13-24.csv",
        index_col=0,
        parse_dates=True
    )

    combined = returns_df.join(spy_returns_df[["SPY"]], how="inner")

    daily_cov = returns_df.cov() * TRADING_DAYS
    cov_matrix_full = daily_cov.loc[tickers, tickers].to_numpy()

    std = np.sqrt(np.diag(cov_matrix_full))
    corr = cov_matrix_full / np.outer(std, std)

    # ------------------------------------------
    # VOLATILITY SMOOTHING (TARGET VOLATILITY)
    # ------------------------------------------
    target_vol = 0.14  # 14% annual portfolio-level volatility (industry norm)
    cov_matrix = corr * (target_vol ** 2)

    idiosyncratic = 0.4   # 40% idio risk weight (industry norm)
    systemic = 0.6        # 60% systemic risk weight

    cov_matrix = (
        systemic   * corr * (target_vol ** 2)
        + idiosyncratic * np.eye(n_assets) * (target_vol ** 2)
    )
    
    # monthly covariance
    cov_monthly = cov_matrix / 12.0
    
    # --- Ensure Positive Semidefinite Covariance ---
    eigvals, eigvecs = np.linalg.eigh(cov_monthly)
    eigvals_clipped = np.clip(eigvals, 1e-8, None)  # small positive floor
    cov_monthly_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T

    L = np.linalg.cholesky(cov_monthly_psd)


    # ------------------------------------------
    # FORWARD-LOOKING EXPECTED RETURNS (CAPM)
    # ------------------------------------------
    combined = returns_df.join(spy_returns_df[["SPY"]], how="inner")
    monthly = combined.resample("M").mean()

    equity_risk_premium = 0.055
    rf = ANNUAL_RISK_FREE_RATE

    mu_annual = []
    for t in tickers:
        stock_monthly = monthly[t]
        market_monthly = monthly["SPY"]

        beta = stock_monthly.cov(market_monthly) / market_monthly.var()
        er = rf + beta * equity_risk_premium
        mu_annual.append(np.log1p(er))

    mu_annual_log_vec = np.array(mu_annual).reshape(-1, 1)


    # ---------------------------------------------------------
    # RETURN SHRINKAGE (Forward-looking dampening of extremes)
    # ---------------------------------------------------------
    shrink_lambda = 0.40        # 40% shrink toward anchor (industry norm)
    anchor_return = np.log1p(0.07)   # 7% nominal long-run return
    mu_annual_log_vec = (
        shrink_lambda * mu_annual_log_vec +
        (1 - shrink_lambda) * anchor_return
    )

    # recalc monthly log returns after shrinkage
    mu_monthly_log_vec = mu_annual_log_vec / 12.0


    # apply advisor fee (reduces drift)
    fee_monthly_log = np.log1p(-advisor_fee / 12)
    mu_monthly_log_vec = mu_monthly_log_vec + fee_monthly_log


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

    shocks = epsilon @ L.T
    stock_log_returns = mu_monthly_log_vec.T + shocks
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
    holdings = w_vec[:, None] * np.ones((1, num_paths))
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
    rf_monthly = np.log1p(ANNUAL_RISK_FREE_RATE) / 12
    cash_growth = np.exp(np.cumsum(np.full((horizon, num_paths), rf_monthly), axis=0))

    port_growth = (1 - cash_percent) * stock_growth + cash_percent * cash_growth

    pct = lambda q: np.percentile(port_growth, q, axis=1).tolist()

    # ------------------------------------------
    # MONTHLY RETURNS → STATS
    # ------------------------------------------
    port_monthly_returns = np.zeros((horizon, num_paths))
    port_monthly_returns[0] = port_growth[0] - 1.0
    for t in range(1, horizon):
        port_monthly_returns[t] = port_growth[t] / port_growth[t-1] - 1.0

    years = horizon / 12.0
    port_annualized_paths = port_growth[-1]**(1.0 / years) - 1.0

    port_mean_annual = float(np.mean(port_annualized_paths))
    monthly_vol = np.std(port_monthly_returns.flatten())
    port_vol_annual = float(monthly_vol * np.sqrt(12))

    p50 = pct(50)
    runmax = np.maximum.accumulate(p50)
    port_max_dd = float(np.min((np.array(p50) - runmax) / runmax))

    port_sharpe = (port_mean_annual - ANNUAL_RISK_FREE_RATE) / (port_vol_annual + 1e-12)

    # -------------------------
    # SP500 SIMULATION (FORWARD-LOOKING, LIKE PORTFOLIO)
    # -------------------------
    spy_sigma_monthly = np.sqrt(cov_monthly_psd.mean())

    # forward-looking CAPM expected return for SPY (beta ≈ 1)
    spy_mu_annual = rf + equity_risk_premium        # nominal expected return
    spy_mu_annual_log = np.log1p(spy_mu_annual)

    # apply shrinkage toward long-term equity expected return
    shrink_lambda = 0.40
    anchor_return = np.log1p(0.07)   # 7% nominal long-run equity return

    spy_mu_annual_log = (
        shrink_lambda * spy_mu_annual_log +
        (1 - shrink_lambda) * anchor_return
    )

    spy_mu_monthly_log = spy_mu_annual_log / 12.0


    z_spy = np.random.normal(size=(horizon, num_paths))
    crashes_spy = np.random.rand(horizon, num_paths) < CRASH_PROB
    z_spy[crashes_spy] = CRASH_MULTIPLIER * z_spy[crashes_spy] + CRASH_SHIFT

    spy_rand = spy_mu_monthly_log + spy_sigma_monthly * z_spy
    spy_growth = np.exp(np.cumsum(spy_rand, axis=0))

    pct_spy = lambda q: np.percentile(spy_growth, q, axis=1).tolist()

    spy_monthly_returns = np.zeros((horizon, num_paths))
    spy_monthly_returns[0] = spy_growth[0] - 1.0
    for t in range(1, horizon):
        spy_monthly_returns[t] = spy_growth[t] / spy_growth[t-1] - 1.0

    spy_annualized_paths = spy_growth[-1]**(1.0 / years) - 1.0

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
    spy_df = pd.read_csv(
        "data_science/data/universal_data/sp500_timeseries_13-24.csv",
        index_col=0,
        parse_dates=True
    )
    last_date = spy_df.index[-1]
    dates = pd.date_range(start=last_date, periods=horizon+1, freq="M")[1:].strftime("%Y-%m-%d").tolist()

    # ------------------------------------------
    # RETURN STRUCTURE (UNCHANGED)
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
    }
