import numpy as np
import pandas as pd
from typing import Dict

ANNUAL_RISK_FREE_RATE = 0.0153

def run_monte_carlo_simulation(
    portfolio: Dict[str, float],
    cash_percent: float,
    horizon: int,
    num_paths: int,
    portfolio_weighing_scheme: str = "Equal Weights",
    rebalancing_rule: str = "annual",
    compounding_type: str = "logarithmic",
    ):

    tickers = list(portfolio.keys())
    n = len(tickers)
    weights = np.array(list(portfolio.values())).reshape(n, 1)

    # Determine weighting scheme and parameters
    if portfolio_weighing_scheme == "Equal Weights":
        # Equal weights → mock covariance
        cov_matrix = np.eye(n) * 0.0001  # small daily variance placeholder

    elif portfolio_weighing_scheme in ["Baseline Markowitz", "Optimized Markowitz"]:
        # Reformat portfolio_weighing_scheme
        portfolio_weighing_scheme = (
            portfolio_weighing_scheme.lower().replace(" ", "_")
        )
        
        # Load covariance matrices
        base_path = f"data_science/data/covariance_models/{portfolio_weighing_scheme}"
        adjusted_cov_matrix = pd.read_csv(f"{base_path}/sp500_adjusted_cov_matrix.csv")
        adjusted_cov_matrix.set_index('ticker', inplace=True)

        # Keep tickers aligned with current portfolio
        adjusted_cov_matrix = adjusted_cov_matrix.loc[tickers, tickers].to_numpy()
        cov_matrix = adjusted_cov_matrix
    else:
        raise ValueError(f"Invalid portfolio weighting scheme: {portfolio_weighing_scheme}")
    
    # Estimate expected return
    performance_summaries = pd.read_csv("data_science/data/universal_data/sp500_performance_summaries.csv")
    annual_returns = performance_summaries[tickers].iloc[0]

    mean_log_returns = annual_returns/252

    # Compute mu and sigma (daily)
    port_mu = float(np.dot(weights.T, mean_log_returns))
    port_sigma = float(np.sqrt(weights.T @ cov_matrix @ weights))

    # Scale to monthly stats for simulation
    mu_monthly = (1 + port_mu * 252) ** (1/12) - 1
    sigma_monthly = port_sigma * np.sqrt(252 / 12)

    # Monte Carlo Simulation
    rand_rets = np.random.normal(mu_monthly, sigma_monthly, (horizon, num_paths))

    if compounding_type == "logarithmic":
        growth_paths = np.exp(np.cumsum(rand_rets, axis=0))
    else:
        growth_paths = (1 + rand_rets).cumprod(axis=0)

    portfolio_growth = (1 - cash_percent) * growth_paths + cash_percent
    
    # Return data for front end
    spy_df = pd.read_csv("data_science/data/universal_data/sp500_timeseries_13-24.csv", parse_dates=['date'])
    last_date = spy_df["date"].iloc[-1]

    dates = pd.date_range(start=last_date, periods=horizon+1, freq="M")[1:]
    dates = [d.strftime("%Y-%m-%d") for d in dates]
    pct = lambda q: np.percentile(portfolio_growth, q, axis=1).tolist()

    terminal = portfolio_growth[-1, :]
    mean_return = np.mean(terminal) - 1
    volatility = np.std(terminal)
    sharpe = (mean_return - 0.03) / (volatility + 1e-8)
    max_dd = float(np.min((np.array(pct(50)) - np.maximum.accumulate(pct(50))) / np.maximum.accumulate(pct(50))))

    return {
        "times": dates,
        "pctBands": {
            "p5": pct(5),
            "p25": pct(25),
            "p50": pct(50),
            "p75": pct(75),
            "p95": pct(95),
        },
        "stats": {
            "mean_return": float(mean_return),
            "volatility": float(volatility),
            "max_drawdown": float(max_dd),
            "sharpe_ratio": float(sharpe),
            "method": portfolio_weighing_scheme,
        }
    }