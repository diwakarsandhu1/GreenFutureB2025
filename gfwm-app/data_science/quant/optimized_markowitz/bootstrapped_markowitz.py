"""
Optimized Markowitz frontier with Michaud resampling (SP500 universe).

Relies on your existing data layout:

  data_science/data/universal_data:
    - sp500_timeseries_13-24.csv
    - spy_timeseries_13-24.csv

  data_science/data/covariance_models/optimized_markowitz:
    - sp500_raw_cov_matrix.csv
    - sp500_adjusted_cov_matrix.csv

Outputs (in this folder's `outputs/` subdirectory):

  markowitz_frontier_optimized.csv
  markowitz_frontier_optimized_michaud.csv
  markowitz_random_optimized.csv
  tangent_portfolio_optimized.csv
  tangent_portfolio_optimized_meta.json
  tangent_portfolio_optimized_michaud.csv
  tangent_portfolio_optimized_michaud_meta.json
  markowitz_frontier_optimized.png
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cvxopt as opt
from cvxopt import matrix, solvers, blas


# ---------------------------------------------------------------------
# CVXOPT settings (keep quiet, cap iterations)
# ---------------------------------------------------------------------

solvers.options["show_progress"] = False
solvers.options["maxiters"] = 80


# ---------------------------------------------------------------------
# Markowitz helpers (baseline logic, but more defensive)
# ---------------------------------------------------------------------


def calculate_optimal_portfolios(
    true_mean_returns: np.ndarray,
    adjusted_mean_returns: np.ndarray,
    true_cov: np.ndarray,
    adjusted_cov: np.ndarray,
    target_returns: np.ndarray,
    annual_risk_free_rate: float = 0.02,
    bounds: Optional[Tuple[float, float]] = None,
    calculate_best_fit: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, np.polynomial.Polynomial]:
    """
    Your original Markowitz function with three important changes:

      1) Uses a safe QP wrapper: any non-'optimal' status is treated as failure
         (we log a warning and skip that target).
      2) `bounds` default to [0, 2/n] if not provided.
      3) Returns a DataFrame with rows only for successful QP solves.
    """
    # ensure 1D float arrays
    true_mean_returns = np.asarray(true_mean_returns, dtype=float).ravel()
    adjusted_mean_returns = np.asarray(adjusted_mean_returns, dtype=float).ravel()
    target_returns = np.asarray(target_returns, dtype=float).ravel()

    n = len(true_mean_returns)

    # convert to daily risk-free, consistent with original code
    risk_free_rate = (1.0 + annual_risk_free_rate) ** (1.0 / 365.0) - 1.0

    # only consider targets above RF
    target_returns = target_returns[target_returns > risk_free_rate]

    # set up result frame (weights will be filled later)
    empty_col = np.empty(len(target_returns), dtype=object)
    optimal_portfolios = pd.DataFrame(
        {
            "target_return": target_returns,
            "annual_return": np.nan,
            "annual_volatility": np.nan,
            "weights": empty_col.copy(),
            "diversification": np.nan,
        }
    )

    # bounds: long-only, mildly diversified unless overridden
    if bounds is None or len(bounds) != 2:
        bounds = (0.0, 2.0 / n)
    else:
        bounds = (float(bounds[0]), float(bounds[1]))
    lb, ub = bounds

    # G w <= h  for bounds:  -w <= -lb,  w <= ub
    G = opt.matrix(0.0, (2 * n, n))
    for i in range(n):
        G[i, i] = -1.0
        G[n + i, i] = 1.0

    eps = 1e-8
    h = opt.matrix(-lb - eps, (2 * n, 1))
    for i in range(n):
        h[i + n] = ub + eps

    # A w = b: [mu_adj; 1] * w = [target_return; 1]
    A = opt.matrix(1.0, (2, n))
    for i in range(n):
        A[0, i] = adjusted_mean_returns[i]

    # CVXOPT matrices
    true_mean_vec = matrix(true_mean_returns)
    true_cov_mat = matrix(true_cov)
    adjusted_cov_mat = matrix(adjusted_cov)
    q = matrix([0.0] * n)

    def solve_qp(target_return: float):
        try:
            sol = solvers.qp(
                P=adjusted_cov_mat,
                q=q,
                G=G,
                h=h,
                A=A,
                b=matrix([target_return, 0.999999], (2, 1)),
            )
        except Exception as e:
            print(f"[WARN] QP failed for target={target_return:.6e}: {e}")
            return None

        status = sol.get("status", None)
        if status != "optimal":
            print(f"[WARN] QP status={status} for target={target_return:.6e}")
            return None

        return sol["x"]

    # run QP across the grid
    optimal_portfolios["weights"] = optimal_portfolios["target_return"].map(solve_qp)

    # drop infeasible targets (where solver failed)
    optimal_portfolios.dropna(subset=["weights"], inplace=True)

    # return / vol / diversification (same formulas as original)
    def _annual_return(w):
        return 252.0 * blas.dot(true_mean_vec, w)

    def _annual_vol(w):
        return np.sqrt(252.0 * blas.dot(w, true_cov_mat * w))

    optimal_portfolios["annual_return"] = optimal_portfolios["weights"].map(
        _annual_return
    )
    optimal_portfolios["annual_volatility"] = optimal_portfolios["weights"].map(
        _annual_vol
    )
    optimal_portfolios["diversification"] = optimal_portfolios["weights"].map(
        lambda w: 1.0
        - 1e4 * np.sum((np.array(w).ravel() - 1.0 / n) ** 4)
    )

    if calculate_best_fit:
        best_fit = np.polynomial.Polynomial.fit(
            optimal_portfolios["annual_return"],
            optimal_portfolios["annual_volatility"],
            2,
            domain=[0.0, 0.5],
        )
        return optimal_portfolios, best_fit

    return optimal_portfolios


def montecarlo_random_portfolios(
    mean_returns: np.ndarray,
    cov: np.ndarray,
    bounds: Optional[Tuple[float, float]] = None,
    iterations: int = int(1e4),
) -> pd.DataFrame:
    """
    Your original Monte Carlo random portfolio generator.
    """
    rng = np.random.default_rng()
    n = len(mean_returns)
    iterations = int(iterations)

    mean_returns = opt.matrix(mean_returns)
    cov = opt.matrix(cov)

    montecarlo_portfolios = pd.DataFrame(index=range(iterations))

    if bounds is None:
        bounds = (0.0, n / 4.0)

    montecarlo_portfolios["random_weights"] = list(
        rng.uniform(low=bounds[0], high=bounds[1], size=(iterations, n))
    )
    montecarlo_portfolios["random_weights"] = montecarlo_portfolios[
        "random_weights"
    ].map(lambda w: opt.matrix(np.array(w) / sum(w)))

    montecarlo_portfolios["annual_return"] = montecarlo_portfolios[
        "random_weights"
    ].map(lambda w: 252 * blas.dot(w, mean_returns))

    montecarlo_portfolios["annual_volatility"] = montecarlo_portfolios[
        "random_weights"
    ].map(lambda w: np.sqrt(252.0 * blas.dot(w, cov * w)))

    return montecarlo_portfolios


# ---------------------------------------------------------------------
# Michaud resampling layer (built on top of the Markowitz function)
# ---------------------------------------------------------------------


def michaud_resampled_frontier(
    returns_df: pd.DataFrame,
    true_cov: np.ndarray,
    adjusted_cov: np.ndarray,
    target_returns: np.ndarray,
    annual_risk_free_rate: float = 0.02,
    bounds: Optional[Tuple[float, float]] = None,
    n_resamples: int = 80,
    seed: int = 123,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Classic Michaud:

      1) Run *baseline* Markowitz once to get the feasible frontier grid.
      2) For each bootstrap resample of the returns:
          - recompute means
          - run Markowitz using those means
          - for each target that solved, accumulate weights
      3) Average weights per target and evaluate (return, vol) under the
         *true* mean / true covariance.
    """
    n_days, n_assets = returns_df.shape
    rng = np.random.default_rng(seed)

    # true mean used for evaluation (and for the baseline classical frontier)
    true_mean_full = returns_df.mean(axis=0).values

    # 1) baseline frontier to find the feasible targets
    base_frontier = calculate_optimal_portfolios(
        true_mean_returns=true_mean_full,
        adjusted_mean_returns=true_mean_full,
        true_cov=true_cov,
        adjusted_cov=adjusted_cov,
        target_returns=target_returns,
        annual_risk_free_rate=annual_risk_free_rate,
        bounds=bounds,
        calculate_best_fit=False,
    )

    feasible_targets = base_frontier["target_return"].values
    n_targets = len(feasible_targets)

    if n_targets == 0:
        raise RuntimeError(
            "No feasible frontier points found in base Markowitz run."
        )

    print(
        f"[MICHAUD] Starting resampling with {n_resamples} resamples, "
        f"{n_targets} feasible targets."
    )

    # accumulators
    sum_w = np.zeros((n_targets, n_assets))
    count_w = np.zeros(n_targets)

    true_cov_np = np.asarray(true_cov, dtype=float)
    true_mean_np = np.asarray(true_mean_full, dtype=float)

    for b in range(n_resamples):
        # bootstrap resample of time indices
        idx = rng.integers(0, n_days, size=n_days)
        boot = returns_df.iloc[idx]
        boot_mean = boot.mean(axis=0).values

        # Markowitz with bootstrapped means
        boot_frontier = calculate_optimal_portfolios(
            true_mean_returns=true_mean_full,  # evaluation means stay "true"
            adjusted_mean_returns=boot_mean,   # optimization uses bootstrapped μ
            true_cov=true_cov,
            adjusted_cov=adjusted_cov,
            target_returns=feasible_targets,
            annual_risk_free_rate=annual_risk_free_rate,
            bounds=bounds,
            calculate_best_fit=False,
        )

        # align targets by value
        for _, row in boot_frontier.iterrows():
            t = row["target_return"]
            j = np.where(np.isclose(feasible_targets, t))[0]
            if j.size == 0:
                continue
            j = j[0]
            w = np.array(row["weights"]).ravel()
            sum_w[j] += w
            count_w[j] += 1

        if (b + 1) % 10 == 0 or b == n_resamples - 1:
            avg_resamples = count_w.mean()
            print(
                f"[MICHAUD] Resample {b+1}/{n_resamples} "
                f"(avg successful resamples/target: {avg_resamples:.1f})"
            )

    # build resampled frontier from averaged weights
    records: List[Dict[str, object]] = []
    for j, t in enumerate(feasible_targets):
        if count_w[j] == 0:
            continue

        w_avg = sum_w[j] / count_w[j]
        ann_ret = float(252.0 * np.dot(true_mean_np, w_avg))
        ann_vol = float(np.sqrt(252.0 * (w_avg @ true_cov_np @ w_avg)))

        records.append(
            {
                "target_return": t,
                "annual_return": ann_ret,
                "annual_volatility": ann_vol,
                "weights": w_avg,
                "n_resamples_used": int(count_w[j]),
            }
        )

    if not records:
        raise RuntimeError("Michaud resampling produced no valid frontier points.")

    michaud_frontier = pd.DataFrame(records)
    return base_frontier, michaud_frontier


# ---------------------------------------------------------------------
# Tangent portfolio + plotting
# ---------------------------------------------------------------------


def compute_tangent_from_frontier(
    frontier_df: pd.DataFrame,
    annual_risk_free_rate: float,
) -> Dict[str, object]:
    """
    Find the max-Sharpe point on a frontier DataFrame that has
    'annual_return' and 'annual_volatility'.
    """
    rf = annual_risk_free_rate
    excess = frontier_df["annual_return"] - rf
    sharpe = excess / frontier_df["annual_volatility"]

    idx = sharpe.idxmax()
    row = frontier_df.loc[idx]

    return {
        "index": int(idx),
        "annual_return": float(row["annual_return"]),
        "annual_volatility": float(row["annual_volatility"]),
        "sharpe": float(sharpe.loc[idx]),
        "weights": row["weights"],
    }


def plot_frontier_and_cloud(
    frontier_df: pd.DataFrame,
    random_df: pd.DataFrame,
    spy_series: pd.Series,
    michaud_df: Optional[pd.DataFrame],
    output_path: Path,
) -> None:
    spy_daily = spy_series.values
    spy_ret_ann = float(spy_daily.mean() * 252.0)
    spy_vol_ann = float(spy_daily.std() * np.sqrt(252.0))

    plt.figure(figsize=(8, 6))

    # random cloud
    plt.scatter(
        random_df["annual_volatility"],
        random_df["annual_return"],
        alpha=0.25,
        s=6,
        label="Random portfolios",
    )

    # classic frontier
    plt.plot(
        frontier_df["annual_volatility"],
        frontier_df["annual_return"],
        linewidth=2.0,
        label="Markowitz frontier",
    )

    # Michaud frontier (if provided)
    if michaud_df is not None and not michaud_df.empty:
        plt.plot(
            michaud_df["annual_volatility"],
            michaud_df["annual_return"],
            linewidth=2.0,
            linestyle="--",
            label="Michaud resampled frontier",
        )

    # SPY
    plt.scatter([spy_vol_ann], [spy_ret_ann], marker="*", s=120, label="SPY")

    plt.xlabel("Annualized Volatility")
    plt.ylabel("Annualized Return")
    plt.title("Markowitz Efficient Frontier (optimized)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"[PLOT] Saved frontier plot to: {output_path}")


# ---------------------------------------------------------------------
# Main: optimized Markowitz + Michaud
# ---------------------------------------------------------------------


def main() -> None:
    # paths relative to this script
    this_dir = Path(__file__).resolve().parent
    data_science_dir = this_dir.parents[1]          # gfwm-app/data_science
    data_root = data_science_dir / "data"
    universal_dir = data_root / "universal_data"
    cov_opt_dir = data_root / "covariance_models" / "optimized_markowitz"

    output_dir = this_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # load data
    sp500_ts_path = universal_dir / "sp500_timeseries_13-24.csv"
    spy_ts_path = universal_dir / "spy_timeseries_13-24.csv"
    raw_cov_path = cov_opt_dir / "sp500_raw_cov_matrix.csv"
    adj_cov_path = cov_opt_dir / "sp500_adjusted_cov_matrix.csv"

    print(f"[PATH] sp500_timeseries: {sp500_ts_path}")
    print(f"[PATH] spy_timeseries  : {spy_ts_path}")
    print(f"[PATH] raw_cov_path    : {raw_cov_path}")
    print(f"[PATH] adj_cov_path    : {adj_cov_path}")

    returns_df = pd.read_csv(sp500_ts_path, index_col=0, parse_dates=True)
    spy_df = pd.read_csv(spy_ts_path, index_col=0, parse_dates=True)
    cov_raw_df = pd.read_csv(raw_cov_path, index_col=0)
    cov_adj_df = pd.read_csv(adj_cov_path, index_col=0)

    # align tickers between returns and covariance
    tickers_ret = returns_df.columns
    tickers_cov = cov_adj_df.columns
    common = tickers_ret.intersection(tickers_cov)

    if len(common) == 0:
        raise ValueError(
            "No overlapping tickers between returns and covariance matrices."
        )
    if len(common) != len(tickers_ret) or len(common) != len(tickers_cov):
        print(
            f"[WARN] Aligning on {len(common)} common tickers "
            f"(returns: {len(tickers_ret)}, cov: {len(tickers_cov)})"
        )

    returns_df = returns_df[common]
    cov_raw_df = cov_raw_df.loc[common, common]
    cov_adj_df = cov_adj_df.loc[common, common]
    tickers: List[str] = list(common)

    n_days, n_assets = returns_df.shape
    print(
        f"[INFO] Running optimized Markowitz with {n_assets} assets "
        f"and {n_days} days."
    )

    mu_daily = returns_df.mean(axis=0).values
    mu_annual = mu_daily * 252.0

    # build a sensible target-return grid from the cross-section of means
    mu_clean = mu_annual[~np.isnan(mu_annual)]
    if mu_clean.size == 0:
        raise RuntimeError("All asset mean returns are NaN; check input data.")

    min_ann = float(np.nanpercentile(mu_clean, 20))
    max_ann = float(np.nanpercentile(mu_clean, 80) * 1.5)
    target_ann = np.linspace(min_ann, max_ann, 60)
    target_daily = target_ann / 252.0

    annual_rf = 0.02
    bounds = (0.0, 2.0 / n_assets)

    # --- classic frontier + Michaud ---
    base_frontier, michaud_frontier = michaud_resampled_frontier(
        returns_df=returns_df,
        true_cov=cov_raw_df.values,
        adjusted_cov=cov_adj_df.values,
        target_returns=target_daily,
        annual_risk_free_rate=annual_rf,
        bounds=bounds,
        n_resamples=80,
        seed=123,
    )

    print(
        f"[INFO] Classic frontier size: {len(base_frontier)}, "
        f"Michaud frontier size: {len(michaud_frontier)}"
    )

    # random portfolios cloud
    random_df = montecarlo_random_portfolios(
        mean_returns=mu_daily,
        cov=cov_raw_df.values,
        bounds=bounds,
        iterations=1000,
    )

    # tangent portfolios
    tangent_classic = compute_tangent_from_frontier(
        frontier_df=base_frontier, annual_risk_free_rate=annual_rf
    )
    tangent_michaud = compute_tangent_from_frontier(
        frontier_df=michaud_frontier, annual_risk_free_rate=annual_rf
    )

    # save CSVs
    frontier_path = output_dir / "markowitz_frontier_optimized.csv"
    frontier_michaud_path = output_dir / "markowitz_frontier_optimized_michaud.csv"
    random_path = output_dir / "markowitz_random_optimized.csv"
    tangent_path = output_dir / "tangent_portfolio_optimized.csv"
    tangent_meta_path = output_dir / "tangent_portfolio_optimized_meta.json"
    tangent_m_path = output_dir / "tangent_portfolio_optimized_michaud.csv"
    tangent_m_meta_path = (
        output_dir / "tangent_portfolio_optimized_michaud_meta.json"
    )

    base_frontier.to_csv(frontier_path, index=False)
    michaud_frontier.to_csv(frontier_michaud_path, index=False)
    random_df.to_csv(random_path, index=False)

    # tangent weights
    tangent_weights_df = pd.DataFrame(
        {"ticker": tickers, "weight": np.asarray(tangent_classic["weights"]).ravel()}
    )
    tangent_weights_df.to_csv(tangent_path, index=False)
    with open(tangent_meta_path, "w") as f:
        json.dump(
            {k: v for k, v in tangent_classic.items() if k != "weights"},
            f,
            indent=2,
        )

    tangent_m_weights_df = pd.DataFrame(
        {"ticker": tickers, "weight": np.asarray(tangent_michaud["weights"]).ravel()}
    )
    tangent_m_weights_df.to_csv(tangent_m_path, index=False)
    with open(tangent_m_meta_path, "w") as f:
        json.dump(
            {k: v for k, v in tangent_michaud.items() if k != "weights"},
            f,
            indent=2,
        )

    print(f"[INFO] Saved frontier CSV to: {frontier_path}")
    print(f"[INFO] Saved Michaud frontier CSV to: {frontier_michaud_path}")
    print(f"[INFO] Saved random cloud CSV to: {random_path}")
    print(f"[INFO] Saved tangent weights to: {tangent_path}")
    print(f"[INFO] Saved tangent meta to: {tangent_meta_path}")
    print(f"[INFO] Saved Michaud tangent weights to: {tangent_m_path}")
    print(f"[INFO] Saved Michaud tangent meta to: {tangent_m_meta_path}")

    # plot
    plot_path = output_dir / "markowitz_frontier_optimized.png"
    plot_frontier_and_cloud(
        frontier_df=base_frontier,
        random_df=random_df,
        spy_series=spy_df.iloc[:, 0],
        michaud_df=michaud_frontier,
        output_path=plot_path,
    )

    # simple diagnostics
    spy_daily = spy_df.iloc[:, 0].values
    spy_ret_ann = float(spy_daily.mean() * 252.0)
    spy_vol_ann = float(spy_daily.std() * np.sqrt(252.0))
    spy_sharpe = (spy_ret_ann - annual_rf) / spy_vol_ann

    min_row = base_frontier.loc[base_frontier["annual_volatility"].idxmin()]
    max_row = base_frontier.loc[base_frontier["annual_return"].idxmax()]

    print("\n[DIAG] optimized (classic) frontier")
    print(
        f"   # points: {len(base_frontier)}\n"
        f"   Min-vol:   ret={min_row['annual_return']:.4f}, "
        f"vol={min_row['annual_volatility']:.4f}\n"
        f"   Max-return:ret={max_row['annual_return']:.4f}, "
        f"vol={max_row['annual_volatility']:.4f}\n"
        f"   Tangent:   ret={tangent_classic['annual_return']:.4f}, "
        f"vol={tangent_classic['annual_volatility']:.4f}, "
        f"sharpe={tangent_classic['sharpe']:.3f}\n"
        f"   SPY:       ret={spy_ret_ann:.4f}, "
        f"vol={spy_vol_ann:.4f}, "
        f"sharpe={spy_sharpe:.3f}"
    )

    min_row_m = michaud_frontier.loc[
        michaud_frontier["annual_volatility"].idxmin()
    ]
    max_row_m = michaud_frontier.loc[
        michaud_frontier["annual_return"].idxmax()
    ]

    print("\n[DIAG] optimized (Michaud) frontier")
    print(
        f"   # points: {len(michaud_frontier)}\n"
        f"   Min-vol:   ret={min_row_m['annual_return']:.4f}, "
        f"vol={min_row_m['annual_volatility']:.4f}\n"
        f"   Max-return:ret={max_row_m['annual_return']:.4f}, "
        f"vol={max_row_m['annual_volatility']:.4f}\n"
        f"   Tangent:   ret={tangent_michaud['annual_return']:.4f}, "
        f"vol={tangent_michaud['annual_volatility']:.4f}, "
        f"sharpe={tangent_michaud['sharpe']:.3f}\n"
        f"   SPY:       ret={spy_ret_ann:.4f}, "
        f"vol={spy_vol_ann:.4f}, "
        f"sharpe={spy_sharpe:.3f}"
    )


if __name__ == "__main__":
    main()
