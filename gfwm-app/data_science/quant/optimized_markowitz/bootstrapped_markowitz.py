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

# ------------------------- CVXOPT options ------------------------------------
solvers.options["show_progress"] = False
solvers.options["maxiters"] = 80


# ------------------------- Bounds helpers (Σ-driven) -------------------------

def make_bounds_from_sigma(
    cov_adj: np.ndarray,
    max_cap: float = 0.05,
    k: float = 5.0,
    floor_frac: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-asset caps proportional to inverse volatility.
    lb_i = 0 (long-only)
    ub_i = min(max_cap, floor + k * inv_vol_i)
    """
    n = cov_adj.shape[0]
    sigma = np.sqrt(np.clip(np.diag(cov_adj), 1e-12, None))
    inv_vol = 1.0 / sigma
    inv_vol /= inv_vol.sum()  # inverse-vol weights (sum=1)

    floor = (floor_frac / n)
    ub_vec = np.minimum(max_cap, k * inv_vol + floor)
    lb_vec = np.zeros(n)
    return lb_vec, ub_vec


def _qp_min_variance(cov_adj: np.ndarray, lb_vec: np.ndarray, ub_vec: np.ndarray) -> Optional[np.ndarray]:
    """Min-variance under box constraints and 1' w = 1."""
    n = cov_adj.shape[0]
    G = matrix(np.vstack([-np.eye(n), np.eye(n)]))
    h = matrix(np.concatenate([-(lb_vec - 1e-9), (ub_vec + 1e-9)]))
    A = matrix(np.ones((1, n)))
    b = matrix(1.0)
    P = matrix(cov_adj)
    q = matrix(np.zeros(n))
    sol = solvers.qp(P, q, G, h, A, b)
    if sol["status"] != "optimal":
        return None
    return np.array(sol["x"]).reshape(-1)


def lift_caps_above_minvar(
    ub_vec: np.ndarray,
    w_minvar: Optional[np.ndarray],
    slack: float = 1.2,
    hard_cap: float = 0.10,
) -> np.ndarray:
    """Ensure ub >= slack * w_minvar (so min-var is feasible), but <= hard_cap."""
    if w_minvar is None:
        return ub_vec
    ub = np.maximum(ub_vec, slack * w_minvar)
    return np.minimum(ub, hard_cap)


def feasible_return_range(
    mu_daily: np.ndarray,
    lb_vec: np.ndarray,
    ub_vec: np.ndarray,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute min/max daily return achievable under box constraints and 1'w=1.
    Uses a tiny quadratic term to keep the QP well-posed.
    """
    n = len(mu_daily)
    eps = 1e-10
    P = matrix(np.eye(n) * eps)
    G = matrix(np.vstack([-np.eye(n), np.eye(n)]))
    h = matrix(np.concatenate([-(lb_vec - 1e-9), (ub_vec + 1e-9)]))
    A = matrix(np.ones((1, n)))
    b = matrix(1.0)

    # max mu'w  <=>  min -mu'w
    q = matrix(-mu_daily)
    sol_max = solvers.qp(P, q, G, h, A, b)
    if sol_max["status"] != "optimal":
        return None, None
    w_max = np.array(sol_max["x"]).reshape(-1)
    r_max = float(mu_daily @ w_max)

    # min mu'w
    q2 = matrix(+mu_daily)
    sol_min = solvers.qp(P, q2, G, h, A, b)
    if sol_min["status"] != "optimal":
        return None, None
    w_min = np.array(sol_min["x"]).reshape(-1)
    r_min = float(mu_daily @ w_min)
    return r_min, r_max


# ------------------------- Markowitz (with vector bounds) ---------------------

def calculate_optimal_portfolios(
    true_mean_returns: np.ndarray,
    adjusted_mean_returns: np.ndarray,
    true_cov: np.ndarray,
    adjusted_cov: np.ndarray,
    target_returns: np.ndarray,
    annual_risk_free_rate: float = 0.02,
    bounds: Optional[Tuple[float, float] | Tuple[np.ndarray, np.ndarray]] = None,
    calculate_best_fit: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, np.polynomial.Polynomial]:
    """
    Your original Markowitz function with:
      - Safe QP wrapper (skip non-optimal).
      - Supports either scalar bounds (lb, ub) or per-asset vectors (lb_vec, ub_vec).
      - Returns DF only with feasible targets.
    """
    true_mean_returns = np.asarray(true_mean_returns, dtype=float).ravel()
    adjusted_mean_returns = np.asarray(adjusted_mean_returns, dtype=float).ravel()
    target_returns = np.asarray(target_returns, dtype=float).ravel()
    n = len(true_mean_returns)

    # daily RF (as in your original)
    rf_daily = (1.0 + annual_risk_free_rate) ** (1.0 / 252.0) - 1.0
    target_returns = target_returns[target_returns > rf_daily]

    optimal_portfolios = pd.DataFrame(
        {
            "target_return": target_returns,
            "annual_return": np.nan,
            "annual_volatility": np.nan,
            "weights": np.empty(len(target_returns), dtype=object),
            "diversification": np.nan,
        }
    )

    # Bounds handling
    lb_vec: np.ndarray
    ub_vec: np.ndarray
    if bounds is None:
        lb_vec = np.zeros(n)
        ub_vec = np.ones(n) * (2.0 / n)
    else:
        if isinstance(bounds[0], np.ndarray):
            lb_vec = np.asarray(bounds[0], dtype=float).ravel()
            ub_vec = np.asarray(bounds[1], dtype=float).ravel()
        else:
            lb, ub = float(bounds[0]), float(bounds[1])
            lb_vec = np.ones(n) * lb
            ub_vec = np.ones(n) * ub

    # G w <= h with per-asset bounds
    G = opt.matrix(np.vstack([-np.eye(n), np.eye(n)]))
    eps = 1e-9
    h = opt.matrix(np.concatenate([-(lb_vec - eps), (ub_vec + eps)]))

    # A w = b: [mu_adj; 1] * w = [target_return; 1]
    A = opt.matrix(1.0, (2, n))
    for i in range(n):
        A[0, i] = adjusted_mean_returns[i]

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
        if sol.get("status", None) != "optimal":
            print(f"[WARN] QP status={sol.get('status')} for target={target_return:.6e}")
            return None
        return sol["x"]

    optimal_portfolios["weights"] = optimal_portfolios["target_return"].map(solve_qp)
    optimal_portfolios.dropna(subset=["weights"], inplace=True)

    def _annual_return(w): return 252.0 * blas.dot(true_mean_vec, w)
    def _annual_vol(w):    return np.sqrt(252.0 * blas.dot(w, true_cov_mat * w))

    optimal_portfolios["annual_return"] = optimal_portfolios["weights"].map(_annual_return)
    optimal_portfolios["annual_volatility"] = optimal_portfolios["weights"].map(_annual_vol)
    optimal_portfolios["diversification"] = optimal_portfolios["weights"].map(
        lambda w: 1.0 - 1e4 * np.sum((np.array(w).ravel() - 1.0 / n) ** 4)
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


# ------------------------- Monte Carlo (unchanged) ----------------------------

def montecarlo_random_portfolios(
    mean_returns: np.ndarray,
    cov: np.ndarray,
    bounds: Optional[Tuple[float, float]] = None,
    iterations: int = int(1e4),
) -> pd.DataFrame:
    rng = np.random.default_rng()
    n = len(mean_returns)
    iterations = int(iterations)

    mean_returns = opt.matrix(mean_returns)
    cov = opt.matrix(cov)

    df = pd.DataFrame(index=range(iterations))
    if bounds is None:
        bounds = (0.0, n / 4.0)

    df["random_weights"] = list(
        rng.uniform(low=bounds[0], high=bounds[1], size=(iterations, n))
    )
    df["random_weights"] = df["random_weights"].map(lambda w: opt.matrix(np.array(w) / sum(w)))
    df["annual_return"] = df["random_weights"].map(lambda w: 252 * blas.dot(w, mean_returns))
    df["annual_volatility"] = df["random_weights"].map(lambda w: np.sqrt(252.0 * blas.dot(w, cov * w)))
    return df


# ------------------------- Michaud resampling ---------------------------------

def michaud_resampled_frontier(
    returns_df: pd.DataFrame,
    true_cov: np.ndarray,
    adjusted_cov: np.ndarray,
    target_returns: np.ndarray,
    annual_risk_free_rate: float = 0.02,
    bounds: Optional[Tuple[float, float] | Tuple[np.ndarray, np.ndarray]] = None,
    n_resamples: int = 80,
    seed: int = 123,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    - Run base Markowitz to get feasible targets.
    - Bootstrap means, solve Markowitz per bootstrap, average weights target-wise.
    - Evaluate using *true* μ and *true* Σ.
    """
    n_days, n_assets = returns_df.shape
    rng = np.random.default_rng(seed)
    true_mean_full = returns_df.mean(axis=0).values

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
        raise RuntimeError("No feasible base frontier points.")

    print(f"[MICHAUD] Starting resampling with {n_resamples} resamples, {n_targets} feasible targets.")

    sum_w = np.zeros((n_targets, n_assets))
    count_w = np.zeros(n_targets)
    true_cov_np = np.asarray(true_cov, dtype=float)
    true_mean_np = np.asarray(true_mean_full, dtype=float)

    for b in range(n_resamples):
        idx = rng.integers(0, n_days, size=n_days)
        boot = returns_df.iloc[idx]
        boot_mean = boot.mean(axis=0).values

        boot_frontier = calculate_optimal_portfolios(
            true_mean_returns=true_mean_full,   # eval μ
            adjusted_mean_returns=boot_mean,    # optimize μ
            true_cov=true_cov,
            adjusted_cov=adjusted_cov,
            target_returns=feasible_targets,
            annual_risk_free_rate=annual_risk_free_rate,
            bounds=bounds,
            calculate_best_fit=False,
        )

        # accumulate weights by matching target values
        for _, row in boot_frontier.iterrows():
            t = row["target_return"]
            j = np.where(np.isclose(feasible_targets, t))[0]
            if j.size == 0:
                continue
            j = j[0]
            sum_w[j] += np.array(row["weights"]).ravel()
            count_w[j] += 1

        if (b + 1) % 10 == 0 or b == n_resamples - 1:
            print(f"[MICHAUD] Resample {b+1}/{n_resamples} (avg successful resamples/target: {count_w.mean():.1f})")

    records: List[Dict[str, object]] = []
    for j, t in enumerate(feasible_targets):
        if count_w[j] == 0:
            continue
        w_avg = sum_w[j] / count_w[j]
        ann_ret = float(252.0 * (true_mean_np @ w_avg))
        ann_vol = float(np.sqrt(252.0 * (w_avg @ true_cov_np @ w_avg)))
        records.append(
            {
                "target_return": float(t),
                "annual_return": ann_ret,
                "annual_volatility": ann_vol,
                "weights": w_avg,
                "n_resamples_used": int(count_w[j]),
            }
        )

    if not records:
        raise RuntimeError("Michaud resampling produced no valid points.")

    michaud_frontier = pd.DataFrame(records)
    return base_frontier, michaud_frontier


# ------------------------- Tangent + Plotting ---------------------------------

def compute_tangent_from_frontier(frontier_df: pd.DataFrame, annual_risk_free_rate: float) -> Dict[str, object]:
    rf = annual_risk_free_rate
    sharpe = (frontier_df["annual_return"] - rf) / frontier_df["annual_volatility"]
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
    plt.scatter(random_df["annual_volatility"], random_df["annual_return"], alpha=0.25, s=6, label="Random portfolios")
    plt.plot(frontier_df["annual_volatility"], frontier_df["annual_return"], linewidth=2.0, label="Markowitz frontier")
    if michaud_df is not None and not michaud_df.empty:
        plt.plot(michaud_df["annual_volatility"], michaud_df["annual_return"], linewidth=2.0, linestyle="--", label="Michaud resampled frontier")
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


# ------------------------- Main -----------------------------------------------

def main() -> None:
    # paths
    this_dir = Path(__file__).resolve().parent
    data_science_dir = this_dir.parents[1]
    data_root = data_science_dir / "data"
    universal_dir = data_root / "universal_data"
    cov_opt_dir = data_root / "covariance_models" / "optimized_markowitz"

    output_dir = this_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # align tickers
    tickers_ret = returns_df.columns
    tickers_cov = cov_adj_df.columns
    common = tickers_ret.intersection(tickers_cov)
    if len(common) == 0:
        raise ValueError("No overlapping tickers between returns and covariance matrices.")
    if len(common) != len(tickers_ret) or len(common) != len(tickers_cov):
        print(f"[WARN] Aligning on {len(common)} common tickers (returns: {len(tickers_ret)}, cov: {len(tickers_cov)})")

    returns_df = returns_df[common]
    cov_raw_df = cov_raw_df.loc[common, common]
    cov_adj_df = cov_adj_df.loc[common, common]
    tickers: List[str] = list(common)

    n_days, n_assets = returns_df.shape
    print(f"[INFO] Running optimized Markowitz with {n_assets} assets and {n_days} days.")

    mu_daily = returns_df.mean(axis=0).values
    annual_rf = 0.02

    # --------- NEW: Σ-informed per-asset bounds + feasibility checks ----------
    lb_vec, ub_vec = make_bounds_from_sigma(cov_adj_df.values, max_cap=0.05, k=5.0, floor_frac=0.5)
    # ensure min-var is feasible (lift caps if needed, cap hard at 10%)
    w_minvar = _qp_min_variance(cov_adj_df.values, lb_vec, np.ones_like(ub_vec))
    ub_vec = lift_caps_above_minvar(ub_vec, w_minvar, slack=1.2, hard_cap=0.10)

    # compute feasible daily return range under these bounds
    rmin, rmax = feasible_return_range(mu_daily, lb_vec, ub_vec)
    rf_daily = (1 + annual_rf) ** (1 / 252.0) - 1.0

    if (rmin is not None) and (rmax is not None) and (rmax > rf_daily):
        # target grid inside feasible range and above RF
        lo = max(rf_daily, rmin + 1e-8)
        hi = max(lo + 1e-8, rmax - 1e-8)
        target_daily = np.linspace(lo, hi, 60)
    else:
        # fallback: percentile grid on μ
        mu_annual = mu_daily * 252.0
        mu_clean = mu_annual[~np.isnan(mu_annual)]
        min_ann = float(np.nanpercentile(mu_clean, 20))
        max_ann = float(np.nanpercentile(mu_clean, 80) * 1.5)
        target_daily = np.linspace(min_ann / 252.0, max_ann / 252.0, 60)

    # ---------------- Classic + Michaud using vector bounds -------------------
    base_frontier, michaud_frontier = michaud_resampled_frontier(
        returns_df=returns_df,
        true_cov=cov_raw_df.values,
        adjusted_cov=cov_adj_df.values,
        target_returns=target_daily,
        annual_risk_free_rate=annual_rf,
        bounds=(lb_vec, ub_vec),        # <-- per-asset bounds used everywhere
        n_resamples=80,
        seed=123,
    )

    print(f"[INFO] Classic frontier size: {len(base_frontier)}, Michaud frontier size: {len(michaud_frontier)}")

    # random cloud (keep simple scalar bounds here; it's only for visualization)
    random_df = montecarlo_random_portfolios(
        mean_returns=mu_daily,
        cov=cov_raw_df.values,
        bounds=(0.0, n_assets / 4.0),
        iterations=1000,
    )

    # tangent portfolios
    tangent_classic = compute_tangent_from_frontier(base_frontier, annual_rf)
    tangent_michaud = compute_tangent_from_frontier(michaud_frontier, annual_rf)

    # save CSVs
    output_dir = this_dir / "outputs"
    frontier_path = output_dir / "markowitz_frontier_optimized.csv"
    frontier_michaud_path = output_dir / "markowitz_frontier_optimized_michaud.csv"
    random_path = output_dir / "markowitz_random_optimized.csv"
    tangent_path = output_dir / "tangent_portfolio_optimized.csv"
    tangent_meta_path = output_dir / "tangent_portfolio_optimized_meta.json"
    tangent_m_path = output_dir / "tangent_portfolio_optimized_michaud.csv"
    tangent_m_meta_path = output_dir / "tangent_portfolio_optimized_michaud_meta.json"

    base_frontier.to_csv(frontier_path, index=False)
    michaud_frontier.to_csv(frontier_michaud_path, index=False)
    random_df.to_csv(random_path, index=False)

    tangent_weights_df = pd.DataFrame({"ticker": tickers, "weight": np.asarray(tangent_classic["weights"]).ravel()})
    tangent_weights_df.to_csv(tangent_path, index=False)
    with open(tangent_meta_path, "w") as f:
        json.dump({k: v for k, v in tangent_classic.items() if k != "weights"}, f, indent=2)

    tangent_m_weights_df = pd.DataFrame({"ticker": tickers, "weight": np.asarray(tangent_michaud["weights"]).ravel()})
    tangent_m_weights_df.to_csv(tangent_m_path, index=False)
    with open(tangent_m_meta_path, "w") as f:
        json.dump({k: v for k, v in tangent_michaud.items() if k != "weights"}, f, indent=2)

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

    # diagnostics
    spy_daily = spy_df.iloc[:, 0].values
    spy_ret_ann = float(spy_daily.mean() * 252.0)
    spy_vol_ann = float(spy_daily.std() * np.sqrt(252.0))
    spy_sharpe = (spy_ret_ann - annual_rf) / spy_vol_ann

    min_row = base_frontier.loc[base_frontier["annual_volatility"].idxmin()]
    max_row = base_frontier.loc[base_frontier["annual_return"].idxmax()]
    print("\n[DIAG] optimized (classic) frontier")
    print(
        f"   # points: {len(base_frontier)}\n"
        f"   Min-vol:   ret={min_row['annual_return']:.4f}, vol={min_row['annual_volatility']:.4f}\n"
        f"   Max-return:ret={max_row['annual_return']:.4f}, vol={max_row['annual_volatility']:.4f}\n"
        f"   Tangent:   ret={tangent_classic['annual_return']:.4f}, "
        f"vol={tangent_classic['annual_volatility']:.4f}, "
        f"sharpe={tangent_classic['sharpe']:.3f}\n"
        f"   SPY:       ret={spy_ret_ann:.4f}, vol={spy_vol_ann:.4f}, sharpe={spy_sharpe:.3f}"
    )

    min_row_m = michaud_frontier.loc[michaud_frontier["annual_volatility"].idxmin()]
    max_row_m = michaud_frontier.loc[michaud_frontier["annual_return"].idxmax()]
    print("\n[DIAG] optimized (Michaud) frontier")
    print(
        f"   # points: {len(michaud_frontier)}\n"
        f"   Min-vol:   ret={min_row_m['annual_return']:.4f}, vol={min_row_m['annual_volatility']:.4f}\n"
        f"   Max-return:ret={max_row_m['annual_return']:.4f}, vol={max_row_m['annual_volatility']:.4f}\n"
        f"   Tangent:   ret={tangent_michaud['annual_return']:.4f}, "
        f"vol={tangent_michaud['annual_volatility']:.4f}, "
        f"sharpe={tangent_michaud['sharpe']:.3f}\n"
        f"   SPY:       ret={spy_ret_ann:.4f}, vol={spy_vol_ann:.4f}, sharpe={spy_sharpe:.3f}"
    )


if __name__ == "__main__":
    main()
