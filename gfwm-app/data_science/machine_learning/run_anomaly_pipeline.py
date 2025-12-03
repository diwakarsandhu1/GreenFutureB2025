"""
run_anomaly_pipeline.py

End-to-end anomaly pipeline:
- PCA (models/pca.py)
- Isolation Forest (models/isolation_forest.py)
- Autoencoder (models/autoencoder.py)

Model-specific outputs remain in:
    data_science/machine_learning/models/outputs/

Composite outputs go into:
    data_science/machine_learning/outputs/

Function API:
    run_anomaly_pipeline(csv_path=...) -> dict[ticker, composite_score_0to1]

Direct execution:
    - Saves composite CSV
    - Generates PCA/IF/AE/composite histograms
    - Top-50 composite bar chart
    - All-tickers composite scatter
    - Composite vs Annual Return scatter
    - Markdown report with embedded charts
"""

from __future__ import annotations

import os
import math
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# Import model detectors
# -------------------------------
from models.pca import run_pca_anomaly_detection
from models.isolation_forest import run_isolation_forest
from models.auto_encoder import run_autoencoder

# -------------------------------
# Paths
# -------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Composite outputs folder (one level above models/)
COMPOSITE_OUTPUT_DIR = os.path.join(THIS_DIR, "pipeline_outputs")
os.makedirs(COMPOSITE_OUTPUT_DIR, exist_ok=True)

# Default data path (Option A)
DEFAULT_CSV = os.path.normpath(
    os.path.join(THIS_DIR, "..", "preprocess_and_filter", "preprocessed_refinitiv.csv")
)


# -------------------------------
# Utilities
# -------------------------------
def _zscore(series: pd.Series) -> pd.Series:
    mu = series.mean()
    sigma = series.std(ddof=0)
    if sigma == 0 or math.isclose(sigma, 0.0):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mu) / sigma


def _minmax(series: pd.Series) -> pd.Series:
    min_v = series.min()
    max_v = series.max()
    if max_v == min_v:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - min_v) / (max_v - min_v)


def _combine(df: pd.DataFrame, weights=(1/3, 1/3, 1/3)) -> pd.Series:
    """
    Combine z-scored per-model scores into a raw composite,
    then caller min-max scales to 0–1.
    """
    w1, w2, w3 = weights
    zp = _zscore(df["pca_score"])
    zi = _zscore(df["if_score"])
    za = _zscore(df["ae_score"])
    return w1 * zp + w2 * zi + w3 * za


def _align(pca: dict, iso: dict, ae: dict) -> pd.DataFrame:
    """
    Build tidy DF of model scores aligned on ticker.
    """
    df = pd.DataFrame({"ticker": list(pca.keys()), "pca_score": list(pca.values())})
    df = df.merge(
        pd.DataFrame({"ticker": list(iso.keys()), "if_score": list(iso.values())}),
        on="ticker",
        how="inner",
    )
    df = df.merge(
        pd.DataFrame({"ticker": list(ae.keys()), "ae_score": list(ae.values())}),
        on="ticker",
        how="inner",
    )
    return df


def _attach_returns(df_scores: pd.DataFrame, csv_path: str) -> pd.DataFrame:
    """
    Join in annual_return (and anything else needed) from the source CSV by ticker.
    If annual_return is missing, the plot that depends on it will be skipped.
    """
    try:
        src = pd.read_csv(csv_path)
        cols = ["ticker", "annual_return"]
        cols = [c for c in cols if c in src.columns]
        if "ticker" not in cols:
            return df_scores
        src_small = src[cols].drop_duplicates(subset=["ticker"])
        return df_scores.merge(src_small, on="ticker", how="left")
    except Exception:
        return df_scores


# -------------------------------
# PUBLIC FUNCTION
# -------------------------------
def run_anomaly_pipeline(csv_path: str = DEFAULT_CSV,
                         weights=(1/3, 1/3, 1/3)) -> dict[str, float]:
    """
    Return {ticker: composite_score_0to1} only (no charts/report).
    """
    # Run models
    pca = run_pca_anomaly_detection(csv_path)
    iso = run_isolation_forest(csv_path)
    ae  = run_autoencoder(csv_path)

    # Merge
    df = _align(pca, iso, ae)

    # Composite: z-scored -> weighted -> min-max to 0–1
    df["composite_score_raw"] = _combine(df, weights=weights)
    df["composite_score"] = _minmax(df["composite_score_raw"])

    # Return 0–1 composite as dict
    return {t: float(s) for t, s in zip(df["ticker"], df["composite_score"])}


# -------------------------------
# Plotting utilities
# -------------------------------
def _hist(series, title, fname, xlabel):
    plt.figure(figsize=(9, 5))
    plt.hist(series, bins=40, alpha=0.85)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.grid(axis="y")
    plt.tight_layout()
    out = os.path.join(COMPOSITE_OUTPUT_DIR, fname)
    plt.savefig(out)
    plt.close()
    return out


def _scatter(series, labels, title, fname, ylabel="Composite Score (0–1)"):
    plt.figure(figsize=(13, 5))
    x = np.arange(len(series))
    plt.scatter(x, series, s=10, alpha=0.7)
    plt.title(title)
    plt.xlabel("Ticker Index (sorted by ticker)")
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    out = os.path.join(COMPOSITE_OUTPUT_DIR, fname)
    plt.savefig(out)
    plt.close()
    return out


def _bar_top(series, labels, title, fname, top_n=50, xlabel="Composite Score (0–1)"):
    order = np.argsort(series.values)[::-1][:top_n]
    vals = series.values[order]
    labs = labels.values[order]

    plt.figure(figsize=(12, 10))
    y = np.arange(len(vals))
    plt.barh(y, vals)
    plt.yticks(y, labs)
    plt.gca().invert_yaxis()
    plt.title(f"{title} (Top {top_n})")
    plt.xlabel(xlabel)
    plt.tight_layout()
    out = os.path.join(COMPOSITE_OUTPUT_DIR, fname)
    plt.savefig(out)
    plt.close()
    return out


def _scatter_returns(df: pd.DataFrame, fname="anomaly_vs_returns.png"):
    """
    Scatter plot: Annual Return (x) vs Composite Score 0–1 (y)
    """
    if "annual_return" not in df.columns or df["annual_return"].isna().all():
        return None

    plt.figure(figsize=(10, 6))
    plt.scatter(df["annual_return"], df["composite_score"], alpha=0.7)
    plt.xlabel("Annual Return")
    plt.ylabel("Composite Anomaly Score (0–1)")
    plt.title("Composite Anomaly Score (0–1) vs Annual Return")
    plt.grid(True, linestyle="--", alpha=0.4)
    out = os.path.join(COMPOSITE_OUTPUT_DIR, fname)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out


# -------------------------------
# REPORT CREATION
# -------------------------------
def _report(stats, paths):
    report_path = os.path.join(COMPOSITE_OUTPUT_DIR, "anomaly_report.md")

    md = [
        "# Composite Anomaly Detection Report",
        "",
        "## Summary Stats",
        "```json",
        json.dumps(stats, indent=2),
        "```",
        "",
        "## PCA Distribution",
        f"![]({os.path.basename(paths['pca'])})",
        "",
        "## Isolation Forest Distribution",
        f"![]({os.path.basename(paths['if'])})",
        "",
        "## Autoencoder Distribution",
        f"![]({os.path.basename(paths['ae'])})",
        "",
        "## Composite Distribution (0–1)",
        f"![]({os.path.basename(paths['composite'])})",
        "",
        "## Top Composite Anomalies",
        f"![]({os.path.basename(paths['top'])})",
        "",
        "## All Composite Scores",
        f"![]({os.path.basename(paths['scatter'])})",
    ]

    if paths.get("returns"):
        md += [
            "",
            "## Composite Anomaly vs Annual Return",
            f"![]({os.path.basename(paths['returns'])})",
        ]

    md += [
        "",
        "---",
        "_Generated by run_anomaly_pipeline.py_",
        "",
    ]

    with open(report_path, "w") as f:
        f.write("\n".join(md))

    return report_path


# -------------------------------
# DIRECT EXECUTION
# -------------------------------
if __name__ == "__main__":
    print("Running full anomaly pipeline (PCA + IF + AE) with 0–1 composite...")

    # 1) Run models
    print(" - PCA...")
    pca = run_pca_anomaly_detection(DEFAULT_CSV)
    print(" - Isolation Forest...")
    iso = run_isolation_forest(DEFAULT_CSV)
    print(" - Autoencoder...")
    ae  = run_autoencoder(DEFAULT_CSV)

    # 2) Align / composite
    df = _align(pca, iso, ae)
    df["composite_score_raw"] = _combine(df, weights=(0.50, 0.40, 0.10))
    df["composite_score"] = _minmax(df["composite_score_raw"])

    # 3) Attach returns (for scatter)
    df = _attach_returns(df, DEFAULT_CSV)

    # 4) Persist CSV (sorted by composite 0–1 desc)
    CSV_OUT = os.path.join(COMPOSITE_OUTPUT_DIR, "composite_scores.csv")
    df.sort_values("composite_score", ascending=False).to_csv(CSV_OUT, index=False)

    # 5) Per-model dists + composite dist (0–1)
    pca_hist_path = _hist(df["pca_score"], "PCA Anomaly Score (Higher = More Anomalous)",
                          "pca_hist.png", "PCA Score")
    if_hist_path  = _hist(df["if_score"], "Isolation Forest Anomaly Score (Higher = More Anomalous)",
                          "if_hist.png", "IF Score")
    ae_hist_path  = _hist(df["ae_score"], "Autoencoder Reconstruction Error (Higher = More Anomalous)",
                          "ae_hist.png", "AE Score")
    comp_hist_path = _hist(df["composite_score"], "Composite Anomaly Score (0–1)",
                           "composite_hist.png", "Composite Score (0–1)")

    # 6) Visualize all tickers & Top-N (composite 0–1)
    top_bar_path = _bar_top(df["composite_score"], df["ticker"],
                            "Composite Anomaly Scores", "composite_top50.png", top_n=50,
                            xlabel="Composite Score (0–1)")
    df_sorted = df.sort_values("ticker")
    all_scatter_path = _scatter(df_sorted["composite_score"], df_sorted["ticker"],
                                "Composite Anomaly Scores — All Tickers",
                                "composite_scatter.png", ylabel="Composite Score (0–1)")

    # 7) Anomaly vs Returns scatter
    returns_scatter_path = _scatter_returns(df)

    # 8) Summary stats & report
    stats = {
        "count": int(df.shape[0]),
        "pca_mean": float(df["pca_score"].mean()),
        "if_mean": float(df["if_score"].mean()),
        "ae_mean": float(df["ae_score"].mean()),
        "comp_mean_0to1": float(df["composite_score"].mean()),
    }

    top_pct = 0.05
    n = int(len(df) * top_pct)

    top_pca = set(df.nlargest(n, "pca_score")["ticker"])
    top_if  = set(df.nlargest(n, "if_score")["ticker"])
    top_ae  = set(df.nlargest(n, "ae_score")["ticker"])

    agreement = {
        "PCA ∩ IF": len(top_pca & top_if),
        "PCA ∩ AE": len(top_pca & top_ae),
        "IF ∩ AE": len(top_if & top_ae),
        "PCA ∩ IF ∩ AE": len(top_pca & top_if & top_ae),
    }
    print(agreement)


    report_path = _report(
        stats,
        paths={
            "pca": pca_hist_path,
            "if": if_hist_path,
            "ae": ae_hist_path,
            "composite": comp_hist_path,
            "top": top_bar_path,
            "scatter": all_scatter_path,
            "returns": returns_scatter_path,
        },
    )