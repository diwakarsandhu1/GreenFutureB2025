"""
analyze_merged_models.py

Quick analysis script to compare baseline vs tuned results for:
- PCA
- Isolation Forest
- Autoencoder

Focuses only on:
    • Improvement percentages
    • Mean improvement magnitudes
    • Summary of tuning effectiveness
"""

import pandas as pd
from pathlib import Path

# ================================
# Load merged comparison file
# ================================
CSV_PATH = Path("hyper_outputs/merged_comparison_all_models.csv")

if not CSV_PATH.exists():
    raise FileNotFoundError(f"Could not find file: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

print("====================================")
print("Loaded merged comparison results")
print("====================================")
print(f"Rows: {len(df)}\nColumns:\n{list(df.columns)}\n")


# ================================
# Improvement metrics
# ================================

print("====================================")
print(" 1) Mean Improvement (Tuned - Baseline)")
print("====================================")

improvements = {}

if "diff_pca" in df.columns:
    improvements["PCA_mean_improvement"] = df["diff_pca"].mean()

if "diff_if" in df.columns:
    improvements["IF_mean_change"] = df["diff_if"].mean()

if "diff_ae" in df.columns:
    improvements["AE_mean_improvement"] = df["diff_ae"].mean()

for k, v in improvements.items():
    print(f"{k}: {v:.6f}")


# ================================
# Improvement percentages
# ================================
print("\n====================================")
print(" 2) Improvement Percentages")
print("====================================")

if "diff_pca" in df.columns:
    pca_improvement_rate = (df["diff_pca"] < 0).mean() * 100
    print(f"PCA improved for {pca_improvement_rate:.2f}% of tickers (lower = better)")

if "diff_ae" in df.columns:
    ae_improvement_rate = (df["diff_ae"] < 0).mean() * 100
    print(f"AE improved for {ae_improvement_rate:.2f}% of tickers (lower = better)")

print("(Isolation Forest uses a stability metric, so lower != better.\n"
      "We check magnitude instead.)")


# ================================
# IF magnitude change
# ================================
print("\n====================================")
print(" 3) Isolation Forest Change Magnitude")
print("====================================")

if "diff_if" in df.columns:
    if_change_mag = df["diff_if"].abs().mean()
    print(f"Mean |difference| in IF anomaly score: {if_change_mag:.6f}")
else:
    print("IF columns not found.")


# ================================
# Summary
# ================================
print("\n====================================")
print(" SUMMARY OF TUNING EFFECTIVENESS")
print("====================================")

if "diff_pca" in df.columns:
    print(f"✔ PCA tuning improved {pca_improvement_rate:.1f}% of tickers")

if "diff_ae" in df.columns:
    print(f"✔ AE tuning improved {ae_improvement_rate:.1f}% of tickers")

if "diff_if" in df.columns:
    print(f"✔ IF tuning caused avg score shift of {if_change_mag:.6f}")

print("\nDone.\n")
