# ============================================================
# save_iris_csv.py — Save Iris Dataset as CSV for DVC tracking
# Run this ONCE locally to generate the data file
# Then: dvc add data/iris.csv → git add data/iris.csv.dvc
# ============================================================

import os

import pandas as pd
from sklearn.datasets import load_iris

# ── Load Iris Dataset ─────────────────────────────────────────
iris = load_iris()

# ── Convert to DataFrame ──────────────────────────────────────
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target          # 0=setosa, 1=versicolor, 2=virginica

# ── Save to CSV ───────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
df.to_csv("data/iris.csv", index=False)

print("✅ Saved data/iris.csv — 150 rows, 5 columns")
