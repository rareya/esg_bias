import pandas as pd, numpy as np
from itertools import combinations

SEED, B = 42, 5000
oof = pd.read_csv("oof_predictions_tuned2.csv")
oof["e"] = (oof["ESG_Overall"] - oof["pred"]).abs()
oof["one"] = 1

c = oof.groupby("CompanyID").agg(region=("Region", "first"),
                                 n=("one", "sum"), e=("e", "sum"))
regions = sorted(c["region"].unique())
N = len(c)
n = c["n"].values.astype(float)
e = c["e"].values.astype(float)
masks = {r: (c["region"].values == r).astype(float) for r in regions}

rng = np.random.default_rng(SEED)
idx = rng.integers(0, N, size=(B, N))          # company-level resamples

def mae(r, ix):
    m = masks[r]
    return (e * m)[ix].sum(-1) / (n * m)[ix].sum(-1)

point = {r: (e * masks[r]).sum() / (n * masks[r]).sum() for r in regions}
boot = {r: mae(r, idx) for r in regions}

# 1) Per-region MAE with 95% CI
rows = []
for r in regions:
    lo, hi = np.percentile(boot[r], [2.5, 97.5])
    rows.append({"Region": r, "Companies": int(masks[r].sum()),
                 "MAE": point[r], "CI_low": lo, "CI_high": hi})
reg = pd.DataFrame(rows).sort_values("MAE")
print("=== Per-region MAE (company bootstrap, 95% CI) ===")
print(reg.round(4).to_string(index=False))
reg.round(4).to_csv("table_region_bootstrap.csv", index=False)

# 2) All pairwise differences, Holm-adjusted
pairs = []
for a, b in combinations(regions, 2):
    d = boot[a] - boot[b]
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    lo, hi = np.percentile(d, [2.5, 97.5])
    pairs.append({"A": a, "B": b, "Diff_A_minus_B": point[a] - point[b],
                  "CI_low": lo, "CI_high": hi, "p_raw": p})
pw = pd.DataFrame(pairs).sort_values("p_raw").reset_index(drop=True)

m = len(pw)
holm = np.maximum.accumulate((m - np.arange(m)) * pw["p_raw"].values)
pw["p_holm"] = np.minimum(1.0, holm)
pw["Significant_Holm_0.05"] = pw["p_holm"] < 0.05

print("\n=== Pairwise differences (21 pairs, Holm-adjusted) ===")
print(pw.round(4).to_string(index=False))
print("\nPairs significant after Holm:", int(pw["Significant_Holm_0.05"].sum()), "/", m)
print("Pairs with raw p < 0.05    :", int((pw["p_raw"] < 0.05).sum()), "/", m)
pw.round(4).to_csv("table_region_pairwise.csv", index=False)

# 3) Within-North and within-South spread (does the N/S split explain the pattern?)
north = {"Europe", "North America", "Oceania"}
print("\nRegion MAE ordering:", " < ".join(reg["Region"]))
print("North regions:", sorted(north & set(regions)),
      "| South regions:", sorted(set(regions) - north))
print("Saved: table_region_bootstrap.csv, table_region_pairwise.csv")