import pandas as pd, numpy as np, json
from xgboost import XGBRegressor

SEED, B = 42, 5000
DATA_PATH = "data\cleaned_esg.csv"   # <- same path
df = pd.read_csv(DATA_PATH)
ID, TARGET = "CompanyID", "ESG_Overall"
north = ["Europe", "North America", "Oceania"]
df["North"] = df["Region"].isin(north).astype(int)

NUM = ["Year", "Revenue", "ProfitMargin", "MarketCap", "GrowthRate",
       "CarbonEmissions", "WaterUsage", "EnergyConsumption"]
eps = 1e-9
df["Carbon_per_Rev"] = df["CarbonEmissions"] / (df["Revenue"] + eps)
df["Water_per_Rev"]  = df["WaterUsage"] / (df["Revenue"] + eps)
df["Energy_per_Rev"] = df["EnergyConsumption"] / (df["Revenue"] + eps)
df["Carbon_per_Energy"] = df["CarbonEmissions"] / (df["EnergyConsumption"] + eps)
INTENS = ["Carbon_per_Rev", "Water_per_Rev", "Energy_per_Rev", "Carbon_per_Energy"]
BASE = NUM + INTENS

VARIANTS = {
    "full":          (BASE, ["Industry", "Region", "Market_Type"]),
    "no_MarketType": (BASE, ["Industry", "Region"]),
    "no_geography":  (BASE, ["Industry"]),
    "region_only":   ([],   ["Region"]),
}

cfg = json.load(open("final_config_v2.json"))
y = df[TARGET].values
groups = df[ID].values

# same outer folds as every earlier step
rng = np.random.default_rng(SEED)
companies = np.array(sorted(df[ID].unique()))
rng.shuffle(companies)
fold_of = {c: i % 5 for i, c in enumerate(companies)}
fold_id = np.array([fold_of[c] for c in groups])

# ---- out-of-fold predictions for each variant ----
oof = {}
for name, (num, cat) in VARIANTS.items():
    X = pd.get_dummies(df[num + cat], columns=cat, drop_first=False)
    pred = np.zeros(len(df))
    for k in range(5):
        tr, te = np.where(fold_id != k)[0], np.where(fold_id == k)[0]
        m = XGBRegressor(**cfg, random_state=SEED, n_jobs=-1).fit(X.iloc[tr], y[tr])
        pred[te] = m.predict(X.iloc[te])
    oof[name] = pred
    print(f"{name:14s} done | pooled R2 {1 - ((y-pred)**2).sum()/((y-y.mean())**2).sum():.4f}", flush=True)

# ---- per-company sums for a fast company-level bootstrap ----
order = np.unique(groups)
N = len(order)
def csum(v): return pd.Series(v).groupby(groups).sum().reindex(order).values.astype(float)
north_c = df.groupby(ID)["North"].first().reindex(order).values.astype(float)
south_c = 1 - north_c
n_c, y_c, y2_c = csum(np.ones(len(df))), csum(y), csum(y**2)

C = {}
for name, pred in oof.items():
    r = y - pred
    C[name] = {"e": csum(np.abs(r)), "se": csum(r**2), "s": csum(r)}

def stats(name, idx):
    a = C[name]
    g = lambda v: v[idx].sum(axis=-1)
    nt = g(n_c)
    mae_n = g(a["e"] * north_c) / g(n_c * north_c)
    mae_s = g(a["e"] * south_c) / g(n_c * south_c)
    return {"R2": 1 - g(a["se"]) / (g(y2_c) - g(y_c)**2 / nt),
            "MAE": g(a["e"]) / nt,
            "Gap": mae_s - mae_n,
            "Bias_North": g(a["s"] * north_c) / g(n_c * north_c),
            "Bias_South": g(a["s"] * south_c) / g(n_c * south_c)}

pt_idx = np.arange(N)
brng = np.random.default_rng(SEED)
idx = brng.integers(0, N, size=(B, N))
point = {k: stats(k, pt_idx) for k in C}
boot = {k: stats(k, idx) for k in C}

print("\n=== Each model: estimate [95% CI]  (Bias = mean of actual minus predicted; negative = over-prediction) ===")
rows = []
for k in C:
    line = [f"{k:14s}"]
    for q in ["R2", "MAE", "Gap", "Bias_North", "Bias_South"]:
        lo, hi = np.percentile(boot[k][q], [2.5, 97.5])
        line.append(f"{q} {point[k][q]:+.3f} [{lo:+.3f},{hi:+.3f}]")
        rows.append({"model": k, "quantity": q, "estimate": point[k][q], "ci_low": lo, "ci_high": hi})
    print(" | ".join(line))
pd.DataFrame(rows).round(4).to_csv("table_ablation_models.csv", index=False)

print("\n=== Paired difference vs full model (variant minus full) ===")
rows = []
for k in ["no_MarketType", "no_geography", "region_only"]:
    for q in ["R2", "MAE", "Gap"]:
        d = boot[k][q] - boot["full"][q]
        p = point[k][q] - point["full"][q]
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"{k:14s} d{q:4s} {p:+.4f} [{lo:+.4f}, {hi:+.4f}] | share <0: {(d < 0).mean():.3f}")
        rows.append({"model": k, "quantity": "d" + q, "diff": p, "ci_low": lo, "ci_high": hi})
pd.DataFrame(rows).round(4).to_csv("table_ablation_paired.csv", index=False)

pd.DataFrame({"CompanyID": df[ID], "Region": df["Region"], "North": df["North"],
              "y": y, **{f"pred_{k}": v for k, v in oof.items()}}
             ).to_csv("oof_predictions_ablation.csv", index=False)
print("\nSaved: table_ablation_models.csv, table_ablation_paired.csv, oof_predictions_ablation.csv")