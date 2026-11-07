import pandas as pd, numpy as np, json
import xgboost as xgb
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
BASE = NUM + ["Carbon_per_Rev", "Water_per_Rev", "Energy_per_Rev", "Carbon_per_Energy"]

VARIANTS = {"full": ["Industry", "Region", "Market_Type"],
            "no_MarketType": ["Industry", "Region"]}

cfg = json.load(open("final_config_v2.json"))
y = df[TARGET].values
groups = df[ID].values
rng = np.random.default_rng(SEED)
companies = np.array(sorted(df[ID].unique())); rng.shuffle(companies)
fold_of = {c: i % 5 for i, c in enumerate(companies)}
fold_id = np.array([fold_of[c] for c in groups])

def parent(col):
    for p in ["Industry_", "Region_", "Market_Type_"]:
        if col.startswith(p):
            return p[:-1]
    return col

order = np.unique(groups); N = len(order)
def csum(v): return pd.Series(v).groupby(groups).sum().reindex(order).values.astype(float)
north_c = df.groupby(ID)["North"].first().reindex(order).values.astype(float)
south_c = 1 - north_c
n_c = csum(np.ones(len(df)))

results = {}
for name, cat in VARIANTS.items():
    X = pd.get_dummies(df[BASE + cat], columns=cat, drop_first=False).astype(float)
    cols = list(X.columns)
    contrib = np.zeros((len(df), len(cols)))
    pred = np.zeros(len(df)); pred_from_shap = np.zeros(len(df))
    for k in range(5):
        tr, te = np.where(fold_id != k)[0], np.where(fold_id == k)[0]
        m = XGBRegressor(**cfg, random_state=SEED, n_jobs=-1).fit(X.iloc[tr], y[tr])
        c = m.get_booster().predict(xgb.DMatrix(X.iloc[te]), pred_contribs=True)
        contrib[te] = c[:, :-1]
        pred[te] = m.predict(X.iloc[te])
        pred_from_shap[te] = c.sum(axis=1)
    print(f"{name}: additivity check, max |pred - sum(SHAP)| = "
          f"{np.abs(pred - pred_from_shap).max():.2e}")
    S = pd.DataFrame(contrib, columns=cols)
    G = S.T.groupby([parent(c) for c in cols]).sum().T     # grouped SHAP per row
    results[name] = (G, pred)

# ---- 1) Global importance, grouped ----
print("\n=== Mean |SHAP| by feature (out-of-fold, all 11,000 rows) ===")
imp = pd.DataFrame({n: results[n][0].abs().mean() for n in results}).fillna(0)
imp = imp.sort_values("full", ascending=False)
print(imp.round(3).to_string())

# ---- 2) Combined geography ----
Gf = results["full"][0]
geo_full = Gf["Region"] + Gf["Market_Type"]            # signed sum per row
geo_nm = results["no_MarketType"][0]["Region"]
print("\n=== Combined geography importance (mean |SHAP|) ===")
print(f"full  (Region + Market_Type summed per row): {geo_full.abs().mean():.3f}")
print(f"full  Region alone: {Gf['Region'].abs().mean():.3f} | "
      f"Market_Type alone: {Gf['Market_Type'].abs().mean():.3f}")
print(f"no_MarketType (Region only): {geo_nm.abs().mean():.3f}")

# ---- 3) Geography contribution to the North-South prediction gap ----
isN = df["North"].values == 1
print("\n=== Mean geography SHAP by group (points of predicted ESG score) ===")
for lab, g in [("full", geo_full.values), ("no_MarketType", geo_nm.values)]:
    print(f"{lab:14s} North {g[isN].mean():+.2f} | South {g[~isN].mean():+.2f} | "
          f"North minus South {g[isN].mean() - g[~isN].mean():+.2f}")
pf = results["full"][1]
print(f"Predicted score gap (N - S): {pf[isN].mean() - pf[~isN].mean():.2f} | "
      f"actual gap: {y[isN].mean() - y[~isN].mean():.2f}")

# Company-bootstrap CI on the geography-attributed gap (full model)
gc = csum(geo_full.values)
brng = np.random.default_rng(SEED)
idx = brng.integers(0, N, size=(B, N))
gap_b = (gc * north_c)[idx].sum(-1) / (n_c * north_c)[idx].sum(-1) \
      - (gc * south_c)[idx].sum(-1) / (n_c * south_c)[idx].sum(-1)
print(f"Geography-attributed N-S gap (full): "
      f"{(gc*north_c).sum()/(n_c*north_c).sum() - (gc*south_c).sum()/(n_c*south_c).sum():+.2f} "
      f"[{np.percentile(gap_b, 2.5):+.2f}, {np.percentile(gap_b, 97.5):+.2f}]")

# ---- 4) By region ----
print("\n=== Mean geography SHAP by region (full model) ===")
print(pd.Series(geo_full.values).groupby(df["Region"].values).mean().sort_values(ascending=False).round(2).to_string())

imp.round(4).to_csv("table_shap_grouped.csv")
pd.DataFrame({"geo_full": geo_full.values, "geo_noMT": geo_nm.values,
              "Region": df["Region"], "North": df["North"]}).to_csv("shap_geography_rows.csv", index=False)
print("\nSaved: table_shap_grouped.csv, shap_geography_rows.csv")