import pandas as pd, numpy as np, json
import xgboost as xgb
from xgboost import XGBRegressor

SEED, R = 42, 5
DELTAS = [0, 2, 5, 10, 16]
DATA_PATH = "data\cleaned_esg.csv"   # <- same path
df = pd.read_csv(DATA_PATH)
ID, TARGET = "CompanyID", "ESG_Overall"
north = ["Europe", "North America", "Oceania"]
df["North"] = df["Region"].isin(north).astype(int)
is_n = df["North"].values == 1
south = 1.0 - df["North"].values
groups = df[ID].values
order = np.unique(groups)
comp_idx = np.searchsorted(order, groups)
NC, T = len(order), 11

# ---------- generator calibrated on the real data ----------
LOGC = ["Revenue", "MarketCap", "CarbonEmissions", "WaterUsage", "EnergyConsumption"]
for c in LOGC:
    df["log_" + c] = np.log(df[c])
df["GR_f"] = df["GrowthRate"].fillna(df["GrowthRate"].median())
GEN = ["log_Revenue", "ProfitMargin", "log_MarketCap", "GR_f",
       "log_CarbonEmissions", "log_WaterUsage", "log_EnergyConsumption"]
ind = pd.get_dummies(df["Industry"], drop_first=True).astype(float).values
Xg = np.column_stack([np.ones(len(df)), df[GEN].values, ind])
y_real = df[TARGET].values

Xfit = np.column_stack([Xg, df["North"].values])
coef = np.linalg.lstsq(Xfit, y_real, rcond=None)[0]
beta = coef[:-1].copy()
beta[0] += coef[-1] * df["North"].mean()        # centre the clean score
resid = y_real - Xfit @ coef
cm = pd.Series(resid).groupby(groups).transform("mean").values
s_w = np.sqrt(((resid - cm) ** 2).sum() / (len(df) - NC))
var_cm = pd.Series(resid).groupby(groups).mean().var()
s_u = np.sqrt(max(var_cm - s_w ** 2 / T, 0))
print(f"Generator: R2 on real data {1 - (resid**2).sum()/((y_real-y_real.mean())**2).sum():.3f} | "
      f"SD feature part {(Xg @ beta).std():.2f} | company-effect SD {s_u:.2f} | yearly-noise SD {s_w:.2f}")

# ---------- models ----------
NUM = ["Year", "Revenue", "ProfitMargin", "MarketCap", "GrowthRate",
       "CarbonEmissions", "WaterUsage", "EnergyConsumption"]
eps = 1e-9
df["Carbon_per_Rev"] = df["CarbonEmissions"] / (df["Revenue"] + eps)
df["Water_per_Rev"] = df["WaterUsage"] / (df["Revenue"] + eps)
df["Energy_per_Rev"] = df["EnergyConsumption"] / (df["Revenue"] + eps)
df["Carbon_per_Energy"] = df["CarbonEmissions"] / (df["EnergyConsumption"] + eps)
BASE = NUM + ["Carbon_per_Rev", "Water_per_Rev", "Energy_per_Rev", "Carbon_per_Energy"]
X_full = pd.get_dummies(df[BASE + ["Industry", "Region", "Market_Type"]],
                        columns=["Industry", "Region", "Market_Type"]).astype(float)
X_nogeo = pd.get_dummies(df[BASE + ["Industry"]], columns=["Industry"]).astype(float)
geo_idx = [i for i, c in enumerate(X_full.columns)
           if c.startswith("Region_") or c.startswith("Market_Type_")]
cfg = json.load(open("final_config_v2.json"))

rng0 = np.random.default_rng(SEED)
comps = order.copy(); rng0.shuffle(comps)
fold_of = {c: i % 5 for i, c in enumerate(comps)}
fold_id = np.array([fold_of[c] for c in groups])

mask = df["GrowthRate"].notna().values          # B-O sample (drops first-year rows)
def bo(X, y, nm):
    Xn, Xs, yn, ys = X[nm], X[~nm], y[nm], y[~nm]
    bn = np.linalg.lstsq(Xn, yn, rcond=None)[0]
    bs = np.linalg.lstsq(Xs, ys, rcond=None)[0]
    xn, xs = Xn.mean(0), Xs.mean(0)
    return yn.mean() - ys.mean(), (xn - xs) @ bs, xs @ (bn - bs)

rows = []
for delta in DELTAS:
    for r in range(R):
        rng = np.random.default_rng(SEED + 1000 * r + 7)      # same noise across deltas
        u = rng.normal(0, s_u, NC)[comp_idx]
        e = rng.normal(0, s_w, len(df))
        y = Xg @ beta + u + e - delta * south

        tot, expl, unexpl = bo(Xg[mask], y[mask], is_n[mask])

        pf, pn, geo = np.zeros(len(df)), np.zeros(len(df)), np.zeros(len(df))
        for k in range(5):
            tr, te = np.where(fold_id != k)[0], np.where(fold_id == k)[0]
            m = XGBRegressor(**cfg, random_state=SEED, n_jobs=-1).fit(X_full.iloc[tr], y[tr])
            c = m.get_booster().predict(xgb.DMatrix(X_full.iloc[te]), pred_contribs=True)
            geo[te] = c[:, geo_idx].sum(1)
            pf[te] = m.predict(X_full.iloc[te])
            m2 = XGBRegressor(**cfg, random_state=SEED, n_jobs=-1).fit(X_nogeo.iloc[tr], y[tr])
            pn[te] = m2.predict(X_nogeo.iloc[te])

        ef, en = np.abs(y - pf), y - pn
        rows.append({"delta": delta, "rep": r,
                     "raw_gap": y[is_n].mean() - y[~is_n].mean(),
                     "BO_unexplained": unexpl, "BO_explained": expl,
                     "SHAP_geo_gap": geo[is_n].mean() - geo[~is_n].mean(),
                     "noGeo_bias_diff": en[is_n].mean() - en[~is_n].mean(),
                     "full_MAE_gap": ef[~is_n].mean() - ef[is_n].mean(),
                     "full_R2": 1 - ((y - pf) ** 2).sum() / ((y - y.mean()) ** 2).sum()})
    print(f"delta {delta} done", flush=True)

res = pd.DataFrame(rows)
res.to_csv("sim_results_raw.csv", index=False)
cols = ["raw_gap", "BO_unexplained", "BO_explained", "SHAP_geo_gap",
        "noGeo_bias_diff", "full_MAE_gap", "full_R2"]
m, s = res.groupby("delta")[cols].mean(), res.groupby("delta")[cols].std()
print("\n=== Mean over 5 replicates (true bias = delta) ===")
print(m.round(2).to_string())
print("\n=== SD over replicates ===")
print(s.round(2).to_string())
print("\n=== Recovery ratios: measure / delta (delta > 0) ===")
rec = pd.DataFrame({c: m[c] / m.index.to_series() for c in
                    ["BO_unexplained", "SHAP_geo_gap", "noGeo_bias_diff"]}).drop(index=0)
print(rec.round(3).to_string())
m.round(3).to_csv("sim_summary.csv")
print("\nSaved: sim_results_raw.csv, sim_summary.csv")