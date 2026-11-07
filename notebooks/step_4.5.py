import pandas as pd, numpy as np, json
from xgboost import XGBRegressor

SEED, R = 42, 5
DELTAS = [0, 10, 16]
ALPHAS = [0.05, 0.10, 0.175, 0.25, 0.35, 0.431]
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
regions = df["Region"].values

DII = {"Oceania": 1.0, "North America": 0.826, "Europe": 0.615,
       "Latin America": 0.282, "Asia": 0.257, "Middle East": 0.242, "Africa": 0.0}
dii = df["Region"].map(DII).values

# ---------- generator (same calibration as step 4.4) ----------
for c in ["Revenue", "MarketCap", "CarbonEmissions", "WaterUsage", "EnergyConsumption"]:
    df["log_" + c] = np.log(df[c])
df["GR_f"] = df["GrowthRate"].fillna(df["GrowthRate"].median())
GEN = ["log_Revenue", "ProfitMargin", "log_MarketCap", "GR_f",
       "log_CarbonEmissions", "log_WaterUsage", "log_EnergyConsumption"]
ind = pd.get_dummies(df["Industry"], drop_first=True).astype(float).values
Xg = np.column_stack([np.ones(len(df)), df[GEN].values, ind])
y_real = df[TARGET].values
Xfit = np.column_stack([Xg, df["North"].values])
coef = np.linalg.lstsq(Xfit, y_real, rcond=None)[0]
beta = coef[:-1].copy(); beta[0] += coef[-1] * df["North"].mean()
resid = y_real - Xfit @ coef
cm = pd.Series(resid).groupby(groups).transform("mean").values
s_w = np.sqrt(((resid - cm) ** 2).sum() / (len(df) - NC))
var_cm = pd.Series(resid).groupby(groups).mean().var()
s_u = np.sqrt(max(var_cm - s_w ** 2 / T, 0))

# ---------- features ----------
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
cfg = json.load(open("final_config_v2.json"))

rng0 = np.random.default_rng(SEED)
comps = order.copy(); rng0.shuffle(comps)
fold_of = {c: i % 5 for i, c in enumerate(comps)}
fold_id = np.array([fold_of[c] for c in groups])
mask = df["GrowthRate"].notna().values

def bo_unexplained(X, y, nm):
    Xn, Xs, yn, ys = X[nm], X[~nm], y[nm], y[~nm]
    bn = np.linalg.lstsq(Xn, yn, rcond=None)[0]
    bs = np.linalg.lstsq(Xs, ys, rcond=None)[0]
    return Xs.mean(0) @ (bn - bs)

def fit_pred(X, target, tr, te):
    m = XGBRegressor(**cfg, random_state=SEED, n_jobs=-1).fit(X.iloc[tr], target[tr])
    return m.predict(X.iloc[te])

METHODS = ["oracle", "uncorrected", "drop_geo", "BO_shift"] + [f"DAES_{a}" for a in ALPHAS]
rows = []
for delta in DELTAS:
    for r in range(R):
        rng = np.random.default_rng(SEED + 1000 * r + 7)
        u = rng.normal(0, s_u, NC)[comp_idx]
        e = rng.normal(0, s_w, len(df))
        clean = Xg @ beta + u + e
        y_obs = clean - delta * south

        P = {m: np.zeros(len(df)) for m in METHODS}
        for k in range(5):
            tr, te = np.where(fold_id != k)[0], np.where(fold_id == k)[0]
            P["oracle"][te] = fit_pred(X_full, clean, tr, te)
            P["uncorrected"][te] = fit_pred(X_full, y_obs, tr, te)
            P["drop_geo"][te] = fit_pred(X_nogeo, y_obs, tr, te)
            trm = tr[mask[tr]]
            uhat = bo_unexplained(Xg[trm], y_obs[trm], is_n[trm])   # training companies only
            P["BO_shift"][te] = fit_pred(X_full, y_obs + uhat * south, tr, te)
            for a in ALPHAS:
                P[f"DAES_{a}"][te] = fit_pred(X_full, y_obs * (1 + a * (1 - dii)), tr, te)

        for mth in METHODS:
            err = clean - P[mth]
            reg_means = pd.Series(err).groupby(regions).mean()
            rows.append({"delta": delta, "rep": r, "method": mth,
                         "RMSE_clean": np.sqrt((err ** 2).mean()),
                         "RMSE_obs": np.sqrt(((y_obs - P[mth]) ** 2).mean()),
                         "level": err.mean(),
                         "bias_diff": err[is_n].mean() - err[~is_n].mean(),
                         "region_dev": (reg_means - err.mean()).abs().max()})
        print(f"delta {delta} rep {r} done", flush=True)

res = pd.DataFrame(rows)
res.to_csv("sim_daes_raw.csv", index=False)
MET = ["RMSE_clean", "RMSE_obs", "level", "bias_diff", "region_dev"]
summ = []
for delta in DELTAS:
    sub = res[res.delta == delta].groupby("method")[MET].agg(["mean", "std"]).reindex(METHODS)
    print(f"\n=== delta = {delta}  (mean ± SD over {R} replicates; truth = clean score) ===")
    out = pd.DataFrame({c: [f"{sub[(c, 'mean')][m]:+.2f}±{sub[(c, 'std')][m]:.2f}" for m in METHODS]
                        for c in MET}, index=METHODS)
    print(out.to_string())
    best = sub[("RMSE_clean", "mean")].filter(like="DAES").idxmin()
    print(f"Best DAES by RMSE_clean: {best}")
    s = sub.copy(); s.columns = [f"{a}_{b}" for a, b in s.columns]; s["delta"] = delta
    summ.append(s)
pd.concat(summ).round(3).to_csv("sim_daes_summary.csv")
print("\nSaved: sim_daes_raw.csv, sim_daes_summary.csv")