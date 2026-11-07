import pandas as pd, numpy as np, json
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

SEED = 42
DATA_PATH = "data\cleaned_esg.csv"   # <- same path
df = pd.read_csv(DATA_PATH)
ID, TARGET = "CompanyID", "ESG_Overall"
north = ["Europe", "North America", "Oceania"]
df["North"] = df["Region"].isin(north).astype(int)
is_n = df["North"].values == 1
south = 1.0 - df["North"].values
y = df[TARGET].values
groups = df[ID].values
regions = df["Region"].values

NUM = ["Year", "Revenue", "ProfitMargin", "MarketCap", "GrowthRate",
       "CarbonEmissions", "WaterUsage", "EnergyConsumption"]
eps = 1e-9
df["Carbon_per_Rev"] = df["CarbonEmissions"] / (df["Revenue"] + eps)
df["Water_per_Rev"] = df["WaterUsage"] / (df["Revenue"] + eps)
df["Energy_per_Rev"] = df["EnergyConsumption"] / (df["Revenue"] + eps)
df["Carbon_per_Energy"] = df["CarbonEmissions"] / (df["EnergyConsumption"] + eps)
BASE = NUM + ["Carbon_per_Rev", "Water_per_Rev", "Energy_per_Rev", "Carbon_per_Energy"]
LOGC = ["Revenue", "MarketCap", "CarbonEmissions", "WaterUsage", "EnergyConsumption",
        "Carbon_per_Rev", "Water_per_Rev", "Energy_per_Rev", "Carbon_per_Energy"]
for c in LOGC:
    df["l_" + c] = np.log(np.clip(df[c], 1e-12, None))
LIN = [c for c in BASE if c not in LOGC] + ["l_" + c for c in LOGC]

X_tree = pd.get_dummies(df[BASE + ["Industry", "Region", "Market_Type"]],
                        columns=["Industry", "Region", "Market_Type"]).astype(float)
X_nogeo = pd.get_dummies(df[BASE + ["Industry"]], columns=["Industry"]).astype(float)
X_lin = pd.get_dummies(df[LIN + ["Industry", "Region", "Market_Type"]],
                       columns=["Industry", "Region", "Market_Type"]).astype(float)
cfg = json.load(open("final_config_v2.json"))

rng = np.random.default_rng(SEED)
order = np.unique(groups); comps = order.copy(); rng.shuffle(comps)
fold_of = {c: i % 5 for i, c in enumerate(comps)}
fold_id = np.array([fold_of[c] for c in groups])

def metrics(pred, idx=None):
    e = y - pred
    sl = slice(None) if idx is None else idx
    return e

# ============ 5.1 Multi-model comparison ============
MODELS = {
    "Ridge": (X_lin, lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                           RidgeCV(alphas=np.logspace(-2, 3, 12)))),
    "RandomForest": (X_tree, lambda: make_pipeline(SimpleImputer(strategy="median"),
        RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5,
                              n_jobs=-1, random_state=SEED))),
    "MLP": (X_lin, lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-2, early_stopping=True,
                     max_iter=500, random_state=SEED))),
    "XGBoost(final)": (X_tree, lambda: XGBRegressor(**cfg, random_state=SEED, n_jobs=-1)),
}
rows = []
for name, (X, make) in MODELS.items():
    for k in range(5):
        tr, te = np.where(fold_id != k)[0], np.where(fold_id == k)[0]
        p = make().fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
        err = np.abs(y[te] - p); n = is_n[te]
        rows.append({"model": name, "fold": k,
                     "R2": 1 - ((y[te] - p) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum(),
                     "MAE": err.mean(), "RMSE": np.sqrt(((y[te] - p) ** 2).mean()),
                     "Gap": err[~n].mean() - err[n].mean()})
    print(name, "done", flush=True)
res = pd.DataFrame(rows)
print("\n=== 5.1 Multi-model comparison (5-fold, company-level; mean ± SD) ===")
out = res.groupby("model").agg(["mean", "std"])[["R2", "MAE", "RMSE", "Gap"]]
for m in MODELS:
    s = out.loc[m]
    print(f"{m:15s} R2 {s[('R2','mean')]:.4f}±{s[('R2','std')]:.4f} | MAE {s[('MAE','mean')]:.3f}±{s[('MAE','std')]:.3f}"
          f" | RMSE {s[('RMSE','mean')]:.3f} | Gap {s[('Gap','mean')]:+.3f}±{s[('Gap','std')]:.3f}")
xr = res[res.model == "XGBoost(final)"].set_index("fold")["R2"]
for m in ["Ridge", "RandomForest", "MLP"]:
    d = xr - res[res.model == m].set_index("fold")["R2"]
    print(f"XGBoost minus {m}: dR2 {d.mean():+.4f} | folds XGBoost better: {(d > 0).sum()}/5")
res.round(4).to_csv("table_multimodel_folds.csv", index=False)

# ============ 5.2 Real-data DAES + baselines ============
DII = {"Oceania": 1.0, "North America": 0.826, "Europe": 0.615,
       "Latin America": 0.282, "Asia": 0.257, "Middle East": 0.242, "Africa": 0.0}
dii = df["Region"].map(DII).values
ind = pd.get_dummies(df["Industry"], drop_first=True).astype(float).values
FIN = ["l_Revenue", "ProfitMargin", "l_MarketCap", "GrowthRate",
       "l_CarbonEmissions", "l_WaterUsage", "l_EnergyConsumption"]
Xbo = np.column_stack([np.ones(len(df)), df[FIN].values, ind])
ok = df["GrowthRate"].notna().values

def bo_unexplained(X, yy, nm):
    bn = np.linalg.lstsq(X[nm], yy[nm], rcond=None)[0]
    bs = np.linalg.lstsq(X[~nm], yy[~nm], rcond=None)[0]
    return X[~nm].mean(0) @ (bn - bs)

ALPHAS = [0.05, 0.10, 0.175, 0.25, 0.35, 0.431]
METHODS = ["uncorrected"] + [f"DAES_{a}" for a in ALPHAS] + ["drop_geo", "BO_shift"]
P = {m: np.zeros(len(df)) for m in METHODS}
for k in range(5):
    tr, te = np.where(fold_id != k)[0], np.where(fold_id == k)[0]
    fit = lambda X, t: XGBRegressor(**cfg, random_state=SEED, n_jobs=-1).fit(X.iloc[tr], t[tr]).predict(X.iloc[te])
    P["uncorrected"][te] = fit(X_tree, y)
    for a in ALPHAS:
        P[f"DAES_{a}"][te] = fit(X_tree, y * (1 + a * (1 - dii)))
    P["drop_geo"][te] = fit(X_nogeo, y)
    trm = tr[ok[tr]]
    uhat = bo_unexplained(Xbo[trm], y[trm], is_n[trm])
    P["BO_shift"][te] = fit(X_tree, y + uhat * south)

rows = []
for m in METHODS:
    p = P[m]; e = y - p
    reg = pd.Series(e).groupby(regions).mean()
    rows.append({"method": m,
                 "RMSE_obs": np.sqrt((e ** 2).mean()),
                 "R2_obs": 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum(),
                 "level": e.mean(),
                 "bias_N": e[is_n].mean(), "bias_S": e[~is_n].mean(),
                 "pred_gap_N_minus_S": p[is_n].mean() - p[~is_n].mean(),
                 "MAE_gap_S_minus_N": np.abs(e)[~is_n].mean() - np.abs(e)[is_n].mean(),
                 "region_dev": (reg - e.mean()).abs().max()})
tab = pd.DataFrame(rows).set_index("method")
pd.set_option("display.width", 220)
print("\n=== 5.2 Real data: all methods evaluated against the ORIGINAL ESG_Overall ===")
print("(bias = actual minus predicted; observed actual gap N-S = %.2f)" % (y[is_n].mean() - y[~is_n].mean()))
print(tab.round(3).to_string())
tab.round(4).to_csv("table_daes_real.csv")
print("\nSaved: table_multimodel_folds.csv, table_daes_real.csv")