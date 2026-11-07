import pandas as pd, numpy as np, json
from collections import Counter
from sklearn.model_selection import GroupKFold, ParameterSampler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

SEED = 42
DATA_PATH = "data\cleaned_esg.csv"   # <- same path
df = pd.read_csv(DATA_PATH)

ID_COL, TARGET = "CompanyID", "ESG_Overall"
north = ["Europe", "North America", "Oceania"]
df["North"] = df["Region"].isin(north).astype(int)

NUM = ["Year", "Revenue", "ProfitMargin", "MarketCap", "GrowthRate",
       "CarbonEmissions", "WaterUsage", "EnergyConsumption"]
CAT = ["Industry", "Region", "Market_Type"]

eps = 1e-9
df["Carbon_per_Rev"] = df["CarbonEmissions"] / (df["Revenue"] + eps)
df["Water_per_Rev"]  = df["WaterUsage"] / (df["Revenue"] + eps)
df["Energy_per_Rev"] = df["EnergyConsumption"] / (df["Revenue"] + eps)
df["Carbon_per_Energy"] = df["CarbonEmissions"] / (df["EnergyConsumption"] + eps)
INTENS = ["Carbon_per_Rev", "Water_per_Rev", "Energy_per_Rev", "Carbon_per_Energy"]

X = pd.get_dummies(df[NUM + INTENS + CAT], columns=CAT, drop_first=False)
y = df[TARGET].values
groups = df[ID_COL].values
is_north = df["North"].values

# same outer folds as before
rng = np.random.default_rng(SEED)
companies = np.array(sorted(df[ID_COL].unique()))
rng.shuffle(companies)
fold_of = {c: i % 5 for i, c in enumerate(companies)}
fold_id = np.array([fold_of[c] for c in groups])

SPACE = {
    "n_estimators":     [150, 300, 500, 800],
    "learning_rate":    [0.02, 0.03, 0.05],
    "max_depth":        [2, 3, 4],
    "min_child_weight": [3, 5, 10, 20],
    "subsample":        [0.7, 0.85, 1.0],
    "colsample_bytree": [0.4, 0.5, 0.6, 0.8],
    "reg_lambda":       [5, 10, 20, 50],
}
prev = json.load(open("final_config.json"))          # config from step 2.2/2.3
configs = [prev] + list(ParameterSampler(SPACE, n_iter=25, random_state=SEED + 1))

prev_res = pd.read_csv("step2_2_results.csv").set_index("fold")

rows, oof_pred = [], np.zeros(len(df))
for k in range(5):
    tr, te = np.where(fold_id != k)[0], np.where(fold_id == k)[0]
    Xtr, ytr, gtr = X.iloc[tr], y[tr], groups[tr]
    inner = GroupKFold(n_splits=3)
    best_cfg, best_score = None, -np.inf
    for cfg in configs:
        sc = []
        for itr, iva in inner.split(Xtr, ytr, gtr):
            m = XGBRegressor(**cfg, random_state=SEED, n_jobs=-1)
            m.fit(Xtr.iloc[itr], ytr[itr])
            sc.append(r2_score(ytr[iva], m.predict(Xtr.iloc[iva])))
        if np.mean(sc) > best_score:
            best_score, best_cfg = np.mean(sc), cfg

    m = XGBRegressor(**best_cfg, random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
    p = m.predict(X.iloc[te]); oof_pred[te] = p
    err = np.abs(y[te] - p); n = is_north[te] == 1
    rows.append({"fold": k, "inner_R2": best_score,
                 "R2_new": r2_score(y[te], p),
                 "R2_prev": prev_res.loc[k, "R2_tuned"],
                 "MAE_new": mean_absolute_error(y[te], p),
                 "RMSE_new": np.sqrt(mean_squared_error(y[te], p)),
                 "Gap_new": err[~n].mean() - err[n].mean(),
                 "best_cfg": json.dumps(best_cfg, sort_keys=True, default=int)})
    print(f"fold {k} done | new R2 {rows[-1]['R2_new']:.4f} vs prev {rows[-1]['R2_prev']:.4f}", flush=True)

res = pd.DataFrame(rows)
d = res.R2_new - res.R2_prev
print("\n", res.drop(columns="best_cfg").round(4).to_string(index=False))
print(f"\nNew  R2: {res.R2_new.mean():.4f} ± {res.R2_new.std(ddof=1):.4f}")
print(f"Prev R2: {res.R2_prev.mean():.4f} ± {res.R2_prev.std(ddof=1):.4f}")
print(f"Paired diff: {d.mean():+.4f} | folds improved: {(d > 0).sum()}/5")
print(f"MAE {res.MAE_new.mean():.4f} | RMSE {res.RMSE_new.mean():.4f} | "
      f"Gap {res.Gap_new.mean():.4f} ± {res.Gap_new.std(ddof=1):.4f}")
adopt = (d > 0).sum() >= 4 and d.mean() >= 0.005
print("\nDECISION:", "ADOPT new config" if adopt else "KEEP step 2.2 config")
print("\nChosen configs per fold:")
for i, c in enumerate(res.best_cfg): print(i, c)

res.to_csv("step2_4_results.csv", index=False)
oof = df[[ID_COL, "Region", "North", TARGET]].copy()
oof["pred"] = oof_pred; oof["fold"] = fold_id
oof.to_csv("oof_predictions_tuned2.csv", index=False)
modal = Counter(res.best_cfg).most_common(1)[0]
json.dump(json.loads(modal[0]), open("final_config_v2.json", "w"), indent=2)
print("Saved: step2_4_results.csv, oof_predictions_tuned2.csv, final_config_v2.json")