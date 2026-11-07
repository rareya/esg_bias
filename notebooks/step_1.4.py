import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

SEED = 42
DATA_PATH = "data\cleaned_esg.csv"   # <- same path
df = pd.read_csv(DATA_PATH)

ID_COL = "CompanyID"
TARGET = "ESG_Overall"
north = ["Europe", "North America", "Oceania"]
df["North"] = df["Region"].isin(north).astype(int)

NUM = ["Year", "Revenue", "ProfitMargin", "MarketCap", "GrowthRate",
       "CarbonEmissions", "WaterUsage", "EnergyConsumption"]
CAT = ["Industry", "Region", "Market_Type"]

X = pd.get_dummies(df[NUM + CAT], columns=CAT, drop_first=False)
y = df[TARGET].values
groups = df[ID_COL].values
is_north = df["North"].values

# Shuffled, reproducible company -> fold assignment (200 companies per fold)
rng = np.random.default_rng(SEED)
companies = np.array(sorted(df[ID_COL].unique()))
rng.shuffle(companies)
fold_of = {c: i % 5 for i, c in enumerate(companies)}
fold_id = np.array([fold_of[c] for c in groups])

rows = []
oof_pred = np.zeros(len(df))

for k in range(5):
    tr = np.where(fold_id != k)[0]
    te = np.where(fold_id == k)[0]

    model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                         subsample=0.8, colsample_bytree=0.8,
                         random_state=SEED, n_jobs=-1)
    model.fit(X.iloc[tr], y[tr])
    pred = model.predict(X.iloc[te])
    oof_pred[te] = pred

    err = np.abs(y[te] - pred)
    n = is_north[te] == 1
    rows.append({
        "fold": k,
        "n_companies": len(np.unique(groups[te])),
        "R2": r2_score(y[te], pred),
        "MAE": mean_absolute_error(y[te], pred),
        "RMSE": np.sqrt(mean_squared_error(y[te], pred)),
        "MAE_North": err[n].mean(),
        "MAE_South": err[~n].mean(),
        "Gap_S_minus_N": err[~n].mean() - err[n].mean(),
    })

res = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(res.round(4).to_string(index=False))

print("\n=== Mean ± Std over 5 folds ===")
for c in ["R2", "MAE", "RMSE", "MAE_North", "MAE_South", "Gap_S_minus_N"]:
    print(f"{c:14s}: {res[c].mean():.4f} ± {res[c].std(ddof=1):.4f}")

print("\nFolds where South error > North error:",
      int((res["Gap_S_minus_N"] > 0).sum()), "/ 5")

# Save for substep 1.5 and for later steps
res.to_csv("cv_results_baseline.csv", index=False)
df_oof = df[[ID_COL, "Region", "North", TARGET]].copy()
df_oof["pred"] = oof_pred
df_oof["fold"] = fold_id
df_oof.to_csv("oof_predictions_baseline.csv", index=False)
print("\nSaved: cv_results_baseline.csv, oof_predictions_baseline.csv")