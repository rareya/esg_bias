import pandas as pd
import numpy as np
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

# --- engineered features (use only the firm's own features, never the target) ---
eps = 1e-9
df["Carbon_per_Rev"] = df["CarbonEmissions"] / (df["Revenue"] + eps)
df["Water_per_Rev"]  = df["WaterUsage"] / (df["Revenue"] + eps)
df["Energy_per_Rev"] = df["EnergyConsumption"] / (df["Revenue"] + eps)
df["Carbon_per_Energy"] = df["CarbonEmissions"] / (df["EnergyConsumption"] + eps)
INTENS = ["Carbon_per_Rev", "Water_per_Rev", "Energy_per_Rev", "Carbon_per_Energy"]

base_cols = ["Revenue", "ProfitMargin", "MarketCap", "GrowthRate",
             "CarbonEmissions", "WaterUsage", "EnergyConsumption"]
for c in base_cols:
    df[c + "_cmean"] = df.groupby(ID_COL)[c].transform("mean")
CMEAN = [c + "_cmean" for c in base_cols]

FEATSETS = {
    "A_baseline":        NUM,
    "B_+intensity":      NUM + INTENS,
    "C_+company_means":  NUM + CMEAN,
    "D_+both":           NUM + INTENS + CMEAN,
}

y = df[TARGET].values
groups = df[ID_COL].values
is_north = df["North"].values

# SAME fold assignment as step 1.4
rng = np.random.default_rng(SEED)
companies = np.array(sorted(df[ID_COL].unique()))
rng.shuffle(companies)
fold_of = {c: i % 5 for i, c in enumerate(companies)}
fold_id = np.array([fold_of[c] for c in groups])

PARAMS = dict(n_estimators=300, learning_rate=0.05, max_depth=6,
              subsample=0.8, colsample_bytree=0.8,
              random_state=SEED, n_jobs=-1)

all_rows = []
for name, cols in FEATSETS.items():
    X = pd.get_dummies(df[cols + CAT], columns=CAT, drop_first=False)
    for k in range(5):
        tr, te = np.where(fold_id != k)[0], np.where(fold_id == k)[0]
        m = XGBRegressor(**PARAMS).fit(X.iloc[tr], y[tr])
        p = m.predict(X.iloc[te])
        err = np.abs(y[te] - p)
        n = is_north[te] == 1
        all_rows.append({"set": name, "fold": k,
                         "R2": r2_score(y[te], p),
                         "MAE": mean_absolute_error(y[te], p),
                         "RMSE": np.sqrt(mean_squared_error(y[te], p)),
                         "Gap": err[~n].mean() - err[n].mean()})

res = pd.DataFrame(all_rows)
summ = res.groupby("set").agg(R2=("R2", "mean"), R2_sd=("R2", "std"),
                              MAE=("MAE", "mean"), RMSE=("RMSE", "mean"),
                              Gap=("Gap", "mean")).round(4)
print(summ.to_string())

# paired fold differences vs baseline
base = res[res.set == "A_baseline"].set_index("fold")["R2"]
print("\nPaired R2 difference vs baseline (per fold):")
for name in FEATSETS:
    if name == "A_baseline":
        continue
    d = res[res.set == name].set_index("fold")["R2"] - base
    print(f"{name:18s} mean {d.mean():+.4f} | folds improved: {(d > 0).sum()}/5 | per fold: {d.round(4).tolist()}")

res.to_csv("step2_1_results.csv", index=False)