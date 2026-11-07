import pandas as pd, numpy as np, ast
from sklearn.metrics import r2_score
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

# same outer folds as before
rng = np.random.default_rng(SEED)
companies = np.array(sorted(df[ID_COL].unique()))
rng.shuffle(companies)
fold_of = {c: i % 5 for i, c in enumerate(companies)}
fold_id = np.array([fold_of[c] for c in groups])

res = pd.read_csv("step2_2_results.csv").set_index("fold")
oof_pred = np.zeros(len(df))

for k in range(5):
    cfg = ast.literal_eval(res.loc[k, "best_cfg"])
    tr, te = np.where(fold_id != k)[0], np.where(fold_id == k)[0]
    m = XGBRegressor(**cfg, random_state=SEED, n_jobs=-1).fit(X.iloc[tr], y[tr])
    oof_pred[te] = m.predict(X.iloc[te])
    r2 = r2_score(y[te], oof_pred[te])
    print(f"fold {k}: R2 {r2:.4f} (step 2.2 reported {res.loc[k, 'R2_tuned']:.4f})")

oof = df[[ID_COL, "Region", "North", TARGET]].copy()
oof["pred"] = oof_pred
oof["fold"] = fold_id
oof.to_csv("oof_predictions_tuned.csv", index=False)
print("Saved: oof_predictions_tuned.csv")