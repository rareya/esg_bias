import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
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

# Check: is GrowthRate missing only in each company's first year?
first_year = df.groupby(ID_COL)["Year"].transform("min")
print("GrowthRate NaN only in first year:",
      (df.loc[df.GrowthRate.isna(), "Year"] == first_year[df.GrowthRate.isna()]).all())

X = pd.get_dummies(df[NUM + CAT], columns=CAT, drop_first=False)
y = df[TARGET].values
groups = df[ID_COL].values

# Company-level split: 800 train / 200 test
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
tr, te = next(gss.split(X, y, groups))
print("Train companies:", len(np.unique(groups[tr])),
      "| Test companies:", len(np.unique(groups[te])),
      "| Overlap:", len(set(groups[tr]) & set(groups[te])))
print("Train rows:", len(tr), "| Test rows:", len(te))

model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                     subsample=0.8, colsample_bytree=0.8,
                     random_state=SEED, n_jobs=-1)
model.fit(X.iloc[tr], y[tr])
pred = model.predict(X.iloc[te])

print("\nR2  :", round(r2_score(y[te], pred), 4))
print("MAE :", round(mean_absolute_error(y[te], pred), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(y[te], pred)), 4))

# North/South MAE gap
err = np.abs(y[te] - pred)
n_mask = df["North"].values[te] == 1
print("\nMAE North:", round(err[n_mask].mean(), 4))
print("MAE South:", round(err[~n_mask].mean(), 4))
print("Gap (S - N):", round(err[~n_mask].mean() - err[n_mask].mean(), 4))
print("Test companies North/South:",
      df.iloc[te][df.iloc[te].North == 1][ID_COL].nunique(), "/",
      df.iloc[te][df.iloc[te].North == 0][ID_COL].nunique())