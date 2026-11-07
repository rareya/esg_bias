import pandas as pd
import numpy as np

SEED = 42
DATA_PATH = "data\cleaned_esg.csv"   # <- same path as before
df = pd.read_csv(DATA_PATH)

ID_COL = "CompanyID"
TARGET = "ESG_Overall"
north = ["Europe", "North America", "Oceania"]
df["North"] = df["Region"].isin(north).astype(int)

# Predictors from the paper (no ESG sub-pillars, no ID, no name)
NUM = ["Year", "Revenue", "ProfitMargin", "MarketCap", "GrowthRate",
       "CarbonEmissions", "WaterUsage", "EnergyConsumption"]
CAT = ["Industry", "Region", "Market_Type"]

# Sanity checks
print("Missing values:\n", df[NUM + CAT + [TARGET]].isna().sum())
print("\nDtypes:\n", df[NUM + CAT].dtypes)

# Target check: is ESG_Overall the mean of the 3 pillars?
pillar_mean = df[["ESG_Environmental", "ESG_Social", "ESG_Governance"]].mean(axis=1)
print("\nMax |ESG_Overall - pillar mean|:", (df[TARGET] - pillar_mean).abs().max())

print("\nTarget describe:\n", df[TARGET].describe())
print("\nMean ESG by North (1) / South (0):\n", df.groupby("North")[TARGET].mean())
print("Raw score gap (North - South):",
      df[df.North == 1][TARGET].mean() - df[df.North == 0][TARGET].mean())

# Build X (one-hot encoding), y, groups
X = pd.get_dummies(df[NUM + CAT], columns=CAT, drop_first=False)
y = df[TARGET].values
groups = df[ID_COL].values

print("\nX shape:", X.shape)
print("y shape:", y.shape)
print("Number of groups:", len(np.unique(groups)))
print("First 10 columns of X:", list(X.columns[:10]))