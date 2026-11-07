import pandas as pd
import numpy as np


SEED = 42
np.random.seed(SEED)

DATA_PATH = "data\cleaned_esg.csv"   # <- change this
df = pd.read_csv(DATA_PATH)

print("Shape:", df.shape)
print("Columns:", list(df.columns))
print(df.head(3))

# Find the company ID column (edit the name if yours differs)
ID_COL = "CompanyID"
print("\nUnique companies:", df[ID_COL].nunique())
print("Rows per company (min/max):",
      df.groupby(ID_COL).size().min(), df.groupby(ID_COL).size().max())

# Check the North/South grouping
north = ["Europe", "North America", "Oceania"]
df["North"] = df["Region"].isin(north).astype(int)
print("\nRegion counts:\n", df["Region"].value_counts())
print("\nNorth vs Market_Type crosstab:\n",
      pd.crosstab(df["North"], df["Market_Type"]))

# Confirm no company appears in both groups
print("\nCompanies with >1 region:",
      (df.groupby(ID_COL)["Region"].nunique() > 1).sum())