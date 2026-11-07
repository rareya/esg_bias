import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

res = pd.read_csv("cv_results_baseline.csv")
oof = pd.read_csv("oof_predictions_baseline.csv")

oof["abs_err"] = (oof["ESG_Overall"] - oof["pred"]).abs()

# 1) Pooled out-of-fold metrics
y, p = oof["ESG_Overall"], oof["pred"]
print("=== Pooled OOF (all 1000 companies, 11000 rows) ===")
print("R2  :", round(r2_score(y, p), 4))
print("MAE :", round(mean_absolute_error(y, p), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(y, p)), 4))
n = oof[oof.North == 1]["abs_err"].mean()
s = oof[oof.North == 0]["abs_err"].mean()
print("MAE North:", round(n, 4), "| MAE South:", round(s, 4),
      "| Gap:", round(s - n, 4))

# 2) Per-region table (OOF, all companies)
reg = (oof.groupby("Region")
          .agg(Companies=("CompanyID", "nunique"),
               MAE=("abs_err", "mean"),
               RMSE=("abs_err", lambda e: np.sqrt((e**2).mean())))
          .sort_values("MAE"))
print("\n=== Per-region OOF error ===")
print(reg.round(4).to_string())

# 3) Paper-ready summary table (mean ± std over folds)
rows = []
for c in ["R2", "MAE", "RMSE", "MAE_North", "MAE_South", "Gap_S_minus_N"]:
    rows.append({"Metric": c, "Mean ± SD": f"{res[c].mean():.4f} ± {res[c].std(ddof=1):.4f}"})
summary = pd.DataFrame(rows)
summary.to_csv("table_cv_summary.csv", index=False)
reg.round(4).to_csv("table_region_oof.csv")
print("\n=== LaTeX (summary) ===")
print(summary.to_latex(index=False))
print("Saved: table_cv_summary.csv, table_region_oof.csv")