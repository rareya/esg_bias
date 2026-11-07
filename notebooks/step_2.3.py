import pandas as pd, numpy as np, json, ast
from collections import Counter
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

base = pd.read_csv("cv_results_baseline.csv").set_index("fold")
s21  = pd.read_csv("step2_1_results.csv")
bun  = s21[s21.set == "B_+intensity"].set_index("fold")
tun  = pd.read_csv("step2_2_results.csv").set_index("fold")

T = {
  "A baseline":  {"R2": base.R2, "MAE": base.MAE, "RMSE": base.RMSE, "Gap": base.Gap_S_minus_N},
  "B untuned":   {"R2": bun.R2,  "MAE": bun.MAE,  "RMSE": bun.RMSE,  "Gap": bun.Gap},
  "B tuned":     {"R2": tun.R2_tuned, "MAE": tun.MAE_tuned, "RMSE": tun.RMSE_tuned, "Gap": tun.Gap_tuned},
}

print("=== Mean ± SD over 5 folds ===")
for name, d in T.items():
    print(f"{name:12s}", " | ".join(f"{m} {v.mean():.4f}±{v.std(ddof=1):.4f}" for m, v in d.items()))

print("\n=== Paired difference: B tuned minus A baseline (95% t-interval, 4 df) ===")
tcrit = stats.t.ppf(0.975, 4)
for m in ["R2", "MAE", "RMSE", "Gap"]:
    d = T["B tuned"][m] - T["A baseline"][m]
    se = d.std(ddof=1) / np.sqrt(5)
    print(f"{m:5s} mean {d.mean():+.4f} | 95% CI [{d.mean()-tcrit*se:+.4f}, {d.mean()+tcrit*se:+.4f}] "
          f"| folds improved: {(d > 0).sum() if m=='R2' else (d < 0).sum()}/5")

g = T["B tuned"]["Gap"]; se = g.std(ddof=1) / np.sqrt(5)
print(f"\nTuned N-S gap: {g.mean():.4f} | 95% CI [{g.mean()-tcrit*se:.4f}, {g.mean()+tcrit*se:.4f}] "
      f"| positive in {(g > 0).sum()}/5 folds")

# Pooled OOF + per-region, baseline vs tuned (same row order)
ob = pd.read_csv("oof_predictions_baseline.csv")
ot = pd.read_csv("oof_predictions_tuned.csv")
y, p = ot["ESG_Overall"], ot["pred"]
print("\n=== Pooled OOF (tuned) ===")
print("R2:", round(r2_score(y, p), 4), "| MAE:", round(mean_absolute_error(y, p), 4),
      "| RMSE:", round(np.sqrt(mean_squared_error(y, p)), 4))

ob["e"] = (ob.ESG_Overall - ob.pred).abs()
ot["e"] = (ot.ESG_Overall - ot.pred).abs()
reg = pd.DataFrame({
    "Companies": ot.groupby("Region").CompanyID.nunique(),
    "MAE_baseline": ob.groupby("Region").e.mean(),
    "MAE_tuned": ot.groupby("Region").e.mean(),
}).assign(Change=lambda d: d.MAE_tuned - d.MAE_baseline).sort_values("MAE_tuned")
print("\n=== Per-region MAE ===")
print(reg.round(4).to_string())
reg.round(4).to_csv("table_region_tuned.csv")

# Save the most frequently chosen config
cfgs = [ast.literal_eval(c) for c in tun.best_cfg]
modal = Counter(json.dumps(c, sort_keys=True) for c in cfgs).most_common(1)[0]
final = json.loads(modal[0])
json.dump(final, open("final_config.json", "w"), indent=2)
print("\nFinal config (chosen in", modal[1], "of 5 folds):", final)

print("\n=== LaTeX ===")
print(r"""\begin{tabular}{lccc}
\toprule
Model & $R^2$ & MAE & RMSE \\
\midrule""")
for name, d in T.items():
    print(f"{name} & {d['R2'].mean():.4f} $\\pm$ {d['R2'].std(ddof=1):.4f} & "
          f"{d['MAE'].mean():.4f} & {d['RMSE'].mean():.4f} \\\\")
print(r"""\bottomrule
\end{tabular}""")