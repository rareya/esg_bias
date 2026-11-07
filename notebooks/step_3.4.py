import pandas as pd, numpy as np

ov  = pd.read_csv("bootstrap_overall.csv")
reg = pd.read_csv("table_region_bootstrap.csv")
pb  = pd.read_csv("table_paired_bootstrap.csv").set_index("Quantity")

def ci(x): return f"[{np.percentile(x, 2.5):.3f}, {np.percentile(x, 97.5):.3f}]"

# Table A: overall metrics with bootstrap CIs (point = bootstrap median is NOT used;
# use the point estimates printed in step 3.1)
pt = {"R2": 0.5772, "MAE": 8.3497, "RMSE": 10.3348,
      "MAE_North": 7.9237, "MAE_South": 8.6723, "Gap_S_minus_N": 0.7487}
labels = {"R2": "$R^2$", "MAE": "MAE", "RMSE": "RMSE", "MAE_North": "MAE (North)",
          "MAE_South": "MAE (South)", "Gap_S_minus_N": "MAE gap (S$-$N)"}
print("% ---- Table A: overall ----")
print(r"\begin{tabular}{lcc}\toprule")
print(r"Metric & Estimate & 95\% CI \\ \midrule")
for k, v in pt.items():
    print(f"{labels[k]} & {v:.4f} & {ci(ov[k])} \\\\")
print(r"\bottomrule\end{tabular}")

# Table B: per-region
print("\n% ---- Table B: per-region ----")
print(r"\begin{tabular}{lccc}\toprule")
print(r"Region & Companies & MAE & 95\% CI \\ \midrule")
for _, r in reg.iterrows():
    print(f"{r.Region} & {int(r.Companies)} & {r.MAE:.3f} & [{r.CI_low:.3f}, {r.CI_high:.3f}] \\\\")
print(r"\bottomrule\end{tabular}")

# Table C: paired improvement
print("\n% ---- Table C: tuned vs baseline (paired) ----")
names = {"dR2": "$\\Delta R^2$", "dMAE": "$\\Delta$MAE", "dRMSE": "$\\Delta$RMSE",
         "dGap": "$\\Delta$ gap (S$-$N)"}
print(r"\begin{tabular}{lcc}\toprule")
print(r"Quantity & Change & 95\% CI \\ \midrule")
for k, lab in names.items():
    r = pb.loc[k]
    print(f"{lab} & {r.Point:+.4f} & [{r.CI_low:+.3f}, {r.CI_high:+.3f}] \\\\")
for k in pb.index:
    if k.startswith("dMAE_"):
        r = pb.loc[k]
        print(f"$\\Delta$MAE, {k[5:]} & {r.Point:+.3f} & [{r.CI_low:+.3f}, {r.CI_high:+.3f}] \\\\")
print(r"\bottomrule\end{tabular}")

print("\n=== FINAL NUMBERS FOR THE PAPER ===")
print("Final model R2 / MAE / RMSE (pooled OOF): 0.5772 / 8.3497 / 10.3348")
print("Fold mean ± SD R2: 0.5719 ± 0.0555")
print("N-S gap: 0.7487 [0.0535, 1.4769], 8.97% of MAE, bootstrap p = 0.033")
print("Improvement vs baseline: dR2 +0.1423, dMAE -1.3111, dRMSE -1.6131 (all CIs exclude 0)")
print("Gap change vs baseline: -0.1331 [-0.5769, 0.3282]")
print("Region pairs significant after Holm: 0/21")