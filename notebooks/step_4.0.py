import pandas as pd, numpy as np
from scipy import stats
import statsmodels.formula.api as smf

DATA_PATH = "data\cleaned_esg.csv"   # <- same path
df = pd.read_csv(DATA_PATH)
ID, Y = "CompanyID", "ESG_Overall"
PILLARS = ["ESG_Environmental", "ESG_Social", "ESG_Governance"]
north = ["Europe", "North America", "Oceania"]
df["North"] = df["Region"].isin(north).astype(int)

DII = {"Oceania": 1.0, "North America": 0.826, "Europe": 0.615,
       "Latin America": 0.282, "Asia": 0.257, "Middle East": 0.242, "Africa": 0.0}
eps = 1e-9
df["Carbon_per_Rev"] = df.CarbonEmissions / (df.Revenue + eps)
df["Water_per_Rev"] = df.WaterUsage / (df.Revenue + eps)
df["Energy_per_Rev"] = df.EnergyConsumption / (df.Revenue + eps)
df["Carbon_per_Energy"] = df.CarbonEmissions / (df.EnergyConsumption + eps)

print("=== 1. Region x Market_Type ===")
print(pd.crosstab(df.Region, df.Market_Type))

print("\n=== 2. Mean scores by region vs DII ===")
reg = df.groupby("Region")[[Y] + PILLARS].mean()
reg["DII"] = pd.Series(DII)
reg = reg.sort_values(Y, ascending=False)
print(reg.round(2).to_string())
rho, p = stats.spearmanr(reg.DII, reg[Y])
print(f"Spearman(DII, mean ESG) across 7 regions: rho={rho:.3f}, p={p:.3f}")
pm = df.groupby("North")[PILLARS].mean()
print("North minus South, by pillar:\n", (pm.loc[1] - pm.loc[0]).round(2).to_string())

print("\n=== 3. Variance structure ===")
cm = df.groupby(ID)[Y].transform("mean")
between = df.groupby(ID)[Y].mean().std()
within = (df[Y] - cm).std()
print(f"SD between companies: {between:.2f} | SD within company over time: {within:.2f}")
comp = df.groupby(ID).agg(m=(Y, "mean"), Region=("Region", "first"))
r2_region = smf.ols("m ~ C(Region)", comp).fit().rsquared
print(f"Share of company-mean variance explained by Region: {r2_region:.3f}")

# ---- standardized data for regressions (drop first-year NaN rows) ----
d = df.dropna(subset=["GrowthRate"]).copy()
NUMF = ["Revenue", "ProfitMargin", "MarketCap", "GrowthRate", "CarbonEmissions",
        "WaterUsage", "EnergyConsumption", "Carbon_per_Rev", "Water_per_Rev",
        "Energy_per_Rev", "Carbon_per_Energy", "Year"]
for c in NUMF:
    d[c] = (d[c] - d[c].mean()) / d[c].std()
FEAT = " + ".join(NUMF) + " + C(Industry)"
cl = {"cov_type": "cluster", "cov_kwds": {"groups": d[ID]}}

print("\n=== 4. Conditional North-South gap (OLS, SE clustered by company) ===")
m0 = smf.ols(f"{Y} ~ North", d).fit(**cl)
m1 = smf.ols(f"{Y} ~ North + {FEAT}", d).fit(**cl)
for name, m in [("raw", m0), ("controlling for firm features", m1)]:
    lo, hi = m.conf_int().loc["North"]
    print(f"{name:32s}: North coef {m.params['North']:.2f} [{lo:.2f}, {hi:.2f}] | R2 {m.rsquared:.3f}")
print(f"Share of raw gap NOT explained by features: {100*m1.params['North']/m0.params['North']:.1f}%")

print("\n=== 5. Region effects given features (reference = Africa) vs DII ===")
m2 = smf.ols(f"{Y} ~ C(Region, Treatment('Africa')) + {FEAT}", d).fit(**cl)
eff = {k.split("T.")[1].rstrip("]"): v for k, v in m2.params.items() if "Region" in k}
eff["Africa"] = 0.0
e = pd.DataFrame({"effect_vs_Africa": pd.Series(eff), "DII": pd.Series(DII)}).sort_values("effect_vs_Africa", ascending=False)
print(e.round(3).to_string())
rho2, p2 = stats.spearmanr(e.DII, e.effect_vs_Africa)
print(f"Spearman(DII, region effect): rho={rho2:.3f}, p={p2:.3f}")

print("\n=== 6. Do feature slopes differ between North and South? ===")
KEY = ["Revenue", "CarbonEmissions", "Carbon_per_Rev", "WaterUsage", "EnergyConsumption"]
for c in NUMF:
    d["NxS_" + c] = d["North"] * d[c]
inter = [f"NxS_{c}" for c in NUMF]
mf = smf.ols(f"{Y} ~ North + {FEAT} + " + " + ".join(inter), d).fit(**cl)
w = mf.wald_test(", ".join(f"{n} = 0" for n in inter), use_f=True, scalar=True)
print(f"Joint test, all North x feature interactions = 0: F={float(w.statistic):.2f}, p={float(w.pvalue):.4f}")
for grp, lab in [(1, "North"), (0, "South")]:
    mg = smf.ols(f"{Y} ~ {FEAT}", d[d.North == grp]).fit()
    print(f"{lab}: R2 {mg.rsquared:.3f} | " + " | ".join(f"{k} {mg.params[k]:+.2f}" for k in KEY))

print("\n=== 7. Raw gap by year ===")
yr = df.groupby(["Year", "North"])[Y].mean().unstack()
print((yr[1] - yr[0]).round(2).to_string())

print("\n=== 8. Industry mix, share of companies (North vs South) ===")
ind = df.drop_duplicates(ID).groupby("North").Industry.value_counts(normalize=True).unstack(0).round(3)
ind.columns = ["South", "North"]
print(ind.to_string())