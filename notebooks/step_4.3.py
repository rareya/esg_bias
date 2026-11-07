import pandas as pd, numpy as np

SEED, B = 42, 2000
DATA_PATH = "data\cleaned_esg.csv"   # <- same path
df = pd.read_csv(DATA_PATH)
ID = "CompanyID"
north = ["Europe", "North America", "Oceania"]
df["North"] = df["Region"].isin(north).astype(int)

d = df.dropna(subset=["GrowthRate"]).reset_index(drop=True)
print("Rows used:", len(d), "(dropped", len(df) - len(d), "first-year rows)")

LOGC = ["Revenue", "MarketCap", "CarbonEmissions", "WaterUsage", "EnergyConsumption"]
print("Minimum raw values (must be > 0 for logs):")
print(d[LOGC].min().round(3).to_string())
for c in LOGC:
    d["log_" + c] = np.log(np.clip(d[c], 1e-9, None))
d["log_Carbon_per_Rev"]    = np.log(np.clip(d.CarbonEmissions / d.Revenue, 1e-12, None))
d["log_Water_per_Rev"]     = np.log(np.clip(d.WaterUsage / d.Revenue, 1e-12, None))
d["log_Energy_per_Rev"]    = np.log(np.clip(d.EnergyConsumption / d.Revenue, 1e-12, None))
d["log_Carbon_per_Energy"] = np.log(np.clip(d.CarbonEmissions / d.EnergyConsumption, 1e-12, None))

FIN = ["log_Revenue", "ProfitMargin", "log_MarketCap", "GrowthRate",
       "log_CarbonEmissions", "log_WaterUsage", "log_EnergyConsumption"]
INT = ["log_Carbon_per_Rev", "log_Water_per_Rev", "log_Energy_per_Rev", "log_Carbon_per_Energy"]
SPECS = {"paper": FIN, "paper+intensity": FIN + INT}
OUTCOMES = ["ESG_Overall", "ESG_Environmental", "ESG_Social", "ESG_Governance"]

ind = pd.get_dummies(d["Industry"], drop_first=True).astype(float)
mats = {}
for s, cols in SPECS.items():
    mats[s] = np.column_stack([np.ones(len(d)), d[cols].values.astype(float), ind.values])

is_n = d["North"].values == 1
Y = {o: d[o].values.astype(float) for o in OUTCOMES}

def bo(X, y, n_mask):
    Xn, Xs, yn, ys = X[n_mask], X[~n_mask], y[n_mask], y[~n_mask]
    bn = np.linalg.lstsq(Xn, yn, rcond=None)[0]
    bs = np.linalg.lstsq(Xs, ys, rcond=None)[0]
    xn, xs = Xn.mean(0), Xs.mean(0)
    return np.array([yn.mean() - ys.mean(),      # total gap (North - South)
                     (xn - xs) @ bs,             # explained
                     xs @ (bn - bs),             # unexplained
                     (xn - xs) @ (bn - bs)])     # interaction

point = {(s, o): bo(mats[s], Y[o], is_n) for s in SPECS for o in OUTCOMES}

by_c = d.groupby(ID).indices
comps = np.array(list(by_c.keys()))
rng = np.random.default_rng(SEED)
boot = {k: np.zeros((B, 4)) for k in point}
for b in range(B):
    pick = rng.choice(comps, size=len(comps), replace=True)
    rows = np.concatenate([by_c[c] for c in pick])
    nm = is_n[rows]
    for s in SPECS:
        Xb = mats[s][rows]
        for o in OUTCOMES:
            boot[(s, o)][b] = bo(Xb, Y[o][rows], nm)
    if (b + 1) % 500 == 0:
        print("bootstrap", b + 1, "/", B, flush=True)

names = ["Total gap", "Explained", "Unexplained", "Interaction"]
rows_out = []
for s in SPECS:
    for o in OUTCOMES:
        p = point[(s, o)]
        print(f"\n--- {s} | {o} ---")
        for i, nme in enumerate(names):
            lo, hi = np.percentile(boot[(s, o)][:, i], [2.5, 97.5])
            print(f"{nme:12s} {p[i]:+8.3f}  [{lo:+8.3f}, {hi:+8.3f}]")
            rows_out.append({"spec": s, "outcome": o, "component": nme,
                             "estimate": p[i], "ci_low": lo, "ci_high": hi})
        print(f"Unexplained share of total gap: {100 * p[2] / p[0]:.1f}%")
pd.DataFrame(rows_out).round(4).to_csv("table_bo_bootstrap.csv", index=False)
print("\nSaved: table_bo_bootstrap.csv")