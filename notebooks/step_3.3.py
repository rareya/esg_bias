import pandas as pd, numpy as np

SEED, B = 42, 5000
ob = pd.read_csv("oof_predictions_baseline.csv")
ot = pd.read_csv("oof_predictions_tuned2.csv")

# Row alignment check: same companies, same targets, same order
assert (ob.CompanyID.values == ot.CompanyID.values).all()
assert np.allclose(ob.ESG_Overall.values, ot.ESG_Overall.values)

d = ot[["CompanyID", "Region", "North", "ESG_Overall"]].copy()
d["eb"]   = (d.ESG_Overall - ob.pred).abs()
d["et"]   = (d.ESG_Overall - ot.pred).abs()
d["se_b"] = (d.ESG_Overall - ob.pred) ** 2
d["se_t"] = (d.ESG_Overall - ot.pred) ** 2
d["y"]    = d.ESG_Overall
d["y2"]   = d.ESG_Overall ** 2
d["one"]  = 1

c = d.groupby("CompanyID").agg(region=("Region", "first"), north=("North", "first"),
                               n=("one", "sum"), eb=("eb", "sum"), et=("et", "sum"),
                               se_b=("se_b", "sum"), se_t=("se_t", "sum"),
                               y=("y", "sum"), y2=("y2", "sum"))
N = len(c)
north = c["north"].values.astype(float); south = 1 - north
n, eb, et, se_b, se_t, y, y2 = (c[k].values.astype(float)
                                for k in ["n", "eb", "et", "se_b", "se_t", "y", "y2"])
regions = sorted(c["region"].unique())
rmask = {r: (c["region"].values == r).astype(float) for r in regions}

def diffs(idx):
    g = lambda a: a[idx].sum(axis=-1)
    nt = g(n)
    sst = g(y2) - g(y) ** 2 / nt
    out = {
        "dR2":   (g(se_b) - g(se_t)) / sst,                  # tuned - baseline
        "dMAE":  g(et) / nt - g(eb) / nt,
        "dRMSE": np.sqrt(g(se_t) / nt) - np.sqrt(g(se_b) / nt),
    }
    gap_b = g(eb * south) / g(n * south) - g(eb * north) / g(n * north)
    gap_t = g(et * south) / g(n * south) - g(et * north) / g(n * north)
    out["Gap_baseline"], out["Gap_tuned"], out["dGap"] = gap_b, gap_t, gap_t - gap_b
    for r in regions:
        m = rmask[r]
        out["dMAE_" + r] = g(et * m) / g(n * m) - g(eb * m) / g(n * m)
    return out

point = diffs(np.arange(N))
rng = np.random.default_rng(SEED)
boot = diffs(rng.integers(0, N, size=(B, N)))

print(f"{'Quantity':22s} {'Point':>9s} {'95% CI':>22s} {'Resamples <0':>13s}")
rows = []
for k in point:
    lo, hi = np.percentile(boot[k], [2.5, 97.5])
    share = (boot[k] < 0).mean()
    rows.append({"Quantity": k, "Point": point[k], "CI_low": lo, "CI_high": hi,
                 "share_below_0": share})
    print(f"{k:22s} {point[k]:9.4f}   [{lo:8.4f}, {hi:8.4f}] {share:13.4f}")

res = pd.DataFrame(rows)
res.round(4).to_csv("table_paired_bootstrap.csv", index=False)
print("\nNote: dX = tuned minus baseline. dR2 > 0 and dMAE/dRMSE < 0 mean improvement.")
print("Saved: table_paired_bootstrap.csv")