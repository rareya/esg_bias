import pandas as pd, numpy as np

SEED, B = 42, 5000
oof = pd.read_csv("oof_predictions_tuned2.csv")
oof["err"] = oof["ESG_Overall"] - oof["pred"]
oof["e"]  = oof["err"].abs()
oof["se"] = oof["err"] ** 2
oof["y"]  = oof["ESG_Overall"]
oof["y2"] = oof["y"] ** 2
oof["one"] = 1

# Per-company sums (fast bootstrap: resample companies, not rows)
c = oof.groupby("CompanyID").agg(north=("North", "first"), n=("one", "sum"),
                                 e=("e", "sum"), se=("se", "sum"),
                                 y=("y", "sum"), y2=("y2", "sum"))
north = c["north"].values.astype(float)
south = 1 - north
n, e, se, y, y2 = (c[k].values.astype(float) for k in ["n", "e", "se", "y", "y2"])
N = len(c)
print("Companies:", N, "| North:", int(north.sum()), "| South:", int(south.sum()))

def stats(idx):
    """idx: array of company indices (B x N or N)."""
    g = lambda a: a[idx].sum(axis=-1)
    N_tot, E, SE, Y, Y2 = g(n), g(e), g(se), g(y), g(y2)
    mae = E / N_tot
    rmse = np.sqrt(SE / N_tot)
    r2 = 1 - SE / (Y2 - Y**2 / N_tot)
    mae_n = g(e * north) / g(n * north)
    mae_s = g(e * south) / g(n * south)
    return {"R2": r2, "MAE": mae, "RMSE": rmse,
            "MAE_North": mae_n, "MAE_South": mae_s,
            "Gap_S_minus_N": mae_s - mae_n,
            "Gap_relative_%": 100 * (mae_s - mae_n) / mae}

point = stats(np.arange(N))
rng = np.random.default_rng(SEED)
boot = stats(rng.integers(0, N, size=(B, N)))

print(f"\n{'Metric':16s} {'Point':>9s} {'95% CI':>22s}")
for k in point:
    lo, hi = np.percentile(boot[k], [2.5, 97.5])
    print(f"{k:16s} {point[k]:9.4f}   [{lo:8.4f}, {hi:8.4f}]")

gap = boot["Gap_S_minus_N"]
p_two = 2 * min((gap <= 0).mean(), (gap >= 0).mean())
print(f"\nBootstrap share of resamples with gap > 0: {(gap > 0).mean():.4f}")
print(f"Approx two-sided bootstrap p-value (gap = 0): {p_two:.4f}")

pd.DataFrame({k: boot[k] for k in boot}).to_csv("bootstrap_overall.csv", index=False)
print("Saved: bootstrap_overall.csv")