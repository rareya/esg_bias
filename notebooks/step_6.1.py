"""
Step 6.1: controlled-bias scenarios (null, A, B). Run from the repo root.

Generator: clean pillar scores = features@beta_k + company effect + yearly noise,
calibrated on the real data, with JOINT draws across the three pillars using the
real residual covariance (so the overall score = mean of pillars keeps realistic
variance). Observed score = clean score - planted bias.

Usage:  python step_6.1.py <replicates> [scenarios comma-separated]
        python step_6.1.py 2 N0,A,B        # smoke test
Output: sim_scen_raw.csv (appended per scenario), sim_scen_log.txt
"""
import sys, json, time, platform
import numpy as np, pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor

R = int(sys.argv[1]) if len(sys.argv) > 1 else 2
SCEN = sys.argv[2].split(",") if len(sys.argv) > 2 else ["N0", "A", "B"]
SEED = 42
DATA_PATH = "data/cleaned_esg.csv"
OUT_RAW = "sim_scen_raw.csv"

df = pd.read_csv(DATA_PATH)
ID, TARGET = "CompanyID", "ESG_Overall"
PILLARS = ["ESG_Environmental", "ESG_Social", "ESG_Governance"]
north = ["Europe", "North America", "Oceania"]
df["North"] = df["Region"].isin(north).astype(int)
is_n = df["North"].values == 1
south = 1.0 - df["North"].values
regions = df["Region"].values
groups = df[ID].values
order = np.unique(groups)
comp_idx = np.searchsorted(order, groups)
NC, T = len(order), 11

# ---------------- generator calibration (joint over pillars) ----------------
for c in ["Revenue", "MarketCap", "CarbonEmissions", "WaterUsage", "EnergyConsumption"]:
    df["log_" + c] = np.log(df[c])
df["GR_f"] = df["GrowthRate"].fillna(df["GrowthRate"].median())
GEN = ["log_Revenue", "ProfitMargin", "log_MarketCap", "GR_f",
       "log_CarbonEmissions", "log_WaterUsage", "log_EnergyConsumption"]
ind = pd.get_dummies(df["Industry"], drop_first=True).astype(float).values
Xg = np.column_stack([np.ones(len(df)), df[GEN].values, ind])
Xfit = np.column_stack([Xg, df["North"].values])

Ypil = df[PILLARS].values.astype(float)
coef = np.linalg.lstsq(Xfit, Ypil, rcond=None)[0]            # (p+1) x 3
beta = coef[:-1].copy()
beta[0] += coef[-1] * df["North"].mean()                      # centre the clean score
resid = Ypil - Xfit @ coef
cm = pd.DataFrame(resid).groupby(groups).transform("mean").values
Sw = ((resid - cm).T @ (resid - cm)) / (len(df) - NC)         # within-company covariance
cm_c = pd.DataFrame(resid).groupby(groups).mean().values
Su = np.cov(cm_c.T) - Sw / T                                   # between-company covariance
w_, v_ = np.linalg.eigh(Su); Su = (v_ * np.clip(w_, 0, None)) @ v_.T   # force PSD

def clean_pillars(rng):
    u = rng.multivariate_normal(np.zeros(3), Su, size=NC)[comp_idx]
    e = rng.multivariate_normal(np.zeros(3), Sw, size=len(df))
    return Xg @ beta + u + e                                   # n x 3

# ---------------- scenarios: planted bias per pillar ----------------
A_OFF = {"Africa": 16.0, "Middle East": 12.0, "Asia": 8.0, "Latin America": 4.0}
off_A = np.array([A_OFF.get(r, 0.0) for r in regions])         # per-row overall offset

def planted(scn):
    """returns n x 3 matrix of per-pillar offsets (subtracted from clean)."""
    P = np.zeros((len(df), 3))
    if scn == "N0":
        return P
    if scn == "A":                      # same offset on every pillar -> overall offset = region offset
        return np.repeat(off_A[:, None], 3, axis=1)
    if scn == "B":                      # Social and Governance -15 on South, Environmental 0
        P[:, 1] = 15.0 * south; P[:, 2] = 15.0 * south
        return P
    raise ValueError(scn)

# ---------------- models & measures ----------------
NUM = ["Year", "Revenue", "ProfitMargin", "MarketCap", "GrowthRate",
       "CarbonEmissions", "WaterUsage", "EnergyConsumption"]
eps = 1e-9
df["Carbon_per_Rev"] = df["CarbonEmissions"] / (df["Revenue"] + eps)
df["Water_per_Rev"] = df["WaterUsage"] / (df["Revenue"] + eps)
df["Energy_per_Rev"] = df["EnergyConsumption"] / (df["Revenue"] + eps)
df["Carbon_per_Energy"] = df["CarbonEmissions"] / (df["EnergyConsumption"] + eps)
BASE = NUM + ["Carbon_per_Rev", "Water_per_Rev", "Energy_per_Rev", "Carbon_per_Energy"]
X_full = pd.get_dummies(df[BASE + ["Industry", "Region", "Market_Type"]],
                        columns=["Industry", "Region", "Market_Type"]).astype(float)
X_nogeo = pd.get_dummies(df[BASE + ["Industry"]], columns=["Industry"]).astype(float)
geo_idx = [i for i, c in enumerate(X_full.columns) if c.startswith("Region_") or c.startswith("Market_Type_")]
cfg = json.load(open("final_config_v2.json"))

rng0 = np.random.default_rng(SEED)
comps = order.copy(); rng0.shuffle(comps)
fold_of = {c: i % 5 for i, c in enumerate(comps)}
fold_id = np.array([fold_of[c] for c in groups])
mask = df["GrowthRate"].notna().values                          # B-O sample (drops first-year rows)

def bo(X, y, nm):
    Xn, Xs, yn, ys = X[nm], X[~nm], y[nm], y[~nm]
    bn = np.linalg.lstsq(Xn, yn, rcond=None)[0]
    bs = np.linalg.lstsq(Xs, ys, rcond=None)[0]
    xn, xs = Xn.mean(0), Xs.mean(0)
    return yn.mean() - ys.mean(), (xn - xs) @ bs, xs @ (bn - bs)   # total, explained, unexplained

def truth_overall(P):
    """mean overall offset over South rows (B-O sample)."""
    off = P.mean(axis=1)
    sel = mask & ~is_n
    return off[sel].mean()

rows = []
def add(scn, rep, measure, value, truth=np.nan):
    rows.append({"scenario": scn, "rep": rep, "measure": measure, "value": float(value), "truth": truth})

t0 = time.time()
for scn in SCEN:
    P = planted(scn)
    T_all = truth_overall(P)
    for r in range(R):
        rng = np.random.default_rng(SEED + 1000 * r + 7)
        clean = clean_pillars(rng)
        obs = clean - P
        y = obs.mean(axis=1)                                     # overall = mean of pillars

        # --- Blinder-Oaxaca (reference = South), overall + pillars
        tot, ex, un = bo(Xg[mask], y[mask], is_n[mask])
        add(scn, r, "BO_unexpl_overall", un, T_all)
        add(scn, r, "BO_expl_overall", ex, 0.0)
        for k, nm in enumerate(["Env", "Soc", "Gov"]):
            _, _, unk = bo(Xg[mask], obs[mask, k], is_n[mask])
            truth_k = P[:, k][mask & ~is_n].mean()
            add(scn, r, f"BO_unexpl_{nm}", unk, truth_k)

        # --- XGBoost full / no-geography, out-of-fold
        pf, pn, geo = np.zeros(len(df)), np.zeros(len(df)), np.zeros(len(df))
        for k in range(5):
            tr, te = np.where(fold_id != k)[0], np.where(fold_id == k)[0]
            m = XGBRegressor(**cfg, random_state=SEED, n_jobs=-1).fit(X_full.iloc[tr], y[tr])
            c = m.get_booster().predict(xgb.DMatrix(X_full.iloc[te]), pred_contribs=True)
            geo[te] = c[:, geo_idx].sum(1); pf[te] = m.predict(X_full.iloc[te])
            pn[te] = XGBRegressor(**cfg, random_state=SEED, n_jobs=-1).fit(X_nogeo.iloc[tr], y[tr]).predict(X_nogeo.iloc[te])
        ef, en = np.abs(y - pf), y - pn
        off_row = P.mean(axis=1)
        T_row = off_row[~is_n].mean()                            # all South rows
        add(scn, r, "SHAP_gap", geo[is_n].mean() - geo[~is_n].mean(), T_row)
        add(scn, r, "noGeo_bias_diff", en[is_n].mean() - en[~is_n].mean(), T_row)
        add(scn, r, "MAE_gap", ef[~is_n].mean() - ef[is_n].mean())
        add(scn, r, "raw_gap", y[is_n].mean() - y[~is_n].mean(), T_row)

        # --- per-region recovery (all scenarios; truth = that region's mean offset)
        for reg in ["Africa", "Middle East", "Asia", "Latin America"]:
            rm = regions == reg
            T_reg = off_row[rm].mean()
            sel = mask & (is_n | rm)
            _, _, unr = bo(Xg[sel], y[sel], is_n[sel])
            add(scn, r, f"BO_unexpl_reg_{reg}", unr, T_reg)
            add(scn, r, f"SHAP_reg_{reg}", geo[is_n].mean() - geo[rm].mean(), T_reg)
            add(scn, r, f"noGeo_bias_reg_{reg}", en[is_n].mean() - en[rm].mean(), T_reg)
        print(f"{scn} rep {r+1}/{R} done ({time.time()-t0:.0f}s)", flush=True)

res = pd.DataFrame(rows)
try:
    old = pd.read_csv(OUT_RAW)
    old = old[~old.scenario.isin(SCEN)]                          # rerunning a scenario replaces it
    res = pd.concat([old, res], ignore_index=True)
except FileNotFoundError:
    pass
res.to_csv(OUT_RAW, index=False)
with open("sim_scen_log.txt", "a") as f:
    f.write(f"{time.ctime()} | scenarios={SCEN} R={R} | {time.time()-t0:.0f}s | python {platform.python_version()} "
            f"xgboost {xgb.__version__} numpy {np.__version__} pandas {pd.__version__}\n")
print("Saved", OUT_RAW)