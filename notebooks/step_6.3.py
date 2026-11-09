"""
Step 6.3: scenarios with a LEGITIMATE regional difference (C1, C2) and covariate
imbalance with group-specific slopes (D). Run from the repo root, one scenario per call.

  python step_6.3.py 20 C1_6      # observed legitimate gap of 6 points + planted bias 10
  python step_6.3.py 20 C2_6      # omitted legitimate gap of 6 points + planted bias 10
  python step_6.3.py 20 D         # imbalance (+0.5 SD) and 30% steeper North slopes + bias 10
Scenario names: C1_<L>, C2_<L> with L the legitimate gap in points.
Appends to sim_scen_raw.csv (replaces the same scenario if rerun) and sim_scen_log.txt.
"""
import sys, json, time, platform
import numpy as np, pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor

R = int(sys.argv[1]); SCEN = sys.argv[2].split(",")
SEED, DELTA = 42, 10.0
OUT_RAW = "sim_scen_raw.csv"

df0 = pd.read_csv("data/cleaned_esg.csv")
ID = "CompanyID"
PILLARS = ["ESG_Environmental", "ESG_Social", "ESG_Governance"]
df0["North"] = df0["Region"].isin(["Europe", "North America", "Oceania"]).astype(int)
is_n = df0["North"].values == 1
south = 1.0 - df0["North"].values
groups = df0[ID].values
order = np.unique(groups)
comp_idx = np.searchsorted(order, groups)
NC, T = len(order), 11
mask = df0["GrowthRate"].notna().values
cfg = json.load(open("final_config_v2.json"))

GEN = ["log_Revenue", "ProfitMargin", "log_MarketCap", "GR_f",
       "log_CarbonEmissions", "log_WaterUsage", "log_EnergyConsumption"]
LOGSRC = {"log_Revenue": "Revenue", "log_MarketCap": "MarketCap", "log_CarbonEmissions": "CarbonEmissions",
          "log_WaterUsage": "WaterUsage", "log_EnergyConsumption": "EnergyConsumption"}

def features(d):
    """design matrices from a (possibly shifted) frame"""
    d = d.copy()
    for k, src in LOGSRC.items(): d[k] = np.log(d[src])
    d["GR_f"] = d["GrowthRate"].fillna(d["GrowthRate"].median())
    ind = pd.get_dummies(d["Industry"], drop_first=True).astype(float).values
    Xg = np.column_stack([np.ones(len(d)), d[GEN].values, ind])
    eps = 1e-9
    d["Carbon_per_Rev"] = d["CarbonEmissions"] / (d["Revenue"] + eps)
    d["Water_per_Rev"] = d["WaterUsage"] / (d["Revenue"] + eps)
    d["Energy_per_Rev"] = d["EnergyConsumption"] / (d["Revenue"] + eps)
    d["Carbon_per_Energy"] = d["CarbonEmissions"] / (d["EnergyConsumption"] + eps)
    base = ["Year", "Revenue", "ProfitMargin", "MarketCap", "GrowthRate", "CarbonEmissions", "WaterUsage",
            "EnergyConsumption", "Carbon_per_Rev", "Water_per_Rev", "Energy_per_Rev", "Carbon_per_Energy"]
    Xf = pd.get_dummies(d[base + ["Industry", "Region", "Market_Type"]],
                        columns=["Industry", "Region", "Market_Type"]).astype(float)
    Xn = pd.get_dummies(d[base + ["Industry"]], columns=["Industry"]).astype(float)
    return Xg, Xf, Xn

# ---- calibration on the ORIGINAL data (identical to step_6.1.py) ----
Xg0, _, _ = features(df0)
Xfit = np.column_stack([Xg0, df0["North"].values])
Ypil = df0[PILLARS].values.astype(float)
coef = np.linalg.lstsq(Xfit, Ypil, rcond=None)[0]
beta = coef[:-1].copy(); beta[0] += coef[-1] * df0["North"].mean()
resid = Ypil - Xfit @ coef
cm = pd.DataFrame(resid).groupby(groups).transform("mean").values
Sw = ((resid - cm).T @ (resid - cm)) / (len(df0) - NC)
cm_c = pd.DataFrame(resid).groupby(groups).mean().values
Su = np.cov(cm_c.T) - Sw / T
w_, v_ = np.linalg.eigh(Su); Su = (v_ * np.clip(w_, 0, None)) @ v_.T
# overall-score slope vector (mean of the three pillar slopes), used to size shifts
beta_ov = beta.mean(axis=1)

def shift_frame(d, dx):
    """apply log-unit shifts dx (dict over GEN names) to NORTH firms' raw columns"""
    d = d.copy(); n = d["North"].values == 1
    for k, src in LOGSRC.items():
        if k in dx: d.loc[n, src] = d.loc[n, src] * np.exp(dx[k])
    if "ProfitMargin" in dx: d.loc[n, "ProfitMargin"] = d.loc[n, "ProfitMargin"] + dx["ProfitMargin"]
    if "GR_f" in dx: d.loc[n, "GrowthRate"] = d.loc[n, "GrowthRate"] + dx["GR_f"]
    return d

sd_feat = pd.Series(Xg0[:, 1:8].std(axis=0), index=GEN)
def build(scn):
    """returns dict with frame-derived matrices, slopes per group, bias truth, trait size"""
    out = {"d": df0, "L": 0.0, "beta_S": beta, "beta_N": beta, "trait": 0.0}
    kind = scn.split("_")[0]
    if kind == "C1":
        L = float(scn.split("_")[1])
        k = L / float((np.abs(beta_ov[1:8]) * sd_feat.values).sum())
        dx = {g: k * np.sign(beta_ov[1 + i]) * sd_feat[g] for i, g in enumerate(GEN)}
        out["d"], out["L"] = shift_frame(df0, dx), L
        print(f"[C1] shift size k = {k:.3f} SD per feature to create a {L} point covariate-driven gap", flush=True)
    elif kind == "C2":
        out["trait"] = float(scn.split("_")[1])
    elif kind == "D":
        dx = {g: 0.5 * sd_feat[g] for g in ["log_Revenue", "log_MarketCap", "log_CarbonEmissions", "log_EnergyConsumption"]}
        out["d"] = shift_frame(df0, dx)
        bN = beta.copy(); bN[1:8, :] *= 1.3
        out["beta_N"] = bN
    return out

def bo_refs(X, y, nm):
    Xn, Xs, yn, ys = X[nm], X[~nm], y[nm], y[~nm]
    bn = np.linalg.lstsq(Xn, yn, rcond=None)[0]; bs = np.linalg.lstsq(Xs, ys, rcond=None)[0]
    bp = np.linalg.lstsq(X, y, rcond=None)[0]
    xn, xs = Xn.mean(0), Xs.mean(0)
    return {"total": yn.mean() - ys.mean(), "expl_S": (xn - xs) @ bs,
            "un_S": xs @ (bn - bs), "un_N": xn @ (bn - bs),
            "un_P": xn @ (bn - bp) + xs @ (bp - bs)}

rows = []
def add(scn, rep, measure, value, truth=np.nan):
    rows.append({"scenario": scn, "rep": rep, "measure": measure, "value": float(value), "truth": truth})

rng0 = np.random.default_rng(SEED)
comps = order.copy(); rng0.shuffle(comps)
fold_of = {c: i % 5 for i, c in enumerate(comps)}
fold_id = np.array([fold_of[c] for c in groups])
t0 = time.time()
for scn in SCEN:
    S = build(scn); d = S["d"]
    Xg, Xf, Xn = features(d)
    geo_idx = [i for i, c in enumerate(Xf.columns) if c.startswith("Region_") or c.startswith("Market_Type_")]
    # truth for the explained component = (x_N - x_S)' beta_S on the clean scale (overall slope vector)
    xN, xS = Xg[mask & is_n].mean(0), Xg[mask & ~is_n].mean(0)
    expl_truth = float((xN - xS) @ beta_ov) if S["beta_N"] is beta else np.nan
    for r in range(R):
        rng = np.random.default_rng(SEED + 1000 * r + 7)
        u = rng.multivariate_normal(np.zeros(3), Su, size=NC)[comp_idx]
        e = rng.multivariate_normal(np.zeros(3), Sw, size=len(d))
        mean_part = Xg @ beta
        if S["beta_N"] is not beta:                    # D: North rows use steeper slopes
            mean_part = np.where(is_n[:, None], Xg @ S["beta_N"], Xg @ beta)
        clean = mean_part + u + e
        obs = clean - DELTA * south[:, None]
        if S["trait"] > 0:                             # C2: omitted trait, equal on all pillars
            z = rng.normal(0, 3.0, NC)[comp_idx] + S["trait"] * is_n
            obs = obs + z[:, None]
        y = obs.mean(axis=1)

        b = bo_refs(Xg[mask], y[mask], is_n[mask])
        add(scn, r, "BO_unexpl_overall", b["un_S"], DELTA)
        add(scn, r, "BO_expl_overall", b["expl_S"], expl_truth)
        if scn.startswith("D"):
            add(scn, r, "BO_unexpl_refS", b["un_S"], DELTA)
            add(scn, r, "BO_unexpl_refN", b["un_N"], DELTA)
            add(scn, r, "BO_unexpl_refPooled", b["un_P"], DELTA)
            add(scn, r, "BO_ref_spread", max(b["un_S"], b["un_N"], b["un_P"]) - min(b["un_S"], b["un_N"], b["un_P"]), 0.0)

        pf, pn, geo = np.zeros(len(d)), np.zeros(len(d)), np.zeros(len(d))
        for k in range(5):
            tr, te = np.where(fold_id != k)[0], np.where(fold_id == k)[0]
            m = XGBRegressor(**cfg, random_state=SEED, n_jobs=-1).fit(Xf.iloc[tr], y[tr])
            c = m.get_booster().predict(xgb.DMatrix(Xf.iloc[te]), pred_contribs=True)
            geo[te] = c[:, geo_idx].sum(1); pf[te] = m.predict(Xf.iloc[te])
            pn[te] = XGBRegressor(**cfg, random_state=SEED, n_jobs=-1).fit(Xn.iloc[tr], y[tr]).predict(Xn.iloc[te])
        ef, en = np.abs(y - pf), y - pn
        add(scn, r, "SHAP_gap", geo[is_n].mean() - geo[~is_n].mean(), DELTA)
        add(scn, r, "noGeo_bias_diff", en[is_n].mean() - en[~is_n].mean(), DELTA)
        add(scn, r, "MAE_gap", ef[~is_n].mean() - ef[is_n].mean())
        add(scn, r, "raw_gap", y[is_n].mean() - y[~is_n].mean(), DELTA + S["L"] + S["trait"])
        print(f"{scn} rep {r+1}/{R} ({time.time()-t0:.0f}s)", flush=True)

res = pd.DataFrame(rows)
try:
    old = pd.read_csv(OUT_RAW, keep_default_na=False, na_values=[""])
    res = pd.concat([old[~old.scenario.isin(SCEN)], res], ignore_index=True)
except FileNotFoundError:
    pass
res.to_csv(OUT_RAW, index=False)
with open("sim_scen_log.txt", "a") as f:
    f.write(f"{time.ctime()} | scenarios={SCEN} R={R} | {time.time()-t0:.0f}s | python {platform.python_version()} "
            f"xgboost {xgb.__version__} numpy {np.__version__} pandas {pd.__version__}\n")
print("Saved", OUT_RAW)