"""
Step 6.2: summarise sim_scen_raw.csv and apply the pre-registered pass/fail rules
(sim_scenarios_spec.md). Run from the repo root after step_6.1.py.

Rule 1 (recovery): |m - T| <= max(0.10*|T|, 0.5)  AND  |m - T| <= 1.96*SE + 0.5
Rule 2 (null)    : |m| <= 0.5 for every bias measure when nothing is planted
Rule 4 (B)       : Social/Governance vs 15 by rule 1; Environmental |m| <= 0.5
The explained component is scored against its own value in the no-bias control N0.
Scenarios not yet run (C1, C2, D) are skipped.
"""
import numpy as np, pandas as pd

raw = pd.read_csv("sim_scen_raw.csv", keep_default_na=False, na_values=[""])
N = raw.groupby(["scenario", "measure"]).rep.nunique().min()
g = raw.groupby(["scenario", "measure"]).agg(
    mean=("value", "mean"), sd=("value", "std"), truth=("truth", "mean")).reset_index()
g["se"] = g.sd / np.sqrt(N)

# explained component: truth = its mean under the no-bias control
n0_expl = g[(g.scenario == "N0") & (g.measure == "BO_expl_overall")]["mean"].iloc[0]
g.loc[g.measure == "BO_expl_overall", "truth"] = n0_expl
g["error"] = g["mean"] - g.truth

def rule1(r):
    if pd.isna(r.truth): return ""
    e = abs(r["mean"] - r.truth)
    return "PASS" if (e <= max(0.10 * abs(r.truth), 0.5) and e <= 1.96 * r.se + 0.5) else "FAIL"

g["verdict"] = g.apply(rule1, axis=1)
# measures that are reported but not scored
g.loc[g.measure.isin(["MAE_gap", "raw_gap"]), "verdict"] = ""
# null control: Rule 2 replaces rule 1 (truth is 0 so rule 1 already equals |m|<=0.5 plus SE clause)
nm = (g.scenario == "N0") & ~g.measure.isin(["MAE_gap", "raw_gap", "BO_expl_overall"])
g.loc[nm, "verdict"] = np.where(g.loc[nm, "mean"].abs() <= 0.5, "PASS", "FAIL")
# A: pillar-level BO is informational (the same offset is on every pillar)
g.loc[(g.scenario == "A") & g.measure.isin(["BO_unexpl_Env", "BO_unexpl_Soc", "BO_unexpl_Gov"]), "verdict"] = ""
# B: Environmental must be ~0 (rule 4), Social/Gov by rule 1 (already truth 15)
be = (g.scenario == "B") & (g.measure == "BO_unexpl_Env")
g.loc[be, "verdict"] = np.where(g.loc[be, "mean"].abs() <= 0.5, "PASS", "FAIL")
# B: per-region measures are informational (offset equal across regions by construction)
g.loc[(g.scenario == "B") & g.measure.str.contains("_reg_"), "verdict"] = ""

g = g.round(3)
g.to_csv("sim_scen_summary.csv", index=False)

pd.set_option("display.width", 220, "display.max_rows", 300)
for s in g.scenario.unique():
    print(f"\n===== {s}  (n = {N} replicates)")
    print(g[g.scenario == s].drop(columns="scenario").to_string(index=False))

print("\n=== MAE gap across scenarios (rule 8: does it move by > 0.5?) ===")
m = g[g.measure == "MAE_gap"].set_index("scenario")["mean"]
print(m.to_string(), f"\nrange = {m.max()-m.min():.3f}")
print("\nVerdict counts:", g[g.verdict != ""].groupby(["scenario", "verdict"]).size().unstack(fill_value=0).to_string())