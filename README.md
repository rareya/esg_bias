# 🌍 Auditing Geographic Bias in AI-Driven ESG Scoring

### A SHAP-Based Explainability Analysis and Disclosure-Adjusted ESG Scoring (DAES) Framework for Rating Parity Between Global North and Global South Firms

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)
[![Research](https://img.shields.io/badge/Research-ICAIF%202026-8B5CF6?style=for-the-badge)](https://icaif2026.org/)
[![XAI](https://img.shields.io/badge/Explainable-AI-EC4899?style=for-the-badge)](https://shap.readthedocs.io/)
[![ESG](https://img.shields.io/badge/Domain-ESG%20Fairness-0EA5E9?style=for-the-badge)]()



*Aarya Patankar*

---

## 📌 Project Overview

This repository contains the full end-to-end research pipeline for our paper submitted to **ICAIF 2026**. The project investigates whether AI-based ESG (Environmental, Social, and Governance) scoring systems encode and perpetuate geographic bias against firms from the **Global South** — particularly those in Africa, Asia, Latin America, and the Middle East — relative to firms from the **Global North** (Europe, North America, Oceania).

Beyond auditing baseline model disparities, this repository:
- Uses a **Blinder-Oaxaca (1973) decomposition** to formally test how much of the North-South ESG gap is explained by legitimate covariate differences (firm size, emissions, profitability) versus unexplained by them — i.e., how much survives even after holding firm characteristics constant.
- Implements **Disclosure-Adjusted ESG Scoring (DAES)**, a dynamic, macro-institutional correction framework that leverages real World Bank Regulatory Quality data to compute each region's **Disclosure Infrastructure Index (DII)**, and scales training targets accordingly.

---

## 💡 Research Motivation

ESG scores have become a dominant mechanism through which investors, regulators, and stakeholders assess corporate sustainability. However, the frameworks and data used to generate these scores are predominantly developed by Western rating agencies, calibrated to Western regulatory standards, and evaluated primarily on Global North firms.

This creates a structural risk: AI models trained on such scores may **learn, encode, and amplify** the geographic inequalities baked into the underlying data — not because the models are inherently biased, but because the input signals themselves carry historical and institutional disclosure disadvantages for Global South firms.

Understanding and auditing this bias is critical for:
- **Equitable capital allocation** — ensuring Global South firms are not systematically under-financed due to structural reporting deficits.
- **ESG policy design** — informing how rating agencies should adapt their methodologies for emerging markets.
- **AI accountability & Bias Mitigation** — demonstrating how XAI tools (SHAP), formal decomposition methods (Oaxaca-Blinder), and institutional target-scaling (DAES) can jointly audit and correct geographic bias.

---

## ❓ Research Questions

> 1. *Do AI-driven ESG scoring models exhibit geographic bias by systematically predicting lower scores or producing higher prediction errors for firms from the Global South compared to those from the Global North?*
> 2. *How much of the observed North-South ESG gap is explained by legitimate differences in firm characteristics, versus unexplained by them (i.e., attributable to how firms are scored rather than what they look like)?*
> 3. *How do feature importances shift during sub-pillar ablation when raw operational signals replace aggregated scores?*
> 4. *Can a Disclosure-Adjusted ESG Scoring (DAES) framework, grounded in real institutional data, dynamically shrink the Global North vs. Global South rating gap without distorting global model performance?*

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🤖 **Multi-Model Prediction** | Baseline and ablation-feature ESG prediction using XGBoost, Random Forest, LightGBM, Ridge, and MLP |
| 📊 **Sub-Pillar Ablation Study** | `ablation_shap.py` — removes aggregated ESG sub-scores to audit raw feature reliance |
| ⚖️ **Oaxaca-Blinder Decomposition** | `oaxaca_decomposition.py` — splits the North-South gap into explained (legitimate covariates) vs. unexplained (bias) components, with bootstrap confidence intervals |
| 🌐 **Institutional DII Integration** | `compute_dii.py` — Region-level Disclosure Infrastructure Index (DII) computed from real 2024 World Bank Worldwide Governance Indicators (not asserted) |
| ⚙️ **DAES Mitigation Framework** | `daes.py` — dynamic target scaling to correct institutional disclosure deficits, using real computed DII |
| 📈 **Two Independently-Derived Alphas** | `daes_alpha_sweep.py` (kneedle-verified accuracy/fairness knee, α₀ = 0.275) and `oaxaca_decomposition.py` (bias-closing target, α₀ = 0.431) |
| 🔍 **SHAP Explainability** | Global feature importance (beeswarm, bar) and before/after SHAP shift analysis across baseline, ablation, and DAES-corrected models |
| ⚖️ **Geographic Fairness Audit** | Disaggregated predictions across 7 regions, 2 market types, and Global North vs. South |
| 📐 **Statistical Hypothesis Testing** | `stat_analysis.py` — t-tests, Mann-Whitney U, and one-way ANOVA |

---

## 🗂️ Repository Structure

```
aiml/
│
├── 📓 notebooks/                   # Jupyter notebooks — run in order 01 → 07
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling_xgboost.ipynb
│   ├── 05_modeling_rf_lgbm.ipynb
│   ├── 06_shap_analysis.ipynb
│   └── 07_fairness_evaluation.ipynb
│
├── 📁 data/
│   ├── company_esg_financial_dataset.csv  # Raw dataset (11,000 firms)
│   ├── cleaned_esg.csv                    # Cleaned dataset
│   ├── world_bank_governance.csv          # Real 2024 World Bank Regulatory Quality (222 countries)
│   └── dii_by_region.csv                  # Region-level DII computed by compute_dii.py
│
├── 📁 results/
│   ├── model_comparison.csv
│   ├── shap_feature_importance.csv
│   ├── ablation_shap_importance.csv
│   ├── multimodal_baseline_results.csv    # 5-model comparison on ablation feature set
│   ├── daes_alpha_sweep.csv               # Pareto sweep, alpha_0 = 0.00 to 0.50
│   ├── daes_shap_importance.csv
│   ├── oaxaca_decomposition.csv           # Explained/unexplained/interaction + 95% CIs
│   ├── oaxaca_bootstrap_draws.csv         # Raw 500-iteration bootstrap draws
│   ├── oaxaca_derived_alpha.txt           # Bias-closing alpha derivation
│   ├── dii_old_vs_new_comparison.csv      # Provenance: hardcoded vs. computed DII
│   ├── fairness_metrics.csv
│   ├── fairness_region.csv
│   └── fairness_markettype.csv
│
├── 📁 figures/
│   ├── shap_summary.png / shap_bar.png / top10_shap_features.png
│   ├── ablation_shap_bar.png / ablation_esg_region.png
│   ├── multimodal_xgb_shap.png
│   ├── daes_pareto_frontier.png
│   ├── daes_before_after.png
│   ├── daes_shap_comparison.png
│   └── oaxaca_decomposition.png           # Explained/unexplained bar chart with 95% CIs
│
├── 📄 ablation_shap.py             # Sub-pillar ablation modeling script
├── 📄 compute_dii.py               # Computes real region-level DII from World Bank data
├── 📄 daes_alpha_sweep.py          # Grid search for DAES Pareto knee point
├── 📄 daes.py                      # Production DAES execution (reads computed DII)
├── 📄 oaxaca_decomposition.py      # Blinder-Oaxaca gap decomposition + bootstrap CIs
├── 📄 multimodal_eval.py           # 5-model benchmark on ESG_Overall (ablation feature set)
├── 📄 stat_analysis.py             # Statistical testing script (scipy)
├── 📄 requirements.txt
├── 📄 .gitignore
└── 📄 README.md
```

---

## 🔄 Workflow

```
Raw Dataset (11,000 firms)
                                 │
                                 ▼
                      01–03 · Data & Preprocessing
                                 │
                                 ▼
                     04–05 · Baseline ML Modeling
                 XGBoost, Random Forest, LightGBM benchmark
                                 │
                                 ▼
                     06–07 · Explainability & Fairness
                 SHAP audit & regional disparity testing
                                 │
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                         ▼
📄 ablation_shap.py    📄 oaxaca_decomposition.py   📄 compute_dii.py
Sub-Pillar Ablation     Explained vs. unexplained     Real DII from World
                         gap decomposition             Bank governance data
        │                        │                         │
        │                        │                         ▼
        │                        │                   📄 daes_alpha_sweep.py
        │                        │                   Kneedle-verified alpha
        │                        │                         │
        │                        └────────────┬────────────┘
        │                                      ▼
        │                                📄 daes.py
        │                        Execute DAES (α₀ = 0.275, real DII)
        │                        Generate final SHAP & predictions
        └──────────────────────────┬──────────────────────┘
                                    ▼
                       stat_analysis.py & Paper
```

---

## ⚖️ Disclosure-Adjusted ESG Scoring (DAES)

### Framework Formulation

$$y_{\text{DAES}, i} = y_i \times \left(1 + \alpha_0 \cdot (1 - DII_i)\right)$$

Where $DII_i \in [0, 1]$ is now computed — not asserted — from real 2024 World Bank Worldwide Governance Indicators (`compute_dii.py`), aggregating country-level Regulatory Quality estimates to each of the dataset's 7 regions and min-max normalizing to a 0–1 scale.

### Two Independently-Derived Values of α₀

DAES requires choosing how aggressively to correct. We report two, from two different criteria, rather than asserting one:

| Method | α₀ | Criterion |
|---|---|---|
| **Kneedle algorithm** (Satopaa et al., 2011) on the RMSE-vs-α sweep | **0.275** | Statistical: the point where further correction starts costing disproportionately more model fit |
| **Blinder-Oaxaca decomposition** | **0.431** | Normative: the value that closes the *entire* unexplained (bias-attributable) portion of the gap |

We adopt **α₀ = 0.275** as the conservative operating value and report 0.431 as the theoretical upper bound, discussing the trade-off explicitly rather than picking one silently.

### Pareto Optimization Results (α₀ Sweep)

| α₀ | RMSE ↑ | North Mean | South Mean | Signed Gap (N–S) ↓ | Bias Reduction |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.000 (Baseline) | 8.66 | 62.95 | 47.51 | +15.44 | 0.0% |
| 0.150 | 9.29 | 64.55 | 51.82 | +12.73 | -17.6% |
| **0.275 (Kneedle Knee)** ⭐ | 10.57 | 65.88 | 55.51 | +10.38 | -32.8% |
| 0.500 (Aggressive) | 14.41 | 68.40 | 62.08 | +6.32 | -59.1% |

---

## ⚖️ Blinder-Oaxaca Gap Decomposition

Before assuming the full North-South gap is bias, we test it. Fitting separate OLS models per group on real covariates (Revenue, ProfitMargin, MarketCap, GrowthRate, CarbonEmissions, WaterUsage, EnergyConsumption, Industry — deliberately excluding ESG sub-scores and Region/Market_Type to avoid circularity), we decompose the gap and bootstrap 500 firm-level resamples for confidence intervals:

| Component | Point Estimate | 95% CI | % of Total Gap |
|---|---:|---:|---:|
| **Total gap** | 16.15 | [15.58, 16.65] | 100.0% |
| **Explained** (legitimate covariates) | -0.33 | [-0.68, -0.02] | -2.1% |
| **Unexplained** (bias) | 16.44 | [16.06, 16.78] | 101.8% |
| Interaction | 0.04 | [-0.07, 0.15] | 0.2% |

**Interpretation:** essentially none of the North-South gap is explained by real business fundamentals in this dataset — firms that look identical on paper still score differently by region. This result is only as strong as the covariates tested; it is evidence *consistent with* bias, not proof of it (see Limitations).

---

## 📊 Summary of Key Findings

1. **A 16.15-point baseline gap exists** between Global North and Global South firms (p < 0.0001), and the Blinder-Oaxaca decomposition shows this gap is not explained by legitimate covariate differences.
2. **Sub-pillar scores mask geographic proxies:** in baseline models, aggregated ESG sub-scores dominate SHAP weight; ablating them shifts model reliance onto `Market_Type`, which becomes the #1 SHAP feature.
3. **DAES correction reduces reliance on geography as a mechanism, not just on paper:** after DAES-correcting the training target, `Market_Type` SHAP importance drops from 8.10 (ablation, uncorrected) to 5.47 (post-correction), falling behind real financial signals — direct mechanistic evidence the correction changes what the model actually learns.
4. **Random Forest slightly outperforms XGBoost** on the ablation (no-sub-score) feature set (R² = 0.731 vs. 0.675); Ridge (0.320) and MLP (0.599) lag behind both tree ensembles — reported honestly rather than assuming XGBoost wins by default.
5. **DAES at α₀ = 0.275 closes 32.8% of the gap**; a formally-derived α₀ = 0.431 would close the full Oaxaca-identified unexplained component, at a higher RMSE cost.

---

## 🚀 Quick Start & Script Execution

### 1. Installation

```bash
git clone https://github.com/aaryakulkarnii/aiml.git
cd aiml
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Running Core Scripts (in order)

```bash
# Step 1: Baseline statistical hypothesis testing
python stat_analysis.py

# Step 2: Sub-pillar ablation study
python ablation_shap.py

# Step 3: Multi-model benchmark on ablation feature set
python multimodal_eval.py

# Step 4: Blinder-Oaxaca gap decomposition
python oaxaca_decomposition.py

# Step 5: Compute real region-level DII from World Bank data
python compute_dii.py

# Step 6: DAES hyperparameter sweep (generates Pareto plot)
python daes_alpha_sweep.py

# Step 7: Execute DAES framework at operating alpha (0.275)
python daes.py
```

---

## ⚠️ Limitations

- **"Unexplained" is not proof of bias.** The Oaxaca-Blinder decomposition is only as good as the covariates tested. A legitimate driver of the gap not present in this dataset (e.g. disclosure completeness, reporting lag) would also register as "unexplained." Future work should test additional covariates directly rather than relying on Region as a catch-all proxy.
- **Dataset scope.** Analysis relies on a single firm-level dataset; validation against independently disclosed real-world ESG scores (e.g. MSCI, Sustainalytics) is left to future work.
- **Region-level, not firm-level, DII.** DII is computed and applied at the regional level; a firm-level DII using each company's actual headquarters country would be a natural extension.

---

## 📝 Citation

```bibtex
@inproceedings{kulkarni2026esgbias,
  title     = {Auditing Geographic Bias in AI-Driven ESG Scoring: A SHAP-Based
               Explainability Analysis and Disclosure-Adjusted ESG Scoring (DAES) Framework},
  author    = {Patankar, Aarya and Kulkarni, Aarya},
  booktitle = {Proceedings of the 7th ACM International Conference on AI in Finance (ICAIF 2026)},
  year      = {2026},
  address   = {Milan, Italy},
  url       = {https://github.com/aaryakulkarnii/aiml}
}
```

---

## 👥 Authors

- **Aarya Patankar**
- **Aarya Kulkarni**

---

## 📄 License

This project is licensed under the MIT License — see the LICENSE file for details.

---

**Built with ❤️ for open and equitable AI research**

*ICAIF 2026, Milan, Italy*