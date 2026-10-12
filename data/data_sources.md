# Data Sources and Provenance

This file records the origin, processing, and known limitations of each dataset used in the geographic ESG-scoring audit. Update it whenever a source file or transformation changes. Do not describe a derived file as a raw download.

## 1. World Bank governance data

- **Local file:** `data/world_bank_governance.csv`
- **Source page (confirmed by project author):** [World Bank DataBank — Environment, Social and Governance (ESG) Data](https://databank.worldbank.org/source/environment-social-and-governance-(esg)-data)
- **Download selections reported by the project author:** all countries; Series = Regulatory; Time = 2024.
- **Series present in the downloaded file:** Regulatory Quality - Governance estimate (approximately -2.5 to +2.5); series code `GOV_WGI_RQ.EST`.
- **Underlying database named in the file footer:** Worldwide Governance Indicators (WGI).
- **Observation level:** country/economy, for the 2024 estimate.
- **File notes:** the CSV includes DataBank metadata/footer rows after the country observations and represents missing values with markers such as `..`; processing code must handle these explicitly.
- **Download date/version:** not recorded in the repository; record this on the next verified download.
- **Role in this project:** input to `compute_dii.py`, which derives a region-level proxy for disclosure/institutional context. This file is not the company-level ESG/financial dataset.

## 2. Region-level DII (derived data)

- **Local file:** `data/dii_by_region.csv`
- **Generating script:** `compute_dii.py`
- **Inputs:** `data/world_bank_governance.csv` and the country-to-region rules defined in `compute_dii.py`.
- **Transformation documented in code:**
  1. Read the 2024 Regulatory Quality estimate and coerce values to numeric, excluding missing estimates.
  2. Assign countries/economies to the project's seven broad regions using `pycountry_convert` plus explicit overrides for Middle East, North America, and Oceania.
  3. Calculate the mean Regulatory Quality estimate for each mapped region.
  4. Min-max normalize the seven regional means to a 0–1 scale and write the resulting `Region,DII` table to `data/dii_by_region.csv`.
- **Interpretation:** this is a **derived regional proxy**, not a directly published World Bank DII measure. Higher values indicate higher average Regulatory Quality relative to the other regions in this particular seven-region mapping; they do not directly measure corporate ESG disclosure quality.
- **Important limitations to validate before relying on results:**
  - The company dataset contains broad regions rather than country identifiers, so country-to-region mapping cannot be matched to individual firms.
  - Regional categories are project-defined; the Middle East and Oceania rules are explicit approximations.
  - The script should be audited for DataBank footer rows, duplicate/non-country economies, missing observations, mapping coverage, and whether the seven regions all appear.
  - Min-max normalization is relative to the regions in this sample and year; it is not an absolute or time-invariant scale.
  - The construct is based on Regulatory Quality, which is not synonymous with ESG disclosure infrastructure. The paper should justify the proxy and use appropriately cautious terminology.
- **Reproducibility:** run from the repository root so the relative paths resolve:
  ```bash
  python compute_dii.py
  ```
  The script also writes `results/dii_old_vs_new_comparison.csv`.

## 3. Company-level ESG and financial dataset — provenance unresolved

- **Local file:** `data/company_esg_financial_dataset.csv`
- **Role:** company/firm-year ESG and financial variables used in the main analysis.
- **Observed schema includes:** company identifier/name, industry, region, year, revenue, profit margin, market capitalization, growth rate, overall and E/S/G scores, carbon emissions, water usage, and energy consumption.
- **Original source, download URL, licence, data-generation process, and download date:** **not yet independently verified**. Do not attribute this dataset to the World Bank merely because the separate governance input came from a World Bank DataBank download.
- **Current audit concern:** the file structure reportedly contains 11,000 rows across 2015–2025, with generic company labels. These characteristics raise the possibility that it is synthetic or simulated, but this must be established from source documentation or reproducible matching before making a definitive claim.
- **Required next steps:** trace the file through repository history and source candidates; inspect the full schema, row counts, missingness, duplicate keys, value distributions, and repeated patterns; document whether observations are real, synthetic, or mixed; establish permission/licence and appropriate citation. Until resolved, any empirical claims must clearly disclose the uncertainty.

## 4. Cleaned/processed company data

- **Local files:** `data/cleaned_esg.csv` and root-level `cleaned_esg.csv` (if both remain in use).
- **Generating script/notebook and exact cleaning steps:** not yet documented here; verify before treating either file as authoritative.
- **Required checks:** identify the canonical version; document row/column counts before and after cleaning, filtering rules, type conversions, missing-value handling, duplicate handling, and whether any rows or labels were synthetically generated. Preserve raw inputs unchanged.

## 5. Provenance and research-integrity rules

1. Keep raw downloads separate from derived data.
2. Record source URL, indicator/series code, selection filters, retrieval date, file checksum where practical, licence/terms, and any preprocessing.
3. Never describe a model result as evidence about real-world firms unless the underlying company-level data are verified as real and suitable for that claim.
4. If the company dataset is synthetic, frame the study as a methodological simulation/audit unless a suitable real-world validation dataset is added.
5. Treat the DII as a proxy with assumptions, not as ground truth for firm-level disclosure quality.
6. Re-run data validation and regenerate derived outputs from scripts; do not hand-edit derived CSVs without documenting the change.
