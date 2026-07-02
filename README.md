# Reproducible analysis — prolonged chest-drain duration after lung resection

R code reproducing all statistical results reported in the manuscript
("Who keeps the chest drain after an early air leak? Determinants of prolonged chest-tube duration following pulmonary resection"). All **reported analyses**
are in R; figure/graph rendering of the LiNGAM causal map is done separately (Python,
not required to reproduce the numbers).

## Data

- `../データ.xlsx` — sheet `Sheet2`: early-air-leak cohort, n = 556 (Phase 2); sheet
  `全集団`: whole cohort, n = 1139 (Phase 1, LiNGAM). Outcome `y` = days to
  chest-drain removal. Missing covariates were completed by MICE upstream; the file has
  no missing values. Per-variable missingness (pre-imputation) is in Multimedia Appendix 1.

## Environment

- R 4.5.0. Packages: `readxl, dplyr, car, MASS, lmtest, survival, survminer, pROC, ggplot2`.
- On this machine Rscript is at `C:\Program Files\R\R-4.5.0\bin\Rscript.exe`.

## Scripts (run in order)

| Script | Purpose | Reproduces |
|---|---|---|
| `01_explore.R` | Distribution, missingness, skewness of `y` (skew 4.03 → 0.24 after log) | Results 3.2 |
| `02_analysis.R` | Reads xlsx, builds `data_clean.rds`; log-OLS = lognormal-AFT check; univariate (per-SD) | Multimedia Appendix 3; Methods |
| `03_final_risk.R` | Cutoff/risk-factor exploration (superseded by 05) | — |
| `04_figures_tables.R` | Early figure/table drafts (superseded by 05) | — |
| `05_final.R` | **Final 6-factor model (binary clinical cutoffs), coefficient-weighted risk-points score, 3-group strata (tertiles), KM (Fig 5), box plot (Fig 6)** | Table 2; Results; Figs 5–6; `data_risk.rds` |
| `06_univariate_table.R` | Univariate table, all variables | Multimedia Appendix 3 |
| `07_modelbuilding.R` | Candidate model + VIF + AIC (converges on the pre-specified factors); final binary-cutoff model diagnostics (BP, VIF, AFT-equivalence), joint F test (FEV1.0%<70+DLCO<80), integer-point weighting, sensitivity analyses | Multimedia Appendix 4; Results |
| `08_flowchart.R` | Study-flow diagram (Fig 1) | Fig 1 |
| `09_pipeline_figure.R` | Pipeline schematic (Fig 2) — figure only | Fig 2 |
| `10_baseline_wholecohort.R` | Whole-cohort (n=1139) baseline of LiNGAM variables, compared by early-air-leak status (Wilcoxon/chi-square) | Multimedia Appendix 7 |

Note: `01`–`08` are the reproducibility core. `09` (pipeline schematic) and the LiNGAM
causal-graph rendering are figures, not part of the numerical reproduction.

## Reproduce the reported numbers

```sh
Rscript 02_analysis.R   # creates data_clean.rds; prints univariate + AFT-equivalence
Rscript 07_modelbuilding.R   # candidate model, AIC, final-model diagnostics, joint F, sensitivity
Rscript 05_final.R      # Table 2 model, weighted risk-points score, 3-group strata, Figs 5–6
Rscript 06_univariate_table.R
```

Key reported values regenerated: Table 2 ratios/CIs/P (binary clinical-cutoff factors);
adj R² 0.112, AIC 1204.8, F 12.7 (6,549); joint F (FEV1.0%<70+DLCO<80) P = .04;
Breusch–Pagan P = .13; max VIF 1.11; sensitivity (COPD-replacement) adj R² 0.117;
integer weighting points (suture 6, air leak 5, albumin 5, fissure 3, FEV1.0% 3, DLCO 1; score 0–23);
Spearman ρ 0.35; 3-group strata by score tertiles n = 210/180/166, median 2/3/4 d;
log-rank/Kruskal–Wallis P < .001.

## Notes on method (reproducibility-relevant)

- The multivariable model is **pre-specified** (established air-leak risk factors + LiNGAM
  causal pathway + clinical relevance), reported as confirmatory; AIC/stepwise (in `07`) is a
  supporting analysis that converges on the same factors, not the selection mechanism.
- The bedside model enters the continuous determinants at **pre-specified clinical cutoffs**
  (FEV1.0% <70%, DLCO <80%, Alb <3.8 g/dL), not data-derived, so the model is consistent with
  the bedside score. The continuous-form LiNGAM causal map (Python) is unaffected.
- The risk score is **weighted**: each factor receives integer points proportional to its
  adjusted coefficient in the final model (`round(coef/min coef)`); patients are stratified by
  tertiles of the total score (0–5 / 6–9 / ≥10 points).
- The weighted-score stratification and the KM/box-plot displays are **descriptive risk
  stratification**, not a cross-validated or externally validated prediction model; a formal
  prediction model with external/temporal validation is future work.

---

# Python code — layered LiNGAM causal-discovery analysis

This repository also contains Python code for the layered LiNGAM causal-discovery analysis used in Phase 1 of the manuscript.

The Python workflow reproduces the edge table used to render the layered LiNGAM causal map and is separate from the R workflow above.

## Data

- `data.xlsx` — whole cohort, n = 1139, used for the Phase 1 layered LiNGAM causal-discovery analysis.

The input file should contain the variables specified in `run_analysis.py`.

## Scripts

| Script | Purpose | Reproduces |
|---|---|---|
| `function_analysis.py` | Main functions for layered regression, residualization, DirectLiNGAM, and edge-table construction | Internal analysis functions |
| `run_analysis.py` | Defines variable layers, runs the layered LiNGAM analysis, and exports the edge table | Edge table for the layered LiNGAM causal map |

## Output

The Python workflow outputs `results.xlsx`, which contains the estimated directed edge table used for the layered LiNGAM causal map.

## Notes on method

The Python workflow corresponds to the Phase 1 whole-cohort causal-discovery analysis.

The estimated edge table is used to render a hypothesis-generating causal map that informed, rather than replaced, clinically guided risk-factor identification. The detailed statistical procedure is described in the manuscript.