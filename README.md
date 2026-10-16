# Temporally Layered Association Mapping of Chest-Tube Duration

This repository contains Python code for temporally layered association mapping of chest-tube duration after pulmonary resection. The analysis was designed for 1,139 patients who underwent thoracoscopic surgery for primary lung cancer. It follows patient background, preoperative tests, operative approach, intraoperative findings, early air leak on postoperative day 1 (POD 1), and chest-tube duration.

## Repository files

| File | Purpose |
| --- | --- |
| `run_analysis.py` | Defines the temporal layers and analysis settings, runs the models, and exports the results to Excel. |
| `functions.py` | Implements residualization, DirectLiNGAM, bootstrap edge selection, and forward layered regression. |

## 1. Variables

| Layer | Variables |
| --- | --- |
| L1: Patient background | `Age`, `Sex`, `BMI`, `Smoking index`, `COPD`, `Interstitial pneumonia`, `ASA-PS` |
| L2: Preoperative tests | `%VC`, `FEV1.0%`, `DLCO`, `Tumor size`, `Albumin`, `Hemoglobin`, `eGFR`, `Glucose`, `HbA1c`, `CRP` |
| L3: Operative approach | `Robotic surgery`, `Lobectomy or more`, `Lymph-node dissection` |
| L4: Intraoperative findings | `Pleural adhesion`, `Incomplete fissure`, `Intraoperative air leak` |
| L5: Intermediate variable | `Early air leak (POD1)` |
| L6: Outcome | `Days to drain removal` |

## 2. Set up Python

The reference analysis used Python 3.13.14, pandas 2.3.0, NumPy 2.3.1, scikit-learn 1.6.1, and lingam 1.12.2. The code also uses SciPy and openpyxl for Excel input and output. For example, on macOS or Linux:

```sh
cd /path/to/private/analysis-workdir
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install "pandas==2.3.0" "numpy==2.3.1" "scikit-learn==1.6.1" "lingam==1.12.2" scipy openpyxl
```

## 3. Run the analysis

Run the script from the directory containing `data.xlsx`:

```sh
cd /path/to/private/analysis-workdir
python /path/to/early-air-leak-chest-drain-analysis/run_analysis.py
```

The script adjusts L2 variables for L1, applies DirectLiNGAM to the L2 residuals, and fits layered regressions through L6. It saves `results.xlsx` in the working directory.

## 4. Output

`results.xlsx` contains two worksheets:

| Worksheet | Contents |
| --- | --- |
| `All_Edges` | Regression edges and selected L2 DirectLiNGAM edges. |
| `Model_Fit` | Sample sizes and model-fit values for `Early air leak (POD1)` and `Days to drain removal`. |
