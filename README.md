# EPL448: CERN Dielectron Invariant Mass Prediction

Machine learning project for predicting the invariant mass of dielectron collision events from the CERN/CMS open dataset.

**Team 2**

- Varnavas Tryfonos
- Thrasos Sazeidis
- Andreas Evagorou

## Overview

This repository contains the final notebook, supporting source code, generated figures, and LaTeX report for EPL448 Deliverable 3.

The problem is formulated as **regression**:

- input: kinematic features for electron pairs
- target: invariant mass `M` in GeV

The project evaluates four regressors:

- KNN
- ExtraTrees
- CatBoost
- XGBoost

across multiple dataset versions built from:

- cleaned raw features
- log-transformed features
- log-transformed target
- domain-driven feature selection
- PCA-based dimensionality reduction

## Highlights

- Four regression models screened with 5-fold cross-validation
- Best two models and best two dataset versions selected for `GridSearchCV`
- GPU-enabled CatBoost and XGBoost in the final workflow where available
- Final tuned performance above `R^2 = 0.99`
- Domain-driven feature selection outperformed PCA for this task

Best tuned outcomes:

- highest `R^2`: CatBoost on V2
- best MAE / MAPE: XGBoost on V4

## Repository Layout

```text
EPL448/
|- README.md
|- requirements.txt
|- data/
|  |- README.md
|  `- dielectron.csv
|- notebooks/
|  `- CERN_Electron_Collision_Team_2_Deliverable_3.ipynb
|- outputs/
|  |- fig_feature_importance_best.png
|  |- fig_pred_vs_actual.png
|  |- fig_residual_analysis.png
|  `- fig_screening_results_svd.png
|- report/
|  |- technical_report.tex
|  `- Deliverable3-Assesment.pdf
`- src/
   |- evaluation.py
   |- features.py
   |- models.py
   `- validation.py
```

## Main Files

- notebook: [notebooks/CERN_Electron_Collision_Team_2_Deliverable_3.ipynb](notebooks/CERN_Electron_Collision_Team_2_Deliverable_3.ipynb)
- report source: [report/technical_report.tex](report/technical_report.tex)
- compiled report: [report/Deliverable3-Assesment.pdf](report/Deliverable3-Assesment.pdf)

## Setup

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 3. Download the dataset

If `data/dielectron.csv` is missing, follow the instructions in [data/README.md](data/README.md).

Expected location:

```text
data/dielectron.csv
```

## Running the Notebook

Launch Jupyter from the project root:

```powershell
jupyter notebook
```

Then open:

```text
notebooks/CERN_Electron_Collision_Team_2_Deliverable_3.ipynb
```

The notebook:

- loads and validates the dataset
- rebuilds all dataset versions
- runs initial model screening
- selects the top 2 models and top 2 dataset versions
- performs `GridSearchCV` tuning
- evaluates tuned models on the held-out 20% test split
- writes figures to `outputs/`

## Experimental Setup

The final notebook uses:

- four screened models: `KNN`, `ET`, `CAT`, `XGB`
- top-2 model selection for tuning
- top-2 dataset selection for tuning
- GPU-enabled CatBoost and XGBoost where available
- a held-out 20% test split for final evaluation

## Figures

Generated figures are stored in [outputs](outputs):

- screening comparison
- predicted vs actual plots
- residual analysis
- feature importance

These figures are referenced directly by the LaTeX report.

## Report

Main source:

```text
report/Deliverable3-Assesment.pdf
```

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- CatBoost
- Jupyter Notebook
- LaTeX

## References

1. T. McCauley, "Events with two electrons from 2010", CERN Open Data Portal, 2014.
2. H. Kilic, S. Ozturk, and E. Yildirim, "Machine learning model performances for the Z boson mass identification through dielectron decay channel", *The European Physical Journal Plus*, 138, 2023.
3. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python", *JMLR*, 12, 2011.
4. T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System", *KDD*, 2016.
5. L. Prokhorenkova et al., "CatBoost: unbiased boosting with categorical features", *NeurIPS*, 2018.

## License

This project is licensed under the [MIT License](LICENSE).
