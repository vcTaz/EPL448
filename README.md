# EPL448 – CERN Electron Collision: Invariant Mass Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%E2%89%A51.3-orange.svg)](https://scikit-learn.org/)

This project applies machine learning to predict the invariant mass of dielectron
collision events recorded by the CMS detector at CERN's Large Hadron Collider.
The dataset contains 100,000 events with kinematic features (energy, momentum,
pseudorapidity, azimuthal angle) for each electron pair. The target variable
M (GeV) reveals physics resonances including the J/ψ meson (~3.1 GeV),
Υ meson (~9.5 GeV), and Z boson (~91 GeV).

Four regression models are explored — KNN, SVR, Random Forest, and XGBoost —
across five preprocessed dataset versions combining log-transformation, feature
engineering, standardisation, and PCA.

**Team 2:** Varnavas Tryfonos, Thrasos Sazeidis, Andreas Evagorou
— University of Cyprus, EPL448 Data Mining on the Web.

---

## Repository Structure

```
EPL448/
├── .gitignore
├── README.md
├── requirements.txt
│
├── data/
│   └── README.md          # Download instructions for dielectron.csv
│
├── notebooks/
│   └── CERN_Electron_Collision_Team_2_Deliverable_3.ipynb  # Deliverable 3 – main notebook
│
├── outputs/               # Generated figures (gitignored – re-run notebooks)
│
├── src/
│   ├── __init__.py
│   ├── features.py        # Feature engineering (add_engineered_features, builders)
│   ├── evaluation.py      # Metrics (compute_metrics, CV_SCORING)
│   ├── models.py          # Pipeline builders, hyperparameter grids
│   └── validation.py      # Data validation checks
│
├── Deliverable1/
│   ├── EPL448_Deliverable1.docx
│   ├── Feedback comments - Deliverable1.docx
│   └── Readme.md
│
└── Deliverable2/
    └── Deliverable2_CERN_ElectronCollision_team2.docx
```

---

## Quick Start

### 1. Clone and install dependencies

```bash
git clone https://github.com/vtryfo01/EPL448.git
cd EPL448
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download the dataset

The 14.7 MB CSV is not stored in this repository.
See **[data/README.md](data/README.md)** for download instructions.

Place the file at:

```
data/dielectron.csv
```

### 3. Run the notebooks

```bash
jupyter notebook notebooks/CERN_Electron_Collision_Team_2_Deliverable_3.ipynb
```

Generated figures are saved to `outputs/` (created automatically).

---

## Dataset

- **Source:** CERN Open Data Portal / Kaggle
- **Size:** 100,000 dielectron collision events, 16 kinematic features
- **Target:** Invariant mass M (GeV)
- **Reference:** McCauley, T. (2014). DOI: 10.7483/OPENDATA.CMS.PCSW.AHVG

---

## Models

| Model | Notes |
|-------|-------|
| KNN | Distance-based; requires StandardScaler |
| SVR | Kernel-based; requires StandardScaler; trained on 20K subsample for speed |
| Random Forest | Scale-invariant ensemble |
| XGBoost | Scale-invariant gradient boosting |

All models are wrapped in scikit-learn `Pipeline` objects so preprocessing
is fitted only on training folds during cross-validation (no data leakage).

---

## References

1. McCauley, T. (2014). Events with two electrons from 2010. CERN Open Data Portal.
2. Kilic, H., Ozturk, S., & Yildirim, E. (2023). Machine learning model performances
   for the Z boson mass identification through dielectron decay channel.
   *The European Physical Journal Plus*, 138(1), 87.
3. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
4. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD 2016*.
5. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.

---

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for
guidelines on how to submit issues and pull requests.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).
By participating you agree to abide by its terms.

## License

This project is licensed under the [MIT License](LICENSE).
