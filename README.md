# EPL448 – CERN Electron Collision: Invariant Mass Prediction

This project applies machine learning to predict the invariant mass of dielectron collision events recorded by the CMS detector at CERN's Large Hadron Collider. The dataset contains 100,000 events with kinematic features (energy, momentum, pseudorapidity, azimuthal angle) for each electron pair, with the target variable M (GeV) revealing physics resonances including the J/ψ meson, Υ meson, and Z boson.

Four regression models are explored — KNN, SVR, Random Forest, and XGBoost — across five preprocessed dataset versions combining log-transformation, feature engineering, standardisation, and PCA.

**Team 2:** Varnavas Tryfonos, Thrasos Sazeidis, Andreas Evagorou — University of Cyprus, EPL448 Data Science & Machine Learning.