"""
Feature engineering for CERN dielectron invariant mass prediction.

Provides physics-motivated feature transformations extracted from Deliverable 2
and shared across all notebooks to eliminate code duplication.
"""

import numpy as np
import pandas as pd

# Energy/momentum features that are strictly positive and right-skewed.
# Log-transforming these reduces skewness and helps distance-based models.
LOG_FEATURES = ['E1', 'E2', 'pt1', 'pt2']

# Reduced feature set for V4 – selected by domain knowledge and feature
# importance analysis.  Removes redundant coordinate representations
# (px/py/pz, raw eta/phi, Q1/Q2) and highly correlated log_pt features.
V4_SELECTED = [
    'sum_E', 'sum_pt', 'delta_R', 'delta_eta', 'delta_phi',
    'opposite_sign', 'log_E1', 'log_E2',
]


def add_engineered_features(df_in: pd.DataFrame) -> pd.DataFrame:
    """Add physics-motivated engineered features to a DataFrame.

    Computes:
        delta_eta     – pseudorapidity separation between the two electrons
        delta_phi     – azimuthal separation, folded into [0, π]
        delta_R       – angular distance sqrt(delta_eta² + delta_phi²)
        sum_pt        – scalar sum of transverse momenta
        sum_E         – scalar sum of energies
        opposite_sign – 1 if the two electrons have opposite charge, else 0

    Parameters
    ----------
    df_in : pd.DataFrame
        DataFrame containing at minimum the columns
        eta1, eta2, phi1, phi2, pt1, pt2, E1, E2, Q1, Q2.

    Returns
    -------
    pd.DataFrame
        Copy of *df_in* with the six new columns appended.
    """
    df_out = df_in.copy()
    df_out['delta_eta'] = df_out['eta1'] - df_out['eta2']

    # Vectorised delta_phi – fold into [0, π] without a Python-level loop
    raw_dphi = np.abs(df_out['phi1'] - df_out['phi2'])
    df_out['delta_phi'] = np.minimum(raw_dphi, 2 * np.pi - raw_dphi)

    df_out['delta_R'] = np.sqrt(df_out['delta_eta'] ** 2 + df_out['delta_phi'] ** 2)
    df_out['sum_pt'] = df_out['pt1'] + df_out['pt2']
    df_out['sum_E'] = df_out['E1'] + df_out['E2']
    df_out['opposite_sign'] = ((df_out['Q1'] * df_out['Q2']) < 0).astype(int)
    return df_out


def build_v1(df_clean: pd.DataFrame) -> pd.DataFrame:
    """V1 – Baseline: raw original features, no transforms."""
    return df_clean[[c for c in df_clean.columns if c != 'M']].copy()


def build_v2(df_clean: pd.DataFrame) -> pd.DataFrame:
    """V2 – Log-transformed energy/pt features, raw target."""
    X = df_clean[[c for c in df_clean.columns if c != 'M']].copy()
    for f in LOG_FEATURES:
        X[f'log_{f}'] = np.log1p(X[f])
        X = X.drop(columns=[f])
    return X


def build_v3(df_clean: pd.DataFrame) -> pd.DataFrame:
    """V3 – Same features as V2 but target will be log(M)."""
    return build_v2(df_clean)


def build_v4(df_clean: pd.DataFrame) -> pd.DataFrame:
    """V4 – Reduced 8-feature set: engineered + log-transformed E1/E2 features, raw target M."""
    X_full = add_engineered_features(df_clean.drop(columns=['M']))
    X_full['log_E1'] = np.log1p(X_full['E1'])
    X_full['log_E2'] = np.log1p(X_full['E2'])
    return X_full[V4_SELECTED].copy()
