"""
Data validation for CERN dielectron dataset.

Runs sanity checks on the raw and cleaned DataFrame before any modelling
begins.  Raises AssertionError with a descriptive message if a check fails,
so problems surface immediately rather than silently corrupting results.
"""

import pandas as pd
import numpy as np

# Columns that must be present in the raw CSV (after stripping whitespace)
REQUIRED_COLUMNS = [
    'Run', 'Event',
    'E1', 'px1', 'py1', 'pz1', 'pt1', 'eta1', 'phi1', 'Q1',
    'E2', 'px2', 'py2', 'pz2', 'pt2', 'eta2', 'phi2', 'Q2',
    'M',
]

# Features that must be strictly positive (used in log-transform)
LOG_FEATURES = ['E1', 'E2', 'pt1', 'pt2']


def validate_raw(df: pd.DataFrame) -> None:
    """Validate the DataFrame immediately after loading from CSV.

    Checks
    ------
    1. All required columns are present (handles trailing-space column names).
    2. The target M is strictly positive everywhere.
    3. Log-transform features are strictly positive.
    4. No NaN values in the target column.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame as returned by pd.read_csv().

    Raises
    ------
    AssertionError if any check fails.
    """
    # Strip whitespace from column names to handle trailing-space issue (L1)
    df.columns = df.columns.str.strip()

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    assert not missing, (
        f"CSV is missing expected columns: {sorted(missing)}. "
        f"Found columns: {sorted(df.columns.tolist())}"
    )

    assert df['M'].notna().all(), (
        f"Target column M contains {df['M'].isna().sum()} NaN values."
    )

    assert (df['M'] > 0).all(), (
        f"Target M has {(df['M'] <= 0).sum()} non-positive values. "
        "Invariant mass must be strictly positive."
    )

    for feat in LOG_FEATURES:
        n_bad = (df[feat] <= 0).sum()
        assert n_bad == 0, (
            f"Feature '{feat}' has {n_bad} non-positive values. "
            "Cannot apply log-transform."
        )


def validate_clean(df_clean: pd.DataFrame) -> None:
    """Validate the working DataFrame after identifier columns are dropped.

    Checks
    ------
    1. No NaN values anywhere.
    2. M is strictly positive.
    3. Log-transform features are strictly positive.

    Parameters
    ----------
    df_clean : pd.DataFrame
        Cleaned DataFrame (Run/Event columns already removed).

    Raises
    ------
    AssertionError if any check fails.
    """
    n_nan = df_clean.isnull().sum().sum()
    assert n_nan == 0, (
        f"Cleaned DataFrame contains {n_nan} NaN values:\n"
        f"{df_clean.isnull().sum()[df_clean.isnull().sum() > 0]}"
    )

    assert (df_clean['M'] > 0).all(), (
        "Target M contains non-positive values after cleaning."
    )

    for feat in LOG_FEATURES:
        if feat in df_clean.columns:
            n_bad = (df_clean[feat] <= 0).sum()
            assert n_bad == 0, (
                f"Feature '{feat}' has {n_bad} non-positive values after cleaning."
            )
