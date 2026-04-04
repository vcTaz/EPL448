"""
Regression evaluation metrics for CERN dielectron invariant mass prediction.

Centralises metric computation and cross-validation scoring dictionaries
so every notebook uses identical, consistent evaluation.
"""

import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute all regression metrics used throughout the project.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Ground-truth target values (in original GeV scale, not log-transformed).
    y_pred : array-like of shape (n,)
        Model predictions (in original GeV scale).

    Returns
    -------
    dict with keys:
        RMSE  – root mean squared error (GeV)
        MAE   – mean absolute error (GeV)
        R2    – coefficient of determination
        MAPE  – mean absolute percentage error (%)
    """
    return {
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'MAE':  float(mean_absolute_error(y_true, y_pred)),
        'R2':   float(r2_score(y_true, y_pred)),
        'MAPE': float(mean_absolute_percentage_error(y_true, y_pred) * 100),
    }


# Scoring dictionary for sklearn cross_validate / GridSearchCV.
# Negated scorers are restored to positive values after CV.
CV_SCORING = {
    'R2':   'r2',
    'RMSE': 'neg_root_mean_squared_error',
    'MAE':  'neg_mean_absolute_error',
}
