"""
EPL448 – CERN Dielectron Invariant Mass Prediction
Shared source modules for all notebooks.

Public API
----------
features   : add_engineered_features, build_v1/v2/v3/v4
models     : build_pipeline, PARAM_GRIDS
evaluation : compute_metrics, CV_SCORING
validation : validate_raw, validate_clean
"""

__version__ = "1.0.0"
__authors__ = [
    "Varnavas Tryfonos",
    "Thrasos Sazeidis",
    "Andreas Evagorou",
]

from .features import add_engineered_features, build_v1, build_v2, build_v3, build_v4
from .models import build_pipeline, PARAM_GRIDS
from .evaluation import compute_metrics, CV_SCORING
from .validation import validate_raw, validate_clean

__all__ = [
    "add_engineered_features",
    "build_v1",
    "build_v2",
    "build_v3",
    "build_v4",
    "build_pipeline",
    "PARAM_GRIDS",
    "compute_metrics",
    "CV_SCORING",
    "validate_raw",
    "validate_clean",
]
