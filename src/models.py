"""
Pipeline builders for CERN dielectron invariant mass prediction.

Centralises model construction so both notebooks always build identical
pipelines and hyperparameter grids.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

RANDOM_STATE = 42

# SVR is O(n² … n³) in the number of samples.  For the screening phase we
# subsample the training set to keep wall-clock time manageable.
# NOTE: the subsample is drawn once before cross_validate, so each CV fold
# sees the same fixed subset.  This is a known limitation of the screening
# phase; the full training set is used for final GridSearchCV tuning.
N_SVR_SCREEN = 20_000

# Number of quantile bins used for stratified train/test splitting
N_STRATIFY_BINS = 10

# Number of top models / datasets selected for hyperparameter tuning
N_TOP_MODELS = 2
N_TOP_DATASETS = 2


def build_pipeline(model_name: str, use_pca: bool = False) -> Pipeline:
    """Build a scikit-learn Pipeline for the requested model.

    Scaling rule
    ------------
    - KNN and SVR require a StandardScaler (distance/kernel sensitive).
    - RF and XGBoost are scale-invariant; no scaler is added unless PCA is
      requested (PCA always requires centred, unit-variance features).

    PCA
    ---
    When *use_pca* is True a PCA step retaining 95% of variance is inserted
    after the scaler.  PCA is applied to dense arrays so ``random_state`` is
    intentionally omitted – it only affects the randomised SVD path which is
    not used here.

    Parameters
    ----------
    model_name : {'KNN', 'SVR', 'RF', 'XGB'}
    use_pca    : bool, default False

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    steps = []

    # Scaler: always for distance/kernel models; also required before PCA
    if model_name in ('KNN', 'SVR') or use_pca:
        steps.append(('scaler', StandardScaler()))

    # PCA – random_state omitted intentionally (dense array → exact SVD)
    if use_pca:
        steps.append(('pca', PCA(n_components=0.95)))

    # Model
    if model_name == 'KNN':
        steps.append(('model', KNeighborsRegressor(n_jobs=-1)))
    elif model_name == 'SVR':
        steps.append(('model', SVR()))
    elif model_name == 'RF':
        steps.append(('model', RandomForestRegressor(
            random_state=RANDOM_STATE, n_jobs=-1)))
    elif model_name == 'XGB':
        if XGBOOST_AVAILABLE:
            steps.append(('model', xgb.XGBRegressor(
                random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)))
        else:
            steps.append(('model', GradientBoostingRegressor(
                random_state=RANDOM_STATE)))
    else:
        raise ValueError(f'Unknown model name: {model_name!r}')

    return Pipeline(steps)


# ── Hyperparameter grids ───────────────────────────────────────────────────────
# Keys follow the pipeline convention: 'step_name__param_name'

PARAM_GRIDS = {
    'KNN': {
        'model__n_neighbors': [3, 5, 10, 15, 20],
        'model__weights':     ['uniform', 'distance'],
        'model__metric':      ['euclidean', 'manhattan'],
    },
    'SVR': {
        'model__C':       [1, 10, 100],
        'model__epsilon': [0.1, 0.5, 1.0],
        'model__kernel':  ['rbf'],
        'model__gamma':   ['scale', 'auto'],
    },
    'RF': {
        'model__n_estimators':    [100, 300, 500],
        'model__max_depth':       [10, 20, None],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf':  [1, 2, 4],
    },
    'XGB': {
        'model__n_estimators':    [100, 300, 500],
        'model__learning_rate':   [0.01, 0.05, 0.1],
        'model__max_depth':       [3, 6, 10],
        'model__subsample':       [0.8, 1.0],
        'model__colsample_bytree': [0.8, 1.0],
    },
}
