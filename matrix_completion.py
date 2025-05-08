import numpy as np
from fancyimpute import SoftImpute

def soft_impute_matrix(X_missing, max_iters=100, tol=1e-3):
    """
    Complete a matrix with missing entries using SoftImpute.
    X_missing: 2D numpy array with np.nan for missing values.
    Returns the completed matrix.
    """
    X_filled = SoftImpute(max_iters=max_iters, tol=tol).fit_transform(X_missing)
    return X_filled
