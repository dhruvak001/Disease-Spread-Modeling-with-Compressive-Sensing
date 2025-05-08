import numpy as np
from sklearn.linear_model import Lasso

def generate_measurement_matrix(m, n, random_seed=None):
    """
    Generate a random Gaussian measurement matrix of shape (m, n).
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    Phi = np.random.randn(m, n)
    return Phi


def recover_lasso(Phi, y, alpha=0.1):
    """
    Recover sparse signal from measurements using LASSO.
    Returns the recovered signal vector of length n.
    """
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(Phi, y)
    return lasso.coef_