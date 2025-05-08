import numpy as np
import pandas as pd

def simulate_true_cases(locations, days, lam=5, random_seed=None):
    """
    Simulate true case counts using a Poisson distribution.
    Returns a (locations x days) array.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    true_cases = np.random.poisson(lam=lam, size=(locations, days))
    return true_cases


def mask_data(true_cases, mask_prob=0.5, random_seed=None):
    """
    Apply a random mask to simulate missing data.
    Returns observed_cases and the boolean mask.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    mask = np.random.rand(*true_cases.shape) < mask_prob
    observed_cases = true_cases * mask
    return observed_cases, mask