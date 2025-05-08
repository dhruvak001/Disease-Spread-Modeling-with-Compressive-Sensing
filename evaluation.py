import numpy as np
from sklearn.metrics import mean_squared_error

def compute_rmse(x_true, x_rec):
    """
    Compute RMSE between true and recovered signals.
    """
    return np.sqrt(mean_squared_error(x_true, x_rec))


def plot_matrices(true_matrix, rec_matrix, cmap='viridis'):
    """
    Plot side-by-side comparison of true vs recovered matrices.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(true_matrix, cmap=cmap)
    plt.title("True Case Matrix")
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(rec_matrix, cmap=cmap)
    plt.title("Recovered Case Matrix")
    plt.colorbar()
    plt.show()
