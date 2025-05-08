import numpy as np
import pandas as pd
from data_simulation import simulate_true_cases, mask_data
from transforms import dct2, idct2
from recovery import generate_measurement_matrix, recover_lasso
from evaluation import compute_rmse, plot_matrices


def main():
    # Simulation parameters
    locations, days = 10, 30
    lam = 5
    mask_prob = 0.5
    m = 150        # number of compressive measurements
    alpha = 0.1    # regularization parameter for LASSO

    # Simulate true case data
    true_cases = simulate_true_cases(locations, days, lam, random_seed=42)
    observed_cases, mask = mask_data(true_cases, mask_prob, random_seed=42)

    # Wrap observed data as DataFrame
    df_obs = pd.DataFrame(observed_cases)
    df_obs.fillna(0, inplace=True)
    print("Observed cases sample:")
    print(df_obs.head())

    # Optional: sparse transform
    sparse_rep = dct2(observed_cases)

    # Prepare compressive sensing
    n = locations * days
    x_true_flat = true_cases.flatten()
    Phi = generate_measurement_matrix(m, n, random_seed=42)
    y = Phi.dot(x_true_flat)

    # Recover signal
    x_rec_flat = recover_lasso(Phi, y, alpha)
    x_rec = x_rec_flat.reshape((locations, days))

    # Evaluate
    rmse = compute_rmse(x_true_flat, x_rec_flat)
    print(f"RMSE: {rmse:.2f}")

    # Visualize
    plot_matrices(true_cases, x_rec)

if __name__ == "__main__":
    main()
