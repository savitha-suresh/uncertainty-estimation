import numpy as np
from scipy.stats import gaussian_kde, entropy
from air_quality_uncertainty_estimation import get_data_air_quality
import torch
import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
from joblib import Parallel, delayed


def kl_divergence_kde(P, Q, num_points=1000):
    """
    Computes KL(Q || P) per feature using Gaussian KDE.
    P and Q are arrays of shape (n_samples, n_features)
    """
    n_features = P.shape[1]
    kl_divs = []

    for i in range(n_features):
        p_vals = P[:, i]
        q_vals = Q[:, i]

        # Estimate density
        kde_p = gaussian_kde(p_vals)
        kde_q = gaussian_kde(q_vals)

        # Create evaluation points over a common support
        min_val = min(p_vals.min(), q_vals.min())
        max_val = max(p_vals.max(), q_vals.max())
        x = np.linspace(min_val, max_val, num_points)

        # Evaluate densities
        p_density = kde_p(x)
        q_density = kde_q(x)

        # Avoid zero values for numerical stability
        p_density += 1e-10
        q_density += 1e-10

        # Normalize (to sum to 1, approximating discrete pdf over x)
        p_density /= p_density.sum()
        q_density /= q_density.sum()

        # KL(Q || P)
        kl = entropy(q_density, p_density)
        kl_divs.append(kl)

    return np.array(kl_divs)


X = get_data_air_quality()
feature_list = X.columns.tolist()
print(feature_list)
X = torch.tensor(X.values)
datasets = ['og', 'gen']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#x_last = torch.load('X_last.pt', map_location=device)
x_last = torch.load('x_last_v1.pt', map_location=device)
mc_samples = torch.load('mc_samples_v1.pt', map_location=device)
mc_samples = mc_samples.numpy()  # Convert to NumPy for joblib (if not already)

n_samples = mc_samples.shape[0]

def compute_kl_pair(i, j):
    kl = kl_divergence_kde(mc_samples[i], mc_samples[j])
    return kl

# 🧵 Parallel compute
results = Parallel(n_jobs=-1)(delayed(compute_kl_pair)(i, j)
                              for i in range(n_samples)
                              for j in range(i + 1, n_samples))

# Convert to numpy array
results = np.array(results)
print(f"Total KL pairs computed: {len(results)}")

# Aggregate across all pairs (mean per feature)
mean_kls = results.mean(axis=0)
print(f"Mean KL divergence per feature: {mean_kls}")


def kl_divergence_kde_with_plot(P, 
                                Q, 
                                feature_names=None, num_points=1000, max_features_to_plot=5):
    """
    Computes KL(Q || P) per feature using Gaussian KDE and plots KDEs and a bar chart.
    P and Q are DataFrames with same columns.
    """
    kl_divs = []

    for i, feature in enumerate(feature_names):
        p_vals = P[:, i]
        q_vals = Q[:, i]

        # KDE estimation
        kde_p = gaussian_kde(p_vals)
        kde_q = gaussian_kde(q_vals)

        # Common support
        min_val = min(p_vals.min(), q_vals.min())
        max_val = max(p_vals.max(), q_vals.max())
        x = np.linspace(min_val, max_val, num_points)

        # Evaluate KDEs
        p_density = kde_p(x)
        q_density = kde_q(x)

        # Numerical stability
        p_density += 1e-10
        q_density += 1e-10

        # Normalize
        p_density /= p_density.sum()
        q_density /= q_density.sum()

        # KL(Q || P)
        kl = entropy(q_density, p_density)
        kl_divs.append(kl)

        # Plot KDE for a few features
        if i < max_features_to_plot:
            plt.figure(figsize=(6, 4))
            plt.plot(x, p_density, label='P (reference)', linewidth=2)
            plt.plot(x, q_density, label='Q (target)', linewidth=2)
            plt.fill_between(x, p_density, q_density, color='gray', alpha=0.2)
            plt.title(f"Feature: {feature} — KL(Q || P) = {kl:.4f}")
            plt.xlabel("Value")
            plt.ylabel("Density")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    # Bar chart of KL divergence
    # kl_divs = sorted(kl_divs, reverse=True)
    # feature_names = sorted(feature_names, key=lambda x: kl_divs[feature_names.index(x)], reverse=True)
    plt.figure(figsize=(10, 5))
    plt.bar(feature_names, kl_divs, color='teal', alpha=0.8)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("KL Divergence (Q || P)")
    plt.title("KL Divergence per Feature")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    kl_df =  pd.Series(kl_divs, index=feature_names)
    sorted_series_desc = kl_df.sort_values(ascending=False)
    return sorted_series_desc



kl_series = kl_divergence_kde_with_plot(X, x_last, feature_list)
print(kl_series)
kl_df =  pd.Series(mean_kls, index=feature_list)
sorted_series_desc = kl_df.sort_values(ascending=False)
print(sorted_series_desc)
