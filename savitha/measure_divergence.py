import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.neighbors import KernelDensity
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler

def compute_kde(data, feature_idx, bandwidth=0.1, n_points=1000):
    """Compute KDE for a specific feature"""
    # Extract feature values and reshape for KDE
    if isinstance(data, torch.Tensor):
        feature_values = data[:, feature_idx].cpu().numpy().reshape(-1, 1)
    else:
        feature_values = data[:, feature_idx].reshape(-1, 1)
    
    # Determine evaluation range
    min_val = np.min(feature_values)
    max_val = np.max(feature_values)
    padding = (max_val - min_val) * 0.1  # Add 10% padding
    x_grid = np.linspace(min_val - padding, max_val + padding, n_points).reshape(-1, 1)
    
    # Fit KDE
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(feature_values)
    
    # Evaluate KDE
    log_dens = kde.score_samples(x_grid)
    dens = np.exp(log_dens)
    
    # Normalize density
    dens = dens / np.trapz(dens, x_grid.flatten())
    
    return x_grid.flatten(), dens

def compute_kl_divergence(p_dens, q_dens, x_grid):
    """Compute KL divergence between two probability distributions"""
    # Add small constant to avoid division by zero
    epsilon = 1e-10
    p_dens = np.maximum(p_dens, epsilon)
    q_dens = np.maximum(q_dens, epsilon)
    
    # Calculate KL divergence: ∫ p(x) * log(p(x) / q(x)) dx
    dx = x_grid[1] - x_grid[0]
    kl = np.sum(p_dens * np.log(p_dens / q_dens)) * dx
    
    return kl

def compare_distributions(x_1, x_2, feature_names=None, bandwidth=0.1):
    """
    Compare distributions of two datasets and compute KL divergence for each feature
    
    Parameters:
    -----------
    x_1 : numpy.ndarray or torch.Tensor
        First dataset with shape (n_samples, n_features)
    x_2 : numpy.ndarray or torch.Tensor
        Second dataset with shape (n_samples, n_features)
    feature_names : list, optional
        Names of the features
    bandwidth : float, optional
        Bandwidth for KDE
        
    Returns:
    --------
    kl_divergences : pandas.DataFrame
        DataFrame with KL divergence for each feature
    """
    # Convert to numpy if they're torch tensors
    if isinstance(x_1, torch.Tensor):
        x_1 = x_1.cpu().numpy()
    if isinstance(x_2, torch.Tensor):
        x_2 = x_2.cpu().numpy()
    
    n_features = x_1.shape[1]
    
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(n_features)]
    
    results = []
    
    for i in range(n_features):
        # Compute KDE for both datasets
        x_grid, p_dens = compute_kde(x_1, i, bandwidth)
        _, q_dens = compute_kde(x_2, i, bandwidth)
        
        # Compute KL divergence in both directions
        kl_p_to_q = compute_kl_divergence(p_dens, q_dens, x_grid)
        kl_q_to_p = compute_kl_divergence(q_dens, p_dens, x_grid)
        
        results.append({
            'Feature': feature_names[i],
            'KL(X₁||X₂)': kl_p_to_q,
            'KL(X₂||X₁)': kl_q_to_p,
            'Symmetric KL': (kl_p_to_q + kl_q_to_p) / 2
        })
    
    # Create and return DataFrame
    return pd.DataFrame(results).sort_values('Symmetric KL', ascending=False)

def plot_kl_divergence_by_feature(kl_df, figsize=(10, 6)):
    """Plot KL divergence for each feature"""
    plt.figure(figsize=figsize)
    
    # Create bar plot for symmetric KL divergence
    sns.barplot(x='Feature', y='Symmetric KL', data=kl_df)
    plt.xticks(rotation=45, ha='right')
    plt.title('KL Divergence Between Datasets by Feature')
    plt.tight_layout()
    
    return plt.gcf()

def plot_feature_distributions(x_1, x_2, feature_indices, feature_names=None, 
                              bandwidth=0.1, figsize=(15, 10), dataset_names=None):
    """
    Plot distributions of selected features from two datasets
    
    Parameters:
    -----------
    x_1, x_2 : numpy.ndarray or torch.Tensor
        Datasets to compare
    feature_indices : list
        Indices of features to plot
    feature_names : list, optional
        Names of all features
    bandwidth : float, optional
        Bandwidth for KDE
    figsize : tuple, optional
        Figure size
    dataset_names : list, optional
        Names for the two datasets
    """
    if dataset_names is None:
        dataset_names = ['Dataset 1', 'Dataset 2']
    
    if feature_names is None:
        n_features = max(x_1.shape[1], x_2.shape[1])
        feature_names = [f'Feature {i+1}' for i in range(n_features)]
    
    # Convert to numpy if they're torch tensors
    if isinstance(x_1, torch.Tensor):
        x_1 = x_1.cpu().numpy()
    if isinstance(x_2, torch.Tensor):
        x_2 = x_2.cpu().numpy()
    
    # Calculate number of rows and columns for subplots
    n_features = len(feature_indices)
    n_cols = 2
    n_rows = int(np.ceil(n_features / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, feature_idx in enumerate(feature_indices):
        if i >= len(axes):
            break
            
        ax = axes[i]
        feature_name = feature_names[feature_idx]
        
        # Compute KDE for both datasets
        x_grid, dens_1 = compute_kde(x_1, feature_idx, bandwidth)
        _, dens_2 = compute_kde(x_2, feature_idx, bandwidth)
        
        # Compute KL divergence
        kl_1_to_2 = compute_kl_divergence(dens_1, dens_2, x_grid)
        kl_2_to_1 = compute_kl_divergence(dens_2, dens_1, x_grid)
        symmetric_kl = (kl_1_to_2 + kl_2_to_1) / 2
        
        # Plot KDE curves
        ax.plot(x_grid, dens_1, label=dataset_names[0], color='blue')
        ax.plot(x_grid, dens_2, label=dataset_names[1], color='red')
        
        # Plot histograms
        ax.hist(x_1[:, feature_idx], bins=20, alpha=0.3, density=True, color='blue')
        ax.hist(x_2[:, feature_idx], bins=20, alpha=0.3, density=True, color='red')
        
        # Add title and legend
        ax.set_title(f"{feature_name}\nKL Div: {symmetric_kl:.4f}")
        ax.legend()
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.suptitle('Feature Distributions Comparison', fontsize=16)
    plt.subplots_adjust(top=0.9)
    
    return fig

# Example usage
def compare_two_datasets(x_1, x_2, feature_names=None, dataset_names=None):
    """
    Compare two datasets and visualize their differences
    
    Parameters:
    -----------
    x_1, x_2 : numpy.ndarray or torch.Tensor
        Datasets to compare
    feature_names : list, optional
        Names of features
    dataset_names : list, optional
        Names for the two datasets
        
    Returns:
    --------
    kl_df : pandas.DataFrame
        DataFrame with KL divergence for each feature
    """
    if dataset_names is None:
        dataset_names = ['Dataset 1', 'Dataset 2']
    
    # Compute KL divergence for all features
    kl_df = compare_distributions(x_1, x_2, feature_names)
    
    # Plot KL divergence by feature
    kl_fig = plot_kl_divergence_by_feature(kl_df)
    plt.figure(kl_fig.number)
    plt.title(f'KL Divergence Between {dataset_names[0]} and {dataset_names[1]} by Feature')
    
    # Select top 4 features with highest divergence for detailed plots
    top_features = kl_df.head(4)['Feature'].tolist()
    if feature_names is None:
        feature_indices = [int(f.split()[-1]) - 1 for f in top_features]
    else:
        feature_indices = [feature_names.index(f) for f in top_features]
    
    # Plot distributions of top features
    dist_fig = plot_feature_distributions(
        x_1, x_2, feature_indices, feature_names, 
        dataset_names=dataset_names
    )
    
    return kl_df, kl_fig, dist_fig

# Example of how to call the function:
# feature_names = ['PM2.5', 'PM10', 'O3', 'NO2', 'SO2', 'CO', 'Temperature', 'Humidity', 'Wind Speed', 'Wind Direction']
# dataset_names = ['Original Data', 'Generated Data']
# kl_df, kl_fig, dist_fig = compare_two_datasets(x_1, x_2, feature_names, dataset_names)
# 
# print(kl_df)
# plt.show()

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
X = df
feature_list = X.columns.tolist()
X = X.astype('float32')
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X = torch.tensor(X.values)
datasets = ['og', 'gen']
x_last = torch.load('X_last.pt')
kl_df, kl_fig, dist_fig = compare_two_datasets(X, x_last, feature_list, datasets)
print(kl_df)