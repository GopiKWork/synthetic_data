import os
import os.path
from pathlib import Path 
import glob
import pandas as pd
import numpy as np
import random
import torch
import bitsandbytes as bnb
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from torch.distributions import kl_divergence, Normal
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from scipy.stats import ks_2samp,wasserstein_distance
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


data_dir = 'datasets'
reqd_cols = [f'setting{i}' for i in range(1,4)] + [f's{i}' for i in range(1, 22)]

def get_engine_data(file_name='train_FD002.txt'):
    engine_df = pd.read_csv(f'{data_dir}/{file_name}',sep=' ', header=None)
    engine_df = engine_df.dropna(axis=1)
    engine_df.columns = ['id','cycle'] + reqd_cols
    engine_data = engine_df[reqd_cols]
    return engine_data

def kld(original_df,synthetic_df):
    epsilon = 1e-8
    # Convert data arrays to PyTorch tensors
    original_data_tensor = torch.tensor(original_df.values, dtype=torch.float32)
    synthetic_data_tensor = torch.tensor(synthetic_df.values, dtype=torch.float32)    
    
    # Calculate KL divergence for each column
    kl_divs = []
    for col in range(len(reqd_cols)):
        # Extract the column data
        synthetic_col = synthetic_data_tensor[:, col]
        original_col = original_data_tensor[:, col]
    
        # Calculate the mean and standard deviation for each column
        synthetic_mean = torch.mean(synthetic_col)
        synthetic_std = torch.std(synthetic_col) + epsilon
        original_mean = torch.mean(original_col)
        original_std = torch.std(original_col) + epsilon
    
        # Create Normal distributions for each column
        synthetic_dist = Normal(synthetic_mean, synthetic_std)
        original_dist = Normal(original_mean, original_std)
    
        # Calculate the KL divergence
        kl_div = kl_divergence(synthetic_dist, original_dist).item()
        kl_divs.append({reqd_cols[col]:kl_div})
    return kl_divs

def ks_test(original_df, synthetic_df):
    ks_results = []
    for col in reqd_cols:
        original_col = original_df[col]
        synthetic_col = synthetic_df[col]
        
        statistic, p_value = ks_2samp(original_col, synthetic_col)
        ks_results.append({col: (statistic, p_value)})
    return ks_results

def wd(original_df, synthetic_df):
    wds = []
    for col in reqd_cols:
        original_col = original_df[col]
        synthetic_col = synthetic_df[col]
        distance = wasserstein_distance(original_col, synthetic_col)
        wds.append({col: distance})
    return wds

def kld_mean(original_df, synthetic_df):
    klds = kld(original_df, synthetic_df)
    divergences = [list(kl_div.values())[0] for kl_div in klds]
    scaler = RobustScaler()
    divergences = scaler.fit_transform(np.array(divergences).reshape(-1, 1)).flatten()
    divergences = np.clip(divergences, None, np.percentile(divergences, 95)) ##winsorized_- clip after 80th %
    mean_divergence = sum(divergences) / len(divergences)
    return mean_divergence

def wd_mean(original_df, synthetic_df):
    wds = wd(original_df, synthetic_df)
    distances = [list(distance.values())[0] for distance in wds]
    distances = np.clip(distances, None, np.percentile(distances, 95)) ##winsorized_- clip after 95th %
    mean_distance = sum(distances) / len(distances)
    return mean_distance

def preprocess_data(data):
    if data.isnull().values.any():
        imputer = SimpleImputer(strategy='mean')
        data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    imputer = SimpleImputer(missing_values=0, strategy='mean')
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    data = data.dropna()
    return data

def compute_stats(original_df, synthetic_df):

    original_data = original_df.to_numpy()
    synthetic_data = synthetic_df.to_numpy()
    
    real_stats_df = pd.DataFrame({
        "Min": np.min(original_data, axis=0),
        "Max": np.max(original_data, axis=0),
        "Mean": np.mean(original_data, axis=0),
        "Std": np.std(original_data, axis=0)
    }, index=original_df.columns)
    
    synthetic_stats_df = pd.DataFrame({
        "Min": np.min(synthetic_data, axis=0),
        "Max": np.max(synthetic_data, axis=0),
        "Mean": np.mean(synthetic_data, axis=0),
        "Std": np.std(synthetic_data, axis=0)
    }, index=synthetic_df.columns)
    
    comparison_stats_df = pd.DataFrame({
        "KLD Mean": kld_mean(original_df, synthetic_df),
        "WD Mean": wd_mean(original_df, synthetic_df)
    }, index=["Comparison"])
    
    return real_stats_df, synthetic_stats_df, comparison_stats_df



def plot_kld(original_df,synthetic_df):
    kl_divs = kld(original_df,synthetic_df)
    values = [list(kl_div.values())[0] for kl_div in kl_divs]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(range(len(reqd_cols)), values)
    ax.set_xlabel('Columns')
    ax.set_ylabel('KL Divergence')
    ax.set_title('KL Divergence for Each Column')
    ax.set_xticks(range(len(reqd_cols)))  
    ax.set_xticklabels(reqd_cols, rotation=45, ha='right')
    ax.grid(True)
    ax.set_yscale('log') ##log scale

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 3))
    ax.yaxis.set_major_formatter(formatter)

    plt.tight_layout()
    return plt

def plot_ks_test(original_df, synthetic_df):
    ks_results = ks_test(original_df, synthetic_df)
    statistics = [list(ks_result.values())[0][0] for ks_result in ks_results]
    p_values = [list(ks_result.values())[0][1] for ks_result in ks_results]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 10))
    
    # KS statistic
    ax1.bar(range(len(reqd_cols)), statistics)
    ax1.set_xlabel('Columns')
    ax1.set_ylabel('KS Statistic')
    ax1.set_title('Kolmogorov-Smirnov Statistic for Each Column')
    ax1.set_xticks(range(len(reqd_cols)))  
    ax1.set_xticklabels(reqd_cols, rotation=45, ha='right')
    ax1.grid(True)
    
    # p-values
    ax2.bar(range(len(reqd_cols)), p_values)
    ax2.set_xlabel('Columns')
    ax2.set_ylabel('P-value')
    ax2.set_title('P-values for Each Column')
    ax2.set_xticks(range(len(reqd_cols)))  
    ax2.set_xticklabels(reqd_cols, rotation=45, ha='right')
    ax2.grid(True)
    ax2.set_yscale('log')  
    
    plt.tight_layout()
    return plt

def plot_wd(original_df, synthetic_df):
        ## Wasserstein distance
    wds = wd(original_df, synthetic_df)
    distances = [list(distance.values())[0] for distance in wds]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(range(len(reqd_cols)), distances)
    ax.set_xlabel('Columns')
    ax.set_ylabel('Distance')
    ax.set_title('Wasserstein Distance')
    ax.set_xticks(range(len(reqd_cols))) 
    ax.set_xticklabels(reqd_cols, rotation=45, ha='right')
    ax.grid(True)
    plt.tight_layout()
    return plt


def plot_distribs(original_df,synthetic_df):
    fig,axes = plt.subplots(nrows=8,ncols=3,figsize=(25,25))
    axes = axes.flatten()
    
    for j, col in enumerate(reqd_cols):
        df = pd.DataFrame( {'Real':original_df[col], 'Synthetic':synthetic_df[col]})
        df.plot(ax=axes[j], title=col,secondary_y='Synthetic',style=['-','--'])
    
    plt.tight_layout()
    return plt

def plot_hist(original_df,synthetic_df):
    fig, axes = plt.subplots(nrows=8, ncols=3, figsize=(25, 25))
    axes = axes.flatten()
    
    for j, col in enumerate(reqd_cols):
        sns.histplot(original_df[col], ax=axes[j], color='blue', alpha=0.8, label='Real',linestyle='-')
        sns.histplot(synthetic_df[col], ax=axes[j], color='orange', alpha=0.8, label='Synthetic',linestyle='--')
        axes[j].set_title(col)
        axes[j].legend()
    
    plt.tight_layout()
    return plt


def plot_tsne(original_df, synthetic_df):
    original_df = preprocess_data(original_df)
    synthetic_df = preprocess_data(synthetic_df)
    combined_data = pd.concat([original_df, synthetic_df], axis=0)
    labels = ['Real'] * len(original_df) + ['Synthetic'] * len(synthetic_df)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(combined_data)
    original_tsne = tsne_results[:len(original_df)]
    synthetic_tsne = tsne_results[len(original_df):]
    plt.figure(figsize=(4, 4))
    plt.scatter(original_tsne[:, 0], original_tsne[:, 1], c='blue', label='Real', alpha=0.5)
    plt.scatter(synthetic_tsne[:, 0], synthetic_tsne[:, 1], c='red', label='Synthetic', alpha=0.5)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.title('t-SNE')
    plt.tight_layout()
    return plt

def plot_pca(original_df, synthetic_df):
    original_df = preprocess_data(original_df)
    synthetic_df = preprocess_data(synthetic_df)
    combined_data = pd.concat([original_df, synthetic_df], axis=0)
    labels = ['Real'] * len(original_df) + ['Synthetic'] * len(synthetic_df)
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(combined_data)
    original_pca = pca_results[:len(original_df)]
    synthetic_pca = pca_results[len(original_df):]
    plt.figure(figsize=(4, 4))
    plt.scatter(original_pca[:, 0], original_pca[:, 1], c='blue', label='Real', alpha=0.5)
    plt.scatter(synthetic_pca[:, 0], synthetic_pca[:, 1], c='red', label='Synthetic', alpha=0.5)
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.legend()
    plt.title('PCA')
    plt.tight_layout()
    return plt

def show_n_save(algo_name,plot_type,original_df, synthetic_df,dir_path="metrics"):
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    file_name = f'{dir_path}/{algo_name}-{plot_type}.png'
    if plot_type == "pca":
        p = plot_pca(original_df,synthetic_df)
    elif plot_type == "tsne":
        p = plot_tsne(original_df,synthetic_df)
    elif plot_type == "hist":
        p = plot_hist(original_df,synthetic_df)
    elif plot_type == "distribs":
        p = plot_distribs(original_df,synthetic_df)
    elif plot_type == "kld":
        p = plot_kld(original_df,synthetic_df)
    elif plot_type == "ks_test":
        p = plot_ks_test(original_df,synthetic_df)
    elif plot_type == "wd":
        p = plot_wd(original_df,synthetic_df)
    p.savefig(file_name)
    p.show()  