"""
Module: src/visualization.py
Description: Automated generation and saving of analytical plots (pre and post filtering)
and advanced Machine Learning visualizations (PCA, K-Medoids) with professional MLOps logging.
"""

import sys
import math
import itertools
from pathlib import Path
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import matplotlib.transforms as transforms
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from matplotlib.axes import Axes  

from sklearn.decomposition import PCA

# Add project root to sys.path to allow absolute imports when running as a script
sys.path.append(str(Path.cwd()))
from src.clustering import (normalize_dataset, train_kmedoids_pipeline, 
                            evaluate_cluster_metrics, get_optimal_pca_components, 
                            get_optimal_k)

# Professional logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("Visualization")

# Professional aesthetic configuration
sns.set_theme(style="whitegrid")

# Global Output Directory
OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# PART 1: EXPLORATORY DATA ANALYSIS (EDA) & FILTERING AUDITS
# =============================================================================

def analyze_histogram(data: pd.Series, name: str, n_bins: int | str = 'auto') -> pd.DataFrame:
    clean_data = data.dropna()
    total = len(clean_data)
    if total == 0:
        return pd.DataFrame()

    counts, bin_edges = np.histogram(clean_data, bins=n_bins)
    df_hist = pd.DataFrame({
        'Start (mV)': bin_edges[:-1],
        'End (mV)': bin_edges[1:],
        'Count': counts,
        'Percent (%)': (counts / total) * 100
    })
    return df_hist


def plot_voltage_span_distribution(jv_df: pd.DataFrame, stage_name: str = "Pre-Filter", filename: str = "voltage_span_distribution.png"):
    logger.info(f"Computing voltage span distribution for stage: {stage_name}...")

    volt_diff_series = (jv_df.groupby(['cell_name', 'id_curve'])['Voltage_V'].agg(np.ptp) * 1000)
    visual_bins = 16
    plt.figure(figsize=(12, 7))
    device_names = jv_df['cell_name'].unique()
    plotted_count = 0

    for name in device_names:
        cell_data = volt_diff_series.get(name, pd.Series(dtype=float))
        if not cell_data.empty:
            sns.histplot(cell_data, bins=visual_bins, kde=True, alpha=0.3, label=name)
            plotted_count += 1

    if plotted_count == 0:
        plt.close()
        return

    plt.xlabel(r'$\Delta V = V_{max} - V_{min}$ (mV)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'Distribution of Voltage Spans per Cell ({stage_name})', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = OUTPUT_DIR / filename
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Plot successfully saved to: {output_path}")
    plt.close()


# =============================================================================
# PART 2: MACHINE LEARNING & CLUSTERING VISUALIZATIONS
# =============================================================================

def plot_confidence_ellipse(x: np.ndarray, y: np.ndarray, ax: Axes, n_std: float = 1.0, facecolor: str = 'none', **kwargs):
    if x.size < 2 or y.size < 2: return
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x, ell_radius_y = np.sqrt(1 + pearson), np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, **kwargs)
    scale_x, scale_y = np.sqrt(cov[0, 0]) * n_std, np.sqrt(cov[1, 1]) * n_std
    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(float(np.mean(x)), float(np.mean(y)))
    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)


def plot_pca_variance_and_loadings(pca_full, pca, n_points: int = 50, filename: str = "pca_variance_loadings.png"):
    fig, (ax_var, ax_load) = plt.subplots(1, 2, figsize=(16, 6))
    
    max_comps = min(10, len(pca_full.explained_variance_ratio_))
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)[:max_comps + 1]
    
    ax_var.plot(range(1, len(cum_var) + 1), cum_var, marker='o', lw=3, ms=8, color='teal')
    ax_var.axhline(y=0.90, color='r', linestyle='--', lw=2, label='90% Explained Variance')
    ax_var.set(xlabel='Number of Components', ylabel='Cumulative Explained Variance', 
               title='PCA: Cumulative Explained Variance', xlim=(0.5, max_comps + 1.5))
    ax_var.grid(True, linestyle='--', alpha=0.5)
    ax_var.legend(fontsize=12)

    v_norm = np.linspace(0, 1, n_points)
    cmap_loadings = plt.cm.tab10
    
    for i in range(pca.n_components_):
        pc_rev = pca.components_[i, :n_points]
        pc_fwd = pca.components_[i, n_points:]
        c = cmap_loadings(i % 10)
        ax_load.plot(v_norm, pc_rev, '-', lw=3, color=c, label=f'PC{i+1} (Rev)')
        ax_load.plot(v_norm, pc_fwd, '--', lw=3, color=c, alpha=0.6, label=f'PC{i+1} (Fwd)')

    ax_load.axhline(0, color='black', lw=1, linestyle='--')
    ax_load.set(xlabel='Normalized Voltage', ylabel='Loading Weight', title=f'PCA Loadings ({pca.n_components_} Components)')
    ax_load.grid(True, linestyle='--', alpha=0.3)
    ax_load.legend(fontsize=10, loc='best', ncol=(2 if pca.n_components_ > 2 else 1))
    
    plt.tight_layout()
    if filename:
        output_path = OUTPUT_DIR / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot successfully saved to: {output_path}")
    plt.close()


def plot_cluster_evaluation(cluster_range: list, inertia: list, silhouette: list, n_clusters: int, filename: str = "cluster_evaluation.png"):
    fig, ax_metrics = plt.subplots(figsize=(10, 6))
    
    color_wcss = 'tab:blue'
    ax_metrics.set_xlabel('Number of Clusters (k)', fontsize=14)
    ax_metrics.set_ylabel('Inertia (WCSS)', color=color_wcss, fontsize=14)
    lns1 = ax_metrics.plot(cluster_range, inertia, marker='o', color=color_wcss, lw=3, ms=10, label='Inertia')
    ax_metrics.tick_params(axis='y', labelcolor=color_wcss)

    ax_sil = ax_metrics.twinx()
    color_sil = 'tab:red'
    ax_sil.set_ylabel('Silhouette Score', color=color_sil, fontsize=14)
    lns2 = ax_sil.plot(cluster_range, silhouette, marker='s', color=color_sil, lw=3, ms=10, label='Silhouette')
    ax_sil.tick_params(axis='y', labelcolor=color_sil)
    ax_sil.set_ylim(0, 1)

    ax_metrics.axvline(x=n_clusters, color='k', linestyle='--', alpha=0.5, label=f'Chosen k={n_clusters}')
    
    lns = lns1 + lns2 + [ax_metrics.get_lines()[-1]]
    labs = [str(l.get_label()) for l in lns]
    
    ax_metrics.legend(lns, labs, loc='upper right', fontsize=12)
    ax_metrics.set_title('Optimal Clusters Validation', fontsize=16, fontweight='bold')
    ax_metrics.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    if filename:
        output_path = OUTPUT_DIR / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot successfully saved to: {output_path}")
    plt.close()


def plot_cluster_median_curves(X_real: np.ndarray, labels: np.ndarray, n_clusters: int, n_points: int = 50, filename: str = "cluster_median_curves.png"):
    cols = min(n_clusters, 3)
    rows = math.ceil(n_clusters / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows), dpi=150)
    
    axes_flat = np.array(axes).flatten() if n_clusters > 1 else [axes]
    colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
    v_norm = np.linspace(0, 1, n_points)

    for i in range(n_clusters):
        ax = axes_flat[i]
        mask = labels == i
        n_samples = np.sum(mask)
        if n_samples == 0: continue
        
        median_c = np.median(X_real[mask], axis=0)
        p25_c = np.percentile(X_real[mask], 25, axis=0)
        p75_c = np.percentile(X_real[mask], 75, axis=0)

        ax.plot(v_norm, median_c[:n_points], color=colors[i], linewidth=3.5)
        ax.fill_between(v_norm, p25_c[:n_points], p75_c[:n_points], color=colors[i], alpha=0.25)
        
        ax.plot(v_norm, median_c[n_points:], linestyle='--', color=colors[i], linewidth=3.5, alpha=0.9)
        ax.fill_between(v_norm, p25_c[n_points:], p75_c[n_points:], color=colors[i], alpha=0.15)

        ax.set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05), xlabel=r'$V_{\mathrm{norm}}$', ylabel=r'$J_{\mathrm{norm}}$')
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.text(0.95, 0.95, f'Type {i}\n(n={n_samples})', transform=ax.transAxes, va='top', ha='right',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))

    for j in range(n_clusters, len(axes_flat)):
        fig.delaxes(axes_flat[j])
        
    plt.tight_layout()
    if filename:
        output_path = OUTPUT_DIR / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot successfully saved to: {output_path}")
    plt.close()


def plot_comparison_heatmaps(jv_labeled: pd.DataFrame, n_clusters: int, filename: str = "comparison_heatmaps.png"):
    df_curves = jv_labeled[['id_curve', 'cell_id', 'cell_name', 'Timestamp', 'label_curve']].drop_duplicates(subset=['id_curve']).copy()
    unique_cells = df_curves[['cell_id', 'cell_name']].drop_duplicates().sort_values('cell_id')
    
    cols = min(len(unique_cells), 3)
    rows = math.ceil(len(unique_cells) / cols)
    
    if rows == 0 or cols == 0:
        return
        
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 7 * rows), dpi=150)
    plt.subplots_adjust(hspace=0.4)
    axes_flat = np.array(axes).flatten() if len(unique_cells) > 1 else [axes]
    cmap = plt.cm.viridis

    for i, (_, row) in enumerate(unique_cells.iterrows()):
        ax = axes_flat[i]
        df = df_curves[df_curves['cell_id'] == row['cell_id']].copy()
        df['date'] = pd.to_datetime(df['Timestamp']).dt.date
        df['hour'] = pd.to_datetime(df['Timestamp']).dt.hour
        
        heatmap_data = df.pivot_table(index='date', columns='hour', values='label_curve',
                                      aggfunc=lambda x: x.mode().iloc[0] if not x.empty else None)
        
        sns.heatmap(heatmap_data, cmap=cmap, ax=ax, cbar=False, vmin=0, vmax=n_clusters-1)
        ax.set_title(f"Device: {row['cell_name']}", fontweight='bold')
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Date")

    for j in range(len(unique_cells), len(axes_flat)):
        fig.delaxes(axes_flat[j])
        
    if filename:
        output_path = OUTPUT_DIR / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot successfully saved to: {output_path}")
    plt.close()


def plot_3d_pca_interactive(jv_labeled: pd.DataFrame, curves_normalized: pd.Series, n_clusters: int = 4, filename: str = "pca_3d_interactive.html"):
    df_labels = jv_labeled[['cell_id', 'curve', 'label_curve']].drop_duplicates()
    X = np.stack(curves_normalized.tolist())
    
    pca = PCA(n_components=3).fit(X)
    X_pca = pca.transform(X)
    
    df_pca = curves_normalized.index.to_frame(index=False)
    df_pca[['PC1', 'PC2', 'PC3']] = X_pca
    df_3d = df_pca.merge(df_labels, on=['cell_id', 'curve'], how='inner')
    df_3d['Cluster'] = df_3d['label_curve'].astype(int).astype(str)
    
    colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
    color_map = {str(i): f'rgba({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)}, {c[3]})' 
                 for i, c in enumerate(colors)}
                 
    fig = px.scatter_3d(df_3d, x='PC1', y='PC2', z='PC3', color='Cluster',
                        color_discrete_map=color_map, opacity=0.75,
                        title=f"3D PCA Map ({np.sum(pca.explained_variance_ratio_)*100:.1f}% Total Variance)")
    fig.update_traces(marker=dict(size=3.5))
    
    if filename:
        output_path = OUTPUT_DIR / filename
        fig.write_html(output_path, include_plotlyjs=True)
        logger.info(f"Interactive 3D plot saved to: {output_path}")


# =============================================================================
# AUTOMATED SCRIPT EXECUTION 
# =============================================================================

if __name__ == "__main__":
    PROCESSED_DIR = Path("data/processed/outdoor")
    
    if not PROCESSED_DIR.exists():
        logger.error(f"Directory {PROCESSED_DIR} not found. Run data_processing.py first.")
        sys.exit(1)

    # Stage 1: Pre-filter exploratory data analysis
    device_dirs = [d.name for d in PROCESSED_DIR.iterdir() if d.is_dir()]
    processed_dfs = []
    logger.info(f"Scanning processed outdoor devices for Pre-Filter visualization: {device_dirs}")

    for name in device_dirs:
        parquet_path = PROCESSED_DIR / name / f"{name}_jv.parquet"
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            if df.empty: 
                continue
            df['cell_name'] = name
            is_reverse_start = (df['ScanDirection'] == 'Reverse') & (df['ScanDirection'].shift(1) != 'Reverse')
            is_reverse_start.iloc[0] = True
            df['curve'] = is_reverse_start.cumsum()
            processed_dfs.append(df)

    if processed_dfs:
        jv_raw = pd.concat(processed_dfs, axis=0, ignore_index=True)
        jv_raw['id_curve'] = jv_raw.groupby(['cell_name', 'curve']).ngroup()
        jv_raw = jv_raw[jv_raw['Timestamp'].dt.time.between(pd.to_datetime('06:00').time(), pd.to_datetime('22:00').time())]
        
        plot_voltage_span_distribution(jv_df=jv_raw, stage_name="Pre-Filter (Raw)", filename="voltage_span_pre_filter.png")

    # Stage 2: Post-filter exploratory data analysis and unsupervised clustering
    filtered_parquet_path = PROCESSED_DIR / "jv_dataset_filtered.parquet"

    if filtered_parquet_path.exists():
        logger.info(f"Loading post-filter dataset from: {filtered_parquet_path}")
        jv_filtered = pd.read_parquet(filtered_parquet_path)
        jv_clean = jv_filtered[jv_filtered['is_curve_valid'] == 1].copy()
        
        plot_voltage_span_distribution(jv_df=jv_clean, stage_name="Post-Filter (Clean)", filename="voltage_span_post_filter.png")

        logger.info("Initializing Machine Learning Pipeline...")
        N_POINTS = 50
        curves_norm = normalize_dataset(jv_clean, n_points=N_POINTS)
        logger.info(f"Total physical curves successfully normalized: {len(curves_norm)}")
        
        # Hyperparameter auto-selection sequence
        logger.info("Starting hyperparameter auto-tuning process...")
        
        # Optimal PCA components selection based on target variance
        X_all = np.stack(curves_norm.tolist())
        NUM_PCA = get_optimal_pca_components(X_all, target_variance=0.90)
        logger.info(f"Auto-PCA: Selected {NUM_PCA} components to retain >= 90% variance.")
        
        # Optimal K selection using a fast sample to balance RAM efficiency
        X_temp = np.stack(curves_norm.sample(min(5000, len(curves_norm)), random_state=42).tolist())
        X_pca_temp = PCA(n_components=NUM_PCA, random_state=42).fit_transform(X_temp)
        NUM_CLUSTERS = get_optimal_k(X_pca_temp, max_k=8)
        
        # Model training execution with dynamic parameters
        ml_results = train_kmedoids_pipeline(
            jv_master=jv_clean,
            curves_normalized=curves_norm,
            n_points=N_POINTS,
            n_clusters=NUM_CLUSTERS,
            n_pca_components=NUM_PCA
        )

        pca_full = ml_results["pca_full_model"]
        pca = ml_results["pca_model"]
        cum_var = np.cumsum(pca_full.explained_variance_ratio_)
        
        logger.info("--- PCA Explained Variance ---")
        for i, var in enumerate(pca_full.explained_variance_ratio_[:5]):
            logger.info(f"  PC{i+1}: {var * 100:>5.2f}% (Cumulative: {cum_var[i] * 100:>5.2f}%)")
        logger.info("------------------------------")
        
        logger.info("Generating advanced clustering visualizations...")
        
        plot_pca_variance_and_loadings(pca_full=pca_full, pca=pca, n_points=N_POINTS, filename="01_pca_variance_loadings.png")
        
        # Final metric computation for historical reporting in figures
        k_range, inertia, silhouette, calinski, davies = evaluate_cluster_metrics(X_pca_temp, max_k=8)
        
        logger.info("--- Clustering Metrics Evaluation ---")
        logger.info(f"  {'k':<2} | {'Inertia (WCSS)':<15} | {'Silhouette':<12} | {'Calinski-Harabasz':<18} | {'Davies-Bouldin':<15}")
        logger.info("-" * 75)
        for k_val, wcss_val, sil_val, cal_val, dav_val in zip(k_range, inertia, silhouette, calinski, davies):
            logger.info(f"  {k_val:<2d} | {wcss_val:<15.2f} | {sil_val:<12.4f} | {cal_val:<18.2f} | {dav_val:<15.4f}")
        logger.info("-" * 75)

        plot_cluster_evaluation(
            cluster_range=k_range, inertia=inertia, silhouette=silhouette, 
            n_clusters=NUM_CLUSTERS, filename="02_cluster_evaluation.png"
        )
        
        plot_cluster_median_curves(
            X_real=ml_results["X_real"], labels=ml_results["labels"], 
            n_clusters=NUM_CLUSTERS, n_points=N_POINTS, filename="03_cluster_median_curves.png"
        )
        
        plot_comparison_heatmaps(
            jv_labeled=ml_results["jv_labeled"], n_clusters=NUM_CLUSTERS, filename="04_comparison_heatmaps.png"
        )
        
        plot_3d_pca_interactive(
            jv_labeled=ml_results["jv_labeled"], curves_normalized=curves_norm, 
            n_clusters=NUM_CLUSTERS, filename="05_pca_3d_interactive.html"
        )
        
        # --- FINAL AUTO-TUNING SUMMARY LOG ---
        idx_k = k_range.index(NUM_CLUSTERS)
        logger.info("======================================================")
        logger.info(" FINAL AUTO-TUNING SUMMARY")
        logger.info("======================================================")
        logger.info(f" Retained PCA : {NUM_PCA} components (Variance >= 90%)")
        logger.info(f" Clusters (k) : {NUM_CLUSTERS} (Minimum Davies-Bouldin)")
        logger.info(f" Metrics for k={NUM_CLUSTERS} -> WCSS: {inertia[idx_k]:.2f} | Silhouette: {silhouette[idx_k]:.4f} | Calinski: {calinski[idx_k]:.2f} | Davies-Bouldin: {davies[idx_k]:.4f}")
        logger.info("======================================================")
        
        logger.info("ALL VISUALIZATIONS GENERATED AND SAVED IN 'outputs/figures'!")
    else:
        logger.warning("Filtered dataset not found. Run src/filtering.py first.")
