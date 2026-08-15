"""
Module: src/clustering.py
Description: Feature engineering (normalization), PCA dimensionality reduction, 
and K-Medoids clustering for J-V curve physical state extraction.
"""

import numpy as np
import pandas as pd
import logging
from typing import Any
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn_extra.cluster import KMedoids
from sklearn.ensemble import IsolationForest

logger = logging.getLogger("Clustering")

def normalize_curve(jv: pd.DataFrame, n_points: int = 50) -> np.ndarray:
    """
    Normalizes curves by capturing the complete hysteresis loop (100 points total).
    Returns a 1D array of length n_points*2 or NaN if invalid.
    """
    V = jv['Voltage_V'].to_numpy(dtype=float)
    J = jv['Current_A'].to_numpy(dtype=float)

    if len(V) < 10:
        return np.full(n_points * 2, np.nan)

    v_min, v_max = V.min(), V.max()
    j_min, j_max = J.min(), J.max()
    v_range = v_max - v_min
    j_range = j_max - j_min

    if v_range == 0 or j_range == 0:
        return np.full(n_points * 2, np.nan)

    idx_turn = np.argmax(np.abs(V - V[0]))

    if idx_turn < 2 or idx_turn > len(V) - 3:
        return np.full(n_points * 2, np.nan)

    V_rev, J_rev = V[:idx_turn + 1], J[:idx_turn + 1]
    V_fwd, J_fwd = V[idx_turn:], J[idx_turn:]

    def interpolate_sweep(v_sweep, j_sweep):
        v_norm = (v_sweep - v_min) / v_range
        j_norm = (j_sweep - j_min) / j_range
        sort_idx = np.argsort(v_norm)
        v_n_s, j_n_s = v_norm[sort_idx], j_norm[sort_idx]
        df_t = pd.DataFrame({'v': v_n_s, 'j': j_n_s}).groupby('v', as_index=False).mean()
        
        if len(df_t) < 2:
            return np.full(n_points, np.nan)
            
        # Bypass SciPy's strict stub limitation for tuple fill_values
        f = interp1d(df_t['v'], df_t['j'], kind='linear', bounds_error=False,
                     fill_value=(float(df_t['j'].iloc[0]), float(df_t['j'].iloc[-1]))) # type: ignore
        return f(np.linspace(0, 1, n_points))

    try:
        return np.concatenate([interpolate_sweep(V_rev, J_rev), interpolate_sweep(V_fwd, J_fwd)])
    except Exception:
        return np.full(n_points * 2, np.nan)


def normalize_dataset(jv_df: pd.DataFrame, n_points: int = 50) -> pd.Series:
    """
    Applies normalization globally and drops non-interpolable curves.
    """
    logger.info("Applying local shape normalization to all curves...")
    
    # Explicit wrapper to help Pylance understand the DataFrame passing
    def _wrapper(df: Any) -> np.ndarray:
        return normalize_curve(df, n_points)

    # Bypass Pandas groupby.apply strict typing which fails on complex returns
    curves_norm = jv_df.groupby(['cell_id', 'curve']).apply(_wrapper) # type: ignore
    
    valid_mask = curves_norm.apply(
        lambda x: not (isinstance(x, float) and np.isnan(x)) and not np.isnan(x).any()
    )
    return curves_norm[valid_mask]


def evaluate_cluster_metrics(X_pca_vals: np.ndarray, max_k: int = 8, sample_size: int = 2000) -> tuple:
    """
    Evaluates optimal K using Inertia, Silhouette, Calinski-Harabasz, and Davies-Bouldin.
    """
    logger.info(f"Evaluating cluster metrics up to k={max_k}...")
    inertia = []
    silhouette_avg = []
    calinski = []
    davies = []
    cluster_range = range(2, max_k + 1)
    
    for k in cluster_range:
        km_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbls_temp = km_temp.fit_predict(X_pca_vals)
        
        inertia.append(float(km_temp.inertia_))
        silhouette_avg.append(float(silhouette_score(X_pca_vals, lbls_temp, sample_size=sample_size, random_state=42)))
        calinski.append(float(calinski_harabasz_score(X_pca_vals, lbls_temp)))
        davies.append(float(davies_bouldin_score(X_pca_vals, lbls_temp)))
        
    return list(cluster_range), inertia, silhouette_avg, calinski, davies

def train_kmedoids_pipeline(
    jv_master: pd.DataFrame, 
    curves_normalized: pd.Series, 
    n_points: int = 50, 
    n_clusters: int = 4, 
    n_pca_components: int = 3,
    max_train_samples: int = 10000,
    contamination: float = 0.01
) -> dict:
    """
    Core pipeline: PCA -> K-Medoids -> Pseudo-FF Reordering.
    Returns a dictionary with models and processed datasets for visualization.
    """
    logger.info("Training PCA and K-Medoids pipeline...")
    
    if len(curves_normalized) > max_train_samples:
        curves_train = curves_normalized.sample(n=max_train_samples, random_state=42)
    else:
        curves_train = curves_normalized

    X_train = np.stack(curves_train.tolist())
    X_all = np.stack(curves_normalized.tolist())

    # 1. PCA
    pca_full = PCA().fit(X_train)
    pca = PCA(n_components=n_pca_components, random_state=42)
    pca.fit(X_train)
    
    X_train_pca = pca.transform(X_train)
    X_all_pca = pca.transform(X_all)

    # Outlier removal via Isolation Forest to prune the distribution tail in the PCA space
    logger.info("Filtering anomalies and extreme noise using Isolation Forest...")
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    outlier_preds = iso_forest.fit_predict(X_all_pca)
    
    # Inlier mask generation where 1 represents normal data and -1 indicates outliers
    clean_mask = (outlier_preds == 1)
    
    # Filter arrays to retain only the core dense point cloud
    X_all_cleaned = X_all[clean_mask]
    X_all_pca_cleaned = X_all_pca[clean_mask]
    curves_normalized_cleaned = curves_normalized[clean_mask]
    
    logger.info(f"Original curves: {len(curves_normalized)} | Curves after outlier filtering: {len(curves_normalized_cleaned)}")


    # 2. K-Medoids
    if len(X_all_pca_cleaned) > max_train_samples:
        train_indices = np.random.RandomState(42).choice(len(X_all_pca_cleaned), size=max_train_samples, replace=False)
        X_train_pca = X_all_pca_cleaned[train_indices]
    else:
        X_train_pca = X_all_pca_cleaned

    kmedoids = KMedoids(n_clusters=n_clusters, random_state=42, method='pam')
    kmedoids.fit(X_train_pca)

    # Explicit Type Guard to assure Pylance the model is fitted
    cluster_centers = kmedoids.cluster_centers_
    assert cluster_centers is not None, "Model must be fitted before extracting centers."

    # 3. Physics-based Reordering (Pseudo Fill-Factor)
    v_norm = np.linspace(0, 1, n_points)
    centers_orig = pca.inverse_transform(cluster_centers)
    
    pseudo_ff = [(np.max(v_norm * center[:n_points]) + np.max(v_norm * center[n_points:])) / 2 for center in centers_orig]
    sorted_indices = np.argsort(pseudo_ff)[::-1]
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_indices)}

    # Predict and map labels
    labels_all_raw = kmedoids.predict(X_all_pca_cleaned)
    labels = np.array([label_mapping[l] for l in labels_all_raw])
    kmedoids.cluster_centers_ = cluster_centers[sorted_indices]

    power_curves = X_all_cleaned * np.tile(v_norm, 2)
    pff_all = (np.max(power_curves[:, :n_points], axis=1) + np.max(power_curves[:, n_points:], axis=1)) / 2

    # 4. Consolidate Labeled Dataset
    labels_bridge = curves_normalized_cleaned.index.to_frame(index=False)
    labels_bridge['label_curve'] = labels
    labels_bridge['pseudo_FF'] = pff_all
    
    jv_master_labeled = jv_master.copy()
    if 'label_curve' in jv_master_labeled.columns:
        jv_master_labeled.drop('label_curve', axis=1, inplace=True)
    if 'pseudo_FF' in jv_master_labeled.columns:
        jv_master_labeled.drop('pseudo_FF', axis=1, inplace=True)
        
    # Left merge retains all original curves, including those failing normalization or tagged as outliers
    jv_master_labeled = jv_master_labeled.merge(labels_bridge, on=['cell_id', 'curve'], how='left')
    
    # Fill unclassified curves (outliers and invalid sweeps) with -1
    jv_master_labeled['label_curve'] = jv_master_labeled['label_curve'].fillna(-1).astype(int)

    logger.info(f"Assigned label -1 to {sum(jv_master_labeled['label_curve'] == -1)} anomalous or unparseable curves.")
    logger.info("Pipeline successfully completed.")
    
    return {
        "jv_labeled": jv_master_labeled,
        "X_real": X_all_cleaned,
        "X_pca": X_all_pca_cleaned,
        "labels": labels,
        "pca_model": pca,
        "pca_full_model": pca_full,
        "kmedoids_model": kmedoids
    }

def get_optimal_pca_components(X: np.ndarray, target_variance: float = 0.90) -> int:
    """Finds the minimum number of components needed to reach the target variance."""
    pca = PCA().fit(X)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    optimal_components = int(np.argmax(cum_var >= target_variance) + 1)
    return optimal_components

def get_optimal_k(X_pca: np.ndarray, max_k: int = 8, sample_size: int = 2000) -> int:
    """Finds the optimal K by minimizing the Davies-Bouldin index to avoid the k=2 binary trap."""
    k_range, _, _, _, davies = evaluate_cluster_metrics(X_pca, max_k=max_k, sample_size=sample_size)

    best_idx = int(np.argmin(davies))
    best_k = k_range[best_idx]
    
    return best_k


if __name__ == "__main__":
    from pathlib import Path
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    
    PROCESSED_DIR = Path("data/processed/outdoor")
    filtered_parquet_path = PROCESSED_DIR / "jv_dataset_filtered.parquet"
    
    if not filtered_parquet_path.exists():
        logger.error(f"Filtered dataset not found at: {filtered_parquet_path}. Execution aborted.")
    else:
        logger.info("Initializing clustering pipeline execution...")
        logger.info("Loading filtered dataset...")
        jv_clean = pd.read_parquet(filtered_parquet_path)
        
        curves_norm = normalize_dataset(jv_clean, n_points=50)
        
        X_all = np.stack(curves_norm.tolist())
        num_pca = get_optimal_pca_components(X_all, target_variance=0.90)
        
        X_temp = np.stack(curves_norm.sample(min(1000, len(curves_norm)), random_state=42).tolist())
        X_pca_temp = PCA(n_components=num_pca, random_state=42).fit_transform(X_temp)
        num_clusters = get_optimal_k(X_pca_temp, max_k=8)

        ml_results = train_kmedoids_pipeline(
            jv_master=jv_clean,
            curves_normalized=curves_norm,
            n_points=50,
            n_clusters=num_clusters,
            n_pca_components=num_pca
        )
        
        output_path = PROCESSED_DIR / "jv_dataset_labeled.parquet"
        ml_results["jv_labeled"].to_parquet(output_path, index=False)
        logger.info(f"Labeled dataset successfully exported to: {output_path}")