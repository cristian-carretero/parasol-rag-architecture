"""
Module: src/clustering.py
Description: Feature engineering (normalization), PCA dimensionality reduction, 
and K-Medoids clustering for J-V curve physical state extraction.
Targets the Complete Hysteresis Loop (Reverse + Forward sweeps).
"""

import numpy as np
import pandas as pd
import logging
from typing import Any, Tuple, List, Dict
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn_extra.cluster import KMedoids
from sklearn.ensemble import IsolationForest
from joblib import Parallel, delayed

# Professional MLOps logging configuration
logger = logging.getLogger("Clustering")

def normalize_curve(jv: pd.DataFrame, n_points: int = 50) -> np.ndarray:
    """
    Normalizes curves by capturing the complete hysteresis loop (default 100 points total).
    Returns a 1D array of length n_points*2, or an array of NaNs if the curve is structurally invalid.
    """
    V = jv['Voltage_V'].to_numpy(dtype=float)
    J = jv['Current_A'].to_numpy(dtype=float)

    if len(V) < 10:
        return np.full(n_points * 2, np.nan)

    v_min, v_max = V.min(), V.max()
    j_min, j_max = J.min(), J.max()
    v_range = v_max - v_min
    j_range = j_max - j_min

    # Prevent division by zero on corrupted flat curves
    if v_range == 0 or j_range == 0:
        return np.full(n_points * 2, np.nan)

    # Detect the physical turnaround point of the sweep (Voc)
    idx_turn = int(np.argmax(np.abs(V - V[0])))

    if idx_turn < 2 or idx_turn > len(V) - 3:
        return np.full(n_points * 2, np.nan)

    # Split into Reverse and Forward scans
    V_rev, J_rev = V[:idx_turn + 1], J[:idx_turn + 1]
    V_fwd, J_fwd = V[idx_turn:], J[idx_turn:]

    def interpolate_sweep(v_sweep, j_sweep):
        v_norm = (v_sweep - v_min) / v_range
        j_norm = (j_sweep - j_min) / j_range
        
        # Enforce strict ascending order for robust scipy interpolation
        sort_idx = np.argsort(v_norm)
        v_n_s, j_n_s = v_norm[sort_idx], j_norm[sort_idx]
        
        # Average duplicated voltage readings caused by sensor resolution limits
        df_t = pd.DataFrame({'v': v_n_s, 'j': j_n_s}).groupby('v', as_index=False).mean()
        
        if len(df_t) < 2:
            return np.full(n_points, np.nan)
            
        return np.interp(
            np.linspace(0, 1, n_points),
            df_t['v'].to_numpy(),
            df_t['j'].to_numpy(),
            left=float(df_t['j'].iloc[0]),
            right=float(df_t['j'].iloc[-1])
        )

    try:
        return np.concatenate([interpolate_sweep(V_rev, J_rev), interpolate_sweep(V_fwd, J_fwd)])
    except Exception:
        return np.full(n_points * 2, np.nan)


def normalize_dataset(jv_df: pd.DataFrame, n_points: int = 50) -> pd.Series:
    """
    Applies global shape normalization to the entire dataset and drops non-interpolable curves.
    Leverages Joblib for robust, multi-core processing execution.
    """
    logger.info("Applying full hysteresis shape normalization to all curves (Parallelized via joblib)...")
    
    groups = list(jv_df.groupby(['cell_id', 'curve']))
    keys = [name for name, df in groups]
    
    results = Parallel(n_jobs=2, backend="threading")(
        delayed(normalize_curve)(df, n_points) for name, df in groups
    )
    
    # Cast generator results to list to satisfy strict static typing (Pylance)
    curves_norm = pd.Series(
        list(results), 
        index=pd.MultiIndex.from_tuples(keys, names=['cell_id', 'curve']),
        dtype=object
    )
    
    valid_mask = curves_norm.apply(
        lambda x: not (isinstance(x, float) and np.isnan(x)) and not np.isnan(x).any()
    )
    
    return curves_norm[valid_mask]


def stratified_sample_by_cell(curves_series: pd.Series, max_samples: int) -> pd.Series:
    """
    Performs balanced stratified sampling across all device IDs to prevent 
    highly prolific cells from skewing the statistical PCA/K-Medoids topology space.
    """
    cell_ids = curves_series.index.get_level_values('cell_id').unique()
    samples_per_cell = max(1, max_samples // len(cell_ids))
    
    # Sample equitably across all available devices up to the defined ceiling limit
    sampled = curves_series.groupby(level='cell_id', group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), samples_per_cell), random_state=42)
    )
    return sampled


def evaluate_cluster_metrics(X_pca_vals: np.ndarray, pca_model: PCA, n_points: int = 50, max_k: int = 8, sample_size: int = 2000) -> Tuple[List[int], List[float], List[float], List[float], List[float], List[float]]:
    """
    Evaluates optimal K using statistical bounds (Inertia, Silhouette, Calinski-Harabasz, Davies-Bouldin)
    and custom thermodynamic separation metrics (Min ΔpFF).
    """
    logger.info(f"Evaluating multi-dimensional cluster metrics up to k={max_k}...")
    inertia, silhouette_avg, calinski, davies, min_dpff_list = [], [], [], [], []
    cluster_range = list(range(2, max_k + 1))
    
    actual_sample_size = min(sample_size, len(X_pca_vals))
    v_norm = np.linspace(0, 1, n_points)
    
    for k in cluster_range:
        km_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbls_temp = km_temp.fit_predict(X_pca_vals)
        
        inertia.append(float(km_temp.inertia_))
        silhouette_avg.append(float(silhouette_score(X_pca_vals, lbls_temp, sample_size=actual_sample_size, random_state=42)))
        calinski.append(float(calinski_harabasz_score(X_pca_vals, lbls_temp)))
        davies.append(float(davies_bouldin_score(X_pca_vals, lbls_temp)))
        
        # --- THERMODYNAMIC XAI METRIC: Min ΔpFF ---
        # Ensures identified clusters represent distinct physical states, not just statistical artifacts
        centers_orig = pca_model.inverse_transform(km_temp.cluster_centers_)
        pffs = [(np.max(v_norm * c[:n_points]) + np.max(v_norm * c[n_points:])) / 2 for c in centers_orig]
        min_diff = min([abs(pffs[i] - pffs[j]) for i in range(k) for j in range(i+1, k)])
        min_dpff_list.append(float(min_diff))
        
    return cluster_range, inertia, silhouette_avg, calinski, davies, min_dpff_list


def get_optimal_pca_components(X: np.ndarray, target_variance: float = 0.90) -> int:
    """Dynamically identifies the minimum number of principal components needed to reach the target variance."""
    pca = PCA().fit(X)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    optimal_components = int(np.argmax(cum_var >= target_variance) + 1)
    logger.info(f"PCA dynamic selection: {optimal_components} components successfully capture >= {target_variance*100}% variance.")
    return optimal_components


def get_optimal_k(X_pca: np.ndarray, pca_model: PCA, max_k: int = 8, sample_size: int = 2000, min_valid_k: int = 3, min_dpff_thresh: float = 0.04) -> int:
    """
    Intelligent K-selection algorithm (Smart Auto-K).
    Bypasses trivial clustering splits and applies statistical consensus paired with Thermodynamic Thresholding.
    """
    k_range, inertia, silhouette, calinski, davies, min_dpff = evaluate_cluster_metrics(
        X_pca, pca_model=pca_model, max_k=max_k, sample_size=sample_size
    )

    # 1. Normalized Geometric Elbow formulation
    k_norm = (np.array(k_range) - np.min(k_range)) / (np.max(k_range) - np.min(k_range))
    inertia_norm = (np.array(inertia) - np.min(inertia)) / (np.max(inertia) - np.min(inertia))
    
    coords = np.column_stack((k_norm, inertia_norm))
    line_vec = coords[-1] - coords[0]
    line_vec_norm = line_vec / np.linalg.norm(line_vec)
    
    vecs_from_first = coords - coords[0]
    scalar_projections = np.sum(vecs_from_first * line_vec_norm, axis=1)
    vec_projections = np.outer(scalar_projections, line_vec_norm)
    
    dists_to_line = np.linalg.norm(vecs_from_first - vec_projections, axis=1)

    # 2. Console Visual Diagnostic Block
    print("\n" + "=" * 105)
    print(" [DIAGNOSTICS] AUTOMATED TOPOLOGICAL K-SELECTION SPACE EVALUATION")
    print("=" * 105)
    print(f" {'K':<4} | {'Inertia (↓)':<13} | {'Elbow Dist (↑)':<15} | {'Silhouette (↑)':<15} | {'Davies-Bouldin (↓)':<18} | {'Min ΔpFF (↑)':<12}")
    print("-" * 105)
    
    for i, k in enumerate(k_range):
        flag = " *" if (k >= min_valid_k and min_dpff[i] >= min_dpff_thresh) else "  "
        print(f" {k:<4}{flag}| {inertia[i]:<13.2f} | {dists_to_line[i]:<15.4f} | {silhouette[i]:<15.4f} | {davies[i]:<18.4f} | {min_dpff[i]:<12.4f}")
    print("=" * 105)

    # 3. Consensus Logic (Strictly ignoring physically redundant K values)
    valid_indices = [i for i, k in enumerate(k_range) if k >= min_valid_k and min_dpff[i] >= min_dpff_thresh]
    
    if not valid_indices: 
        logger.warning("No evaluated K satisfies the thermodynamic separation threshold. Falling back to optimal maximum distance.")
        valid_indices = [i for i, k in enumerate(k_range) if k >= min_valid_k]
        if not valid_indices: valid_indices = list(range(len(k_range)))
        
    best_elbow_idx = valid_indices[int(np.argmax(dists_to_line[valid_indices]))]
    best_sil_idx = valid_indices[int(np.argmax(np.array(silhouette)[valid_indices]))]
    best_db_idx = valid_indices[int(np.argmin(np.array(davies)[valid_indices]))]
    
    best_elbow_k = k_range[best_elbow_idx]
    best_sil_k = k_range[best_sil_idx]
    best_db_k = k_range[best_db_idx]

    final_k = best_elbow_k
    consensus = False

    # Allow statistical consensus override if Silhouette and Davies-Bouldin agree against the Elbow method
    if best_sil_k == best_db_k and best_sil_k != best_elbow_k:
        final_k = best_sil_k
        consensus = True

    print(" [DECISION LOGIC]")
    print(f" -> Optimal Geometric Elbow (Valid space) : {best_elbow_k}")
    print(f" -> Max Silhouette Score (Valid space)    : {best_sil_k}")
    print(f" -> Min Davies-Bouldin Index (Valid space): {best_db_k}")
    print("-" * 105)
    if consensus:
        print(f" >> FINAL SELECTED k : {final_k} (Statistical Consensus Override)")
    else:
        print(f" >> FINAL SELECTED k : {final_k} (Geometric Elbow Anchor)")
    print("=" * 105 + "\n")

    return final_k


def analyze_cluster_medoids(centers_orig: np.ndarray, n_points: int = 50) -> Tuple[pd.DataFrame, List[int]]:
    """
    Calculates physical parameters for each cluster's medoid (V: 0 to 1, J: 0 to 1).
    Applies strict Thermodynamic Boundary Rules to flag and reject artificial or corrupted clusters.
    """
    v_norm = np.linspace(0, 1, n_points)
    metrics = []
    invalid_clusters = []
    
    print("\n" + "=" * 95)
    print(" [XAI DIAGNOSTICS] THERMODYNAMIC MEDOID VALIDATION (AUTO-PRUNING)")
    print("=" * 95)
    print(f" {'Cluster':<7} | {'J_sc @ V=0 (≥0.5)':<19} | {'J_oc @ V=Voc (≤0.3)':<21} | {'Pos. Slope % (≤15%)':<20} | {'Status'}")
    print("-" * 95)
    
    for i, center in enumerate(centers_orig):
        # Arrays are strictly ordered V=0 -> V=Voc due to interpolation mapping
        j_rev = center[:n_points]
        
        # 1. Short-circuit Rule (J_sc): At V=0, the photocurrent must be near its maximum.
        j_sc = j_rev[0]
        rule1_fail = j_sc < 0.50  
        
        # 2. Open-circuit Rule (J_oc): At V=Voc, the net current must approach zero.
        j_oc = j_rev[-1]
        rule2_fail = j_oc > 0.30
        
        # 3. Monotonicity Rule: Current should not increase with voltage (violates diode physics)
        dj_dv = np.diff(j_rev) / (np.diff(v_norm) + 1e-9)
        positive_slope_ratio = np.mean(dj_dv > 0.05) 
        rule3_fail = positive_slope_ratio > 0.15 # Max 15% tolerance allocated for environmental sensor noise
        
        is_invalid = rule1_fail or rule2_fail or rule3_fail
        status = "REJECTED ❌" if is_invalid else "PASS ✅"
        
        if is_invalid:
            invalid_clusters.append(i)
            
        metrics.append({
            "cluster_id": i,
            "j_sc": j_sc,
            "j_oc": j_oc,
            "positive_slope_ratio": positive_slope_ratio,
            "is_valid": not is_invalid
        })
        
        print(f" {i:<7} | {j_sc:<19.2f} | {j_oc:<21.2f} | {positive_slope_ratio:<20.2%} | {status}")
        
    print("=" * 95 + "\n")
    return pd.DataFrame(metrics), invalid_clusters


def train_kmedoids_pipeline(
    curves_normalized: pd.Series, 
    n_points: int = 50, 
    n_clusters: int | None = None, 
    n_pca_components: int | None = None,
    max_train_samples: int = 3000,
    contamination: float = 0.01
) -> Dict[str, Any]:
    """
    Core Unsupervised Pipeline: PCA Dimensionality Reduction -> Isolation Forest -> K-Medoids -> Pseudo-FF Reordering.
    Simultaneously evaluates complete hysteresis (reverse and forward sweeps).
    """
    logger.info("Training PCA and K-Medoids pipeline on complete hysteresis loops...")
    
    if len(curves_normalized) > max_train_samples:
        curves_train = stratified_sample_by_cell(curves_normalized, max_train_samples)
        logger.info(f"Stratified sampling enforced: {len(curves_train)} curves extracted uniformly across all devices.")
    else:
        curves_train = curves_normalized

    X_train = np.stack(curves_train.tolist())
    X_all = np.stack(curves_normalized.tolist())

    # 1. Principal Component Analysis (PCA)
    pca_full = PCA().fit(X_train)
    pca = PCA(n_components=n_pca_components, random_state=42)
    pca.fit(X_train)
    
    X_train_pca = pca.transform(X_train)
    X_all_pca = pca.transform(X_all)

    # 2. Outlier removal via Isolation Forest
    logger.info("Filtering structural anomalies and extreme environmental noise using Isolation Forest...")
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    outlier_preds = iso_forest.fit_predict(X_all_pca)
    
    clean_mask = (outlier_preds == 1)
    
    X_all_cleaned = X_all[clean_mask]
    X_all_pca_cleaned = X_all_pca[clean_mask]
    curves_normalized_cleaned = curves_normalized[clean_mask]
    
    logger.info(f"Initial raw curves: {len(curves_normalized)} | Valid curves post outlier filtration: {len(curves_normalized_cleaned)}")

    # 3. K-Medoids (PAM algorithm for robust center extraction)
    if len(X_all_pca_cleaned) > max_train_samples:
        train_indices = np.random.RandomState(42).choice(len(X_all_pca_cleaned), size=max_train_samples, replace=False)
        X_train_pca = X_all_pca_cleaned[train_indices]
    else:
        X_train_pca = X_all_pca_cleaned

    if n_clusters is None:
        raise ValueError("Parameter 'n_clusters' cannot be None. Explicit topology required.")

    kmedoids = KMedoids(n_clusters=n_clusters, random_state=42, method='pam')
    kmedoids.fit(X_train_pca)

    cluster_centers = kmedoids.cluster_centers_
    assert cluster_centers is not None, "Algorithm must be fitted before extracting spatial centers."

    # 4. Physics-based Cluster Reordering (Ranked by Pseudo Fill-Factor magnitude)
    v_norm = np.linspace(0, 1, n_points)
    centers_orig = pca.inverse_transform(cluster_centers)
    
    pseudo_ff = [(np.max(v_norm * center[:n_points]) + np.max(v_norm * center[n_points:])) / 2 for center in centers_orig]
    sorted_indices = np.argsort(pseudo_ff)[::-1]
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_indices)}

    labels_all_raw = kmedoids.predict(X_all_pca_cleaned)
    labels = np.array([label_mapping[l] for l in labels_all_raw])
    kmedoids.cluster_centers_ = cluster_centers[sorted_indices]
    
    centers_orig_sorted = centers_orig[sorted_indices]
    
    cluster_metrics_df, invalid_clusters = analyze_cluster_medoids(centers_orig_sorted, n_points=n_points)

    power_curves = X_all_cleaned * np.tile(v_norm, 2)
    pff_all = (np.max(power_curves[:, :n_points], axis=1) + np.max(power_curves[:, n_points:], axis=1)) / 2

    # --- AUTO-PRUNING LOGIC ---
    # Automatically discards curves assigned to physically impossible cluster spaces
    for inv_c in invalid_clusters:
        labels[labels == inv_c] = -1
        
    if invalid_clusters:
        logger.warning(f"Auto-pruning activated: Clusters {invalid_clusters} rejected due to physical violations. Affected curves routed to label -1.")

    # 5. Export lightweight labels bridging entity
    labels_bridge = curves_normalized_cleaned.index.to_frame(index=False)
    labels_bridge['label_curve'] = labels
    labels_bridge['pseudo_FF'] = pff_all
    
    rejected_curves = int(np.sum(labels == -1))
    logger.info(f"Flagged {rejected_curves} curves as structurally anomalous in this processing iteration.")
    logger.info("Pipeline iteration successfully completed.")
    
    return {
        "labels_bridge": labels_bridge,
        "X_real": X_all_cleaned,
        "X_pca": X_all_pca_cleaned,
        "labels": labels,
        "pca_model": pca,
        "pca_full_model": pca_full,
        "kmedoids_model": kmedoids,
        "cluster_metrics": cluster_metrics_df,
        "invalid_clusters": invalid_clusters   
    }

if __name__ == "__main__":
    from pathlib import Path
    import joblib

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    FILTERED_DIR = Path("data/filtered/outdoor")
    CLUSTERED_DIR = Path("data/clustered/outdoor")
    CLUSTERED_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR = Path("data/clustered/artifacts/outdoor")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    filtered_parquet_path = FILTERED_DIR / "jv_dataset_filtered.parquet"

    if not filtered_parquet_path.exists():
        logger.error(f"Filtered dataset artifact not found at: {filtered_parquet_path}. Execution aborted.")
    else:
        logger.info("Initializing Iterative Clustering pipeline execution...")
        logger.info("Loading filtered dataset into memory (Only physically valid curves retained)...")
        jv_clean = pd.read_parquet(
            filtered_parquet_path,
            filters=[('is_curve_valid', '==', 1)]
        )

        curves_norm = normalize_dataset(jv_clean, n_points=50)

        # Flush raw dataset from RAM to prevent OOM errors during matrix clustering
        del jv_clean
        import gc
        gc.collect()
        
        active_curves = curves_norm.copy()
        max_iterations = 5
        iteration = 1
        
        # Variable initialization to satisfy strict static typing (Pylance architecture checks)
        ml_results = None
        num_clusters = 0
        k_range, inertia, silhouette, calinski, davies, min_dpff = [], [], [], [], [], []

        # Autonomic Self-Healing Loop: Iteratively removes non-physical clusters and retrains
        while iteration <= max_iterations:
            logger.info(f"\n{'='*60}\n STARTING ALGORITHMIC ITERATION {iteration} | Active Curves: {len(active_curves)}\n{'='*60}")

            X_active = np.stack(active_curves.tolist())
            num_pca = get_optimal_pca_components(X_active, target_variance=0.90)

            X_temp_series = stratified_sample_by_cell(active_curves, min(1500, len(active_curves)))
            X_temp = np.stack(X_temp_series.tolist())
            
            pca_model_temp = PCA(n_components=num_pca, random_state=42)
            X_pca_temp = pca_model_temp.fit_transform(X_temp)
            
            num_clusters = get_optimal_k(X_pca_temp, pca_model=pca_model_temp, max_k=8, min_valid_k=3)

            k_range, inertia, silhouette, calinski, davies, min_dpff = evaluate_cluster_metrics(
                X_pca_temp, pca_model=pca_model_temp, max_k=8
            )

            ml_results = train_kmedoids_pipeline(
                curves_normalized=active_curves,
                n_points=50,
                n_clusters=num_clusters,
                n_pca_components=num_pca
            )

            invalid_clusters = ml_results["invalid_clusters"]

            if not invalid_clusters:
                logger.info(f"\n>>> CONVERGENCE REACHED AT ITERATION {iteration} <<<")
                logger.info("All cluster medoids successfully satisfy physical boundary conditions. Terminating optimization loop.")
                break
            else:
                logger.warning(
                    f"Iteration {iteration} flagged non-physical artifact clusters {invalid_clusters}. "
                    "Pruning anomalies and retraining pipeline topology..."
                )

                labels_bridge = ml_results["labels_bridge"]
                valid_curves_df = labels_bridge[labels_bridge['label_curve'] != -1][['cell_id', 'curve']].drop_duplicates()
                valid_keys = pd.MultiIndex.from_frame(valid_curves_df)

                active_curves = active_curves.loc[active_curves.index.intersection(valid_keys)]
                iteration += 1

        if iteration > max_iterations:
            logger.warning("Maximum iteration ceiling reached prior to total convergence. Serializing last valid algorithmic state.")

        # Firewall: If the optimization loop failed entirely, abort safely to prevent downstream corruption
        if ml_results is None:
            logger.error("Pipeline failed to produce valid ML analytical results. Aborting serialization.")
            exit(1)

        logger.info("Executing final projection of topological labels onto the raw dataset (Memory intensive process)...")
        
        # Reload jv_clean from disk as it was previously flushed from RAM to prevent OOM errors
        jv_clean = pd.read_parquet(
            filtered_parquet_path,
            filters=[('is_curve_valid', '==', 1)]
        )
        
        final_labels = ml_results["labels_bridge"]
        jv_labeled = jv_clean.merge(final_labels, on=['cell_id', 'curve'], how='left')
        jv_labeled['label_curve'] = jv_labeled['label_curve'].fillna(-1).astype(int)
        
        output_path = CLUSTERED_DIR / "jv_dataset_labeled.parquet"
        jv_labeled.to_parquet(output_path, engine='pyarrow', index=False)
        logger.info(f"Labeled dataset successfully serialized to: {output_path}")

        # Package ML artifacts for decoupled downstream visualization and explainability
        artifacts = {
            "pca_full_model": ml_results["pca_full_model"],
            "pca_model": ml_results["pca_model"],
            "X_real": ml_results["X_real"],
            "labels": ml_results["labels"],
            "curves_normalized": active_curves,   
            "k_range": k_range,
            "inertia": inertia,
            "silhouette": silhouette,
            "n_clusters": num_clusters,
            "cluster_metrics": ml_results["cluster_metrics"]
        }

        artifacts_path = ARTIFACTS_DIR / "clustering_artifacts.joblib"
        joblib.dump(artifacts, artifacts_path)
        logger.info(f"Clustering artifacts successfully serialized to: {artifacts_path}")