import numpy as np
import pandas as pd
from scipy.sparse import issparse

# ---------- Helpers ----------

def _as_2d_array_from_obsm(adata_subset, key="X_cnv"):
    """Return a 2D ndarray shaped (n_cells, n_bins). Handles sparse, pandas, ragged, and transposed cases."""
    X = adata_subset.obsm[key]

    # Convert to ndarray
    if issparse(X):
        X = X.tocsr()
        X = X.toarray()
    elif isinstance(X, pd.DataFrame):
        X = X.values
    else:
        X = np.asarray(X)

    # Handle ragged/object rows
    if X.dtype == object:
        X = np.vstack([np.asarray(row).ravel() for row in X])

    # Ensure 2D
    if X.ndim == 1:
        X = X[None, :]

    # Make cells on axis 0
    n_obs = adata_subset.n_obs
    if X.shape[0] != n_obs and X.shape[1] == n_obs:
        X = X.T

    if X.shape[0] != n_obs:
        raise ValueError(f"X_cnv has incompatible shape {X.shape} for {n_obs} cells.")

    return X

def _safe_pct_change(new, old):
    return 0.0 if (old is None or np.isclose(old, 0.0)) else (new - old) / old * 100.0

def _per_cell_qabs(X, q=0.75):
    """Per-cell q-quantile of absolute CNV (less sensitive to zeros than median)."""
    return np.quantile(np.abs(X), q, axis=1)

def _rms_per_cell(X):
    """Per-cell root mean square amplitude (penalizes large events)."""
    return np.sqrt(np.mean(X**2, axis=1))

def _robust_std_per_cell_iqr(X):
    """Per-cell robust std via IQR/1.349. Fallback to RMS if IQR==0."""
    q1 = np.quantile(X, 0.25, axis=1)
    q3 = np.quantile(X, 0.75, axis=1)
    iqr = q3 - q1
    robust_std = iqr / 1.349
    # Fallback for degenerate rows
    fallback = _rms_per_cell(X)
    return np.where(np.isclose(robust_std, 0.0), fallback, robust_std)

def _lag1_autocorr_per_cell(X):
    """
    Vectorized lag-1 autocorrelation per row.
    Returns 0 when the variance is zero or when length < 2.
    """
    if X.shape[1] < 2:
        return np.zeros(X.shape[0], dtype=float)

    X0 = X[:, :-1]
    X1 = X[:,  1:]

    X0c = X0 - X0.mean(axis=1, keepdims=True)
    X1c = X1 - X1.mean(axis=1, keepdims=True)

    num = np.sum(X0c * X1c, axis=1)
    den = np.sqrt(np.sum(X0c**2, axis=1) * np.sum(X1c**2, axis=1))
    ac = num / np.where(den == 0, np.nan, den)
    return np.nan_to_num(ac, nan=0.0)

# ---------- Metrics (robust versions) ----------

def calculate_global_cnv_contrast(adata, tumor_cell_type, cell_type_key, q=0.75, mix_with_rms=True, mix_weight=0.5):
    """
    Robust Global CNV Contrast.
    - Base: per-cell q-quantile of |CNV| (default q=0.75) to avoid collapse to zero.
    - Optional mix with RMS to acknowledge amplitude.
    Returns the median across tumor cells.
    """
    tumor_cells = adata[adata.obs[cell_type_key] == tumor_cell_type]
    if tumor_cells.n_obs == 0:
        return 0.0
    X = _as_2d_array_from_obsm(tumor_cells, "X_cnv")

    qabs = _per_cell_qabs(X, q=q)
    if mix_with_rms:
        rms = _rms_per_cell(X)
        per_cell = mix_weight * qabs + (1.0 - mix_weight) * rms
    else:
        per_cell = qabs
    return float(np.median(per_cell))

def calculate_control_noise(adata, control_cell_type, cell_type_key):
    """
    Robust Control Noise.
    - Uses per-cell robust std via IQR/1.349 (fallback to RMS if IQR==0).
    - Median across control cells.
    """
    control_cells = adata[adata.obs[cell_type_key] == control_cell_type]
    if control_cells.n_obs == 0:
        return 0.0
    X = _as_2d_array_from_obsm(control_cells, "X_cnv")
    noise = _robust_std_per_cell_iqr(X)
    return float(np.median(noise))

def calculate_coherence(adata, tumor_cell_type, cell_type_key):
    """
    Coherence / Contiguity via lag-1 autocorrelation across genomic bins per cell.
    Vectorized and stable (returns 0 for flat signals).
    """
    tumor_cells = adata[adata.obs[cell_type_key] == tumor_cell_type]
    if tumor_cells.n_obs == 0:
        return 0.0
    X = _as_2d_array_from_obsm(tumor_cells, "X_cnv")
    ac = _lag1_autocorr_per_cell(X)
    return float(np.median(ac))

def calculate_false_positive_rate(adata, control_cell_type, cell_type_key, threshold=0.3, mode="absolute", k=3.0):
    """
    False-Positive Guard:
    - mode="absolute": fraction of control cell-bin entries with |CNV| > threshold.
    - mode="z": robust z-score per bin across controls (median/IQR); count |Z|>k.
    Returns rate in [0,1].
    """
    ctrl = adata[adata.obs[cell_type_key] == control_cell_type]
    if ctrl.n_obs == 0:
        return 0.0
    X = _as_2d_array_from_obsm(ctrl, "X_cnv")

    if mode == "z":
        med = np.median(X, axis=0)
        q1  = np.quantile(X, 0.25, axis=0)
        q3  = np.quantile(X, 0.75, axis=0)
        iqr = q3 - q1
        sigma = iqr / 1.349
        # Fallback to standard deviation where IQR==0
        sd_fallback = X.std(axis=0, ddof=1)
        sigma = np.where(np.isclose(sigma, 0.0), np.where(np.isclose(sd_fallback, 0.0), 1.0, sd_fallback), sigma)
        Z = (X - med) / sigma
        rate = np.mean(np.abs(Z) > k)
    else:
        rate = np.mean(np.abs(X) > threshold)

    return float(rate)

if __name__ == "__main__":
    import argparse
    import anndata as ad

    parser = argparse.ArgumentParser(description='Evaluate CNV analysis results.')
    parser.add_argument('--raw_adata_path', type=str, required=True, help='Path to the AnnData object with raw data and infercnv results.')
    parser.add_argument('--denoised_adata_path', type=str, required=True, help='Path to the AnnData object with denoised data and infercnv results.')
    parser.add_argument('--cell_type_key', type=str, required=True, help='The key in adata.obs that contains the cell type information.')
    parser.add_argument('--tumor_cell_type', type=str, required=True, help='The label for tumor cells.')
    parser.add_argument('--control_cell_type', type=str, required=True, help='The label for control cells.')
    args = parser.parse_args()

    # Load the AnnData objects
    adata_raw = ad.read_h5ad(args.raw_adata_path)
    adata_denoised = ad.read_h5ad(args.denoised_adata_path)

    # Calculate metrics
    raw_contrast = calculate_global_cnv_contrast(adata_raw, args.tumor_cell_type, args.cell_type_key)
    denoised_contrast = calculate_global_cnv_contrast(adata_denoised, args.tumor_cell_type, args.cell_type_key)

    raw_noise = calculate_control_noise(adata_raw, args.control_cell_type, args.cell_type_key)
    denoised_noise = calculate_control_noise(adata_denoised, args.control_cell_type, args.cell_type_key)

    raw_coherence = calculate_coherence(adata_raw, args.tumor_cell_type, args.cell_type_key)
    denoised_coherence = calculate_coherence(adata_denoised, args.tumor_cell_type, args.cell_type_key)

    raw_fp = calculate_false_positive_rate(adata_raw, args.control_cell_type, args.cell_type_key)
    denoised_fp = calculate_false_positive_rate(adata_denoised, args.control_cell_type, args.cell_type_key)

    # Print scorecard
    print("--- CNV Evaluation Scorecard ---")
    print(f"| Metric                  | Raw Data | Denoised Data | % Change |")
    print(f"|-------------------------|----------|---------------|----------|")
    print(f"| Global CNV Contrast (↑) | {raw_contrast:.4f}   | {denoised_contrast:.4f}       | {_safe_pct_change(denoised_contrast, raw_contrast):.2f}%    |")
    print(f"| Control Noise (↓)       | {raw_noise:.4f}   | {denoised_noise:.4f}        | {_safe_pct_change(denoised_noise, raw_noise):.2f}%     |")
    print(f"| Coherence / Contiguity (↑) | {raw_coherence:.4f}   | {denoised_coherence:.4f}        | {_safe_pct_change(denoised_coherence, raw_coherence):.2f}%     |")
    print(f"| False-Positive Guard (↓)| {raw_fp:.4f}       | {denoised_fp:.4f}             | {_safe_pct_change(denoised_fp, raw_fp):.2f}%     |")