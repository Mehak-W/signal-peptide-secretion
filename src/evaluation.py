"""
Evaluation utilities for signal peptide study.

Computes standard regression metrics: MSE, RMSE, MAE, R2, Spearman, Pearson.
"""
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_metrics(y_true, y_pred):
    """
    Compute regression metrics.

    Returns:
        dict with mse, rmse, mae, r2, spearman_rho, spearman_p, pearson_r, pearson_p
    """
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    sp_rho, sp_p = stats.spearmanr(y_true, y_pred)
    pe_r, pe_p = stats.pearsonr(y_true, y_pred)

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'spearman_rho': float(sp_rho),
        'spearman_p': float(sp_p),
        'pearson_r': float(pe_r),
        'pearson_p': float(pe_p),
        'n_samples': len(y_true),
    }


def format_metrics(metrics, prefix=''):
    """Format metrics dict as a readable string."""
    lines = []
    if prefix:
        lines.append(f"--- {prefix} ---")
    lines.append(f"  MSE:      {metrics['mse']:.4f}")
    lines.append(f"  RMSE:     {metrics['rmse']:.4f}")
    lines.append(f"  MAE:      {metrics['mae']:.4f}")
    lines.append(f"  R2:       {metrics['r2']:.4f}")
    lines.append(f"  Spearman: {metrics['spearman_rho']:.4f} (p={metrics['spearman_p']:.2e})")
    lines.append(f"  Pearson:  {metrics['pearson_r']:.4f} (p={metrics['pearson_p']:.2e})")
    lines.append(f"  N:        {metrics['n_samples']}")
    return '\n'.join(lines)
