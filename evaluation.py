"""calculate MCC."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata

def elementwise_r2(z, hz, use_rank=False):
    """
    Calculates the pairwise R^2 matrix between all elements of z and hz,
    and finds the optimal 1-to-1 matching (Permutation & Scaling).
    """
    z = z.detach().cpu().numpy()
    hz = hz.detach().cpu().numpy()

    if use_rank:
        z = np.apply_along_axis(rankdata, 0, z)
        hz = np.apply_along_axis(rankdata, 0, hz)
        
    k = z.shape[1]
    
    combined_data = np.hstack((z, hz))
    cor_matrix = np.corrcoef(combined_data, rowvar=False)
    
    cross_cor = cor_matrix[:k, k:]
    
    r2_matrix = cross_cor ** 2

    # Handle NaN/Inf caused by collapsed dimensions or non-finite values
    if not np.isfinite(r2_matrix).all():
        print("[R2 warning] r2 matrix contains NaN/Inf")
        print("[R2 warning] z std:", np.std(z, axis=0))
        print("[R2 warning] hz std:", np.std(hz, axis=0))
        r2_matrix = np.nan_to_num(
            r2_matrix,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )
    
    row_ind, col_ind = linear_sum_assignment(-1 * r2_matrix)
    
    optimal_r2_scores = r2_matrix[row_ind, col_ind]
    mean_matched_r2 = optimal_r2_scores.mean()
    
    return mean_matched_r2, r2_matrix, optimal_r2_scores

def MCC(z, hz, k, use_spearman=False, use_floc=False, p=0.5):
    z = z.detach().cpu().numpy()
    hz = hz.detach().cpu().numpy()

    if use_spearman:
        # spearman corr
        z_proc = np.apply_along_axis(rankdata, 0, z)
        hz_proc = np.apply_along_axis(rankdata, 0, hz)
    elif use_floc:
        # 1. Center the data using the Median (robust to heavy tails)
        z_centered = z - np.median(z, axis=0)
        hz_centered = hz - np.median(hz, axis=0)
        
        # 2. Apply the signed fractional power: sign(x) * |x|^p
        z_proc = np.sign(z_centered) * np.power(np.abs(z_centered), p)
        hz_proc = np.sign(hz_centered) * np.power(np.abs(hz_centered), p)
    else:
        z_proc = z
        hz_proc = hz

    cor_abs = np.abs(np.corrcoef(z_proc.T, hz_proc.T))[0:k, k:]

    # Handle NaN/Inf caused by collapsed dimensions or non-finite values
    if not np.isfinite(cor_abs).all():
        print("[MCC warning] correlation matrix contains NaN/Inf")
        print("[MCC warning] z std:", np.std(z_proc, axis=0))
        print("[MCC warning] hz std:", np.std(hz_proc, axis=0))
        print("[MCC warning] NaN count:", np.isnan(cor_abs).sum())
        print("[MCC warning] Inf count:", np.isinf(cor_abs).sum())

        cor_abs = np.nan_to_num(
            cor_abs,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

    assignments = linear_sum_assignment(-1 * cor_abs)

    maxcor = cor_abs[assignments].sum()
    return maxcor, cor_abs


def reorder(A, d):
    B = A * 1.0

    mind = linear_sum_assignment(-1 * A)[1]

    B = np.delete(B, mind, 1)

    Ao = []
    Ao = np.expand_dims(Ao, 0)
    Ao = np.repeat(Ao, d, axis=0)

    # Order the latent variables such that latent variables displaying the highest correlation with the same source feature are together
    for i in range(d):
        Ai = np.array(A[:, mind[i]], ndmin=2).T
        Ao = np.concatenate((Ao, Ai), axis=1)

    Ao = np.concatenate((Ao, B), axis=1)

    return Ao