"""
hausdorff_distance.py - Hausdorff Distance calculation for contour comparison

This module implements the Hausdorff Distance calculation, including percentile-based
variants for robust contour comparison in medical imaging applications.

Authors: 
    Original MATLAB: Jingwei Duan, Ph.D. (duan.jingwei01@gmail.com), Quan Chen, Ph.D.
    
Date: February 2025
Version: 1.0
License: MIT License
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Union
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def hausdorff_dist_pctile(P: np.ndarray, Q: np.ndarray, 
                         pct: Union[float, List[float], np.ndarray] = [90, 100],
                         lmf: Optional[int] = None, 
                         visualize: bool = False) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the Hausdorff Distance and percentile variants between two point sets.
    
    The Directional Hausdorff Distance (dhd) is defined as:
    dhd(P,Q) = max p ∈ P [ min q ∈ Q [ ||p-q|| ] ].
    
    The Hausdorff Distance is defined as max{dhd(P,Q), dhd(Q,P)}
    
    This function allows calculating percentile variants of the Hausdorff Distance,
    which are more robust to outliers than the standard (100th percentile) HD.
    
    Args:
        P: First set of points (N x D array where N is number of points, D is dimensionality)
        Q: Second set of points (M x D array)
        pct: Percentile(s) to calculate (default: [90, 100])
        lmf: Large matrix flag (0: force full matrix, 1: force loop method, None: auto)
        visualize: Whether to visualize the results (only works for D ≤ 3)
        
    Returns:
        Tuple of:
            - Array of Hausdorff distances at each percentile
            - Mean distance between the two point sets
            - Full distance matrix (or empty array if largeMat=True)
            - Minimum distances from P to Q
            - Minimum distances from Q to P
    """
    # Check inputs
    sP = P.shape
    sQ = Q.shape
    
    if sP[1] != sQ[1]:
        raise ValueError("Inputs P and Q must have the same number of columns (dimensions)")
    
    # Convert pct to numpy array for consistency
    if isinstance(pct, (int, float)):
        pct = np.array([pct])
    else:
        pct = np.array(pct)
    
    # Determine whether to use large matrix method
    use_large_mat = False
    if lmf is not None:
        use_large_mat = bool(lmf)
    elif sP[0] * sQ[0] > 2e6:
        # If the distance matrix would be too large, use loop method
        use_large_mat = True
    
    if use_large_mat:
        # Use loop method for large matrices to save memory
        # Initialize distances
        min_p = np.zeros(sP[0])
        min_q = np.zeros(sQ[0])
        
        # Calculate minimum distance from each point in P to Q
        for p in range(sP[0]):
            min_p[p] = np.min(np.sum((P[p, :] - Q) ** 2, axis=1))
        
        # Calculate minimum distance from each point in Q to P
        for q in range(sQ[0]):
            min_q[q] = np.min(np.sum((Q[q, :] - P) ** 2, axis=1))
        
        # Calculate percentile values
        vp = np.percentile(min_p, pct)
        vq = np.percentile(min_q, pct)
        
        # Calculate means
        mean_p = np.mean(np.sqrt(min_p))
        mean_q = np.mean(np.sqrt(min_q))
        
        # Calculate Hausdorff distances (average of both directions)
        hd = (np.sqrt(vp) + np.sqrt(vq)) / 2
        
        # For the full Hausdorff distance (100th percentile),
        # use the maximum of both directions
        if 100 in pct:
            idx = np.where(pct == 100)[0][0]
            hd[idx] = np.sqrt(max(vp[idx], vq[idx]))
        
        # Calculate mean distance
        mean_d = (mean_p + mean_q) / 2
        
        # We don't calculate the full distance matrix in this mode
        D = np.array([])
        
        # Return squared distances for consistency with full matrix method
        min_p = np.sqrt(min_p)
        min_q = np.sqrt(min_q)
    
    else:
        # Use full matrix method for smaller datasets
        # Calculate the full distance matrix
        D = distance_matrix(P, Q)
        
        # Get the minimum distance from each point in P to any point in Q
        min_p = np.min(D, axis=1)
        
        # Get the minimum distance from each point in Q to any point in P
        min_q = np.min(D, axis=0)
        
        # Calculate percentiles
        vp = np.percentile(min_p, pct)
        vq = np.percentile(min_q, pct)
        
        # Calculate means
        mean_p = np.mean(min_p)
        mean_q = np.mean(min_q)
        
        # Calculate Hausdorff distances (average of both directions)
        hd = (vp + vq) / 2
        
        # For the full Hausdorff distance (100th percentile),
        # use the maximum of both directions
        if 100 in pct:
            idx = np.where(pct == 100)[0][0]
            hd[idx] = max(vp[idx], vq[idx])
        
        # Calculate mean distance
        mean_d = (mean_p + mean_q) / 2
    
    # Visualize if requested and dimensionality allows
    if visualize and not use_large_mat and sP[1] <= 3:
        visualize_hausdorff(P, Q, D, min_p, min_q, hd, pct)
    
    return hd, mean_d, D, min_p, min_q


def visualize_hausdorff(P: np.ndarray, Q: np.ndarray, D: np.ndarray, 
                        min_p: np.ndarray, min_q: np.ndarray, 
                        hd: float, pct: np.ndarray) -> None:
    """
    Visualize the Hausdorff distance calculation.
    
    Args:
        P: First set of points
        Q: Second set of points
        D: Distance matrix
        min_p: Minimum distances from P to Q
        min_q: Minimum distances from Q to P
        hd: Hausdorff distance value(s)
        pct: Percentile(s) used
    """
    # Create figure
    fig = plt.figure(figsize=(12, 6))
    
    # Get indices of minimum distances
    idx_p = np.argmin(D, axis=1)
    idx_q = np.argmin(D, axis=0)
    
    # Get the point pair with the maximum minimum distance
    # This represents the Hausdorff distance
    if 100 in pct:
        idx_hd_p = np.argmax(min_p)
        idx_hd_q = np.argmax(min_q)
        
        if min_p[idx_hd_p] > min_q[idx_hd_q]:
            # P->Q direction is the Hausdorff distance
            hd_point1 = P[idx_hd_p]
            hd_point2 = Q[idx_p[idx_hd_p]]
        else:
            # Q->P direction is the Hausdorff distance
            hd_point1 = Q[idx_hd_q]
            hd_point2 = P[idx_q[idx_hd_q]]
    else:
        # If 100th percentile not requested, use the highest available
        max_pct = np.max(pct)
        idx_max_pct = np.where(pct == max_pct)[0][0]
        
        # Find the point corresponding to this percentile
        # This is an approximation as percentiles don't map directly to single points
        pct_val_p = np.percentile(min_p, max_pct)
        pct_val_q = np.percentile(min_q, max_pct)
        
        idx_hd_p = np.argmin(np.abs(min_p - pct_val_p))
        idx_hd_q = np.argmin(np.abs(min_q - pct_val_q))
        
        if min_p[idx_hd_p] > min_q[idx_hd_q]:
            hd_point1 = P[idx_hd_p]
            hd_point2 = Q[idx_p[idx_hd_p]]
        else:
            hd_point1 = Q[idx_hd_q]
            hd_point2 = P[idx_q[idx_hd_q]]
    
    # Determine the dimensionality of the data
    dim = P.shape[1]
    
    # ---- Plot the point clouds and connections ----
    ax1 = fig.add_subplot(121, projection='3d' if dim == 3 else None)
    
    if dim == 1:
        # 1D case: Plot points on a line with an offset
        ax1.plot(P[:, 0], np.zeros_like(P[:, 0]), 'bx', markersize=10, linewidth=3, label='P')
        ax1.plot(Q[:, 0], np.ones_like(Q[:, 0]) * 0.1, 'ro', markersize=8, linewidth=2.5, label='Q')
        
        # Draw minimum distance lines from P to Q
        for i in range(len(P)):
            ax1.plot([P[i, 0], Q[idx_p[i], 0]], [0, 0.1], 'b-', alpha=0.3)
        
        # Draw minimum distance lines from Q to P
        for i in range(len(Q)):
            ax1.plot([Q[i, 0], P[idx_q[i], 0]], [0.1, 0], 'r-', alpha=0.3)
        
        # Highlight Hausdorff distance
        ax1.plot([hd_point1[0], hd_point2[0]], 
                [0 if hd_point1[0] in P[:, 0] else 0.1, 
                0.1 if hd_point2[0] in Q[:, 0] else 0], 
                'k-', linewidth=2, marker='s', markersize=12, label=f'HD ({np.max(pct)}%)')
        
        ax1.set_yticks([])
        ax1.set_xlabel('Dimension 1')
        
    elif dim == 2:
        # 2D case
        ax1.plot(P[:, 0], P[:, 1], 'bx', markersize=10, linewidth=3, label='P')
        ax1.plot(Q[:, 0], Q[:, 1], 'ro', markersize=8, linewidth=2.5, label='Q')
        
        # Draw minimum distance lines from P to Q
        for i in range(len(P)):
            ax1.plot([P[i, 0], Q[idx_p[i], 0]], [P[i, 1], Q[idx_p[i], 1]], 'b-', alpha=0.3)
        
        # Draw minimum distance lines from Q to P
        for i in range(len(Q)):
            ax1.plot([Q[i, 0], P[idx_q[i], 0]], [Q[i, 1], P[idx_q[i], 1]], 'r-', alpha=0.3)
        
        # Highlight Hausdorff distance
        ax1.plot([hd_point1[0], hd_point2[0]], [hd_point1[1], hd_point2[1]], 
                'k-', linewidth=2, marker='s', markersize=12, label=f'HD ({np.max(pct)}%)')
        
        ax1.set_xlabel('Dimension 1')
        ax1.set_ylabel('Dimension 2')
        
    elif dim == 3:
        # 3D case
        ax1.plot(P[:, 0], P[:, 1], P[:, 2], 'bx', markersize=10, linewidth=3, label='P')
        ax1.plot(Q[:, 0], Q[:, 1], Q[:, 2], 'ro', markersize=8, linewidth=2.5, label='Q')
        
        # Draw minimum distance lines from P to Q (only a subset for clarity)
        for i in range(0, len(P), max(1, len(P)//20)):  # Plot ~20 lines
            ax1.plot([P[i, 0], Q[idx_p[i], 0]], 
                    [P[i, 1], Q[idx_p[i], 1]], 
                    [P[i, 2], Q[idx_p[i], 2]], 'b-', alpha=0.3)
        
        # Draw minimum distance lines from Q to P (only a subset for clarity)
        for i in range(0, len(Q), max(1, len(Q)//20)):  # Plot ~20 lines
            ax1.plot([Q[i, 0], P[idx_q[i], 0]], 
                    [Q[i, 1], P[idx_q[i], 1]], 
                    [Q[i, 2], P[idx_q[i], 2]], 'r-', alpha=0.3)
        
        # Highlight Hausdorff distance
        ax1.plot([hd_point1[0], hd_point2[0]], 
                [hd_point1[1], hd_point2[1]], 
                [hd_point1[2], hd_point2[2]], 
                'k-', linewidth=2, marker='s', markersize=12, label=f'HD ({np.max(pct)}%)')
        
        ax1.set_xlabel('Dimension 1')
        ax1.set_ylabel('Dimension 2')
        ax1.set_zlabel('Dimension 3')
    
    # Set title and legend
    ax1.set_title(f'Hausdorff Distance = {hd[-1]:.4f}')
    ax1.legend(loc='best')
    
    # ---- Plot the distance matrix ----
    ax2 = fig.add_subplot(122)
    
    # Create a square matrix for pcolor
    D_ext = np.zeros((D.shape[0] + 1, D.shape[1] + 1))
    D_ext[:-1, :-1] = D
    D_ext[-1, :] = D_ext[-2, :]
    D_ext[:, -1] = D_ext[:, -2]
    
    # Create meshgrid for plotting
    X, Y = np.meshgrid(np.arange(D.shape[1] + 1), np.arange(D.shape[0] + 1))
    
    # Plot the distance matrix
    pc = ax2.pcolor(X - 0.5, Y - 0.5, D_ext, cmap='viridis', alpha=0.8)
    fig.colorbar(pc, ax=ax2, label='Distance')
    
    # Highlight the Hausdorff distance point
    if min_p[idx_hd_p] > min_q[idx_hd_q]:
        # Highlight in the P->Q direction
        ax2.add_patch(plt.Rectangle((idx_p[idx_hd_p] - 0.5, idx_hd_p - 0.5), 1, 1, 
                                   fill=False, edgecolor='white', linewidth=2))
    else:
        # Highlight in the Q->P direction
        ax2.add_patch(plt.Rectangle((idx_hd_q - 0.5, idx_q[idx_hd_q] - 0.5), 1, 1, 
                                   fill=False, edgecolor='white', linewidth=2))
    
    ax2.set_xlabel('Ordered points in Q (o)')
    ax2.set_ylabel('Ordered points in P (x)')
    ax2.set_title('Distance (color) between points in P and Q\nHausdorff distance outlined in white')
    
    plt.tight_layout()
    plt.show()


def bidirectional_local_distance(ref: np.ndarray, test: np.ndarray) -> np.ndarray:
    """
    Calculate the Bidirectional Local Distance (BLD) between two structures.
    
    This finds one distance for every point in the reference structure.
    BLD captures how much a test contour differs from a reference contour.
    
    Args:
        ref: Reference structure (N x D array)
        test: Test structure (M x D array)
        
    Returns:
        N x 1 vector of distances
    """
    N = ref.shape[0]
    M = test.shape[0]
    
    # Initialize distances from test to reference
    d_TR = np.zeros(M)
    d_TRi = np.zeros(M, dtype=int)
    
    # Initialize distances from reference to test
    d = np.full(N, np.inf)
    
    # For each point in test, find closest point in reference
    for i in range(M):
        # Calculate distances from this test point to all reference points
        dists = np.sqrt(np.sum((ref - test[i]) ** 2, axis=1))
        
        # Find minimum distance and corresponding reference point
        d_TR[i] = np.min(dists)
        d_TRi[i] = np.argmin(dists)
        
        # Update minimum distances from reference to test
        d = np.minimum(d, dists)
    
    # For each reference point, find maximum distance where test points match
    for i in range(N):
        # Find test points that are closest to this reference point
        hits = d_TR[d_TRi == i]
        
        # If any test points are closest to this reference point,
        # update the distance to the maximum
        if len(hits) > 0:
            d[i] = max(d[i], np.max(hits))
    
    return d