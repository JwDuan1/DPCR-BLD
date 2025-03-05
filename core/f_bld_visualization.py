"""
f_bld_visualization.py - Visualization of point clouds for medical imaging analysis

This module implements visualization and comparison of reference and template contours
based on point cloud alignment, registration, and disagreement analysis.

Authors: 
    Original MATLAB: Jingwei Duan, Ph.D. (duan.jingwei01@gmail.com), Quan Chen, Ph.D.

    
Date: February 2025
Version: 1.0
License: MIT License
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors
import logging
from scipy.io import loadmat
import traceback
from sklearn.metrics.pairwise import euclidean_distances
from utils.visualization import setup_3d_axes

def align_com(points):
    """
    Align points to their center of mass.
    
    Args:
        points (np.ndarray): Array of points to align
        
    Returns:
        np.ndarray: Aligned points
    """
    # Calculate center of mass
    com = np.mean(points, axis=0)
    
    # Subtract COM from points
    return points - com


def f_bld_visualization(reference_file, template_file, count_threshold=5000, max_iterations=20, point_size=10):
    """
    Visualizes and compares point clouds for medical imaging analysis.
    
    This function performs point cloud visualization and comparison between reference
    and template contours. It includes functionalities for point cloud alignment, 
    downsampling, registration, and visualization of disagreements between contours.
    
    Args:
        reference_file: Path to .mat file containing reference data
        template_file: Path to .mat file containing template data
        count_threshold: Maximum point count before downsampling (default: 5000)
        max_iterations: Maximum iterations for cpdr (default: 20)
    Returns:
        A dictionary of matplotlib figures for visualization
    """
    # Import optional dependencies with fallbacks
    try:
        import h5py
        has_h5py = True
    except ImportError:
        has_h5py = False
        logging.warning("h5py not available. Falling back to scipy.io.loadmat")
    
    try:
        from pycpd import DeformableRegistration
        has_pycpd = True
    except ImportError:
        has_pycpd = False
        logging.warning("pycpd not available. Registration will be limited.")
    
    # Load reference and template data
    try:
        if has_h5py:
            try:
                # Try h5py first for newer MAT files
                ref_data = {}
                with h5py.File(reference_file, 'r') as f:
                    for key in f.keys():
                        ref_data[key] = np.array(f[key])
                
                template_data = {}
                with h5py.File(template_file, 'r') as f:
                    for key in f.keys():
                        template_data[key] = np.array(f[key])
            except Exception as e:
                logging.warning(f"Could not read MAT file with h5py: {e}")
                ref_data = loadmat(reference_file)
                template_data = loadmat(template_file)
        else:
            ref_data = loadmat(reference_file)
            template_data = loadmat(template_file)
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        logging.error(traceback.format_exc())
        return None
    
    # Extract point clouds - handle both h5py and scipy.io formats
    try:
        ref_pts = ref_data['refpts']
        ref_pts_with_bld = ref_data['refptswithbld']
        tst_pts = ref_data['tstpts']
        template_pts = template_data['refpts']
        
    except KeyError as e:
        logging.error(f"Could not find expected data fields in MAT files: {e}")
        return None
    
    # Calculate center of mass for alignment
    ref_pts_com = np.mean(ref_pts, axis=0)
    
    # Align reference points to center of mass
    ref_pts = align_com(ref_pts)
    
    # Align test points to same coordinate system
    tst_pts = tst_pts - ref_pts_com
    
    # Align reference points with BLD
    ref_pts_with_bld[:, 0:3] = align_com(ref_pts_with_bld[:, 0:3])
    
    # Process Template points
    template_pts = align_com(template_pts)
    
    # Check for downsampling
    logging.info(f"Original point counts - Template: {len(template_pts)}, Reference: {len(ref_pts)}")
    
    if len(template_pts) > count_threshold:
        logging.info(f"The OAR points number is {len(template_pts)}, larger than threshold {count_threshold}")
        logging.info("-----------Downsample-------------")
        
        # Start with initial grid size
        grid_size = 0.05
        
        # Function for simple voxel grid downsampling
        def voxel_downsample(points, voxel_size):
            # Create voxel grid
            voxel_indices = np.floor(points / voxel_size).astype(int)
            
            # Find unique voxels
            _, indices = np.unique(voxel_indices, axis=0, return_index=True)
            
            # Return downsampled points
            return points[indices]
        
        # Initialize downsampled point sets
        ref_pts_downsampled = ref_pts
        ref_pts_with_bld_downsampled = ref_pts_with_bld
        tst_pts_downsampled = tst_pts
        template_pts_downsampled = template_pts
        
        # Iteratively downsample until below threshold
        while len(template_pts_downsampled) > count_threshold:
            # Downsample all point clouds
            template_pts_downsampled = voxel_downsample(template_pts, grid_size)
            ref_pts_downsampled = voxel_downsample(ref_pts, grid_size)
            tst_pts_downsampled = voxel_downsample(tst_pts, grid_size)
            
            # For ref_pts_with_bld, we need to keep the BLD values
            temp_points = voxel_downsample(ref_pts_with_bld[:, 0:3], grid_size)
            
            # Find indices in original points (map downsampled points to original)
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(ref_pts_with_bld[:, 0:3])
            indices = nbrs.kneighbors(temp_points, return_distance=False).flatten()
            
            # Keep BLD values from original points
            ref_pts_with_bld_downsampled = ref_pts_with_bld[indices]
            
            # Check if done or increase grid size
            if len(template_pts_downsampled) <= count_threshold:
                logging.info(f"Downsampled OAR points: {len(template_pts_downsampled)}")
                break
            
            grid_size += 0.001
            # logging.info(f"Grid size increased to {grid_size:.4f}")
    else:
        logging.info(f"No downsampling needed, point count under threshold")
        ref_pts_downsampled = ref_pts
        ref_pts_with_bld_downsampled = ref_pts_with_bld
        tst_pts_downsampled = tst_pts
        template_pts_downsampled = template_pts
    
    # Perform CPD registration if available
    if template_pts_downsampled.shape == ref_pts_with_bld_downsampled[:, 0:3].shape and np.allclose(template_pts_downsampled, ref_pts_with_bld_downsampled[:, 0:3], atol=1e-5):
        # Points are identical, no need for registration
        logging.info("Template and reference points are identical. Skipping registration.")
        transformed_moving = ref_pts_with_bld_downsampled[:, 0:3].copy()
        
        # Transfer intensity values directly (no transformation needed)
        template_intensities = ref_pts_with_bld_downsampled[:, 3].copy()
        registration_successful = True
        rmse = 0.0
    elif has_pycpd:
        # Points are different, perform registration
        logging.info("Performing non-rigid CPD registration")
        
        try:
            # Setup registration parameters
            reg = DeformableRegistration(
                X=template_pts_downsampled,
                Y=ref_pts_with_bld_downsampled[:, 0:3],
                alpha=2,  # Regularization weight
                beta=2,   # Width of Gaussian kernel
                max_iterations=max_iterations,     
                tolerance=1e-5  
            )
            
            # Perform registration
            transformed_moving, registration_params = reg.register()
            
            # Calculate RMSE (using proper distance calculation between point sets)
            # Build a nearest neighbor model on the template points
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(template_pts_downsampled)
            # Find the closest template point for each transformed point
            distances, _ = nbrs.kneighbors(transformed_moving)
            # Calculate RMSE from these distances
            rmse = np.sqrt(np.mean(distances**2))
            logging.info(f"Registration RMSE: {rmse:.6f}")
            registration_successful = True
        except Exception as e:
            logging.error(f"Registration failed: {str(e)}")
            transformed_moving = ref_pts_with_bld_downsampled[:, 0:3].copy()
            rmse = -1
            registration_successful = False
    else:
        logging.warning("pycpd library not available, skipping registration")
        transformed_moving = ref_pts_with_bld_downsampled[:, 0:3].copy()
        rmse = -1
        registration_successful = False
    
    # Transfer intensity values from registered points to Template points
    template_intensities = np.zeros(len(template_pts_downsampled))
    
    # Find nearest neighbors to transfer values
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(transformed_moving)
    for j in range(len(template_pts_downsampled)):
        distances, indices = nbrs.kneighbors([template_pts_downsampled[j]])
        template_intensities[j] = ref_pts_with_bld_downsampled[indices[0][0], 3]
    
    # Create visualization figures
    figures = {}
    
    # Helper function for consistent figure creation
    def create_figure_with_points(title, point_sets, colors, labels, intensity=None, cmap='coolwarm'):
        """Helper to create consistent figures."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Calculate overall bounds to set consistent limits across all plots
        all_points = np.vstack([pts for pts in point_sets])
        x_min, y_min, z_min = np.min(all_points, axis=0)
        x_max, y_max, z_max = np.max(all_points, axis=0)
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        
        # Plot each point set
        for i, (pts, c, lbl) in enumerate(zip(point_sets, colors, labels)):
            if intensity is not None and i == 0:  # First set with intensity values
                # Calculate color limits for symmetric scaling
                global_max_abs = max(abs(np.max(intensity)), abs(np.min(intensity)))
                norm = plt.Normalize(vmin=-global_max_abs, vmax=global_max_abs)
                sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=intensity, cmap=cmap, alpha=0.9,norm=norm, s=point_size)
                
                # Add colorbar
                cbar = fig.colorbar(sc, ax=ax, label='Disagreements (mm)')
            else:
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=c, s=point_size, alpha=0.2, label=lbl)
        
        # Set axis limits to show real dimensions
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

        # Display actual dimensions in figure text
        # dimension_text = f"X Range: {x_min:.2f} to {x_max:.2f} cm\n" + \
        #                  f"Y Range: {y_min:.2f} to {y_max:.2f} cm\n" + \
        #                  f"Z Range: {z_min:.2f} to {z_max:.2f} cm"
        # ax.text2D(0.05, 0.05, dimension_text, transform=ax.transAxes)

        ax.set_box_aspect((x_range, y_range, z_range))        
        setup_3d_axes(ax.axes, x_range, y_range, z_range)
        # Add labels and legend if we're not using intensity coloring
        if intensity is None:
            ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=len(point_sets), fontsize=14)
            
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        ax.set_zlabel('Z (cm)')
        ax.view_init(elev=-155, azim=151)  # Similar to MATLAB view
        fig.tight_layout()
        
        return fig
    
    # 1. Reference vs Test Point Clouds
    figures['ref_vs_test'] = create_figure_with_points(
        'Reference vs Test Point Cloud Comparison',
        [ref_pts_downsampled, tst_pts_downsampled],
        ['blue', 'red'],
        ['Reference point cloud', 'Test point cloud']
    )
    
    # 2. Reference vs Template Point Clouds
    figures['ref_vs_template'] = create_figure_with_points(
        'Reference vs Template Point Cloud Comparison',
        [ref_pts_downsampled, template_pts_downsampled],
        ['blue', 'green'],
        ['Reference point cloud - example case', 'Template point cloud']
    )
    
    # 3. Registered Point Clouds
    figures['registration_result'] = create_figure_with_points(
        'Registration Result: Reference vs Template',
        [transformed_moving, template_pts_downsampled],
        ['blue', 'green'],
        ['Reference point cloud - example case', 'Template point cloud']
    )
    
    # 4. Visualize disagreements on reference contour
    figures['ref_disagreements'] = create_figure_with_points(
        'Disagreements on Reference Contour',
        [ref_pts_with_bld_downsampled[:, 0:3]],
        ['blue'],
        ['Reference points'],
        intensity=ref_pts_with_bld_downsampled[:, 3] * 10  # Scale BLD values as in MATLAB
    )
    
    # 5. Visualize disagreements on Template contour
    figures['template_disagreements'] = create_figure_with_points(
        'Disagreements on Template Contour',
        [template_pts_downsampled],
        ['green'],
        ['Template points'],
        intensity=template_intensities * 10 
    )
    
    return figures


if __name__ == "__main__":

    
    reference_file = r"...\output\Brainstem_case1.mat"
    template_file = r"\...\output\_Ref\Brainstem-Ref.mat"
    count_threshold =  5000
    
    figures = f_bld_visualization(reference_file, template_file, count_threshold)
    plt.show()