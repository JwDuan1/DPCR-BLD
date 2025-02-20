"""
3D Medical Data Visualization Tool

This module provides tools for visualizing 3D medical imaging data with color-mapped surfaces.
It supports visualization of contour data with customizable color maps and 3D coordinate systems.
The visualization includes color-mapped surface displays with the following specifications:

- Standard Deviation Display:
    - Purple colormap (cm.Purples)
    - Range: Symmetric range based on max absolute value of 2nd and 98th percentiles
    - One decimal precision on colorbar

- Disagreement Display:
    - Diverging colormap (cm.coolwarm)
    - Range: Symmetric range based on max absolute value of 2nd and 98th percentiles
    - One decimal precision on colorbar

This tool is part of the DR-BLD.

Authors:
    Jingwei Duan, Ph.D. @ duan.jingwei01@gmail.com
    Quan Chen, Ph.D.


Reference:
    Based on BLDMatchViaDCPR algorithm for systematic surface local disaggrements 
    and outliers analysis of medical structure boundaries.

    # Process all local disaggrements files in a directory
    >>> from medical_visualization import process_directory
    >>> # For standard deviation analysis
    >>> process_directory(
    ...     directory="path/to/data/BiasesResult",
    ...     pattern="_*.csv"
    ... )
    
    # Process detected outlier files in a directory
    >>> from medical_visualization import process_directory
    >>> # For standard deviation analysis
    >>> process_directory(
    ...     directory="path/to/data/DetectedError",
    ...     pattern="_DetectError_*.csv"
    ... )
    
Date: February 2025
Version: 1.0
License: MIT
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib import cm
import matplotlib.ticker as ticker
from matplotlib.patches import FancyArrowPatch
import os
import glob
from typing import Tuple, List, Optional


class Arrow3D(FancyArrowPatch):
    """A class for creating 3D arrows in matplotlib plots."""
    
    def __init__(self, xs, ys, zs, *args, **kwargs):
        """Initialize a 3D arrow.
        
        Args:
            xs: List of x coordinates [start_x, end_x]
            ys: List of y coordinates [start_y, end_y]
            zs: List of z coordinates [start_z, end_z]
            *args: Additional positional arguments for FancyArrowPatch
            **kwargs: Additional keyword arguments for FancyArrowPatch
        """
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        """Project the arrow into 3D space."""
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def cart2sph(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert Cartesian coordinates to spherical coordinates.
    
    Args:
        x: X coordinates
        y: Y coordinates
        z: Z coordinates
        
    Returns:
        Tuple of (azimuth, elevation, radius)
    """
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r


def extract_organ_name(filepath: str) -> str:
    """Extract organ name from the filepath.
    
    Args:
        filepath: Path to the data file
        
    Returns:
        Extracted organ name
    """
    parts = filepath.split('_')
    if len(parts) < 4:
        return parts[-2]
    elif len(parts) == 4:
        return parts[-2]
    elif len(parts) == 5:
        return f"{parts[-3]}_{parts[-2]}"
    elif len(parts) > 5:
        return f"{parts[-4]}_{parts[-3]}_{parts[-2]}"
    return 'Test'


def setup_3d_axes(ax: Axes3D, x_range: float, y_range: float, z_range: float,
                  arrow_scale: float = 0.7) -> None:
    """Set up 3D coordinate system with arrows and labels.
    
    Args:
        ax: Matplotlib 3D axes object
        x_range: Range of x-axis
        y_range: Range of y-axis
        z_range: Range of z-axis
        arrow_scale: Scale factor for arrow size (default: 0.7)
    """
    arrow_props = dict(mutation_scale=20, arrowstyle='->', shrinkA=0, shrinkB=0)
    
    # Calculate arrow ranges
    x_arrow_range = arrow_scale * x_range
    y_arrow_range = arrow_scale * y_range
    z_arrow_range = 0.5 * z_range  # Different scale for z-axis
    
    # Add coordinate labels
    ax.text(-x_arrow_range, 0, 0, r'$R$', fontsize=20)
    ax.text(0, -y_arrow_range, 0, r'$A$', fontsize=20)
    ax.text(0, 0, -z_arrow_range, r'$S$', fontsize=20)
    
    # Add coordinate arrows
    arrows = [
        Arrow3D([0, -x_arrow_range], [0, 0], [0, 0], color='c', **arrow_props),
        Arrow3D([0, 0], [0, -y_arrow_range], [0, 0], color='y', **arrow_props),
        Arrow3D([0, 0], [0, 0], [0, -z_arrow_range], color='g', **arrow_props)
    ]
    
    for arrow in arrows:
        ax.add_artist(arrow)


def visualize_contour_data(filepath: str) -> None:
    """Visualize 3D contour data with color mapping.
    
    Args:
        filepath: Path to the CSV file containing contour data
    """
    # Read and process data
    data = pd.read_csv(filepath)
    organ_name = extract_organ_name(filepath)
    
    # Extract coordinates and offset
    x = np.array(data.iloc[:, 0])
    y = np.array(data.iloc[:, 1])
    z = np.array(data.iloc[:, 2])
    offset = np.array(data.iloc[:, 3]) * 10  # Convert to mm
    
    # Set up visualization parameters
    is_std_data = "std" in organ_name
    norm_98 = np.percentile(offset, 98)
    norm_2 = np.percentile(offset, 2)
    max_abs_value = max(abs(norm_2), abs(norm_98))
    
    # Configure colormap based on data type
    if is_std_data:
        colormap = cm.Purples
        # Calculate range based on 98th percentile for std data
        norm_max = np.percentile(np.abs(offset), 98)
        norm_min = 0
        label = 'Standard deviation (mm)'
    else:
        colormap = cm.coolwarm
        # Calculate symmetric range based on 98th percentile of absolute values
        max_abs_98 = np.percentile(np.abs(offset), 98)
        norm_min = -max_abs_98
        norm_max = max_abs_98
        label = 'Disagreements (mm)'
    
    # Create figure and 3D axes
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set up normalization and color mapping
    norm = plt.Normalize(norm_min, norm_max)
    scamap = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
    fcolors = scamap.to_rgba(offset)
    
    # Create 3D scatter plot
    ax.scatter(x, y, z, c=fcolors, cmap=colormap, s=70, linewidth=5,
              alpha=0.9, antialiased=False)
              
    # Increase tick label sizes
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='z', labelsize=14)
    
    # Set axis labels and title
    ax.set_xlabel('X(cm)', fontsize=16)
    ax.set_ylabel('Y(cm)', fontsize=16)
    ax.set_zlabel('Z(cm)', fontsize=16)
    ax.set_title(organ_name, fontsize=18)
    
    # Configure axis ranges and aspect ratio
    x_range = np.max(x) - np.min(x)
    y_range = np.max(y) - np.min(y)
    z_range = np.max(z) - np.min(z)
    ax.set_box_aspect((x_range, y_range, z_range))
    
    # Set up 3D coordinate system
    setup_3d_axes(ax, x_range, y_range, z_range)
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.85, 0.325, 0.025, 0.35])
    cbar = fig.colorbar(scamap, cax=cbar_ax, format=ticker.FormatStrFormatter('%.1f'))
    cbar.set_label(label, rotation=270, fontsize=14, labelpad=20)
    cbar.ax.tick_params(labelsize=16)
    
    # Set view angle
    ax.view_init(-155, 151)
    
    plt.show()


def process_directory(directory: str, pattern: str = "_*") -> None:
    """Process all matching files in a directory.
    
    Args:
        directory: Directory path to search
        pattern: Glob pattern for matching files (default: "_*")
    """
    file_pattern = os.path.join(directory, pattern)
    for filepath in glob.glob(file_pattern):
        try:
            visualize_contour_data(filepath)
        except Exception as e:
            print(f"Error processing {filepath}: {str(e)}")


if __name__ == "__main__":
    # Example usage
    DATA_DIR = r"\\hnas1-dpts\Radiation Oncology\Physicists\Residents\Duan\Researches\Data\Analysis\All_SingleContourAsTemplate\CNS\DetectedError"
    PATTERN = "_*"
    
    process_directory(DATA_DIR, PATTERN)