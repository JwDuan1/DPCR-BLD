"""
Visualization utilities for E-SAFE.

This module provides visualization helpers for the E-SAFE application.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QSizePolicy

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


def cuboid_data(center, size):
    """
    Create the vertices of a cuboid for 3D plotting.
    
    Args:
        center: The center coordinates [x, y, z]
        size: The size in each dimension [l, w, h]
        
    Returns:
        tuple: (x, y, z) arrays of vertices
    """
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the (left, outside, bottom) point
    o = [a - b / 2 for a, b in zip(center, size)]
    # get the length, width, and height
    l, w, h = size
    x = np.array([
        [o[0], o[0] + l, o[0] + l, o[0],
         o[0]],  # x coordinate of points in bottom surface
        [o[0], o[0] + l, o[0] + l, o[0],
         o[0]],  # x coordinate of points in upper surface
        [o[0], o[0] + l, o[0] + l, o[0],
         o[0]],  # x coordinate of points in outside surface
        [o[0], o[0] + l, o[0] + l, o[0], o[0]]
    ])  # x coordinate of points in inside surface
    y = np.array([
        [o[1], o[1], o[1] + w, o[1] + w,
         o[1]],  # y coordinate of points in bottom surface
        [o[1], o[1], o[1] + w, o[1] + w,
         o[1]],  # y coordinate of points in upper surface
        [o[1], o[1], o[1], o[1],
         o[1]],  # y coordinate of points in outside surface
        [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]
    ])  # y coordinate of points in inside surface
    z = np.array([
        [o[2], o[2], o[2], o[2],
         o[2]],  # z coordinate of points in bottom surface
        [o[2] + h, o[2] + h, o[2] + h, o[2] + h,
         o[2] + h],  # z coordinate of points in upper surface
        [o[2], o[2], o[2] + h, o[2] + h,
         o[2]],  # z coordinate of points in outside surface
        [o[2], o[2], o[2] + h, o[2] + h, o[2]]
    ])  # z coordinate of points in inside surface
    return x, y, z


def setup_3d_axes(ax, x_range, y_range, z_range, arrow_scale=0.7):
    """
    Set up 3D coordinate system with arrows and labels.
    
    Args:
        ax: Matplotlib 3D axis to configure
        x_range: Range of x-axis
        y_range: Range of y-axis
        z_range: Range of z-axis
        arrow_scale: Scale factor for arrows (default: 0.7)
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


class MatplotlibCanvas(FigureCanvas):
    """Matplotlib canvas for embedding in Qt with improved performance."""
    def __init__(self, parent=None, width=5, height=4, dpi=100, projection='3d'):
        """
        Initialize a matplotlib canvas.
        
        Args:
            parent: Parent widget
            width: Figure width in inches
            height: Figure height in inches
            dpi: Dots per inch
            projection: Axes projection ('3d' or None)
        """
        # Create figure
        self.fig = plt.Figure(figsize=(width, height), dpi=dpi)
        
        # Create axes with the specified projection
        if projection == '3d':
            self.axes = self.fig.add_subplot(111, projection='3d')
        else:
            self.axes = self.fig.add_subplot(111)
        
        # Use tight layout for better space utilization
        self.fig.tight_layout()
        
        # Initialize canvas
        super(MatplotlibCanvas, self).__init__(self.fig)
        
        # Set parent if provided
        if parent:
            self.setParent(parent)
            
        # Configure for Qt
        from PyQt5.QtWidgets import QSizePolicy
        self.setMinimumSize(100, 100)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()
        
        # Configure for better performance
        self.fig.set_dpi(dpi)
        
        # Set renderer to Agg for better performance
        self.fig.set_canvas(self)

    def safe_draw(self):
        """Safer drawing method to prevent recursive painting."""
        try:
            # Process events before drawing to clear any pending paint events
            from PyQt5.QtWidgets import QApplication
            QApplication.processEvents()
            # Use draw_idle for non-blocking drawing
            self.draw_idle()
        except Exception as e:
            print(f"Error drawing canvas: {e}")
            # Fall back to regular draw if draw_idle fails
            self.draw()