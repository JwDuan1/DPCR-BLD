"""
Visualization tab for E-SAFE GUI.

This module implements the Visualization tab for the E-SAFE GUI.
"""

import os
import fnmatch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QLineEdit, QFileDialog, QGroupBox, QComboBox, QSpinBox,
    QDialog, QTabWidget, QSplitter, QTextEdit, QMessageBox, QDoubleSpinBox, QRadioButton
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from gui.widgets import FileSelectionWidget
from utils.visualization import MatplotlibCanvas, setup_3d_axes


class VisualizationTab(QWidget):
    """
    Tab for visualization of results.
    
    This tab allows the user to visualize and explore results from the registration
    and analysis.
    """
    
    def __init__(self, main_window):
        """
        Initialize the Visualization tab.
        
        Args:
            main_window: The main window instance
        """
        super().__init__(main_window)
        
        self.main_window = main_window
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Split view
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side - Controls
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Data selection
        data_group = QGroupBox("Data Selection")
        data_layout = QVBoxLayout(data_group)
        
        # Root directory
        self.viz_dir_widget = FileSelectionWidget(
            "Results Directory:", "Directory containing results",
            "directory", "", "Select Results Directory"
        )
        data_layout.addWidget(self.viz_dir_widget)
        
        # Visualization type
        viz_type_layout = QHBoxLayout()
        self.viz_type_combo = QComboBox()
        self.viz_type_combo.addItems([
            "Mean BLD",
            "Standard Deviation",
            "Statistical Outliers"
        ])
        self.viz_type_combo.currentIndexChanged.connect(self.viz_type_changed)
        
        viz_type_layout.addWidget(QLabel("Visualization Type:"))
        viz_type_layout.addWidget(self.viz_type_combo)
        data_layout.addLayout(viz_type_layout)
        
        # OAR selection
        oar_layout = QHBoxLayout()
        self.viz_oar_combo = QComboBox()
        self.viz_oar_combo.addItem("Select OAR")
        self.viz_refresh_btn = QPushButton("Refresh")
        self.viz_refresh_btn.clicked.connect(self.refresh_viz_data)
        
        oar_layout.addWidget(QLabel("OAR:"))
        oar_layout.addWidget(self.viz_oar_combo)
        oar_layout.addWidget(self.viz_refresh_btn)
        data_layout.addLayout(oar_layout)
        
        # File selection
        file_layout = QHBoxLayout()
        self.viz_file_combo = QComboBox()
        self.viz_file_combo.addItem("Select file")
        self.viz_file_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.viz_file_combo.setMinimumContentsLength(30)  # Show more text content
        
        file_layout.addWidget(QLabel("File:"))
        file_layout.addWidget(self.viz_file_combo, 1)  # Give file combo more space
        data_layout.addLayout(file_layout)
        
        visualization_btn = QPushButton("Generate Visualization")
        visualization_btn.clicked.connect(self.generate_visualization)
        data_layout.addWidget(visualization_btn)
        
        left_layout.addWidget(data_group)
        
        # Visualization options
        options_group = QGroupBox("Visualization Options")
        options_layout = QVBoxLayout(options_group)
        
        # Color map selection
        colormap_layout = QHBoxLayout()
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems([
            "coolwarm", "viridis", "plasma", "inferno", 
            "magma", "Purples", "Blues", "Reds"
        ])
        
        colormap_layout.addWidget(QLabel("Colormap:"))
        colormap_layout.addWidget(self.colormap_combo)
        options_layout.addLayout(colormap_layout)
        
        # Percentile range
        percentile_layout = QHBoxLayout()
        self.percentile_min_spin = QSpinBox()
        self.percentile_min_spin.setRange(0, 49)
        self.percentile_min_spin.setValue(2)
        
        self.percentile_max_spin = QSpinBox()
        self.percentile_max_spin.setRange(51, 100)
        self.percentile_max_spin.setValue(98)
        
        percentile_layout.addWidget(QLabel("Percentile Range:"))
        percentile_layout.addWidget(self.percentile_min_spin)
        percentile_layout.addWidget(QLabel("to"))
        percentile_layout.addWidget(self.percentile_max_spin)
        
        options_layout.addLayout(percentile_layout)
        
        # Deviation range in mm
        deviation_layout = QHBoxLayout()
        self.deviation_min_spin = QDoubleSpinBox()
        self.deviation_min_spin.setRange(-10.0, 0.0)
        self.deviation_min_spin.setSingleStep(0.5)
        self.deviation_min_spin.setValue(-5.0)
        self.deviation_min_spin.setDecimals(1)

        self.deviation_max_spin = QDoubleSpinBox()
        self.deviation_max_spin.setRange(0.0, 20.0)
        self.deviation_max_spin.setSingleStep(0.5)
        self.deviation_max_spin.setValue(5.0)
        self.deviation_max_spin.setDecimals(1)

        deviation_layout.addWidget(QLabel("Deviation Range (mm):"))
        deviation_layout.addWidget(self.deviation_min_spin)
        deviation_layout.addWidget(QLabel("to"))
        deviation_layout.addWidget(self.deviation_max_spin)
        options_layout.addLayout(deviation_layout)
        
        # Range mode selection
        range_mode_layout = QHBoxLayout()
        self.percentile_mode_radio = QRadioButton("Percentile Mode")
        self.percentile_mode_radio.setChecked(True)
        self.deviation_mode_radio = QRadioButton("Direct Range Mode")
        range_mode_layout.addWidget(self.percentile_mode_radio)
        range_mode_layout.addWidget(self.deviation_mode_radio)
        options_layout.addLayout(range_mode_layout)
        
        # Point size and alpha controls
        point_style_layout = QHBoxLayout()
        self.point_size_spin = QSpinBox()
        self.point_size_spin.setRange(1, 200)
        self.point_size_spin.setValue(50)
        self.point_size_spin.setSingleStep(5)
        
        self.point_alpha_spin = QDoubleSpinBox()
        self.point_alpha_spin.setRange(0.1, 1.0)
        self.point_alpha_spin.setValue(0.9)
        self.point_alpha_spin.setSingleStep(0.1)
        self.point_alpha_spin.setDecimals(1)
        
        point_style_layout.addWidget(QLabel("Point Size:"))
        point_style_layout.addWidget(self.point_size_spin)
        point_style_layout.addWidget(QLabel("Transparency:"))
        point_style_layout.addWidget(self.point_alpha_spin)
        options_layout.addLayout(point_style_layout)
        
        # View angles
        angle_layout = QHBoxLayout()
        self.elevation_spin = QSpinBox()
        self.elevation_spin.setRange(-180, 180)
        self.elevation_spin.setValue(-155)
        
        self.azimuth_spin = QSpinBox()
        self.azimuth_spin.setRange(-180, 180)
        self.azimuth_spin.setValue(151)
        
        angle_layout.addWidget(QLabel("Elevation:"))
        angle_layout.addWidget(self.elevation_spin)
        angle_layout.addWidget(QLabel("Azimuth:"))
        angle_layout.addWidget(self.azimuth_spin)
        options_layout.addLayout(angle_layout)
        
        # Export options
        export_layout = QHBoxLayout()
        self.export_dpi_spin = QSpinBox()
        self.export_dpi_spin.setRange(72, 600)
        self.export_dpi_spin.setValue(300)
        
        export_btn = QPushButton("Export Image")
        export_btn.clicked.connect(self.export_visualization)
        
        export_layout.addWidget(QLabel("Export DPI:"))
        export_layout.addWidget(self.export_dpi_spin)
        export_layout.addWidget(export_btn)
        options_layout.addLayout(export_layout)
        
        left_layout.addWidget(options_group)
        
        # Direct visualization
        direct_viz_group = QGroupBox("Direct File Visualization (Template vs Example Ref)")
        direct_viz_layout = QVBoxLayout(direct_viz_group)

        # Reference file selection
        self.ref_file_widget = FileSelectionWidget(
            "Example File:", "Select Example .mat file",
            "file_open", "MAT Files (*.mat)", "Select Example MAT File"
        )
        direct_viz_layout.addWidget(self.ref_file_widget)

        # Template file selection
        self.template_file_widget = FileSelectionWidget(
            "Template File:", "Select template .mat file",
            "file_open", "MAT Files (*.mat)", "Select Template MAT File"
        )
        direct_viz_layout.addWidget(self.template_file_widget)

        # Parameters layout
        params_layout = QHBoxLayout()

        
        # point size for plotting
        params_layout.addWidget(QLabel("Point Size:"))
        self.viz_individual_point_size_spin = QSpinBox()
        self.viz_individual_point_size_spin.setRange(1, 500)
        self.viz_individual_point_size_spin.setSingleStep(10)
        self.viz_individual_point_size_spin.setValue(10)
        params_layout.addWidget(self.viz_individual_point_size_spin)
        
        # Registration iterations
        params_layout.addWidget(QLabel("Registration Iterations:"))
        self.viz_reg_iterations_spin = QSpinBox()
        self.viz_reg_iterations_spin.setRange(5, 100)
        self.viz_reg_iterations_spin.setSingleStep(5)
        self.viz_reg_iterations_spin.setValue(20)
        params_layout.addWidget(self.viz_reg_iterations_spin)

        direct_viz_layout.addLayout(params_layout)

        # Count threshold for downsampling
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Count Threshold:"))
        self.viz_count_threshold_spin = QSpinBox()
        self.viz_count_threshold_spin.setRange(100, 20000)
        self.viz_count_threshold_spin.setSingleStep(100)
        self.viz_count_threshold_spin.setValue(5000)
        threshold_layout.addWidget(self.viz_count_threshold_spin)
        direct_viz_layout.addLayout(threshold_layout)

        # Launch button
        launch_viz_btn = QPushButton("Launch Direct Visualization")
        launch_viz_btn.clicked.connect(self.launch_direct_visualization)
        direct_viz_layout.addWidget(launch_viz_btn)

        left_layout.addWidget(direct_viz_group)

        # Right side - Visualization
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        viz_group = QGroupBox("Visualization")
        viz_layout = QVBoxLayout(viz_group)
        
        self.viz_canvas = MatplotlibCanvas(parent=self,width=6, height=5, dpi=100)
        viz_layout.addWidget(self.viz_canvas)
        
        self.viz_info_text = QTextEdit(self)
        self.viz_info_text.setReadOnly(True)
        self.viz_info_text.setMaximumHeight(150)
        viz_layout.addWidget(self.viz_info_text)
        
        right_layout.addWidget(viz_group)
        
        # Add widgets to splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([400, 800])
        
        # Add splitter to main layout
        layout.addWidget(splitter)
    
    def set_results_dir(self, results_dir):
        """
        Set the results directory from another tab.
        
        Args:
            results_dir: The results directory path
        """
        if results_dir:
            self.viz_dir_widget.set_path(results_dir)
            self.refresh_viz_data()
    
    def viz_type_changed(self):
        """Called when visualization type is changed."""
        self.refresh_viz_data()
    
    def refresh_viz_data(self):
        """Refresh visualization data sources."""
        root_dir = self.viz_dir_widget.get_path()
        viz_type = self.viz_type_combo.currentText()
        
        if not root_dir or not os.path.isdir(root_dir):
            QMessageBox.warning(self, "Warning", "Please select a valid results directory")
            return
            
        # Clear existing items
        self.viz_oar_combo.clear()
        self.viz_oar_combo.addItem("Select OAR")
        
        self.viz_file_combo.clear()
        self.viz_file_combo.addItem("Select file")
        
        # Find files based on visualization type
        if viz_type == "Mean BLD":
            pattern = "_*_Result.csv"
            exclude_pattern = "-std_Result.csv"
        elif viz_type == "Standard Deviation":
            pattern = "_*-std_Result.csv"
            exclude_pattern = None
        else:  # Statistical Outliers
            pattern = "_DetectError_*.csv"
            exclude_pattern = None
            
            # Check if DetectedError directory exists
            detect_dir = os.path.join(os.path.dirname(root_dir), "DetectedError")
            if os.path.isdir(detect_dir):
                root_dir = detect_dir
            else:
                self.main_window.log_message("DetectedError directory not found", "WARNING")
        
        # Find matching files
        matching_files = []
        for file in os.listdir(root_dir):
            if fnmatch.fnmatch(file, pattern):
                if exclude_pattern is None or not fnmatch.fnmatch(file, exclude_pattern):
                    matching_files.append(file)
        
        # Extract OAR names
        oars = set()
        for file in matching_files:
            parts = file.split('_')
            if len(parts) >= 2:
                if viz_type in ["Mean BLD", "Standard Deviation"]:
                    oar = parts[1].split('-')[0]  # Extract before any "-" character
                else:
                    oar = parts[2].split('-')[0]  # For outliers, OAR name is in different position
                oars.add(oar)
        
        # Add to combo box
        for oar in sorted(oars):
            self.viz_oar_combo.addItem(oar)
            
        # Connect signal for OAR selection
        self.viz_oar_combo.currentIndexChanged.connect(self.viz_oar_selected)
        
        self.main_window.log_message(f"Found {len(oars)} OARs with {viz_type} data")
    
    def viz_oar_selected(self):
        """Called when an OAR is selected for visualization."""
        oar = self.viz_oar_combo.currentText()
        
        if oar == "Select OAR":
            return
            
        root_dir = self.viz_dir_widget.get_path()
        viz_type = self.viz_type_combo.currentText()
        
        if not root_dir:
            return
            
        # Clear existing files
        self.viz_file_combo.clear()
        self.viz_file_combo.addItem("Select file")
        
        # Find matching files for this OAR
        if viz_type == "Mean BLD":
            pattern = f"_{oar}_Result.csv"
        elif viz_type == "Standard Deviation":
            pattern = f"_{oar}-std_Result.csv"
        else:  # Statistical Outliers
            pattern = f"_DetectError_{oar}_*.csv"
            
            # Check if DetectedError directory exists
            detect_dir = os.path.join(os.path.dirname(root_dir), "DetectedError")
            if os.path.isdir(detect_dir):
                root_dir = detect_dir
        
        # Find matching files
        matching_files = []
        for file in os.listdir(root_dir):
            if viz_type == "Statistical Outliers":
                if f"_DetectError_{oar}_" in file:
                    matching_files.append(file)
            else:
                if file == pattern:
                    matching_files.append(file)
        
        # Add to combo box
        for file in matching_files:
            self.viz_file_combo.addItem(file)
            
        # If only one matching file, select it
        if len(matching_files) == 1:
            self.viz_file_combo.setCurrentIndex(1)
        
        self.main_window.log_message(f"Found {len(matching_files)} {viz_type} files for {oar}")
    
    def generate_visualization(self):
        """Generate visualization from selected data."""
        root_dir = self.viz_dir_widget.get_path()
        viz_type = self.viz_type_combo.currentText()
        oar = self.viz_oar_combo.currentText()
        file = self.viz_file_combo.currentText()
        
        if not root_dir or oar == "Select OAR" or file == "Select file":
            QMessageBox.warning(self, "Warning", "Please select OAR and file")
            return
            
        # Determine file path
        if viz_type == "Statistical Outliers":
            detect_dir = os.path.join(os.path.dirname(root_dir), "DetectedError")
            if os.path.isdir(detect_dir):
                file_path = os.path.join(detect_dir, file)
            else:
                file_path = os.path.join(root_dir, file)
        else:
            file_path = os.path.join(root_dir, file)
            
        if not os.path.isfile(file_path):
            QMessageBox.warning(self, "Warning", f"File not found: {file_path}")
            return
            
        # Load the data
        try:
            # Read CSV file
            data = pd.read_csv(file_path, header=None)
            
            # Extract coordinates and values
            x = data.iloc[:, 0].values
            y = data.iloc[:, 1].values
            z = data.iloc[:, 2].values
            values = 10* data.iloc[:, 3].values
            
            # Clear previous plot
            self.viz_canvas.figure.clf()
            self.viz_canvas.axes = self.viz_canvas.figure.add_subplot(111, projection='3d')
            
            # Determine colormap
            colormap_name = self.colormap_combo.currentText()
            colormap = plt.get_cmap(colormap_name)
            
            # Get point size and alpha from spinboxes
            point_size = self.point_size_spin.value()
            point_alpha = self.point_alpha_spin.value()
            
            # Determine normalization
            if viz_type == "Standard Deviation":
                vmin = 0
                if self.percentile_mode_radio.isChecked():
                    p_max = self.percentile_max_spin.value()
                    vmax = np.percentile(values, p_max)
                else:
                    vmax = self.deviation_max_spin.value()
                label = "Standard Deviation (mm)"
            else:
                if self.percentile_mode_radio.isChecked():
                    p_min = self.percentile_min_spin.value()
                    p_max = self.percentile_max_spin.value()
                    
                    v_min = np.percentile(values, p_min)
                    v_max = np.percentile(values, p_max)
                else:
                    v_min = self.deviation_min_spin.value()
                    v_max = self.deviation_max_spin.value()
                
                # For diverging colormaps, make symmetric around zero
                if colormap_name in ['coolwarm', 'RdBu', 'RdBu_r', 'seismic']:
                    vmax = max(abs(v_min), abs(v_max))
                    vmin = -vmax
                else:
                    vmin = v_min
                    vmax = v_max
                    
                label = "Disagreements (mm)"
            
            # Create scatter plot with user-defined size and alpha
            scatter = self.viz_canvas.axes.scatter(
                x, y, z, 
                c=values, 
                cmap=colormap, 
                s=point_size,  
                alpha=point_alpha,
                vmin=vmin, 
                vmax=vmax
            )
            
            # Add colorbar
            cbar = self.viz_canvas.figure.colorbar(scatter, ax=self.viz_canvas.axes)
            cbar.set_label(label)
            
            # Calculate axis bounds
            x_min, x_max = np.min(x), np.max(x)
            y_min, y_max = np.min(y), np.max(y)
            z_min, z_max = np.min(z), np.max(z)
            
            # Set axis limits to show real dimensions
            self.viz_canvas.axes.set_xlim(x_min, x_max)
            self.viz_canvas.axes.set_ylim(y_min, y_max)
            self.viz_canvas.axes.set_zlim(z_min, z_max)
            
            # Calculate ranges
            x_range = x_max - x_min
            y_range = y_max - y_min
            z_range = z_max - z_min
            
            # Arrow
            setup_3d_axes(self.viz_canvas.axes, x_range, y_range, z_range)
            # Set aspect ratio to match data dimensions
            self.viz_canvas.axes.set_box_aspect((x_range, y_range, z_range))
            
            # Set axis labels
            self.viz_canvas.axes.set_xlabel('X (cm)')
            self.viz_canvas.axes.set_ylabel('Y (cm)')
            self.viz_canvas.axes.set_zlabel('Z (cm)')
            
            # Set title
            title = f"{oar} - {viz_type}"
            self.viz_canvas.axes.set_title(title)
            
            # Set view angle
            elevation = self.elevation_spin.value()
            azimuth = self.azimuth_spin.value()
            self.viz_canvas.axes.view_init(elevation, azimuth)
            
            # Set tight layout to prevent clipping
            self.viz_canvas.figure.tight_layout()
            
            # Update canvas
            self.viz_canvas.draw()
            
            # Update info text
            self.viz_info_text.clear()
            self.viz_info_text.append(f"Structure: {oar}")
            self.viz_info_text.append(f"Visualization Type: {viz_type}")
            self.viz_info_text.append(f"File: {file}")
            self.viz_info_text.append(f"Total points: {len(values)}")
            self.viz_info_text.append(f"Value range: {np.min(values):.2f} to {np.max(values):.2f} mm")
            self.viz_info_text.append(f"Mean value: {np.mean(values):.2f} mm")
            self.viz_info_text.append(f"Standard deviation: {np.std(values):.2f} mm")
            
            self.main_window.log_message(f"Visualization generated for {oar}")
            
        except Exception as e:
            self.main_window.log_message(f"Error generating visualization: {str(e)}", "ERROR")
        
    def launch_direct_visualization(self):
        """Launch direct visualization of reference and template files."""
        ref_file = self.ref_file_widget.get_path()
        template_file = self.template_file_widget.get_path()
        count_threshold = self.viz_count_threshold_spin.value()
        
        if not ref_file or not template_file:
            QMessageBox.warning(self, "Warning", "Please select reference and template files")
            return
        
        if not os.path.isfile(ref_file) or not os.path.isfile(template_file):
            QMessageBox.warning(self, "Warning", "One or both selected files do not exist")
            return
        
        self.main_window.log_message(f"Starting direct visualization of {os.path.basename(ref_file)} and {os.path.basename(template_file)}")
        
        # Import the visualization function
        try:
            from core.f_bld_visualization import f_bld_visualization
            
            # Show progress indicator
            self.main_window.progress_bar.setVisible(True)
            self.main_window.update_progress(10)
            # Get registration iterations
            reg_iterations = self.viz_reg_iterations_spin.value()
            individual_point_size =self.viz_individual_point_size_spin.value()
            # Run the visualization function
            figures = f_bld_visualization(ref_file, template_file, count_threshold,reg_iterations,individual_point_size)
            
            # Update progress
            self.main_window.update_progress(90)
            
            # Create a dialog to display the figures
            if figures:
                self.show_visualization_dialog(figures)
            else:
                QMessageBox.warning(self, "Warning", "No visualization data could be generated")
            
            # Hide progress indicator
            self.main_window.progress_bar.setVisible(False)
            
        except ImportError:
            self.main_window.log_message("Module f_bld_visualization not found. Please make sure it's in your PYTHONPATH.", "ERROR")
        except Exception as e:
            self.main_window.log_message(f"Error in visualization: {str(e)}", "ERROR")
            self.main_window.progress_bar.setVisible(False)
    
    def show_visualization_dialog(self, figures):
        """
        Display visualization figures in a dialog.
        
        Args:
            figures: Dictionary of matplotlib figures
        """
        dialog = QDialog(self)
        dialog.setWindowTitle("BLD Visualization")
        dialog.resize(1000, 800)
        
        layout = QVBoxLayout(dialog)
        
        # Create a tab widget to display multiple figures
        tab_widget = QTabWidget(self)
        layout.addWidget(tab_widget)
        
        # Add each figure to a tab
        for name, fig in figures.items():
            # Create a tab
            tab = QWidget()
            tab_layout = QVBoxLayout(tab)
            
            # Convert the matplotlib figure to a Qt canvas
            canvas = FigureCanvasQTAgg(fig)
            canvas.setParent(tab)
            tab_layout.addWidget(canvas)
            
            
            # Add to tab widget with a readable name
            readable_name = ' '.join(name.replace('_', ' ').title().split())
            tab_widget.addTab(tab, readable_name)
        
        # Add a close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        # Show the dialog
        dialog.exec_()
    
    def export_visualization(self):
        """Export visualization as an image file."""
        if not hasattr(self.viz_canvas, 'figure') or self.viz_canvas.figure is None:
            QMessageBox.warning(self, "Warning", "No visualization to export")
            return
            
        file, _ = QFileDialog.getSaveFileName(
            self, "Export Visualization", "", 
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)"
        )
        
        if not file:
            return
            
        dpi = self.export_dpi_spin.value()
        
        try:
            # Save figure safely
            self.viz_canvas.figure.savefig(file, dpi=dpi, bbox_inches='tight')
            self.main_window.log_message(f"Visualization exported to {file}")
        except Exception as e:
            self.main_window.log_message(f"Error exporting visualization: {str(e)}", "ERROR")