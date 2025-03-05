"""
Registration Analysis tab for E-SAFE GUI.

This module implements the Registration Analysis tab for the E-SAFE GUI.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QLineEdit, QFileDialog, QGroupBox, QComboBox, QSpinBox,
    QDoubleSpinBox, QMessageBox
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont
from gui.widgets import FileSelectionWidget, LogDisplay
from utils.visualization import MatplotlibCanvas
from utils.worker_thread import WorkerThread,SafeWorkerThread


class RegistrationAnalysisTab(QWidget):
    """
    Tab for deformable registration and analysis.
    
    This tab allows the user to perform deformable registration and analysis
    on BLD data.
    """
    
    def __init__(self, main_window):
        """
        Initialize the Registration Analysis tab.
        
        Args:
            main_window: The main window instance
        """
        super().__init__(main_window)
        
        self.main_window = main_window
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Create form layout
        form_layout = QVBoxLayout()
        
        # Root directory selection
        root_dir_group = QGroupBox("Root Directory")
        root_dir_layout = QHBoxLayout(root_dir_group)
        
        self.root_dir_widget = FileSelectionWidget(
            "", "Base directory containing BLD data",
            "directory", "", "Select Root Directory"
        )
        root_dir_layout.addWidget(self.root_dir_widget)
        form_layout.addWidget(root_dir_group)
        
        # Parameters
        params_group = QGroupBox("Registration Parameters")
        params_layout = QHBoxLayout(params_group)
        
        # OAR selection
        oar_layout = QVBoxLayout()
        self.oar_combo = QComboBox()
        self.oar_combo.addItem("Process all OARs")
        self.refresh_oar_btn = QPushButton("Refresh OARs")
        self.refresh_oar_btn.clicked.connect(self.refresh_oars)
        
        oar_layout.addWidget(QLabel("Select OAR:"))
        oar_layout.addWidget(self.oar_combo)
        oar_layout.addWidget(self.refresh_oar_btn)
        
        # Grid average
        grid_layout = QVBoxLayout()
        self.grid_average_spin = QDoubleSpinBox()
        self.grid_average_spin.setRange(0.01, 1.0)
        self.grid_average_spin.setSingleStep(0.01)
        self.grid_average_spin.setValue(0.05)
        
        grid_layout.addWidget(QLabel("Grid Average:"))
        grid_layout.addWidget(self.grid_average_spin)
        
        # Count threshold
        count_layout = QVBoxLayout()
        self.count_threshold_spin = QSpinBox()
        self.count_threshold_spin.setRange(100, 20000)
        self.count_threshold_spin.setSingleStep(100)
        self.count_threshold_spin.setValue(5000)
        
        count_layout.addWidget(QLabel("Count Threshold:"))
        count_layout.addWidget(self.count_threshold_spin)
        
        # Max iterations
        max_iter_layout = QVBoxLayout()
        self.max_iterations_spin = QSpinBox()
        self.max_iterations_spin.setRange(5, 100)
        self.max_iterations_spin.setSingleStep(5)
        self.max_iterations_spin.setValue(30)

        max_iter_layout.addWidget(QLabel("Max Iterations:"))
        max_iter_layout.addWidget(self.max_iterations_spin)
        
        # Add all parameter layouts
        params_layout.addLayout(oar_layout)
        params_layout.addLayout(grid_layout)
        params_layout.addLayout(count_layout)
        params_layout.addLayout(max_iter_layout)
        form_layout.addWidget(params_group)
        
        # Process button
        self.process_btn = QPushButton("Run Deformable Registration and Analysis")  # Change from process_btn
        self.process_btn.setMinimumHeight(40)
        self.process_btn.clicked.connect(self.run_registration_analysis)
        
        form_layout.addWidget(self.process_btn)
        

        
        # Results Viewer
        results_group = QGroupBox("Registration Results")
        results_layout = QVBoxLayout(results_group)
        
        # Create disabled label (hidden by default)
        self.disabled_label = QLabel("VISULIZATION DISABLED DURING REGISTRATION")
        self.disabled_label.setAlignment(Qt.AlignCenter)
        self.disabled_label.setStyleSheet("background-color: #FFEEEE; color: #CC0000; font-weight: bold; padding: 10px;")
        self.disabled_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.disabled_label.setFixedHeight(50)
        self.disabled_label.hide()
        results_layout.addWidget(self.disabled_label)
        
        # Results selector
        selector_layout = QHBoxLayout()
        self.results_oar_combo = QComboBox()
        self.results_type_combo = QComboBox()
        self.results_type_combo.addItems(["Mean", "Standard Deviation", "Outliers"])
        view_results_btn = QPushButton("View Results")
        view_results_btn.clicked.connect(self.view_registration_results)
        
        selector_layout.addWidget(QLabel("OAR:"))
        selector_layout.addWidget(self.results_oar_combo)
        selector_layout.addWidget(QLabel("Type:"))
        selector_layout.addWidget(self.results_type_combo)
        selector_layout.addWidget(view_results_btn)
        results_layout.addLayout(selector_layout)
        
        # Results display
        self.results_canvas = MatplotlibCanvas(parent=self,width=6, height=4, dpi=100)
        results_layout.addWidget(self.results_canvas)
        
        # Log display
        self.log_display = LogDisplay("Log")
        
        # Add layouts to main layout
        layout.addLayout(form_layout, 1)
        layout.addWidget(results_group, 2)
        layout.addWidget(self.log_display, 1)
    
    def set_root_dir(self, root_dir):
        """
        Set the root directory from another tab.
        
        Args:
            root_dir: The root directory path
        """
        if root_dir:
            self.root_dir_widget.set_path(root_dir)
            self.refresh_oars()
    
    def refresh_oars(self):
        """Refresh the OAR list from root directory."""
        root_dir = self.root_dir_widget.get_path()
        
        if not root_dir or not os.path.isdir(root_dir):
            QMessageBox.warning(self, "Warning", "Please select a valid root directory")
            return
            
        # Check for _Ref directory
        ref_dir = os.path.join(root_dir, '_Ref')
        if not os.path.isdir(ref_dir):
            QMessageBox.warning(self, "Warning", "_Ref directory not found in root directory")
            return
            
        # Clear existing items
        self.oar_combo.clear()
        self.oar_combo.addItem("Process all OARs")
        
        # Find all template files
        for file in os.listdir(ref_dir):
            if file.endswith('-Ref.mat'):
                parts = file.split('_')
                if len(parts) >= 1:
                    structure = parts[0]
                    self.oar_combo.addItem(structure)
        
        self.main_window.log_message(f"Found {self.oar_combo.count() - 1} OARs with templates")
        
        # Also update the results OAR combo
        self.results_oar_combo.clear()
        for i in range(1, self.oar_combo.count()):
            self.results_oar_combo.addItem(self.oar_combo.itemText(i))
    
    def run_registration_analysis(self):
        """Run the deformable registration and analysis process."""
        root_dir = self.root_dir_widget.get_path()
        
        if not root_dir:
            QMessageBox.warning(self, "Warning", "Please select a root directory")
            return
                
        # Get parameters
        oar_name = None if self.oar_combo.currentIndex() == 0 else self.oar_combo.currentText()
        grid_average = self.grid_average_spin.value()
        count_threshold = self.count_threshold_spin.value()
        max_iterations = self.max_iterations_spin.value()
        
        self.main_window.log_message(f"Starting deformable registration and analysis for {oar_name if oar_name else 'all OARs'}")
        
        try:
            from core.bld_match_via_dcpr import bld_match_via_dcpr
            from utils.worker_thread import launch_safe_worker
            
            # List widgets to disable
            widgets_to_disable = [
                self.root_dir_widget, self.oar_combo, self.refresh_oar_btn,
                self.grid_average_spin, self.count_threshold_spin,
                self.max_iterations_spin, self.results_oar_combo,
                self.results_type_combo
            ]
            
            # Show progress bar
            self.main_window.progress_bar.setVisible(True)
            
            # Launch worker
            self.main_window.current_worker = launch_safe_worker(
                parent=self,
                function=bld_match_via_dcpr,
                args=[root_dir, oar_name, grid_average, count_threshold, max_iterations],
                callback_func=self.registration_analysis_finished,
                progress_callback=self.main_window.update_progress,
                status_callback=self.main_window.update_log_display,
                disable_widgets=widgets_to_disable,
                disable_canvas=self.results_canvas,
                heartbeat_ms=100  # Use slower heartbeat for visualization
            )
            # Change process button to stop button
            self.process_btn.setText("Stop Registration")
            self.process_btn.setStyleSheet("background-color: #ff6666;")
            self.process_btn.clicked.disconnect()
            self.process_btn.clicked.connect(self.stop_registration_analysis)

            # Show disabled message
            self.disabled_label.show()
        except ImportError:
            self.main_window.log_message("Module import error", "ERROR")
            
    def stop_registration_analysis(self):
        """Stop the registration analysis process."""
        if self.main_window.current_worker and self.main_window.current_worker.isRunning():
            reply = QMessageBox.question(
                self, 'Confirm Stop',
                "Are you sure you want to stop the registration analysis?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # Log stopping message
                self.main_window.log_message("Stopping registration analysis...", "WARNING")
                
                # Reset UI immediately to prevent further interactions
                self.reset_ui_after_stop()
                
                # Then, safely terminate the worker (do this last)
                try:
                    if hasattr(self.main_window.current_worker, 'terminate'):
                        self.main_window.current_worker.terminate()
                except Exception as e:
                    self.main_window.log_message(f"Error during termination: {str(e)}", "ERROR")
                    
    def force_terminate_worker(self):
        """Force termination of worker thread if still running."""
        if self.main_window.current_worker and self.main_window.current_worker.isRunning():
            try:
                # Force termination
                self.main_window.current_worker.terminate()
                self.main_window.current_worker.wait(1000)
            except Exception as e:
                self.main_window.log_message(f"Error during forced termination: {str(e)}", "ERROR")
            finally:
                # Always restore UI
                QTimer.singleShot(100, self.reset_ui_after_stop)

    def reset_ui_after_stop(self):
        """Reset UI after worker termination."""
        # Reset button
        self.process_btn.setText("Run Deformable Registration and Analysis")
        self.process_btn.setStyleSheet("")
        
        # Disconnect safely to avoid errors if already disconnected
        try:
            self.process_btn.clicked.disconnect()
        except TypeError:
            pass  # Already disconnected
            
        self.process_btn.clicked.connect(self.run_registration_analysis)
        
        # Hide progress bar and disabled label
        self.main_window.progress_bar.setVisible(False)
        self.disabled_label.hide()
        
        # Re-enable controls
        self.root_dir_widget.setEnabled(True)
        self.oar_combo.setEnabled(True)
        self.refresh_oar_btn.setEnabled(True)
        self.grid_average_spin.setEnabled(True)
        self.count_threshold_spin.setEnabled(True)
        self.max_iterations_spin.setEnabled(True)
        self.results_oar_combo.setEnabled(True)
        self.results_type_combo.setEnabled(True)
        
        # Make sure the view results button is enabled if it exists
        for child in self.findChildren(QPushButton):
            if child.text() == "View Results":
                child.setEnabled(True)
        
        # Ensure results canvas is visible - but use a timer to avoid recursive painting
        QTimer.singleShot(300, self.restore_canvas)
        
        # Log completion
        self.main_window.log_message("Registration analysis stopped by user", "WARNING")

    def restore_canvas(self):
        """Safely restore canvas visibility"""
        if hasattr(self, 'results_canvas') and self.results_canvas:
            # Make sure canvas is visible
            self.results_canvas.setVisible(True)
            
            # Force a complete resize/update of the canvas
            if hasattr(self.results_canvas, 'fig'):
                self.results_canvas.fig.tight_layout()
                self.results_canvas.draw_idle()  # Use draw_idle instead of draw
                
    def reset_registration_button(self):
        """Reset the registration button to its original state."""
        self.process_btn.setText("Run Deformable Registration and Analysis")
        self.process_btn.setStyleSheet("")
        
        # Disconnect safely using try-except to avoid errors if not connected
        try:
            self.process_btn.clicked.disconnect()
        except Exception:
            pass
            
        # Connect to the run method
        self.process_btn.clicked.connect(self.run_registration_analysis)
        
        # Find the results group widget
        results_group = None
        for widget in self.findChildren(QGroupBox):
            if widget.title() == "Registration Results":
                results_group = widget
                break
        
        # Restore the visualization canvas if it was removed and we found the group
        if results_group and self.results_canvas.parent() is None:
            results_layout = results_group.layout()
            if results_layout:
                results_layout.addWidget(self.results_canvas)
                self.results_canvas.show()
                # Queue a canvas update
                QTimer.singleShot(100, self.results_canvas.update)
        
    def registration_analysis_finished(self, success, message):
        """Called when registration and analysis is finished."""
        # Reset UI first - use a slight delay to avoid paint issues
        QTimer.singleShot(100, self.reset_registration_button)
        QTimer.singleShot(100, lambda: self.disabled_label.hide())
        
        # Stop any timers
        if hasattr(self, 'force_update_timer') and self.force_update_timer.isActive():
            self.force_update_timer.stop()
        
        # Use a timer to re-enable widgets safely
        def enable_widgets():
            # Re-enable UI elements
            self.oar_combo.setEnabled(True)
            self.grid_average_spin.setEnabled(True)
            self.count_threshold_spin.setEnabled(True)
            self.max_iterations_spin.setEnabled(True)
            self.refresh_oar_btn.setEnabled(True)
            
            # Ensure results canvas is visible
            if hasattr(self, 'results_canvas') and self.results_canvas:
                self.results_canvas.setVisible(True)
                # Trigger a proper redraw
                if hasattr(self.results_canvas, 'draw_idle'):
                    self.results_canvas.draw_idle()
            
            # Hide progress bar
            self.main_window.progress_bar.setVisible(False)
            
            # Log completion
            if success:
                self.main_window.log_message(message)
                self.log_display.append(f"SUCCESS: {message}")
                
                # Set directory for visualization tab
                root_dir = self.root_dir_widget.get_path()
                if hasattr(self.main_window.visualization_tab, 'set_results_dir'):
                    try:
                        result_dir = os.path.join(root_dir, 'BiasesResult')
                        self.main_window.visualization_tab.set_results_dir(result_dir)
                    except Exception as e:
                        self.main_window.log_message(f"Error setting results directory: {str(e)}", "WARNING")
                
                # Move to next tab with a delay to ensure UI is ready
                QTimer.singleShot(300, lambda: self.main_window.tabs.setCurrentIndex(5))
            else:
                self.main_window.log_message(message, "ERROR")
                self.log_display.append(f"ERROR: {message}")
        
        # Schedule widget re-enabling with a delay to avoid paint conflicts
        QTimer.singleShot(200, enable_widgets)

        
    def handle_status_update(self, message, level="INFO"):
        """Handle status updates from the worker thread."""
        self.main_window.update_log_display(message, level)
        
        # Also update our local log display
        if level.upper() == "ERROR":
            formatted_msg = f"<span style='color:red;'>{message}</span>"
        elif level.upper() == "WARNING":
            formatted_msg = f"<span style='color:#CC9900;'>{message}</span>"
        else:
            formatted_msg = f"<span style='color:black;'>{message}</span>"
        
        self.log_display.append(formatted_msg)
    
    
        
    def view_registration_results(self):
        """View registration results."""
        if self.results_oar_combo.count() == 0:
            QMessageBox.warning(self, "Warning", "No OARs available")
            return
            
        oar = self.results_oar_combo.currentText()
        result_type = self.results_type_combo.currentText()
        root_dir = self.root_dir_widget.get_path()
        
        if not root_dir:
            QMessageBox.warning(self, "Warning", "Please run registration analysis first")
            return
            
        # Determine file path based on result type
        if result_type == "Mean":
            file_path = os.path.join(root_dir, 'BiasesResult', f'_{oar}_Result.csv')
        elif result_type == "Standard Deviation":
            file_path = os.path.join(root_dir, 'BiasesResult', f'_{oar}-std_Result.csv')
        else:  # Outliers
            # Find any outlier files
            detect_error_dir = os.path.join(root_dir, 'DetectedError')
            if not os.path.isdir(detect_error_dir):
                QMessageBox.warning(self, "Warning", "DetectedError directory not found")
                return
                
            outlier_files = []
            for file in os.listdir(detect_error_dir):
                if f"_DetectError_{oar}" in file:
                    outlier_files.append(os.path.join(detect_error_dir, file))
            
            if not outlier_files:
                QMessageBox.warning(self, "Warning", f"No outliers found for {oar}")
                return
                
            file_path = outlier_files[0]  # Use the first outlier file
        
        if not os.path.isfile(file_path):
            QMessageBox.warning(self, "Warning", f"Result file not found: {file_path}")
            return
            
        # Plot the results
        try:
            self.plot_results(file_path, oar, result_type)
        except Exception as e:
            self.main_window.log_message(f"Error plotting results: {str(e)}", "ERROR")
    
    def plot_results(self, file_path, oar, result_type):
        """
        Plot registration results.
        
        Args:
            file_path: Path to results file
            oar: OAR name
            result_type: Type of results to plot
        """
        # Load the CSV file
        data = pd.read_csv(file_path, header=None)
        
        # Extract coordinates and values
        x = data.iloc[:, 0].values
        y = data.iloc[:, 1].values
        z = data.iloc[:, 2].values
        values = data.iloc[:, 3].values
        
        # Properly clear previous plot using figure methods
        if hasattr(self.results_canvas, 'figure'):
            self.results_canvas.figure.clear()
            self.results_canvas.axes = self.results_canvas.figure.add_subplot(111, projection='3d')
        
        # Determine colormap based on result type
        if result_type == "Standard Deviation":
            cmap = plt.cm.Purples
            label = "Standard Deviation (mm)"
            vmin = 0
            vmax = np.percentile(values, 98)
        else:
            cmap = plt.cm.coolwarm
            label = "Disagreements (mm)"
            vmax = np.percentile(np.abs(values), 98)
            vmin = -vmax
        
        # Create scatter plot with smaller point size to improve performance
        scatter = self.results_canvas.axes.scatter(x, y, z, c=values, cmap=cmap, s=15, alpha=0.8, vmin=vmin, vmax=vmax)
        
        # Add colorbar
        cbar = self.results_canvas.figure.colorbar(scatter, ax=self.results_canvas.axes)
        cbar.set_label(label)
        
        # Calculate axis bounds
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        z_min, z_max = np.min(z), np.max(z)
        
        # Set axis limits
        self.results_canvas.axes.set_xlim(x_min, x_max)
        self.results_canvas.axes.set_ylim(y_min, y_max)
        self.results_canvas.axes.set_zlim(z_min, z_max)
        
        # Calculate ranges
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        
        # Set axis aspect ratio
        self.results_canvas.axes.set_box_aspect((x_range, y_range, z_range))
        
        # Set labels
        self.results_canvas.axes.set_xlabel('X (cm)')
        self.results_canvas.axes.set_ylabel('Y (cm)')
        self.results_canvas.axes.set_zlabel('Z (cm)')
        
        # Set title
        title = f"{oar} - {result_type}"
        self.results_canvas.axes.set_title(title)
        
        # Set view angle
        self.results_canvas.axes.view_init(-155, 151)
        
        # Set tight layout to prevent clipping
        self.results_canvas.figure.tight_layout()
        
        # Use draw_idle instead of draw for better thread safety
        if hasattr(self.results_canvas, 'draw_idle'):
            self.results_canvas.draw_idle()
        else:
            self.results_canvas.draw()
        
        # Add statistics to log
        stats_text = f"Statistics for {oar} ({result_type}):\n"
        stats_text += f"Total points: {len(values)}\n"
        stats_text += f"Min value: {np.min(values):.2f} mm\n"
        stats_text += f"Max value: {np.max(values):.2f} mm\n"
        
        self.log_display.append(stats_text)