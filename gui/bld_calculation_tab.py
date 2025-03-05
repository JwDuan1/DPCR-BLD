"""
BLD Calculation tab for E-SAFE GUI.

This module implements the BLD Calculation tab for the E-SAFE GUI.
"""

import os
import pandas as pd
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QLineEdit, QFileDialog, QGroupBox, QRadioButton, QComboBox,
    QMessageBox,QApplication
)
from PyQt5.QtCore import QTimer

from gui.widgets import FileSelectionWidget, LogDisplay
from utils.worker_thread import WorkerThread,SafeWorkerThread


class BLDCalculationTab(QWidget):
    """
    Tab for calculating BLD metrics.
    
    This tab allows the user to calculate BLD metrics between reference and test
    structure sets.
    """
    
    def __init__(self, main_window):
        """
        Initialize the BLD Calculation tab.
        
        Args:
            main_window: The main window instance
        """
        super().__init__(main_window)
        
        self.main_window = main_window
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Create form layout
        form_layout = QVBoxLayout()
        
        # Input file selection
        input_group = QGroupBox("Input Files")
        input_layout = QVBoxLayout(input_group)
        
        self.ref_list_widget = FileSelectionWidget(
            "Reference List:", "Reference list CSV from previous step",
            "file_open", "CSV Files (*.csv)", "Open Reference List"
        )
        input_layout.addWidget(self.ref_list_widget)
        
        self.test_list_widget = FileSelectionWidget(
            "Test List:", "Test list CSV from previous step",
            "file_open", "CSV Files (*.csv)", "Open Test List"
        )
        input_layout.addWidget(self.test_list_widget)
        
        form_layout.addWidget(input_group)
        
        # Output settings
        output_group = QGroupBox("Output Settings")
        output_layout = QVBoxLayout(output_group)
        
        self.result_file_widget = FileSelectionWidget(
            "Results File:", "Results CSV file",
            "file_save", "CSV Files (*.csv)", "Save Results File"
        )
        output_layout.addWidget(self.result_file_widget)
        
        self.bld_dir_widget = FileSelectionWidget(
            "BLD Directory:", "BLD output directory",
             "directory", "", "Select BLD Output Directory"
        )
        output_layout.addWidget(self.bld_dir_widget)
        
        win_prefix_layout = QHBoxLayout()
        win_prefix_layout.addWidget(QLabel("Windows Path Prefix:"))
        self.win_prefix_edit = QLineEdit()
        self.win_prefix_edit.setPlaceholderText("Optional Windows path prefix")
        win_prefix_layout.addWidget(self.win_prefix_edit)
        output_layout.addLayout(win_prefix_layout)
        
        form_layout.addWidget(output_group)
        
        # Processing mode selection
        mode_group = QGroupBox("Processing Mode")
        mode_layout = QVBoxLayout(mode_group)
        
        self.batch_mode_radio = QRadioButton("Batch Mode (Process All Structures)")
        self.batch_mode_radio.setChecked(True)
        self.batch_mode_radio.toggled.connect(self.toggle_bld_mode)
        
        self.single_oar_radio = QRadioButton("Single OAR Mode")
        self.single_oar_radio.toggled.connect(self.toggle_bld_mode)
        
        self.oar_selection_combo = QComboBox()
        self.oar_selection_combo.setEnabled(False)
        self.oar_selection_combo.addItem("Select an OAR after loading input files")
        
        refresh_oars_btn = QPushButton("Refresh OAR List")
        refresh_oars_btn.clicked.connect(self.refresh_oar_list)
        
        mode_layout.addWidget(self.batch_mode_radio)
        mode_layout.addWidget(self.single_oar_radio)
        
        oar_selection_layout = QHBoxLayout()
        oar_selection_layout.addWidget(QLabel("Select OAR:"))
        oar_selection_layout.addWidget(self.oar_selection_combo)
        oar_selection_layout.addWidget(refresh_oars_btn)
        mode_layout.addLayout(oar_selection_layout)
        
        form_layout.addWidget(mode_group)
        
        # Process button
        self.calculate_bld_btn = QPushButton("Calculate BLD Metrics")
        self.calculate_bld_btn.setMinimumHeight(40)
        self.calculate_bld_btn.clicked.connect(self.run_bld_calculation)
        
        form_layout.addWidget(self.calculate_bld_btn)
        
        # Log display
        self.log_display = LogDisplay("Log")
        
        # Add form layout to main layout
        layout.addLayout(form_layout, 1)
        layout.addWidget(self.log_display, 1)
    
    def set_input_files(self, ref_file, test_file):
        """
        Set input files from previous tab.
        
        Args:
            ref_file: Reference list CSV file path
            test_file: Test list CSV file path
        """
        if ref_file:
            self.ref_list_widget.set_path(ref_file)
        
        if test_file:
            self.test_list_widget.set_path(test_file)
        
        # Create default output paths
        if ref_file and test_file:
            try:
                # Determine common parent directory
                ref_dir = os.path.dirname(ref_file)
                test_dir = os.path.dirname(test_file)
                
                # Use either directory if they're the same
                if ref_dir == test_dir:
                    output_dir = ref_dir
                else:
                    # Otherwise, find a common parent
                    output_dir = os.path.dirname(os.path.commonpath([ref_dir, test_dir]))
                
                # Create default output paths
                results_file = os.path.join(output_dir, "bld_results.csv")
                bld_dir = os.path.join(output_dir, "BLD")
                
                # Set default output paths
                self.result_file_widget.set_path(results_file)
                self.bld_dir_widget.set_path(bld_dir)
                
                # Refresh OAR list
                if self.single_oar_radio.isChecked():
                    self.refresh_oar_list()
            except Exception:
                # If any error occurs, just continue without setting defaults
                pass
    
    def toggle_bld_mode(self):
        """Toggle between batch and single OAR modes."""
        if self.single_oar_radio.isChecked():
            self.oar_selection_combo.setEnabled(True)
            self.refresh_oar_list()
        else:
            self.oar_selection_combo.setEnabled(False)
    
    def refresh_oar_list(self):
        """Refresh the list of available OARs from input files."""
        ref_list = self.ref_list_widget.get_path()
        test_list = self.test_list_widget.get_path()
        
        if not ref_list or not test_list:
            QMessageBox.warning(self, "Warning", "Please select reference and test list files first.")
            return
        
        try:
            # Read structure names from test list file
            test_df = pd.read_csv(test_list)
            
            # Clear existing items
            self.oar_selection_combo.clear()
            
            # Get column names skipping the first few non-structure columns
            num_non_strs = 4  # First 4 columns are not structures
            
            if test_df.shape[1] <= num_non_strs:
                self.main_window.log_message("No structure columns found in test list file.", "WARNING")
                self.oar_selection_combo.addItem("No structures found")
                return
            
            # Add each structure name to combo box
            for i in range(num_non_strs, test_df.shape[1]):
                self.oar_selection_combo.addItem(test_df.columns[i])
            
            # Also try getting OARs from BLD directory if it exists
            bld_dir = self.bld_dir_widget.get_path()
            if bld_dir and os.path.isdir(bld_dir):
                try:
                    from core.bld_batch import get_available_oars
                    oars = get_available_oars(bld_dir)
                    
                    # Add any OARs from the directory that aren't already in the list
                    current_oars = [self.oar_selection_combo.itemText(i) 
                                for i in range(self.oar_selection_combo.count())]
                    
                    for oar in oars:
                        if oar not in current_oars:
                            self.oar_selection_combo.addItem(oar)
                except Exception as e:
                    self.main_window.log_message(f"Could not get OARs from BLD directory: {str(e)}", "WARNING")
            
            if self.oar_selection_combo.count() > 0:
                self.oar_selection_combo.setCurrentIndex(0)
                self.main_window.log_message(f"Found {self.oar_selection_combo.count()} structures")
            else:
                self.oar_selection_combo.addItem("No structures found")
                self.main_window.log_message("No structures found in input files", "WARNING")
                
        except Exception as e:
            self.main_window.log_message(f"Error loading structure names: {str(e)}", "ERROR")
            self.oar_selection_combo.clear()
            self.oar_selection_combo.addItem("Error loading structures")
    
    def run_bld_calculation(self):
        """Run the BLD calculation process."""
        ref_list = self.ref_list_widget.get_path()
        test_list = self.test_list_widget.get_path()
        result_file = self.result_file_widget.get_path()
        bld_dir = self.bld_dir_widget.get_path()
        win_prefix = self.win_prefix_edit.text()
        
        if not ref_list or not test_list or not result_file or not bld_dir:
            QMessageBox.warning(self, "Warning", "Please fill in all required fields")
            return
        
        # Determine if we're in single OAR or batch mode
        oar_name = None
        if self.single_oar_radio.isChecked():
            oar_name = self.oar_selection_combo.currentText()
            if oar_name in ["No structures found", "Error loading structures", "Select an OAR after loading input files"]:
                QMessageBox.warning(self, "Warning", "Please select a valid OAR")
                return
            
            self.main_window.log_message(f"Starting BLD calculation for single OAR: {oar_name}")
        else:
            self.main_window.log_message("Starting BLD calculation in batch mode")
        
        # Make sure the output directory exists
        os.makedirs(bld_dir, exist_ok=True)
        
        # Import the module dynamically
        try:
            from core.bld_batch import bld_batch
            from utils.worker_thread import launch_safe_worker
            
            # List widgets to disable during processing
            widgets_to_disable = [
                self.ref_list_widget, self.test_list_widget,
                self.result_file_widget, self.bld_dir_widget,
                self.win_prefix_edit, self.oar_selection_combo,
                self.batch_mode_radio, self.single_oar_radio
            ]
            
            # Change calculate button to stop button
            self.calculate_bld_btn.setText("Stop Calculation")
            self.calculate_bld_btn.setStyleSheet("background-color: #ff6666;")
            self.calculate_bld_btn.clicked.disconnect()
            self.calculate_bld_btn.clicked.connect(self.stop_bld_calculation)
            
            # Show progress bar
            self.main_window.progress_bar.setVisible(True)
            
            # Launch worker
            self.main_window.current_worker = launch_safe_worker(
                parent=self,
                function=bld_batch,
                args=[ref_list, test_list, result_file, bld_dir, win_prefix, oar_name],
                callback_func=self.bld_calculation_finished,
                progress_callback=self.main_window.update_progress,
                status_callback=self.main_window.update_log_display,
                disable_widgets=widgets_to_disable
            )
            
        except ImportError as e:
            self.main_window.log_message(f"Module import error: {str(e)}", "ERROR")
        
    def stop_bld_calculation(self):
        """Stop the BLD calculation process."""
        if self.main_window.current_worker and self.main_window.current_worker.isRunning():
            reply = QMessageBox.question(
                self, 'Confirm Stop',
                "Are you sure you want to stop the BLD calculation?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.main_window.log_message("Stopping BLD calculation...", "WARNING")
                self.main_window.current_worker.terminate()
                self.main_window.current_worker.wait(1000)  # Wait up to 1 second
                self.reset_bld_calculation_button()
                self.main_window.progress_bar.setVisible(False)
                self.main_window.log_message("BLD calculation stopped by user", "WARNING")
    
    def reset_bld_calculation_button(self):
        """Reset the calculate button to its original state."""
        self.calculate_bld_btn.setText("Calculate BLD Metrics")
        self.calculate_bld_btn.setStyleSheet("")
        self.calculate_bld_btn.clicked.disconnect()
        self.calculate_bld_btn.clicked.connect(self.run_bld_calculation)
    
    def bld_calculation_finished(self, success, message):
        """Called when BLD calculation is finished."""
        # Stop the heartbeat timer
        if hasattr(self, 'heartbeat_timer') and self.heartbeat_timer.isActive():
            self.heartbeat_timer.stop()
            
        self.main_window.progress_bar.setVisible(False)
        self.reset_bld_calculation_button()
        
        if success:
            self.main_window.log_message(message)
            
            # Set BLD directory for next tab
            bld_dir = self.bld_dir_widget.get_path()
            if hasattr(self.main_window.template_selection_tab, 'set_bld_dir'):
                self.main_window.template_selection_tab.set_bld_dir(bld_dir)
            
            # Move to next tab
            self.main_window.tabs.setCurrentIndex(3)
        else:
            self.main_window.log_message(message, "ERROR")