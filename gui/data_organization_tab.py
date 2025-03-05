"""
Data Organization tab for E-SAFE GUI.

This module implements the Data Organization tab for the E-SAFE GUI.
"""

import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QLineEdit, QFileDialog, QGroupBox, QMessageBox
)

from gui.widgets import FileSelectionWidget, LogDisplay
from utils.worker_thread import WorkerThread


class DataOrganizationTab(QWidget):
    """
    Tab for organizing and extracting DICOM structure information.
    
    This tab allows the user to extract structure information from DICOM files
    and save it to a CSV file.
    """
    
    def __init__(self, main_window):
        """
        Initialize the Data Organization tab.
        
        Args:
            main_window: The main window instance
        """
        super().__init__(main_window)
        
        self.main_window = main_window
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Create form layout
        form_layout = QVBoxLayout()
        
        # Input folder selection
        input_group = QGroupBox("Input Folder")
        input_layout = QHBoxLayout(input_group)
        
        self.input_folder_widget = FileSelectionWidget(
            "", "Select input folder containing DICOM files",
            "directory", "", "Select Input Folder"
        )
        input_layout.addWidget(self.input_folder_widget)
        form_layout.addWidget(input_group)
        
        # Output file selection
        output_group = QGroupBox("Output CSV File")
        output_layout = QHBoxLayout(output_group)
        
        self.output_file_widget = FileSelectionWidget(
            "", "Select output CSV file location",
            "file_save", "CSV Files (*.csv)", "Save Output File"
        )
        output_layout.addWidget(self.output_file_widget)
        form_layout.addWidget(output_group)
        
        # Process button
        process_btn = QPushButton("Extract Structure Names")
        process_btn.setMinimumHeight(40)
        process_btn.clicked.connect(self.run_structure_name_filtering)
        
        form_layout.addWidget(process_btn)
        
        # Log display
        self.log_display = LogDisplay("Log")
        
        # Add form layout to main layout
        layout.addLayout(form_layout, 1)
        layout.addWidget(self.log_display, 2)
    
    def run_structure_name_filtering(self):
        """Run the structure name filtering process."""
        input_folder = self.input_folder_widget.get_path()
        output_file = self.output_file_widget.get_path()
        
        if not input_folder or not output_file:
            QMessageBox.warning(self, "Warning", "Please select input folder and output file")
            return
        
        self.main_window.log_message(f"Starting structure name filtering from {input_folder} to {output_file}")
        
        # Import the module dynamically to avoid import errors if not available
        try:
            from core.filter_patient_st_name import filter_patient_st_name
            
            # Create worker thread
            self.main_window.current_worker = WorkerThread(
                filter_patient_st_name, 
                [input_folder, output_file, self.main_window.update_progress]
            )
            
            # Connect signals
            self.main_window.current_worker.update_status.connect(self.main_window.update_log_display)
            self.main_window.current_worker.finished_signal.connect(self.structure_filtering_finished)
            
            # Ensure progress bar is visible
            self.main_window.progress_bar.setVisible(True)
            self.main_window.update_progress(0)  # Reset to 0
            
            # Start the thread
            self.main_window.current_worker.start()
            
        except ImportError:
            self.main_window.log_message("Module filter_patient_st_name not found. Please make sure it's in your PYTHONPATH.", "ERROR")
    
    def structure_filtering_finished(self, success, message):
        """Called when structure name filtering is finished."""
        self.main_window.progress_bar.setVisible(False)
        
        if success:
            self.main_window.log_message(message)
            
            # Move to next tab
            self.main_window.tabs.setCurrentIndex(1)
            
            # Pass output file to next tab
            output_file = self.output_file_widget.get_path()
            if hasattr(self.main_window.structure_matching_tab, 'set_input_file'):
                self.main_window.structure_matching_tab.set_input_file(output_file)
        else:
            self.main_window.log_message(message, "ERROR")