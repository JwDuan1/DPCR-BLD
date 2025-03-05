"""
Structure Matching tab for E-SAFE GUI.

This module implements the Structure Matching tab for the E-SAFE GUI.
"""

import os
import csv
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QLineEdit, QFileDialog, QGroupBox, QMessageBox
)

from gui.widgets import FileSelectionWidget, LogDisplay
from utils.worker_thread import WorkerThread


class StructureMatchingTab(QWidget):
    """
    Tab for matching structure names with a lookup table.
    
    This tab allows the user to match structure names from DICOM files with
    a lookup table and create matched structure lists.
    """
    
    def __init__(self, main_window):
        """
        Initialize the Structure Matching tab.
        
        Args:
            main_window: The main window instance
        """
        super().__init__(main_window)
        
        self.main_window = main_window
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Create form layout
        form_layout = QVBoxLayout()
        
        # Lookup table creation heading
        heading = QLabel("Create Lookup Table for Structure Name Matching")
        form_layout.addWidget(heading)
        
        # Lookup table selection
        lookup_group = QGroupBox("Lookup Table CSV")
        lookup_layout = QHBoxLayout(lookup_group)
        
        self.lookup_file_widget = FileSelectionWidget(
            "", "Select lookup table CSV file",
            "file_open", "CSV Files (*.csv)", "Open Lookup Table"
        )
        
        lookup_layout.addWidget(self.lookup_file_widget)
        
        # Create new lookup table button
        create_lookup_btn = QPushButton("Create New")
        create_lookup_btn.clicked.connect(self.create_new_lookup_table)
        lookup_layout.addWidget(create_lookup_btn)
        
        form_layout.addWidget(lookup_group)
        
        # Reference data folder
        ref_group = QGroupBox("Reference Data Folder")
        ref_layout = QHBoxLayout(ref_group)
        
        self.ref_folder_widget = FileSelectionWidget(
            "", "Select reference data folder",
            "directory", "", "Select Reference Data Folder"
        )
        ref_layout.addWidget(self.ref_folder_widget)
        form_layout.addWidget(ref_group)
        
        # Test data folder
        test_group = QGroupBox("Test Data Folder")
        test_layout = QHBoxLayout(test_group)
        
        self.test_folder_widget = FileSelectionWidget(
            "", "Select test data folder",
            "directory", "", "Select Test Data Folder"
        )
        test_layout.addWidget(self.test_folder_widget)
        form_layout.addWidget(test_group)
        
        # Output files
        output_group = QGroupBox("Output Files")
        output_layout = QVBoxLayout(output_group)
        
        self.ref_output_widget = FileSelectionWidget(
            "Reference List:", "Reference output list CSV",
            "file_save", "CSV Files (*.csv)", "Save Reference Output File"
        )
        output_layout.addWidget(self.ref_output_widget)
        
        self.test_output_widget = FileSelectionWidget(
            "Test List:", "Test output list CSV",
            "file_save", "CSV Files (*.csv)", "Save Test Output File"
        )
        output_layout.addWidget(self.test_output_widget)
        
        form_layout.addWidget(output_group)
        
        # Process buttons
        buttons_layout = QHBoxLayout()
        
        process_ref_btn = QPushButton("Process Reference")
        process_ref_btn.clicked.connect(lambda: self.run_structure_matching("ref"))
        
        process_test_btn = QPushButton("Process Test")
        process_test_btn.clicked.connect(lambda: self.run_structure_matching("test"))
        
        process_both_btn = QPushButton("Process Both")
        process_both_btn.clicked.connect(self.run_structure_matching_both)
        process_both_btn.setMinimumHeight(40)
        
        buttons_layout.addWidget(process_ref_btn)
        buttons_layout.addWidget(process_test_btn)
        buttons_layout.addWidget(process_both_btn)
        
        form_layout.addLayout(buttons_layout)
        
        # Log display
        self.log_display = LogDisplay("Log")
        
        # Add form layout to main layout
        layout.addLayout(form_layout, 1)
        layout.addWidget(self.log_display, 1)
    
    def set_input_file(self, file_path):
        """
        Set an input file path from another tab.
        
        Args:
            file_path: The file path to set
        """
        # This could be used to set a default folder based on the output of the previous tab
        pass
    
    def create_new_lookup_table(self):
        """Create a new lookup table template."""
        file, _ = QFileDialog.getSaveFileName(self, "Create Lookup Table", "", "CSV Files (*.csv)")
        if not file:
            return
            
        # Create example lookup table
        try:
            with open(file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Brainstem", "Brain_Stem", "BrainStem", "Brain stem"])
                writer.writerow(["Parotid_L", "Lt_Parotid", "Parotid Lt", "L Parotid"])
                writer.writerow(["Parotid_R", "Rt_Parotid", "Parotid Rt", "R Parotid"])
                writer.writerow(["SpinalCord", "Spinal_Cord", "Cord", ""])
            
            self.lookup_file_widget.set_path(file)
            self.main_window.log_message(f"Created lookup table template at {file}")
            
            # Open in default application
            import subprocess
            import platform
            
            if platform.system() == 'Windows':
                os.startfile(file)
            elif platform.system() == 'Darwin':  # macOS
                subprocess.call(('open', file))
            else:  # Linux
                subprocess.call(('xdg-open', file))
                
        except Exception as e:
            self.main_window.log_message(f"Error creating lookup table: {str(e)}", "ERROR")
    
    def run_structure_matching(self, match_type):
        """
        Run the structure matching process.
        
        Args:
            match_type: Type of matching ('ref' or 'test')
        """
        lookup_file = self.lookup_file_widget.get_path()
        
        if not lookup_file:
            QMessageBox.warning(self, "Warning", "Please select a lookup table file")
            return
        
        if match_type == "ref":
            input_folder = self.ref_folder_widget.get_path()
            output_file = self.ref_output_widget.get_path()
            label = "reference"
        else:
            input_folder = self.test_folder_widget.get_path()
            output_file = self.test_output_widget.get_path()
            label = "test"
            
        if not input_folder or not output_file:
            QMessageBox.warning(self, "Warning", f"Please select {label} input folder and output file")
            return
            
        self.main_window.log_message(f"Starting structure matching for {label} data")
        
        # Import the module dynamically
        try:
            from core.org_study_list import org_study_list
            
            # Create worker thread
            self.main_window.current_worker = WorkerThread(
                org_study_list, 
                [input_folder, lookup_file, output_file]
            )
            
            # Connect signals
            self.main_window.current_worker.update_status.connect(self.main_window.update_log_display)
            self.main_window.current_worker.update_progress.connect(self.main_window.update_progress)
            self.main_window.current_worker.finished_signal.connect(
                lambda success, message: self.structure_matching_finished(success, message, match_type)
            )
            
            # Start the thread
            self.main_window.progress_bar.setVisible(True)
            self.main_window.current_worker.start()
            
        except ImportError:
            self.main_window.log_message("Module org_study_list not found. Please make sure it's in your PYTHONPATH.", "ERROR")
    
    def run_structure_matching_both(self):
        """Run structure matching for both reference and test data."""
        lookup_file = self.lookup_file_widget.get_path()
        ref_folder = self.ref_folder_widget.get_path()
        test_folder = self.test_folder_widget.get_path()
        ref_output = self.ref_output_widget.get_path()
        test_output = self.test_output_widget.get_path()
        
        if not lookup_file:
            QMessageBox.warning(self, "Warning", "Please select a lookup table file")
            return
            
        if not ref_folder or not ref_output:
            QMessageBox.warning(self, "Warning", "Please select reference input folder and output file")
            return
            
        if not test_folder or not test_output:
            QMessageBox.warning(self, "Warning", "Please select test input folder and output file")
            return
        
        self.main_window.log_message("Starting structure matching for both reference and test data")
        
        # Run reference first, then test will be run after reference is finished
        self.run_structure_matching("ref")
    
    def structure_matching_finished(self, success, message, match_type):
        """Called when structure matching is finished."""
        self.main_window.progress_bar.setVisible(False)
        
        if success:
            self.main_window.log_message(message)
            self.log_display.append(f"SUCCESS ({match_type}): {message}")
            
            # If this was reference and we're in batch mode, run test
            if match_type == "ref" and self.test_folder_widget.get_path() and self.test_output_widget.get_path():
                self.run_structure_matching("test")
            else:
                # Move to next tab
                self.main_window.tabs.setCurrentIndex(2)
                
                # Pass output files to next tab
                if hasattr(self.main_window.bld_calculation_tab, 'set_input_files'):
                    self.main_window.bld_calculation_tab.set_input_files(
                        self.ref_output_widget.get_path(), 
                        self.test_output_widget.get_path()
                    )
        else:
            self.main_window.log_message(message, "ERROR")
            self.log_display.append(f"ERROR ({match_type}): {message}")