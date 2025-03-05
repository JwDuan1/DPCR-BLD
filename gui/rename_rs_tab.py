"""
Rename RS tab for E-SAFE GUI.

This module implements the Rename RS tab for the E-SAFE GUI.
"""

import os
import logging
import traceback
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QFileDialog, QGroupBox, QMessageBox
)
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition

from gui.widgets import FileSelectionWidget, LogDisplay


class RenameRSWorker(QThread):
    """Specialized worker thread for renaming RS DICOM files."""
    progress_signal = pyqtSignal(int)
    status_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, object)
    
    def __init__(self, input_folder):
        super().__init__()
        self.input_folder = input_folder
        
    def run(self):
        try:
            from core.rename_dcm_rs import DicomFileProcessor
            
            # Create processor
            processor = DicomFileProcessor(self.input_folder)
            
            # Find all DCM files
            self.status_signal.emit("Finding DICOM files...")
            all_dcm_files = processor.find_all_dcm_files()
            
            total_files = len(all_dcm_files)
            processed_count = 0
            rs_count = 0
            
            # Update initial progress
            self.progress_signal.emit(0)
            
            # Process each file
            for i, file_path in enumerate(all_dcm_files):
                try:
                    # Let the processor determine if it's a RTSTRUCT file
                    base_name = os.path.basename(file_path)
                    # Will return True only if it's a valid RTSTRUCT file
                    if processor.process_file(file_path):
                        rs_count += 1
                        self.status_signal.emit(f"Processing RTSTRUCT file {rs_count}: {base_name}")
                    
                    processed_count += 1
                    
                    # Update progress every 10 files or when processing an RS file
                    if i % 10 == 0 or base_name.startswith('RS'):
                        progress = int((i + 1) / total_files * 100)
                        self.progress_signal.emit(progress)
                        
                except Exception as e:
                    self.status_signal.emit(f"Error processing {file_path}: {str(e)}")
            
            # Ensure 100% progress at end
            self.progress_signal.emit(100)
            
            # Return statistics
            stats = {
                "total_files": total_files,
                "rs_files": rs_count,
                "processed_files": processed_count
            }
            self.finished_signal.emit(True, stats)
            
        except Exception as e:
            self.status_signal.emit(f"Error: {str(e)}")
            self.status_signal.emit(traceback.format_exc())
            self.finished_signal.emit(False, str(e))


class RenameRSTab(QWidget):
    """
    Tab for renaming DICOM structure set files.
    
    This tab allows the user to rename DICOM structure set files (RS*.dcm)
    based on their DICOM tags.
    """
    
    def __init__(self, main_window):
        """
        Initialize the Rename RS tab.
        
        Args:
            main_window: The main window instance
        """
        super().__init__(main_window)
        
        self.main_window = main_window
        self.worker = None
        
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
        
        # Process button
        self.process_btn = QPushButton("Rename Structure Set Files")
        self.process_btn.setMinimumHeight(40)
        self.process_btn.clicked.connect(self.run_rename_rs)
        
        form_layout.addWidget(self.process_btn)
        
        # Log display
        self.log_display = LogDisplay("Log")
        
        # Add form layout to main layout
        layout.addLayout(form_layout, 1)
        layout.addWidget(self.log_display, 2)
    
    def run_rename_rs(self):
        """Run the rename RS process."""
        input_folder = self.input_folder_widget.get_path()
        
        if not input_folder:
            QMessageBox.warning(self, "Warning", "Please select input folder")
            return
        
        self.main_window.log_message(f"Starting RS DICOM file renaming in {input_folder}")
        
        # Disable button during processing
        self.process_btn.setEnabled(False)
        self.process_btn.setText("Processing...")
        
        # Create and configure worker thread
        self.worker = RenameRSWorker(input_folder)
        self.worker.progress_signal.connect(self.main_window.update_progress)
        self.worker.status_signal.connect(self.main_window.update_log_display)
        self.worker.finished_signal.connect(self.rename_rs_finished)
        
        # Show progress bar
        self.main_window.progress_bar.setVisible(True)
        
        # Start the thread
        self.worker.start()
    
    def rename_rs_finished(self, success, result):
        """Called when rename RS process is finished."""
        # Re-enable button
        self.process_btn.setEnabled(True)
        self.process_btn.setText("Rename Structure Set Files")
        
        # Hide progress bar
        self.main_window.progress_bar.setVisible(False)
        
        if success:
            stats = result
            if isinstance(stats, dict):
                self.main_window.log_message(f"Rename RS completed successfully")
                self.main_window.log_message(f"Total DCM files: {stats['total_files']}")
                self.main_window.log_message(f"RS files found: {stats['rs_files']}")
                self.main_window.log_message(f"Files processed: {stats['processed_files']}")
                
                # Log to the tab's log display
                self.log_display.append(f"<span style='color:green;'>SUCCESS: Renamed {stats['rs_files']} structure set files</span>")
            else:
                self.main_window.log_message(f"Rename RS completed: {result}")
            
            # Pass input folder to next tab if available
            input_folder = self.input_folder_widget.get_path()
            if hasattr(self.main_window.data_org_tab, 'input_folder_widget') and input_folder:
                self.main_window.data_org_tab.input_folder_widget.set_path(input_folder)
                
            # Move to next tab
            self.main_window.tabs.setCurrentIndex(1)
        else:
            self.main_window.log_message(result, "ERROR")
            self.log_display.append(f"<span style='color:red;'>ERROR: {result}</span>")