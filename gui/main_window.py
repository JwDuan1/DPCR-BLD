"""
Main window for E-SAFE application.

This module implements the main window and shared UI elements for the E-SAFE application.
"""

import os
import sys
import logging
import datetime
import importlib.util
import traceback

# Add these imports here
from PyQt5.QtGui import QTextCursor
from PyQt5.QtCore import QMetaType

# Try registering QTextCursor with Qt's meta-object system using QMetaType
try:
    QMetaType.type("QTextCursor")
except:
    pass  # If it fails, we can still proceed with the application

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTabWidget, QMessageBox, QApplication
)
from PyQt5.QtCore import Qt

from utils.log_utils import setup_log_redirection, restore_streams
from gui.widgets import StatusBarWidget
from gui.data_organization_tab import DataOrganizationTab
from gui.structure_matching_tab import StructureMatchingTab
from gui.bld_calculation_tab import BLDCalculationTab
from gui.template_selection_tab import TemplateSelectionTab
from gui.registration_analysis_tab import RegistrationAnalysisTab
from gui.visualization_tab import VisualizationTab
from gui.rename_rs_tab import RenameRSTab



class MainWindow(QMainWindow):
    """Main window class for E-SAFE toolkit."""
    
    def __init__(self):
        super().__init__()
        
        # Set up logging redirection
        self.log_handler, self.stdout_redirector, self.stderr_redirector = setup_log_redirection(self)
        
        # Initialize GUI
        self.setWindowTitle("Contouring E-SAFE: Evaluation, Safety Assurance, and Feedback Enhancement")
        self.setGeometry(100, 100, 1200, 800)
        
        # Set up logging
        self.setup_logging()
        
        # Create the central widget and main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create tabs for different workflow steps
        self.tabs = QTabWidget(self)
        self.main_layout.addWidget(self.tabs)
        
        # Create tabs for each step of the workflow
        self.rename_rs_tab = RenameRSTab(self)
        self.data_org_tab = DataOrganizationTab(self)
        self.structure_matching_tab = StructureMatchingTab(self)
        self.bld_calculation_tab = BLDCalculationTab(self)
        self.template_selection_tab = TemplateSelectionTab(self)
        self.registration_analysis_tab = RegistrationAnalysisTab(self)
        self.visualization_tab = VisualizationTab(self)
        
        # Add tabs to the tab widget
        self.tabs.addTab(self.rename_rs_tab, "0. Rename RS.dcm")
        self.tabs.addTab(self.data_org_tab, "1. Data Organization")
        self.tabs.addTab(self.structure_matching_tab, "2. Structure Matching")
        self.tabs.addTab(self.bld_calculation_tab, "3. BLD Calculation")
        self.tabs.addTab(self.template_selection_tab, "4. Template Selection")
        self.tabs.addTab(self.registration_analysis_tab, "5. Registration & Analysis")
        self.tabs.addTab(self.visualization_tab, "6. Visualization")
        
        # Status bar at the bottom
        self.status_widget = StatusBarWidget()
        self.main_layout.addWidget(self.status_widget)
        
        # Access status label and progress bar directly for legacy code compatibility
        self.status_label = self.status_widget.status_label
        self.progress_bar = self.status_widget.progress_bar
        
        # Initialize variables
        self.current_worker = None
        self.log_file = None
        
        # Show the GUI
        self.show()
        self.log_message("Application started")
    
    def setup_logging(self):
        """Set up logging configuration."""
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join("logs", f"esafe_{timestamp}.log")
        
        # Configure file handler
        file_handler = logging.FileHandler(self.log_file)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # Get root logger
        root_logger = logging.getLogger()
        
        # Check if a file handler already exists (from esafe_main.py)
        has_file_handler = any(isinstance(h, logging.FileHandler) for h in root_logger.handlers)
        
        # Only add our handler if one doesn't exist already
        if not has_file_handler:
            root_logger.addHandler(file_handler)
        
        # Return existing or new log file path
        for handler in root_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                return handler.baseFilename
        
        return self.log_file
    
    def update_log_display(self, message, level="INFO"):
        """Update log displays in the GUI."""
        # Strip trailing whitespace and ignore empty messages
        message = message.rstrip()
        if not message:
            return
        
        # Check for log level in message format (2025-02-27 21:25:19,051 - INFO - Message)
        # More robust detection: check for timestamp pattern and log level indicators
        if " - ERROR - " in message:
            level = "ERROR"
        elif " - WARNING - " in message:
            level = "WARNING"
        elif " - INFO - " in message:
            level = "INFO"
        # If level is passed as argument, it overrides the detection from message content
        elif level.upper() in ["ERROR", "WARNING", "INFO"]:
            level = level.upper()
        else:
            # Default to INFO for unrecognized levels
            level = "INFO"
        
        # Format based on level
        if level == "ERROR":
            formatted_msg = f"<span style='color:red;'>{message}</span>"
        elif level == "WARNING":
            formatted_msg = f"<span style='color:#CC9900;'>{message}</span>"  # Yellow color
        else:  # INFO and any other level
            formatted_msg = f"<span style='color:black;'>{message}</span>"
        
        # Update status bar
        self.status_label.setText(message)
        
        # Only update current tab's log
        tab_index = self.tabs.currentIndex()
        log_displays = [
            self.rename_rs_tab.log_display,
            self.data_org_tab.log_display,
            self.structure_matching_tab.log_display,
            self.bld_calculation_tab.log_display,
            None,  # Template selection tab doesn't have a log
            self.registration_analysis_tab.log_display,
            None   # Visualization tab doesn't have a log
        ]
        
        if tab_index < len(log_displays) and log_displays[tab_index] is not None:
            log_displays[tab_index].append(formatted_msg)
            
    def log_message(self, message, level="INFO"):
        """Log a message to both the log file and the status bar."""
        if level.upper() == "INFO":
            logging.info(message)
        elif level.upper() == "WARNING":
            logging.warning(message)
        elif level.upper() == "ERROR":
            logging.error(message)
            QMessageBox.critical(self, "Error", message)
        
        # Update the status label directly
        self.status_label.setText(message)
    
    def update_progress(self, value):
        """Update progress bar value."""
        self.status_widget.set_progress(value)
        # Process events to keep GUI responsive during long operations
        QApplication.processEvents()
    
    def process_events(self):
        """Process pending events to keep GUI responsive."""
        QApplication.processEvents()
    
    def closeEvent(self, event):
        """Called when application is closing."""
        # Check if a worker thread is running
        if self.current_worker is not None and self.current_worker.isRunning():
            reply = QMessageBox.question(
                self, 'Confirm Exit',
                "A process is still running. Are you sure you want to exit?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # Try to terminate the worker thread gracefully first
                if hasattr(self.current_worker, 'terminate'):
                    self.current_worker.terminate()
                    
                # Wait a bit for the thread to terminate
                if not self.current_worker.wait(3000):  # Wait up to 3 seconds
                    # Force termination as a last resort
                    self.current_worker.terminate()
                    self.current_worker.wait(1000)
                    
                # Restore original streams before closing
                restore_streams(self.log_handler, self.stdout_redirector, self.stderr_redirector)
                event.accept()
            else:
                event.ignore()
        else:
            # Restore original streams
            restore_streams(self.log_handler, self.stdout_redirector, self.stderr_redirector)
            event.accept()