"""
Logging utilities for E-SAFE GUI.

This module provides classes and functions for redirecting logs to the GUI.
"""

import sys
import logging
from io import StringIO
from PyQt5.QtCore import QObject, pyqtSignal


class GuiLogHandler(logging.Handler, QObject):
    """Custom log handler that emits a signal with log messages."""
    log_signal = pyqtSignal(str, str)  # message, level
    
    def __init__(self):
        logging.Handler.__init__(self)
        QObject.__init__(self)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.setFormatter(formatter)
    
    def emit(self, record):
        msg = self.format(record)
        # Emit the formatted message and level
        self.log_signal.emit(msg, record.levelname)


class OutputRedirector(StringIO):
    """Redirects stdout/stderr to the GUI log."""
    def __init__(self, gui_instance, stream_type='stdout'):
        super().__init__()
        self.gui_instance = gui_instance
        self.stream_type = stream_type
        self.original_stream = sys.stdout if stream_type == 'stdout' else sys.stderr
    
    def write(self, text):
        if text.strip():  # Only process non-empty text
            # Send to original stream
            self.original_stream.write(text)
            
            # Send to GUI log - handle progress bars specially
            if '\r' in text:  # Progress bar updates often have carriage returns
                # Only update status bar, not log text
                self.gui_instance.status_label.setText(text.strip())
                
                # Try to extract progress percentage
                if "%" in text:
                    try:
                        pct_str = text.split("%")[0].split()[-1]
                        pct = int(float(pct_str))
                        self.gui_instance.update_progress(pct)
                    except (ValueError, IndexError):
                        pass
            else:
                level = "ERROR" if self.stream_type == 'stderr' else "INFO"
                self.gui_instance.update_log_display(text, level)
    
    def flush(self):
        # This needs to be implemented for compatibility
        self.original_stream.flush()


def setup_log_redirection(gui_instance):
    """
    Set up log redirection for a GUI instance.
    
    Args:
        gui_instance: The main GUI instance
        
    Returns:
        tuple: (log_handler, stdout_redirector, stderr_redirector)
    """
    # Create and connect custom log handler
    log_handler = GuiLogHandler()
    log_handler.log_signal.connect(gui_instance.update_log_display)
    
    # Add the handler to the root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(log_handler)
    
    # Redirect stdout and stderr
    stdout_redirector = OutputRedirector(gui_instance, 'stdout')
    stderr_redirector = OutputRedirector(gui_instance, 'stderr')
    
    # Save original streams
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Set new redirectors
    sys.stdout = stdout_redirector
    sys.stderr = stderr_redirector
    
    return log_handler, stdout_redirector, stderr_redirector


def restore_streams(log_handler, stdout_redirector, stderr_redirector):
    """
    Restore original streams and clean up log handlers.
    
    Args:
        log_handler: The custom log handler
        stdout_redirector: The stdout redirector
        stderr_redirector: The stderr redirector
    """
    # Restore original streams
    sys.stdout = stdout_redirector.original_stream
    sys.stderr = stderr_redirector.original_stream
    
    # Remove custom log handler
    root_logger = logging.getLogger()
    root_logger.removeHandler(log_handler)
    
def setup_logging(log_file: str) -> None:
    """Set up logging configuration."""
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logging.info(f"Logging initialized to {log_file}")