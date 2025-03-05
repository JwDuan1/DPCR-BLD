"""
Enhanced worker thread management for E-SAFE.

This module provides thread management utilities for running background tasks
in the E-SAFE application.
"""

from PyQt5.QtCore import QThread, pyqtSignal, QTimer,QEventLoop
import traceback
import time
import threading
import logging

from PyQt5.QtWidgets import QApplication

class WorkerThread(QThread):
    """Enhanced worker thread with result storing and termination support."""
    update_progress = pyqtSignal(int)
    update_status = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)  # Success, Message
    
    def __init__(self, function, args, timeout=30000000):
        super().__init__()
        self.function = function
        self.args = args
        self.timeout = timeout
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.handle_timeout)
        self.result_data = None  # Store the result
        self.termination_flag = threading.Event()
    

    def run(self):
        try:
            # Start timeout timer
            self.timer.start(self.timeout)
            
            # Update status
            self.update_status.emit(f"Starting {self.function.__name__}...")
            
            # Process events to keep GUI responsive
            from PyQt5.QtWidgets import QApplication
            QApplication.processEvents()
            
            # Check if function supports termination checking
            if hasattr(self.function, 'supports_termination'):
                # Add termination check function to args
                args_with_check = list(self.args)
                args_with_check.append(self.check_termination)
                self.result_data = self.function(*args_with_check)
            else:
                # Execute function normally
                self.result_data = self.function(*self.args)
            
            # Cancel timeout timer
            self.timer.stop()
            
            # Process events to keep GUI responsive
            QApplication.processEvents()
            
            # Emit success signal
            self.finished_signal.emit(True, f"{self.function.__name__} completed successfully")
            
        except Exception as e:
            # Cancel timeout timer
            self.timer.stop()
            
            # Log the error
            error_msg = f"Error in {self.function.__name__}: {str(e)}"
            self.update_status.emit(error_msg)
            self.update_status.emit(traceback.format_exc())
            
            # Process events to keep GUI responsive
            from PyQt5.QtWidgets import QApplication
            QApplication.processEvents()
            
            # Emit failure signal
            self.finished_signal.emit(False, f"Error: {str(e)}")
    
    def result(self):
        """Return the result data."""
        return self.result_data
    
    def handle_timeout(self):
        """Handle thread execution timeout."""
        self.update_status.emit(f"Operation timed out after {self.timeout/1000} seconds")
        self.terminate()
        self.wait(1000)  # Wait 1 second for termination
        self.finished_signal.emit(False, f"Operation timed out")
    
    def terminate(self):
        """
        Overridden terminate method that sets termination flag before hard termination.
        This allows the function to clean up if it checks the termination flag.
        """
        # Set the termination flag so the function can exit gracefully if it checks
        self.termination_flag.set()
        self.update_status.emit(f"Requesting termination of {self.name}...")
        
        # Give it a chance to terminate gracefully
        for _ in range(50):  # Increase wait time to 5 seconds (50 * 0.1)
            if not self.isRunning():
                return
            QApplication.processEvents()  # Process events while waiting
            time.sleep(0.1)
        
        # If still running, force termination
        logging.warning("Thread did not terminate gracefully, forcing termination")
        super().terminate()
    
    def check_termination(self):
        """
        Check if termination has been requested.
        Long-running functions should periodically call this to allow clean shutdown.
        
        Returns:
            bool: True if termination requested, False otherwise
        """
        return self.termination_flag.is_set()


# Add a decorator to mark functions that support termination checking
def supports_termination(func):
    """
    Decorator to mark functions that support termination checking.
    This helps the WorkerThread know which functions can be terminated cleanly.
    """
    func.supports_termination = True
    return func

class SafeWorkerThread(QThread):
    """Thread-safe worker that prevents GUI freezing during long operations."""
    update_progress = pyqtSignal(int)
    update_status = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    
    def __init__(self, function, args):
        super().__init__()
        self.function = function
        self.args = args
        self.result_data = None
        
    def run(self):
        try:
            # Store progress updates locally
            last_progress = 0
            last_status_time = time.time()
            
            # Determine which function we're running
            func_name = self.function.__name__ if hasattr(self.function, '__name__') else 'unknown'
            
            # Create monitoring mechanism for bld_match_via_dcpr
            # This function doesn't accept callbacks, so we'll monitor its log messages
            if func_name == 'bld_match_via_dcpr':
                # Just use the original arguments without modifications
                # Periodically emit progress updates based on time
                start_time = time.time()
                estimated_duration = 600  # 10 minutes estimated
                
                # Start the function in a separate thread we can monitor
                import threading
                result_container = [None]
                exception_container = [None]
                
                def run_function():
                    try:
                        result_container[0] = self.function(*self.args)
                    except Exception as e:
                        exception_container[0] = e
                
                thread = threading.Thread(target=run_function)
                thread.start()
                
                # Monitor the thread and emit progress
                while thread.is_alive():
                    elapsed = time.time() - start_time
                    progress = min(95, int(elapsed / estimated_duration * 100))
                    
                    if abs(progress - last_progress) >= 5:
                        self.update_progress.emit(progress)
                        last_progress = progress
                    
                    # Status update based on duration
                    if time.time() - last_status_time > 2:
                        # self.update_status.emit(f"Processing... ({int(elapsed)}s elapsed)")
                        last_status_time = time.time()
                    
                    time.sleep(1)
                    
                # Check if there was an exception
                if exception_container[0]:
                    raise exception_container[0]
                    
                self.result_data = result_container[0]
                
            elif func_name == 'bld_batch':
                # Define wrappers that throttle GUI interactions
                def progress_wrapper(value):
                    nonlocal last_progress
                    if abs(value - last_progress) >= 2 or value >= 100 or value == 0:
                        self.update_progress.emit(value)
                        last_progress = value
                        time.sleep(0.01)
                
                def status_wrapper(message):
                    nonlocal last_status_time
                    now = time.time()
                    if now - last_status_time > 0.5:
                        self.update_status.emit(message)
                        last_status_time = now
                
                # BLD batch expects callbacks as final two arguments
                modified_args = list(self.args)
                if len(modified_args) < 7:  # Check we don't already have callbacks
                    modified_args.extend([progress_wrapper, status_wrapper])
                
                self.result_data = self.function(*modified_args)
            else:
                # For any other function, just call it with original args
                self.result_data = self.function(*self.args)
            
            # Generate appropriate success message
            success_message = "Process completed successfully"
            if func_name == 'bld_batch':
                success_message = "BLD calculation completed successfully"
            elif func_name == 'bld_match_via_dcpr':
                success_message = "Registration analysis completed successfully"
                
            self.finished_signal.emit(True, success_message)
            
        except Exception as e:
            # Capture full traceback for detailed error reporting
            error_details = traceback.format_exc()
            self.update_status.emit(f"Error: {str(e)}\n{error_details}")
            self.finished_signal.emit(False, str(e))

def launch_safe_worker(parent, function, args, callback_func, 
                       
                      progress_callback=None, status_callback=None, 
                      disable_widgets=None, disable_canvas=None, heartbeat_ms=1000):  # Increased from 100
    """
    Universal function to safely launch a worker thread.
    
    Args:
        parent: Parent widget
        function: Function to execute in the thread
        args: Arguments for the function
        callback_func: Function to call when thread finishes
        progress_callback: Function to handle progress updates
        status_callback: Function to handle status updates
        disable_widgets: List of widgets to disable during processing
        disable_canvas: Matplotlib canvas to temporarily remove
        heartbeat_ms: Milliseconds between GUI updates
    
    Returns:
        The worker thread
    """
    # Disable widgets
    if disable_widgets:
        for widget in disable_widgets:
            widget.setEnabled(False)
            
    # Remove canvas from parent if specified
    canvas_parent = None
    canvas_layout = None
    if disable_canvas:
        canvas_parent = disable_canvas.parent()
        # Store parent layout for restoration later
        if canvas_parent:
            canvas_layout = canvas_parent.layout()
            disable_canvas.setParent(None)
    
    # Create a throttled progress callback to reduce UI updates
    def throttled_progress(value):
        if progress_callback:
            # Store last update time and value in parent
            current_time = time.time()
            if (not hasattr(parent, 'last_progress_time') or 
                not hasattr(parent, 'last_progress_value') or 
                current_time - parent.last_progress_time > 1.0 or    # Only update UI every 1 second
                abs(value - parent.last_progress_value) >= 5 or      # Or if progress changes by 5%
                value >= 100 or value == 0):                         # Or at start/end
                parent.last_progress_time = current_time
                parent.last_progress_value = value
                progress_callback(value)
    
    # Create worker thread
    worker = SafeWorkerThread(function, args)
    
    # Connect signals with throttled progress
    if progress_callback:
        worker.update_progress.connect(throttled_progress)
    if status_callback:
        worker.update_status.connect(status_callback)
    
    # Create a wrapped callback that handles cleanup
    def finished_wrapper(success, message):
        # Stop heartbeat timer
        if hasattr(parent, 'heartbeat_timer') and parent.heartbeat_timer.isActive():
            parent.heartbeat_timer.stop()
            
        # Re-enable widgets
        if disable_widgets:
            for widget in disable_widgets:
                widget.setEnabled(True)
                
        # Restore canvas
        if disable_canvas and canvas_parent and canvas_layout:
            # Use a timer to safely restore canvas
            def restore_canvas():
                if disable_canvas.parent() is None:
                    canvas_layout.addWidget(disable_canvas)
                    
            QTimer.singleShot(100, restore_canvas)
        
        # Call the original callback
        callback_func(success, message)
    
    # Connect the wrapped callback
    worker.finished_signal.connect(finished_wrapper)
    
    # Create a heartbeat timer to keep GUI responsive
    parent.heartbeat_timer = QTimer(parent)
    parent.heartbeat_timer.timeout.connect(QApplication.processEvents)
    parent.heartbeat_timer.start(heartbeat_ms)
    
    # Start worker with a slight delay
    QTimer.singleShot(50, worker.start)
    
    return worker


class LoggingWorker:
    """
    A context manager for handling logging during long-running processes.
    
    This provides a clean way to capture and report progress from functions
    that don't explicitly emit signals but do write to logs.
    
    Example:
        with LoggingWorker("Processing files", total=100) as worker:
            for i in range(100):
                # Do some work
                worker.update(i+1)
    """
    def __init__(self, description, total=100):
        self.description = description
        self.total = total
        self.current = 0
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        logging.info(f"Starting {self.description}...")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        if exc_type is not None:
            logging.error(f"Failed {self.description}: {exc_val}")
            logging.error(traceback.format_exc())
        else:
            logging.info(f"Completed {self.description} in {elapsed_time:.2f} seconds")
        
    def update(self, current, additional_info=""):
        """Update progress."""
        self.current = current
        percentage = int(100 * current / self.total)
        if additional_info:
            logging.info(f"Progress: {percentage}% - {current}/{self.total} - {additional_info}")
        else:
            logging.info(f"Progress: {percentage}% - {current}/{self.total}")