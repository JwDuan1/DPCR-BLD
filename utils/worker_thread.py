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
                      disable_widgets=None, disable_canvas=None, heartbeat_ms=1000):  # Increased heartbeat interval
    """
    Universal function to safely launch a worker thread with improved stability.
    """
    # Disable widgets
    if disable_widgets:
        for widget in disable_widgets:
            widget.setEnabled(False)
            
    # Instead of removing canvas, just hide it
    if disable_canvas:
        disable_canvas.setVisible(False)
    
    # Create a more aggressively throttled progress callback
    def throttled_progress(value):
        if progress_callback:
            # Use a more aggressive throttling approach
            current_time = time.time()
            if not hasattr(parent, '_progress_data'):
                parent._progress_data = {'last_time': 0, 'last_value': -1}
                
            # Only update if significant time passed or significant progress change
            if (current_time - parent._progress_data['last_time'] > 2.5 or    # Increased to 2.5 seconds
                abs(value - parent._progress_data['last_value']) >= 10 or     # Increased to 10%
                value >= 100 or value == 0):
                parent._progress_data['last_time'] = current_time
                parent._progress_data['last_value'] = value
                
                # Use QMetaObject.invokeMethod for thread safety if available
                try:
                    from PyQt5.QtCore import QMetaObject, Qt, Q_ARG
                    QMetaObject.invokeMethod(parent, "update_progress", 
                                            Qt.QueuedConnection,
                                            Q_ARG(int, value))
                except (ImportError, AttributeError):
                    # Fallback to direct call if method not available
                    progress_callback(value)
    
    # Create worker thread
    worker = SafeWorkerThread(function, args)
    
    # Connect signals with throttled progress
    if progress_callback:
        worker.update_progress.connect(throttled_progress)
    if status_callback:
        # Create a throttled status callback too
        def throttled_status(message):
            # Only forward critical messages or throttle regular updates
            if "ERROR" in message or "WARNING" in message or not hasattr(parent, '_last_status_time'):
                parent._last_status_time = time.time()
                status_callback(message)
            elif time.time() - parent._last_status_time > 3.0:  # Only update every 3 seconds
                parent._last_status_time = time.time()
                status_callback(message)
        
        worker.update_status.connect(throttled_status)
    
    # Create a wrapped callback that handles cleanup
    def finished_wrapper(success, message):
        # Stop heartbeat timer
        if hasattr(parent, 'heartbeat_timer') and parent.heartbeat_timer.isActive():
            parent.heartbeat_timer.stop()
            
        # Re-enable widgets
        if disable_widgets:
            for widget in disable_widgets:
                if widget and not widget.isDestroyed():  # Check if widget still exists
                    widget.setEnabled(True)
                
        # Show canvas instead of re-adding it
        if disable_canvas and not disable_canvas.isDestroyed():
            disable_canvas.setVisible(True)
        
        # Force garbage collection before callback
        import gc
        gc.collect()
        
        # Call the original callback
        callback_func(success, message)
    
    # Connect the wrapped callback
    worker.finished_signal.connect(finished_wrapper)
    
    # Create a less aggressive heartbeat timer
    parent.heartbeat_timer = QTimer(parent)
    parent.heartbeat_timer.timeout.connect(QApplication.processEvents)
    parent.heartbeat_timer.start(heartbeat_ms)
    
    # Add periodic memory cleanup
    if not hasattr(parent, 'cleanup_timer'):
        parent.cleanup_timer = QTimer(parent)
        def force_cleanup():
            import gc
            gc.collect()
        parent.cleanup_timer.timeout.connect(force_cleanup)
        parent.cleanup_timer.start(30000)  # Run garbage collection every 30 seconds
    
    # Start worker with a slight delay
    QTimer.singleShot(100, worker.start)
    
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