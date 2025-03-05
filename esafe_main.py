"""
esafe_main.py - Launcher for Contouring E-SAFE application

This module implements the main window and shared UI elements for the E-SAFE application.

Authors: 
    Jingwei Duan, Ph.D. (duan.jingwei01@gmail.com), Quan Chen, Ph.D.


Date: March 2025
Version: 1.0
License: MIT License
"""

import os
import sys
import logging
import datetime
import importlib.util
import traceback

# Set up basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def is_package_available(package_name):
    """
    Check if a package can be imported without actually importing it.
    
    Args:
        package_name: Name of the package to check
        
    Returns:
        bool: True if package is available, False otherwise
    """
    try:
        spec = importlib.util.find_spec(package_name)
        return spec is not None
    except (ImportError, AttributeError, ValueError):
        return False

def main():
    """Main entry point for the application."""
    
    # Required packages
    required_packages = [
        'PyQt5', 'numpy', 'pandas', 'matplotlib', 'h5py', 
        'pydicom', 'scipy', 'skimage', 'tqdm','trimesh','rtree','cv2','psutil','pycpd','sklearn'
    ]

    # Check for missing packages
    logging.info("Checking installed packages...")
    missing_packages = []
    for package in required_packages:
        available = is_package_available(package)
        logging.info(f"  - {package}: {'Available' if available else 'Not available'}")
        if not available:
            missing_packages.append(package)

    if missing_packages:
        logging.warning("The following required packages are missing:")
        for package in missing_packages:
            logging.warning(f"  - {package}")
        logging.warning("\nSome functionality may be limited. Please install them with:")
        logging.warning(f"pip install {' '.join(missing_packages)}")
        
        if 'PyQt5' in missing_packages:
            logging.error("PyQt5 is required for the GUI. Cannot continue.")
            sys.exit(1)
        else:
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                sys.exit(1)

    os.makedirs("logs", exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join("logs", f"esafe_{timestamp}.log")
    
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logging.root.addHandler(file_handler)
    
    logging.info("Starting E-SAFE application")

    try:
        # Debugging: Print messages before GUI starts
        logging.info("Step 1: Importing GUI modules...")
        
        from gui.main_window import MainWindow  # Potential error here
        from PyQt5.QtWidgets import QApplication

        logging.info("Step 2: Initializing QApplication...")
        
        app = QApplication(sys.argv)

        logging.info("Step 3: Creating MainWindow instance...")

        window = MainWindow()  # Check for parent argument issue

        logging.info("Step 4: Starting application loop...")
        
        sys.exit(app.exec_())

    except ImportError as e:
        logging.error(f"Error importing required modules: {e}")
        logging.error("Please ensure all required packages are installed.")
        sys.exit(1)

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        
        import traceback
        traceback.print_exc()  # Print full error details to terminal
        logging.error(traceback.format_exc())  # Log full traceback
        
        sys.exit(1)

if __name__ == "__main__":
    main()
