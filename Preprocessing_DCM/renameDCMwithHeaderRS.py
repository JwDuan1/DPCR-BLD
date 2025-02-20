import os
import glob
import logging
from typing import Optional, Tuple
from datetime import datetime
import pydicom
from pydicom.errors import InvalidDicomError

class DicomFileProcessor:
    def __init__(self, base_folder: str, log_file: Optional[str] = None):
        self.base_folder = base_folder
        self.invalid_chars = "<>:\"/\\|?*"
        self.total_dcm_files = 0
        self.rs_files_count = 0
        self._setup_logging(log_file)
        
    def _setup_logging(self, log_file: Optional[str]) -> None:
        """Configure logging to both file and console."""
        # Create a logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        # Create formatters and handlers
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    def find_all_dcm_files(self):
        """Find all DCM files in directory and count them."""
        dcm_files = []
        for root, _, files in os.walk(self.base_folder):
            for file in files:
                if file.endswith('.dcm'):
                    dcm_files.append(os.path.join(root, file))
        self.total_dcm_files = len(dcm_files)
        return dcm_files

    def sanitize_filename(self, filename: str) -> str:
        for char in self.invalid_chars:
            filename = filename.replace(char, '_')
        return filename

    def get_dicom_tags(self, ds: pydicom.dataset.FileDataset) -> Tuple[str, ...]:
        patient_id = getattr(ds, 'PatientID', '')
        structure_set_label = getattr(ds, 'StructureSetLabel', '')
        route_id = getattr(ds, 'RouteID', '')
        structure_set_date = getattr(ds, 'StructureSetDate', '')
        structure_set_time = getattr(ds, 'StructureSetTime', '').replace(':', '')
        
        return (
            patient_id,
            structure_set_label,
            route_id,
            structure_set_date,
            structure_set_time
        )

    def generate_new_filename(self, ds: pydicom.dataset.FileDataset) -> Optional[str]:
        tags = self.get_dicom_tags(ds)
        patient_id, structure_set_label, route_id, set_date, set_time = tags
        
        if not patient_id:
            return None
            
        components = []
        components.append(patient_id)
        
        if structure_set_label:
            components.append(structure_set_label)
        if route_id:
            components.append(route_id)
        if set_date:
            components.append(set_date)
        if set_time:
            components.append(set_time)
            
        if len(components) > 1:
            return f"{'_'.join(components)}.dcm"
        return None

    def process_file(self, file_path: str) -> bool:
        base_name = os.path.basename(file_path)
        if not base_name.startswith('RS'):
            return False

        self.rs_files_count += 1
        logging.info(f"\nProcessing RS file {self.rs_files_count} of {self.total_dcm_files} total DCM files")
        logging.info(f"Current file: {base_name}")

        try:
            ds = pydicom.dcmread(file_path, force=True)
            new_filename = self.generate_new_filename(ds)
            
            if not new_filename:
                logging.warning(f"Required DICOM tags missing in {file_path}")
                return True
                
            new_filename = self.sanitize_filename(new_filename)
            new_path = os.path.join(os.path.dirname(file_path), new_filename)
            
            logging.info(f"Previous name: {base_name}")
            logging.info(f"New name: {new_filename}")
            
            if os.path.normcase(file_path) != os.path.normcase(new_path):
                if not os.path.exists(new_path):
                    os.rename(file_path, new_path)
                    logging.info(f'Successfully renamed: {base_name} -> {new_filename}')
                else:
                    logging.warning(f'Target file already exists: {new_filename}')
            else:
                logging.info(f'File already has correct name: {base_name}')
                
        except InvalidDicomError as e:
            logging.error(f'Invalid DICOM file: {base_name}. Error: {str(e)}')
        except Exception as e:
            logging.error(f'Error processing {base_name}: {str(e)}')
            
        return True

    def process_directory(self) -> None:
        start_time = datetime.now()
        logging.info(f"Starting DICOM processing in: {self.base_folder}")
        
        # First find all DCM files
        all_dcm_files = self.find_all_dcm_files()
        logging.info(f"Found {self.total_dcm_files} DCM files in total")
        
        processed_count = 0
        error_count = 0
        
        for file_path in all_dcm_files:
            try:
                if self.process_file(file_path):
                    processed_count += 1
            except Exception as e:
                error_count += 1
                logging.error(f"Unexpected error processing {file_path}: {str(e)}")
        
        duration = datetime.now() - start_time
        logging.info("\nProcessing Summary:")
        logging.info(f"Total DCM files found: {self.total_dcm_files}")
        logging.info(f"Files starting with RS: {self.rs_files_count}")
        logging.info(f"Processing completed in: {duration}")
        logging.info(f"Files processed: {processed_count}")
        logging.info(f"Errors encountered: {error_count}")

def main():
    # Define the input folder
    infolder = r'\\hnas1-dpts\Radiation Oncology\Physicists\Residents\Duan\Researches\Data\GithubTest\AI'
    
    # Create log file in the same directory as the script
    log_file = os.path.join(os.path.dirname(__file__), 'Rename_RSdicom_processing.log')
    
    # Initialize and run the processor
    processor = DicomFileProcessor(infolder, log_file)
    processor.process_directory()

if __name__ == "__main__":
    main()