"""
DICOM Structure Set File Processor

This module processes DICOM structure set files (RS*.dcm) and renames them based on their DICOM tags.
The new filename format is: PatientID_StructureSetLabel_RouteID_Date_Time.dcm
"""

import os
import logging
from typing import Optional, Tuple, List
from datetime import datetime
import pydicom
from pydicom.errors import InvalidDicomError

class DicomFileProcessor:
    def __init__(self, base_folder: str):
        self.base_folder = base_folder
        self.invalid_chars = "<>:\"/\\|?*"
        self.total_dcm_files = 0
        self.rs_files_count = 0
        
    def find_all_dcm_files(self) -> List[str]:
        """Find all DCM files in directory and count them."""
        dcm_files = []
        for root, _, files in os.walk(self.base_folder):
            for file in files:
                if file.endswith('.dcm'):
                    dcm_files.append(os.path.join(root, file))
        self.total_dcm_files = len(dcm_files)
        return dcm_files

    def sanitize_filename(self, filename: str) -> str:
        """Remove invalid characters from filename."""
        for char in self.invalid_chars:
            filename = filename.replace(char, '_')
        return filename

    def get_dicom_tags(self, ds: pydicom.dataset.FileDataset) -> Tuple[str, ...]:
        """Extract relevant DICOM tags for filename creation."""
        patient_id = getattr(ds, 'PatientID', '')
        structure_set_label = getattr(ds, 'StructureSetLabel', '')
        # Try fallback options if StructureSetLabel is missing
        if not structure_set_label and hasattr(ds, 'SeriesDescription'):
            structure_set_label = ds.SeriesDescription
            
        route_id = getattr(ds, 'RouteID', '')
        structure_set_date = getattr(ds, 'StructureSetDate', '')
        if not structure_set_date and hasattr(ds, 'SeriesDate'):
            structure_set_date = ds.SeriesDate
            
        structure_set_time = getattr(ds, 'StructureSetTime', '')
        if not structure_set_time and hasattr(ds, 'SeriesTime'):
            structure_set_time = ds.SeriesTime
            
        # Clean up time format
        structure_set_time = structure_set_time.replace(':', '')
        
        return (
            patient_id,
            structure_set_label,
            route_id,
            structure_set_date,
            structure_set_time
        )

    def generate_new_filename(self, ds: pydicom.dataset.FileDataset) -> Optional[str]:
        """Generate a new filename based on DICOM tags."""
        tags = self.get_dicom_tags(ds)
        patient_id, structure_set_label, route_id, set_date, set_time = tags
        
        # Require at least patient ID
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
        """Process a single DICOM file if it's a RTSTRUCT file."""
        base_name = os.path.basename(file_path)
        
        try:
            # Try to read the DICOM header without loading pixel data
            ds = pydicom.dcmread(file_path, force=True, stop_before_pixels=True)
            
            # Check if it's a structure set file by examining the Modality tag
            if not hasattr(ds, 'Modality') or ds.Modality != 'RTSTRUCT':
                return False
                
            # It's a valid RTSTRUCT file
            self.rs_files_count += 1
            logging.info(f"Processing RTSTRUCT file: {base_name}")
        except InvalidDicomError:
            logging.warning(f"Not a valid DICOM file: {base_name}")
            return False
        except Exception as e:
            logging.error(f"Error reading DICOM file {base_name}: {str(e)}")
            return False

        try:
            # Read DICOM file
            ds = pydicom.dcmread(file_path, force=True)
            
            # Generate new filename
            new_filename = self.generate_new_filename(ds)
            
            if not new_filename:
                logging.warning(f"Required DICOM tags missing in {file_path}")
                return True
                
            # Sanitize filename
            new_filename = self.sanitize_filename(new_filename)
            new_path = os.path.join(os.path.dirname(file_path), new_filename)
            
            logging.info(f"Previous name: {base_name}")
            logging.info(f"New name: {new_filename}")
            
            # Rename file if needed
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