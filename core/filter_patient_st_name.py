"""
filter_patient_st_name.py - Extract RTSTRUCT information from DICOM files

This script recursively searches through folders to find RTSTRUCT 
DICOM files and extracts relevant structure information into a CSV file.

Authors: 
    Original MATLAB: Jingwei Duan, Ph.D. (duan.jingwei01@gmail.com), Quan Chen, Ph.D.

    
Date: February 2025
Version: 1.0
License: MIT License
"""

import os
import csv
import logging
import traceback
from typing import List, Optional, TextIO
import pydicom

def filter_patient_st_name(
    in_root_folder: str, 
    out_csv: str, 
    progress_callback: Optional[callable] = None
) -> None:
    """
    Recursively search through folders to find RTSTRUCT DICOM files
    and extract structure information into a CSV file.
    """
    try:
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        logging.info(f"Starting DICOM file search in: {in_root_folder}")
        logging.info(f"Output will be saved to: {out_csv}")
        
        # Validate input paths
        if not os.path.exists(in_root_folder):
            raise ValueError(f"Input folder does not exist: {in_root_folder}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        
        # Call progress callback with initial progress
        if progress_callback:
            progress_callback(0)
        
        # Find all DICOM files
        all_files = []
        for root, _, files in os.walk(in_root_folder):
            for file in files:
                if file.lower().endswith(('.dcm', '.DCM')):
                    all_files.append(os.path.join(root, file))
        
        total_files = len(all_files)
        logging.info(f"Found {total_files} potential DICOM files")
        
        # Identify RTSTRUCT files
        rtstruct_files = []
        processed_files = 0
        
        # First stage: Find RTSTRUCT files
        for file_path in all_files:
            try:
                # Update progress (first 50%)
                processed_files += 1
                if progress_callback:
                    progress = int((processed_files / total_files) * 50)
                    progress_callback(progress)
                
                # Read DICOM header
                try:
                    dicom_info = pydicom.dcmread(file_path, stop_before_pixels=True)
                except Exception as read_error:
                    logging.warning(f"Could not read DICOM header for {file_path}: {read_error}")
                    continue
                
                # Check if RTSTRUCT
                if hasattr(dicom_info, 'Modality') and dicom_info.Modality == 'RTSTRUCT':
                    rtstruct_files.append(file_path)
                    logging.info(f"Found RTSTRUCT file: {file_path}")
            
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")
                logging.error(traceback.format_exc())
        
        logging.info(f"Found {len(rtstruct_files)} RTSTRUCT files")
        
        # Process RTSTRUCT files
        with open(out_csv, 'w', newline='', encoding='utf-8') as csvfile:
            # Write CSV header
            csvfile.write('filepath,structureName,modality,studydate,StudyInstanceUID,ImageSeriesInstanceUID,structures\n')
            
            # Second stage: Extract and write RTSTRUCT details
            for idx, file_path in enumerate(rtstruct_files, 1):
                try:
                    # Update progress (second 50%)
                    if progress_callback:
                        progress = 50 + int((idx / len(rtstruct_files)) * 50)
                        progress_callback(progress)
                    logging.info(f"Processing RTSTRUCT file: {file_path}")
                    # Read full DICOM info
                    dicom_info = pydicom.dcmread(file_path)
                    
                    # Extract structure set name (if available)
                    structure_set_name = ''
                    if hasattr(dicom_info, 'StructureSetName'):
                        structure_set_name = dicom_info.StructureSetName
                    elif hasattr(dicom_info, 'StructureSetLabel'):
                        structure_set_name = dicom_info.StructureSetLabel
                    
                    # Extract individual structure names
                    structure_names = []
                    if hasattr(dicom_info, 'StructureSetROISequence'):
                        structure_names = [
                            roi.ROIName for roi in dicom_info.StructureSetROISequence 
                            if hasattr(roi, 'ROIName')
                        ]
                    
                    # Safely extract metadata
                    modality = getattr(dicom_info, 'Modality', 'Unknown')
                    study_date = getattr(dicom_info, 'StudyDate', 'Unknown')
                    study_uid = getattr(dicom_info, 'StudyInstanceUID', 'Unknown')
                    series_uid = getattr(dicom_info, 'SeriesInstanceUID', 'Unknown')
                    
                    # Get Referenced Series UID
                    image_series_uid = ''
                    if hasattr(dicom_info, 'ReferencedFrameOfReferenceSequence'):
                        for frame_ref in dicom_info.ReferencedFrameOfReferenceSequence:
                            if hasattr(frame_ref, 'RTReferencedStudySequence'):
                                for rt_study in frame_ref.RTReferencedStudySequence:
                                    if hasattr(rt_study, 'RTReferencedSeriesSequence'):
                                        for rt_series in rt_study.RTReferencedSeriesSequence:
                                            if hasattr(rt_series, 'SeriesInstanceUID'):
                                                image_series_uid = rt_series.SeriesInstanceUID
                                                break
                    
                    # Write to CSV
                    safe_structures = [
                        name.replace(',', ',').replace('\n', ' ') 
                        for name in structure_names
                    ]
                    csvfile.write(
                        f'{file_path},{structure_set_name},{modality},{study_date},{study_uid},{image_series_uid},'
                        f'{",".join(safe_structures)}\n'
                    )
                    
                except Exception as e:
                    logging.error(f"Error processing RTSTRUCT file {file_path}: {e}")
                    logging.error(traceback.format_exc())
        
        # Final progress update
        if progress_callback:
            progress_callback(100)
        
        logging.info(f"Processing complete. Results saved to {out_csv}")
    
    except Exception as e:
        logging.error(f"Critical error in filter_patient_st_name: {e}")
        logging.error(traceback.format_exc())
        
        # Ensure progress callback is called in case of error
        if progress_callback:
            progress_callback(0)
        
        raise
 
    
if __name__ == "__main__":
    def print_progress(progress):
        print(f"Progress: {progress}%")
    # Example usage
    filter_patient_st_name(
        r'...\FolderwithAIgeneratedRSdcm', 
        r'...\AI_output_structures.csv',
        progress_callback=print_progress
    )