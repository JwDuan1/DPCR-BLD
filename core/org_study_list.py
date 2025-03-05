"""
org_study_list.py - Organize and match DICOM structure names with lookup table

This script reads DICOM RTSTRUCT files and matches structure names with 
a provided lookup table, creating a CSV report of matched structures.

Authors: 
    Original MATLAB: Jingwei Duan, Ph.D. (duan.jingwei01@gmail.com), Quan Chen, Ph.D.
    
Date: February 2025
Version: 1.0
License: MIT License
"""

import os
import csv
import logging
import pandas as pd
import pydicom
from pydicom.errors import InvalidDicomError
from typing import List, Dict, Optional, Any


def org_study_list(root_folder: str, lookup_file: str, list_file: str) -> None:
    """
    Read DICOM RTSTRUCT files and match structure names with a lookup table,
    creating a CSV report of matched structures.
    
    Args:
        root_folder: Directory containing RTSTRUCT DICOM files
        lookup_file: Path to CSV file containing structure name lookup table
        list_file: Output CSV file path for matched structures
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logging.info(f"Processing RTSTRUCT files in: {root_folder}")
    logging.info(f"Using lookup table: {lookup_file}")
    
    # Read lookup table
    logging.info("Reading lookup table...")
    lookup_table = pd.read_csv(lookup_file, header=None)
    num_structures, num_aliases = lookup_table.shape
    
    # Get file list
    file_list = [f for f in os.listdir(root_folder) 
                if os.path.isfile(os.path.join(root_folder, f)) and f.lower().endswith('.dcm')]
    
    logging.info(f"Found {len(file_list)} DICOM files")
    
    # Open output file
    try:
        with open(list_file, 'w', newline='') as outfile:
            # Write header
            outfile.write('Fname,StudyInstanceUID,ImageSeriesInstanceUID,RTSTRUCTInstanceUID')
            
            # Write structure name headers from lookup table
            for i in range(num_structures):
                outfile.write(f',{lookup_table.iloc[i, 0]}')
            outfile.write('\n')
            
            # Process each RTSTRUCT file
            for filename in file_list:
                try:
                    full_path = os.path.join(root_folder, filename)
                    logging.info(f"Processing: {filename}")
                    
                    # Read DICOM info
                    dicom_info = pydicom.dcmread(full_path, force=True)
                    
                    # Skip if not RTSTRUCT
                    if not hasattr(dicom_info, 'Modality') or dicom_info.Modality != 'RTSTRUCT':
                        logging.info(f"Skipping non-RTSTRUCT file: {filename}")
                        continue
                    
                    # Extract ROI names
                    roi_names = extract_roi_names(dicom_info)
                    
                    # Write file information
                    outfile.write(f'{full_path},{dicom_info.StudyInstanceUID},')
                    
                    # Get referenced series UID from ReferencedFrameOfReferenceSequence
                    referenced_series_uid = ""
                    if hasattr(dicom_info, 'ReferencedFrameOfReferenceSequence'):
                        ref_frame = dicom_info.ReferencedFrameOfReferenceSequence[0]
                        if hasattr(ref_frame, 'RTReferencedStudySequence'):
                            rt_study = ref_frame.RTReferencedStudySequence[0]
                            if hasattr(rt_study, 'RTReferencedSeriesSequence'):
                                rt_series = rt_study.RTReferencedSeriesSequence[0]
                                referenced_series_uid = rt_series.SeriesInstanceUID
                    
                    outfile.write(f'{referenced_series_uid},{dicom_info.SeriesInstanceUID}')
                    
                    # Match and write structure names
                    for i in range(num_structures):
                        found = False
                        
                        for j in range(num_aliases):
                            if j >= lookup_table.shape[1]:
                                break
                                
                            name = lookup_table.iloc[i, j]
                            if pd.isna(name):
                                break
                                
                            # Check for matching structure name
                            if name in roi_names:
                                outfile.write(f',{name}')
                                found = True
                                break
                        
                        if not found:
                            outfile.write(', ')
                    
                    outfile.write('\n')
                    
                except InvalidDicomError:
                    logging.error(f"Invalid DICOM file: {filename}")
                except Exception as e:
                    logging.error(f"Error processing file {filename}: {e}")
            
        logging.info(f"Processing complete. Results saved to {list_file}")
                    
    except IOError as e:
        logging.error(f"Error opening output file {list_file}: {e}")
        raise


def extract_roi_names(dicom_info: pydicom.dataset.FileDataset) -> List[str]:
    """
    Extract ROI names from DICOM structure.
    
    Args:
        dicom_info: DICOM dataset
        
    Returns:
        List of ROI names
    """
    roi_names = []
    
    if hasattr(dicom_info, 'StructureSetROISequence'):
        for roi in dicom_info.StructureSetROISequence:
            roi_names.append(roi.ROIName)
    
    return roi_names


def dicom_recursive_finder(in_folder: str, mod_tag: str) -> List[str]:
    """
    Recursively find DICOM files of specified modality.
    
    Args:
        in_folder: Path to search directory
        mod_tag: DICOM modality to search for (e.g., 'RTSTRUCT')
        
    Returns:
        List of matching file paths
    """
    matching_files = []
    
    for root, dirs, files in os.walk(in_folder):
        for file in files:
            if file.lower().endswith('.dcm'):
                try:
                    file_path = os.path.join(root, file)
                    dicom_info = pydicom.dcmread(file_path, force=True)
                    
                    if hasattr(dicom_info, 'Modality') and dicom_info.Modality == mod_tag:
                        matching_files.append(file_path)
                        
                except Exception as e:
                    logging.warning(f"Error reading file {file}: {e}")
    
    return matching_files


if __name__ == "__main__":
    # Example usage
    org_study_list(
        r'...\FolderwithAIgeneratedRSdcm', 
        r'...\LookupList.csv', 
        r'...\AImatched_list.csv'
    )