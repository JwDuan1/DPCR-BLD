"""
bld_batch.py - Evaluate Medical Image Segmentation Performance

This module compares test and reference DICOM structure sets and calculates
various metrics including Dice score, Hausdorff distance, and surface Dice.
Also, it generates the BLD calculation on Ref-test pairs.

Authors: 
    Original MATLAB: Jingwei Duan, Ph.D. (duan.jingwei01@gmail.com), Quan Chen, Ph.D.

    
Date: February 2025
Version: 1.0
License: MIT License
"""

import os
import sys
import pandas as pd
import numpy as np
import pydicom
import logging
import datetime
import csv
import traceback
from typing import Tuple, List, Dict, Optional, Any, Callable
import scipy.ndimage
from scipy.spatial import distance_matrix
import h5py
from PyQt5.QtWidgets import QApplication
# Import local modules

try:
    from core.hausdorff_distance import hausdorff_dist_pctile, bidirectional_local_distance
except ImportError:
    from hausdorff_distance import hausdorff_dist_pctile, bidirectional_local_distance



def bld_batch(reflist: str, tstlist: str, resultfile: str, bldpath: str, 
              winsuffix: str = '', oar_name: str = None, 
              progress_callback: Callable[[int], None] = None,
              status_callback: Callable[[str], None] = None) -> None:
    """
    Compare reference and test contours, calculating various similarity metrics.
    
    Args:
        reflist: Path to CSV file containing reference (e.g., Manual) structure list
        tstlist: Path to CSV file containing test (e.g., AI) structure list
        resultfile: Path to output CSV file for metrics
        bldpath: Path to save intermediate results
        winsuffix: Optional prefix for Windows path (default: '')
        oar_name: Optional name of a specific OAR to process (default: None, which processes all)
        progress_callback: Optional callback function to report progress (0-100)
        status_callback: Optional callback function to report status messages
    """
    # Create output directory
    os.makedirs(bldpath, exist_ok=True)
    
    # Setup logging to both file and console
    log_file = os.path.join(bldpath, 'bld_batch_log.txt')
    setup_logging(log_file)
    
    # Custom dual_log function that can also update GUI
    def log_message(message):
        dual_log(message)
        if status_callback:
            status_callback(message)
    
    try:
        log_message('=== Starting BLD_Batch processing ===')
        if oar_name:
            log_message(f'Processing single OAR: {oar_name}')
        else:
            log_message('Processing all OARs (batch mode)')
            
        log_message(f'Reference list: {reflist}')
        log_message(f'Test list: {tstlist}')
        log_message(f'Result file: {resultfile}')
        
        # Read input tables
        log_message('Reading input tables...')
        ref_table = pd.read_csv(reflist)
        tst_table = pd.read_csv(tstlist)
        
        # Get table dimensions
        num_non_strs = 4  # First 4 columns are not structures
        num_strs = tst_table.shape[1] - num_non_strs
        
        # If processing single OAR, find its column index
        oar_index = None
        if oar_name:
            for i in range(num_non_strs, tst_table.shape[1]):
                if tst_table.columns[i] == oar_name:
                    oar_index = i - num_non_strs
                    break
            
            if oar_index is None:
                log_message(f'ERROR: OAR {oar_name} not found in test table')
                return
        
        # Define surface Dice thresholds
        surface_dice_threshold_in_cm = np.arange(0.05, 0.65, 0.05)
        num_surface_dice_threshold_in_cm = len(surface_dice_threshold_in_cm)
        
        # Initialize metric labels
        ind_labels = ['dice', 'recall', 'precision']
        for i in range(90, 101):
            ind_labels.append(f'hausdorff{i}(cm)')
        ind_labels.append('meanSurDist(cm)')
        for i in range(num_surface_dice_threshold_in_cm):
            ind_labels.append(f'SurfaceDice{surface_dice_threshold_in_cm[i]*10:.1f}mm')
        for label in ['tstNumObjects', 'refNumObjects', 'COMDistance(cm)']:
            ind_labels.append(label)
        
        # Determine which structures to process
        structure_indices = [oar_index] if oar_index is not None else range(num_strs)
        
        # Open results file and write headers
        with open(resultfile, 'w', newline='') as fp:
            # Write header
            fp.write('tstfname,reffname')
            
            if oar_name:
                # Single OAR mode - write headers only for this OAR
                for label in ind_labels:
                    fp.write(f',{oar_name}_{label}')
            else:
                # Batch mode - write headers for all OARs
                for i in range(num_strs):
                    st_name = tst_table.columns[num_non_strs + i]
                    for label in ind_labels:
                        fp.write(f',{st_name}_{label}')
            
            fp.write('\n')
            
            # Process each case in chunks
            total_cases = tst_table.shape[0]
            log_message(f'Beginning case processing ({total_cases} total cases)')
            
            # Process in chunks to avoid memory issues
            chunk_size = 1  # Process 5 cases at a time
            for chunk_start in range(0, total_cases, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total_cases)
                log_message(f'Processing chunk {chunk_start//chunk_size + 1}: cases {chunk_start+1}-{chunk_end}')
                
                # Process each case in this chunk
                for i in range(chunk_start, chunk_end):
                    fname = tst_table.iloc[i, 0]
                    log_message(f'-> Processing case {i+1}/{total_cases}: {fname}')
                    
                    # Find matching reference
                    found = -1
                    for j in range(ref_table.shape[0]):
                        if ref_table.iloc[j, 2] == tst_table.iloc[i, 2]:  # Compare StudyInstanceUID
                            found = j
                            break
                            
                    if found == -1:
                        log_message(f'WARNING: No matching reference found for: {fname}')
                        continue
                        
                    # Load DICOM info
                    ref_fname = ref_table.iloc[found, 0]
                    try:
                        # Construct DICOM paths
                        if winsuffix:
                            tst_dicom_path = winsuffix + fname[5:]
                            ref_dicom_path = winsuffix + ref_fname[5:]
                        else:
                            tst_dicom_path = fname
                            ref_dicom_path = ref_fname
                            
                        log_message(f'   Loading test DICOM: {tst_dicom_path}')
                        log_message(f'   Loading reference DICOM: {ref_dicom_path}')
                        
                        info_tst = pydicom.dcmread(tst_dicom_path)
                        info_ref = pydicom.dcmread(ref_dicom_path)
                        
                        log_message('   Successfully loaded both DICOM files')
                        
                        # Write case identifiers first
                        fp.write(f'{fname}, {ref_fname} ')
                        
                        # Initialize metric storage arrays
                        metrics_values = []
                        
                        # Process each structure
                        for idx in structure_indices:
                            k = idx + num_non_strs
                            st_name = tst_table.columns[k]
                            log_message(f'   Processing structure: {st_name}')
                            
                            # Get structure data
                            struct_tst = None
                            struct_ref = None
                            
                            tst_name = tst_table.iloc[i, k]
                            ref_name = ref_table.iloc[found, k]
                            
                            if pd.notna(tst_name) and pd.notna(ref_name):
                                try:
                                    struct_tst = get_struct_by_name(info_tst, tst_name)
                                    struct_ref = get_struct_by_name(info_ref, ref_name)
                                except Exception as e:
                                    log_message(f'      Error getting structures: {str(e)}')
                                    continue
                            
                            # Default metrics for empty structures
                            default_metrics = [np.nan, np.nan, np.nan] + [np.nan]*11 + [np.nan] + \
                                              [np.nan]*num_surface_dice_threshold_in_cm + \
                                              [np.nan, np.nan, np.nan]
                            
                            # Handle empty structures
                            if not struct_ref or len(struct_ref) == 0:
                                log_message('      Empty reference structure')
                                metrics_values.extend(default_metrics)
                                continue
                                
                            if not struct_tst or len(struct_tst) == 0:
                                log_message('      Empty test structure')
                                metrics_values.extend(default_metrics)
                                continue
                            
                            # Determine structure type
                            flag = determine_structure_type(st_name)
                            
                            try:
                                # Compute overlap metrics
                                metrics = compute_overlap(
                                    struct_ref, struct_tst, flag, surface_dice_threshold_in_cm, 
                                    bldpath, st_name, info_ref.PatientID,
                                    status_callback=status_callback
                                )
                                
                                # Compute and format metrics
                                inter, refonly, tstonly = metrics['inter'], metrics['refonly'], metrics['tstonly']
                                
                                # Dice
                                dice = 2 * inter / (2 * inter + refonly + tstonly)
                                tpr = inter / (inter + refonly)
                                ppv = inter / (inter + tstonly)
                                
                                # Combine metrics
                                case_metrics = [
                                    dice, 
                                    tpr, 
                                    ppv
                                ]
                                
                                # Add Hausdorff distances
                                case_metrics.extend(metrics['hd'])
                                
                                # Add mean surface distance
                                case_metrics.append(metrics['mean_d'])
                                
                                # Add surface Dice
                                case_metrics.extend(metrics['surface_dice'])
                                
                                # Add number of objects and center distance
                                case_metrics.extend([
                                    metrics['tst_num_objects'], 
                                    metrics['ref_num_objects'], 
                                    metrics['center_distance']
                                ])
                                
                                metrics_values.extend(case_metrics)
                                
                                log_message('      Metrics computed successfully')
                                import gc
                                gc.collect()
                            except Exception as e:
                                log_message(f'      Error computing metrics: {str(e)}')
                                metrics_values.extend(default_metrics)
                        
                        # Write metrics values
                        format_str = ','.join(['%f'] * len(metrics_values))
                        fp.write(',' + format_str % tuple(metrics_values))
                        fp.write('\n')
                        fp.flush()  # Ensure writing after each case
                        
                        log_message('   Case processing completed')
                        import gc
                        gc.collect()  # Force GC after every case
                        if i % 1 == 0:  # Process events every other case
                            QApplication.processEvents()
                        if i % 10 == 0:  # Every 10 cases
                            import psutil
                            mem = psutil.Process().memory_info().rss / (1024 * 1024)
                            log_message(f"Memory usage: {mem:.1f} MB")
                    except Exception as e:
                        log_message(f'ERROR: Failed to load DICOM data - {str(e)}')
                        continue
                
                # Force garbage collection after each chunk
                import gc
                gc.collect()
                
                # Allow GUI to process events between chunks
                if progress_callback:
                    progress_callback(int(100 * chunk_end / total_cases))
                QApplication.processEvents()
        
        # Final progress update
        if progress_callback:
            progress_callback(100)
            
        log_message('=== Processing completed successfully ===')
        
        # Return log file path
        return log_file
                    
    except Exception as e:
        log_message(f'ERROR: {str(e)}')
        log_message(f'Stack trace: {traceback.format_exc()}')
        raise


def compute_overlap(struct_ref, struct_tst, flag, surface_dice_threshold_in_cm, bld_path, st_name, mrn, 
                   status_callback=None):
    """
    Compute overlap metrics between reference and test structures.
    Adjusted to match MATLAB implementation.
    """
    # Helper function for status updates
    def update_status(message):
        if status_callback:
            status_callback(message)
    
    update_status(f"Analyzing structure extents...")
    # Analyze structures to get extent info
    ref_mins, ref_maxs, ref_slice_thickness, struct_ref = analy_struct(struct_ref)
    tst_mins, tst_maxs, tst_slice_thickness, struct_tst = analy_struct(struct_tst)
    
    num_sd_sc_threshold_in_cm = len(surface_dice_threshold_in_cm)
    
    # Determine volume extents - EXACTLY match MATLAB calculation
    all_mins = np.minimum(ref_mins, tst_mins)
    all_maxs = np.maximum(ref_maxs, tst_maxs)
    
    # Adjust z extents based on structure type
    if flag.lower() == 'refonly':  # Use lowercase comparison like MATLAB's strcmpi
        all_mins[2] = ref_mins[2]
        all_maxs[2] = ref_maxs[2]
    elif flag.lower() == 'overlap':
        all_mins[2] = max(ref_mins[2], tst_mins[2])
        all_maxs[2] = min(ref_maxs[2], tst_maxs[2])
    
    # Setup header for mask computation - EXACTLY match MATLAB
    header = {
        'x_pixdim': 1/10,
        'y_pixdim': 1/10,
        'z_pixdim': max(ref_slice_thickness, tst_slice_thickness)/10
    }
    
    # Handle case where slice thickness is unknown
    if header['z_pixdim'] == 0:
        header['z_pixdim'] = 0.1
    
    # Setup coordinate system
    header['x_start'] = all_mins[0]/10
    header['y_start'] = all_mins[1]/10
    header['z_start'] = -all_maxs[2]/10
    
    # Setup dimensions - EXACTLY match MATLAB's ceil function and spacing
    header['x_dim'] = int(np.ceil((all_maxs[0] - all_mins[0])/10/header['x_pixdim'])) + 2
    header['y_dim'] = int(np.ceil((all_maxs[1] - all_mins[1])/10/header['y_pixdim'])) + 2
    header['z_dim'] = int(np.ceil((all_maxs[2] - all_mins[2])/10/header['z_pixdim'])) + 1
    
    # update_status(f"Computing reference mask...")
    # Compute masks - ensure compute_mask_from_struct is correctly implemented
    ref_mask = compute_mask_from_struct(struct_ref, header, 'HFS')
    B_ref, _ = bwboundaries_2d_stack_fast(ref_mask)

    # update_status(f"Computing test mask...")
    tst_mask = compute_mask_from_struct(struct_tst, header, 'HFS')
    B_tst, _ = bwboundaries_2d_stack_fast(tst_mask)
    
    # Check if test mask is empty (match MATLAB error handling)
    if not B_tst:
        return {
            'inter': np.nan,
            'refonly': np.nan,
            'tstonly': np.nan,
            'union': np.nan,
            'hd': np.ones(11) * np.nan,  # Match MATLAB's ones(1,11)*NaN
            'mean_d': np.nan,
            'surface_dice': np.ones(num_sd_sc_threshold_in_cm) * np.nan,
            'ref_num_objects': np.nan,
            'tst_num_objects': np.nan,
            'center_distance': np.nan
        }
    
    # Extract boundary points in the EXACT same way
    # update_status(f"Extracting boundary points...")
    ref_pts = []
    for boundary in B_ref:
        ref_pts.append(boundary)
    
    if not ref_pts:
        raise ValueError("No reference boundary points found")
    ref_pts = np.vstack(ref_pts)
    
    tst_pts = []
    for boundary in B_tst:
        tst_pts.append(boundary)
    tst_pts = np.vstack(tst_pts)
    
    # Store original points
    raw_ref_pts = ref_pts.copy()
    raw_tst_pts = tst_pts.copy()
    
    # Scale points by pixel dimensions - EXACT match to MATLAB
    num_ref_pts = ref_pts.shape[0]
    num_tst_pts = tst_pts.shape[0]
    
    # Match MATLAB's scaling exactly:
    # refpts = refpts .* (ones(numrefpts,1) * [header.x_pixdim, header.y_pixdim, header.z_pixdim]);
    ref_pts = ref_pts * np.tile([header['x_pixdim'], header['y_pixdim'], header['z_pixdim']], (num_ref_pts, 1))
    tst_pts = tst_pts * np.tile([header['x_pixdim'], header['y_pixdim'], header['z_pixdim']], (num_tst_pts, 1))
    
    # Calculate Hausdorff metrics
    # update_status(f"Calculating Hausdorff distances...")
    percentiles = list(range(90, 101))  # Match MATLAB's 90:100
    hd_result, mean_d, _, dp, dq = hausdorff_dist_pctile(ref_pts, tst_pts, percentiles)
    
    # Count connected components
    # update_status(f"Counting connected components...")
    # Match MATLAB's bwconncomp exactly
    cc_ref = scipy.ndimage.label(ref_mask)[1]  # Only need the count
    cc_tst = scipy.ndimage.label(tst_mask)[1]
    
    # Find centers of mass
    # update_status(f"Finding centers of mass...")
    ref_center = find_mass_center(ref_mask)
    ref_mask_center = ref_center * np.array([header['x_pixdim'], header['y_pixdim'], header['z_pixdim']])
    
    tst_center = find_mass_center(tst_mask)
    tst_mask_center = tst_center * np.array([header['x_pixdim'], header['y_pixdim'], header['z_pixdim']])
    
    # Calculate center distance - use pdist2 equivalent for exact match
    center_distance = np.sqrt(np.sum((tst_mask_center - ref_mask_center)**2))
    
    # Calculate surface Dice at different thresholds
    # update_status(f"Calculating Surface Dice metrics...")
    surface_dice_values = np.zeros(num_sd_sc_threshold_in_cm)
    for i in range(num_sd_sc_threshold_in_cm):
        threshold = surface_dice_threshold_in_cm[i]
        # EXACT match to MATLAB: sum(dp<threshold) + sum(dq<threshold)) / (numel(dp) + numel(dq))
        surface_dice_values[i] = (np.sum(dp < threshold) + np.sum(dq < threshold)) / (len(dp) + len(dq))
    
    # Check which reference points are inside test structure
    # update_status(f"Computing reference points inside test mask...")
    in_mask = ref_inside_tst_pts(tst_mask, raw_ref_pts)
    
    # Calculate BLD
    # update_status(f"Calculating Bidirectional Local Distance...")
    bld_values = bidirectional_local_distance(ref_pts, tst_pts)
    
    # Points outside the test mask get negative BLD values - EXACTLY match MATLAB
    bld_values[~in_mask] = -1 * bld_values[~in_mask]
    
    # Store reference points with BLD values
    ref_pts_with_bld = np.column_stack((ref_pts, bld_values))
    
    # Compute volumetric overlap - match MATLAB's bitset approach
    # update_status(f"Computing volumetric overlap...")
    bit_mask = np.zeros(ref_mask.shape, dtype=np.uint8)  # Same as MATLAB's uint8
    
    # Mimic MATLAB's bitset function
    bit_mask[ref_mask] |= 1  # Sets bit 1 where ref_mask is True
    bit_mask[tst_mask] |= 2  # Sets bit 2 where tst_mask is True
    
    # Calculate overlap statistics EXACTLY as in MATLAB
    inter = np.sum(bit_mask == 3)  # Both ref and test
    refonly = np.sum(bit_mask == 1)  # Only ref
    tstonly = np.sum(bit_mask == 2)  # Only test
    union = np.sum(bit_mask > 0)  # Either ref or test
    
    # Save results - match MATLAB's data
    update_status(f"Saving results...")
    output_file = os.path.join(bld_path, f'{st_name}_{mrn}.mat')
    save_variables(output_file, {
        'refptswithbld': ref_pts_with_bld,
        'refmask': ref_mask,
        'tstmask': tst_mask,
        'header': header,
        'tstpts': tst_pts,
        'refpts': ref_pts,
        'rawtstpts': raw_tst_pts,
        'rawrefpts': raw_ref_pts
    })
    
    # Return metrics in the same format as MATLAB would
    return {
        'inter': inter,
        'refonly': refonly,
        'tstonly': tstonly,
        'union': union,
        'hd': hd_result,
        'mean_d': mean_d,
        'surface_dice': surface_dice_values,
        'ref_num_objects': cc_ref,
        'tst_num_objects': cc_tst,
        'center_distance': center_distance
    }


def find_mass_center(mask):
    """
    Find the center of mass of a binary mask.
    Implemented to match MATLAB's precise calculation.
    """
    if not np.any(mask):
        return np.zeros(3)
        
    # Create coordinate arrays
    y_coords, x_coords, z_coords = np.nonzero(mask)
    
    # Calculate center of mass (match MATLAB's computation exactly)
    total = np.sum(mask)
    center_y = np.sum(y_coords) / total
    center_x = np.sum(x_coords) / total
    center_z = np.sum(z_coords) / total
    
    return np.array([center_y, center_x, center_z])


def dual_log(message: str) -> None:
    """
    Write timestamped message to both log file and console.
    
    Args:
        message: Message to log
    """
    logging.info(message)


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

def determine_structure_type(st_name: str) -> str:
    """
    Determine structure type based on name.
    
    Args:
        st_name: Structure name
        
    Returns:
        Structure type ('overlap', 'refonly', or 'default')
    """
    # Special cases with specific handling requirements
    overlap_structures = [
        'SpinalCord1', 'Rectum1', 'FemoralHead_R', 'FemoralHead_L',
        'SpinalCanal', 'Aorta', 'V_Venacava_I'
    ]
    
    refonly_structures = ['Esophagus']
    
    if st_name in overlap_structures:
        return 'overlap'
    elif st_name in refonly_structures:
        return 'refonly'
    else:
        return 'default'


def compute_mask_from_struct(struct1, header, pos='HFS'):
    """
    Compute a 3D mask from a structure's contour data.
    
    Args:
        struct1 (Sequence): Contour sequence from DICOM RTSTRUCT
        header (dict): Image header with dimensions and spacing
        pos (str, optional): Patient position. Defaults to 'HFS' (Head First Supine)
    
    Returns:
        numpy.ndarray: 3D binary mask of the structure
    """
    # Create empty mask - initialize as zeros to match MATLAB behavior
    mask = np.zeros((header['x_dim'], header['y_dim'], header['z_dim']), dtype=bool)
    
    # Skip if struct1 is empty
    if not struct1:
        return mask
    
    prev_z = -100000  # Track previous slice position
    
    # Process each contour
    for item in struct1:
        isxor = 0
        
        # Skip if item doesn't have required attributes
        if not hasattr(item, 'NumberOfContourPoints') or not hasattr(item, 'ContourData'):
            continue
            
        num_points = item.NumberOfContourPoints
        
        # Extract contour data as (N,3) array
        contour_data = np.array(item.ContourData).reshape(-1, 3)
        
        # Check if this contour is on the same slice as the previous
        if abs(contour_data[0, 2] - prev_z) < 1e-5:
            isxor = 1
            
        prev_z = contour_data[0, 2]
        
        # Convert physical coordinates to image coordinates
        # IMPORTANT: Transpose to match MATLAB's 3×N format
        pts = _physical_to_image_coords(contour_data.T, header, pos)
        
        # Get z-index (use exact value, not rounded)
        z_idx = int(pts[2, 0])
        
        # Skip if slice is out of range
        if z_idx < 1 or z_idx > header['z_dim']:
            continue
        
        # Create 2D mask for this slice
        poly_mask = _poly2mask(pts[0, :], pts[1, :], header['x_dim'], header['y_dim'])
        
        # Apply XOR if needed (for handling overlapping contours on same slice)
        if isxor:
            mask[:, :, z_idx-1] = np.logical_xor(mask[:, :, z_idx-1], poly_mask)
        else:
            # Direct assignment like MATLAB - not logical OR
            mask[:, :, z_idx-1] = poly_mask
    
    return mask


def _physical_to_image_coords(pts, header, pos):
    """
    Convert physical coordinates to image coordinates.
    
    Args:
        pts (numpy.ndarray): Physical coordinates in 3×N format
                             (row 0=x, row 1=y, row 2=z)
        header (dict): Image header information
        pos (str): Patient position (HFS, HFP, FFP, FFS)
    
    Returns:
        numpy.ndarray: Image coordinates in 3×N format
    """
    # Initialize the output points array
    pti = np.zeros_like(pts)
    
    # Use MATLAB-like rounding behavior
    def matlab_round(x):
        return np.floor(x + 0.5)
    
    # Coordinate conversion based on patient position
    if pos == 'HFS':
        # EXACTLY match MATLAB code:
        # Z coordinate: Map from physical space to slice index
        pti[2, :] = matlab_round(-1 * (pts[2, :]/10 + header['z_start']) / header['z_pixdim']) + 1
        # X coordinate
        pti[0, :] = matlab_round((pts[0, :]/10 - header['x_start']) / header['x_pixdim']) + 1
        # Y coordinate
        pti[1, :] = matlab_round((pts[1, :]/10 - header['y_start']) / header['y_pixdim']) + 1
    elif pos == 'HFP':
        pti[2, :] = matlab_round(-1 * (pts[2, :]/10 + header['z_start']) / header['z_pixdim']) + 1
        pti[0, :] = matlab_round(header['x_dim'] - (pts[0, :]/10 - header['x_start']) / header['x_pixdim'])
        pti[1, :] = matlab_round(header['y_dim'] - (pts[1, :]/10 - header['y_start']) / header['y_pixdim'])
    elif pos == 'FFP':
        pti[2, :] = matlab_round(-1 * (pts[2, :]/10 + header['z_start']) / header['z_pixdim']) + 1
        pti[0, :] = matlab_round((pts[0, :]/10 - header['x_start']) / header['x_pixdim']) + 1
        pti[1, :] = matlab_round(header['y_dim'] - (pts[1, :]/10 - header['y_start']) / header['y_pixdim'])
    elif pos == 'FFS':
        pti[2, :] = matlab_round(-1 * (pts[2, :]/10 + header['z_start']) / header['z_pixdim']) + 1
        pti[0, :] = matlab_round((-1 * pts[0, :]/10 - header['x_start']) / header['x_pixdim']) + 1
        pti[1, :] = matlab_round((pts[1, :]/10 - header['y_start']) / header['y_pixdim']) + 1
    else:
        raise ValueError(f"Unsupported patient position: {pos}")
    
    return pti


def _poly2mask(xs, ys, width, height):
    """
    Create a binary mask from polygon coordinates.
    
    Args:
        xs (numpy.ndarray): X coordinates of polygon vertices
        ys (numpy.ndarray): Y coordinates of polygon vertices
        width (int): Mask width
        height (int): Mask height
    
    Returns:
        numpy.ndarray: Binary mask
    """
    # Create empty mask with shape (height, width) for OpenCV
    mask_cv = np.zeros((height, width), dtype=np.uint8)
    
    # Convert coordinates to integers with MATLAB-like rounding
    xs_int = np.floor(xs + 0.5).astype(np.int32)
    ys_int = np.floor(ys + 0.5).astype(np.int32)
    
    # Clip to ensure coordinates are within bounds
    xs_int = np.clip(xs_int, 0, width-1)
    ys_int = np.clip(ys_int, 0, height-1)
    
    # Create array of polygon vertices for OpenCV
    vertices = np.column_stack((xs_int, ys_int))
    
    # Ensure polygon is closed (first and last points are the same)
    if vertices.shape[0] > 1 and not np.array_equal(vertices[0], vertices[-1]):
        vertices = np.vstack((vertices, vertices[0:1]))
    
    # Fill the polygon
    if len(vertices) > 2:  # Need at least 3 points to create a polygon
        import cv2
        cv2.fillPoly(mask_cv, [vertices], 1)
    
        # Draw all boundary lines to ensure periphery points are included
        for i in range(len(vertices) - 1):
            cv2.line(mask_cv, tuple(vertices[i]), tuple(vertices[i+1]), 1, thickness=1)
    
    # Transpose to get mask with shape (width, height)
    mask = mask_cv.T
    
    # Convert to boolean
    return mask.astype(bool)


def analy_struct(instruct):
    """
    Analyze structure contour data.
    
    Args:
        instruct (dict): Structure contour sequence
    
    Returns:
        tuple: (mins, maxs, slice_thickness, updated_instruct)
    """
    # Get fieldnames 
    if not instruct:
        return None, None, 0, {}
    
    # Initialize variables
    slice_z = []
    mins = None
    maxs = None
    
    # Iterate through contour items
    for item in instruct:
        # Skip empty items
        if not item:
            continue
        
        # Extract number of points
        num_points = item.NumberOfContourPoints
        
        # Skip if fewer than 3 points
        if num_points < 3:
            continue
        
        # Extract contour data and reshape
        pts = np.array(item.ContourData).reshape(-1, 3)
        
        # Calculate mins and maxs
        if pts.shape[0] > 1:
            tmp_mins = np.min(pts, axis=0)
            tmp_maxs = np.max(pts, axis=0)
        else:
            tmp_mins = pts[0]
            tmp_maxs = pts[0]
        
        # Store z slice positions
        slice_z.append(tmp_mins[2])
        
        # Update mins and maxs
        if mins is None:
            mins = tmp_mins
            maxs = tmp_maxs
        else:
            mins = np.minimum(mins, tmp_mins)
            maxs = np.maximum(maxs, tmp_maxs)
    
    # Determine slice thickness
    if len(slice_z) > 1:
        slice_thickness = np.median(np.diff(np.sort(np.unique(slice_z))))
    else:
        slice_thickness = 0
    
    return mins, maxs, slice_thickness, instruct


def bwboundaries_2d_stack_fast(bw: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Find boundaries of all objects in a 3D binary volume.
    
    Args:
        bw: 3D binary volume with shape (x_dim, y_dim, z_dim)
        
    Returns:
        Tuple of (list of boundary points for each object, labeled array)
    """
    # Find connected components
    labeled, num_features = scipy.ndimage.label(bw)
    B = []
    
    # Process each component
    for i in range(1, num_features + 1):
        # Create binary image for this component
        temp_im = (labeled == i)
        
        # Find coordinates where this object exists
        # numpy.nonzero returns indices in order of dimensions (x, y, z for a 3D array)
        x_indices, y_indices, z_indices = np.nonzero(temp_im)
        
        if len(x_indices) == 0:  # Skip empty components
            continue
            
        # Find bounding box
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        z_min, z_max = np.min(z_indices), np.max(z_indices)
        
        # Extract ROI
        roi = temp_im[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
        
        # Find 3D perimeter using erosion (equivalent to MATLAB's bwperim)
        eroded = scipy.ndimage.binary_erosion(roi)
        perim = roi & ~eroded
        
        # Map back to full size
        full_perim = np.zeros_like(temp_im, dtype=bool)
        full_perim[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1] = perim
        
        # Get coordinates of perimeter voxels using numpy's convention
        perim_x, perim_y, perim_z = np.where(full_perim)
        
        # Skip if no perimeter voxels
        if len(perim_x) == 0:
            continue
        
        # Store coordinates as [x y z] to match MATLAB convention
        # This matches the format of the original MATLAB code's B{i}=[x y z]
        B.append(np.column_stack([perim_x, perim_y, perim_z]))
    
    if not B:  # Handle case with no boundaries
        return B, labeled
        
    return B, labeled


def ref_inside_tst_pts(tst_mask: np.ndarray, raw_ref_pts: np.ndarray) -> np.ndarray:
    """
    Determine which reference points are inside the test structure.
    
    This implementation closely matches the MATLAB version that uses
    isosurface and inpolyhedron functionality.
    
    Args:
        tst_mask: Binary mask of test structure
        raw_ref_pts: Reference points (in voxel coordinates)
        
    Returns:
        Binary array indicating which points are inside
    """
    try:
        from skimage import measure
    except ImportError:
        logging.warning("skimage not available, using simplified method for point-in-mesh detection")
        # Initialize result array
        in_mask = np.zeros(len(raw_ref_pts), dtype=bool)
        
        # Check each point directly
        for i, point in enumerate(raw_ref_pts):
            # Convert to integers for indexing
            y, x, z = np.round(point).astype(int)
            
            # Check bounds
            if (0 <= y < tst_mask.shape[0] and 
                0 <= x < tst_mask.shape[1] and 
                0 <= z < tst_mask.shape[2]):
                # Check if point is inside the mask
                in_mask[i] = tst_mask[y, x, z]
                
        return in_mask
    
    # Get dimensions of tst_mask
    rows, cols, _ = tst_mask.shape
    
    # Create a single layer of zeros
    zero_layer = np.zeros((rows, cols))
    
    # Add zero layers to beginning and end (exactly as in MATLAB version)
    padded_mask = np.concatenate([
        zero_layer[:,:,np.newaxis], 
        tst_mask, 
        zero_layer[:,:,np.newaxis]
    ], axis=2)
    
    # Adjust z coordinates
    pts = raw_ref_pts.copy()
    pts[:, 2] = pts[:, 2] + 1
    
    try:
        # Create isosurface from the mask (similar to MATLAB's isosurface)
        verts, faces, _, _ = measure.marching_cubes(padded_mask, level=0.5)
        
        # Flip faces to ensure normals point OUT (as in MATLAB)
        faces = np.fliplr(faces)
        
        try:
            # Try using trimesh (best match to MATLAB's inpolyhedron)
            import trimesh
            mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            in_mask = mesh.contains(pts)
            
        except ImportError:
            # Fallback to ray casting algorithm if trimesh is not available
            logging.warning("trimesh not available, using scipy's Delaunay for point-in-polyhedron. The accuracy of close boundary would be inaccurate.")
            from scipy.spatial import Delaunay
            hull = Delaunay(verts)
            in_mask = hull.find_simplex(pts) >= 0
            
    except Exception as e:
        # Fallback to direct checking if mesh creation fails
        logging.warning(f"Could not create isosurface: {str(e)}. Using simplified implementation.")
        
        # Initialize result array
        in_mask = np.zeros(len(pts), dtype=bool)
        
        # Check each point
        for i, point in enumerate(pts):
            # Convert to integers for indexing
            y, x, z = np.round(point).astype(int)
            
            # Check bounds
            if (0 <= y < padded_mask.shape[0] and 
                0 <= x < padded_mask.shape[1] and 
                0 <= z < padded_mask.shape[2]):
                # Check if point is inside the mask
                in_mask[i] = padded_mask[y, x, z]
    
    # Validate result length matches input (same check as in MATLAB)
    if len(in_mask) != len(raw_ref_pts):
        logging.error('Error! code: refinsidetstpts')
        
    import gc
    gc.collect()
    return in_mask


def get_struct_by_name(roi, name):
    """
    Find a structure by name in a DICOM RTSTRUCT dataset.
    
    Args:
        roi (pydicom.dataset.FileDataset): DICOM RTSTRUCT dataset
        name (str): Name of the structure to find
    
    Returns:
        dict or None: Structure contour sequence if found, None otherwise
    """
    # Extract ROI names and check for match
    for item in roi.StructureSetROISequence:
        if hasattr(item, 'ROIName') and item.ROIName.lower() == name.lower():
            roi_number = item.ROINumber
            
            # Find matching contour sequence
            for contour_item in roi.ROIContourSequence:
                if (hasattr(contour_item, 'ReferencedROINumber') and 
                    contour_item.ReferencedROINumber == roi_number and 
                    hasattr(contour_item, 'ContourSequence')):
                    return contour_item.ContourSequence
    
    return None


def save_variables(file_path: str, variables: Dict[str, Any]) -> None:
    """
    Save variables to a MATLAB-compatible format.
    
    Args:
        file_path: Path to save the file
        variables: Dictionary of variables to save
    """
    # Use h5py to save in a MATLAB-compatible format
    with h5py.File(file_path, 'w') as f:
        for name, data in variables.items():
            if isinstance(data, np.ndarray):
                f.create_dataset(name, data=data)
            elif isinstance(data, dict):
                # Create a group for the dictionary
                group = f.create_group(name)
                for key, value in data.items():
                    group.create_dataset(key, data=value)
            else:
                # Try to convert to numpy array
                try:
                    f.create_dataset(name, data=np.array(data))
                except:
                    logging.warning(f"Could not save variable '{name}' of type {type(data)}")


def get_available_oars(bld_dir: str) -> List[str]:
    """
    Extract unique OAR names from files in the BLD directory.
    
    Args:
        bld_dir: Path to the BLD directory
    
    Returns:
        List of unique OAR names
    """
    oars = set()
    for file in os.listdir(bld_dir):
        if file.endswith('.mat'):
            parts = file.split('_')
            if len(parts) >= 1:
                oars.add(parts[0])
    return sorted(list(oars))


if __name__ == "__main__":
    # Example usage for single OAR mode (Brainstem)
    bld_batch(
        r'...\Manual_matched.csv',
        r'...\AI_matched.csv',
        r'...\results.csv',
        r'\...\output',
        '',
        # oar_name='Brainstem'
    )