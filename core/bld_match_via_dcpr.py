"""
bld_match_via_dcpr.py - Bidirectional Local Distance Match via Deformable point could Registration

This module performs point cloud registration and error analysis for medical 
structure boundaries using Coherent Point Drift (CPD) registration.

Authors: 
    Original MATLAB: Jingwei Duan, Ph.D. (duan.jingwei01@gmail.com), Quan Chen, Ph.D.

    
Date: February 2025
Version: 1.1
License: MIT License
"""

import sys
import os
import glob
import numpy as np
import pandas as pd
import h5py
import logging
import datetime
from typing import List, Dict, Optional, Any, Tuple, Union
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from pycpd import DeformableRegistration
from sklearn.neighbors import NearestNeighbors
import traceback
# For progress tracking in notebooks or console
import matplotlib
from PyQt5.QtWidgets import QApplication
matplotlib.use('Agg')  # Use non-interactive backend
try:
    from tqdm.auto import tqdm
except ImportError:
    # Simple tqdm alternative if not available
    def tqdm(iterable, *args, **kwargs):
        return iterable


class PointCloud:
    """A simple point cloud class with location and intensity data."""
    def __init__(self, location, intensity=None):
        """
        Initialize a point cloud.
        
        Args:
            location: Nx3 array of point coordinates (each row is a point)
            intensity: Optional intensity values for each point
        """
        self.location = np.asarray(location)
        
        # Check and fix dimensions if needed
        if self.location.size > 0:
            # If location has shape (3, N) and N > 3, it's likely transposed
            if self.location.shape[0] == 3 and self.location.shape[1] > 3:
                # logging.warning(f"Points appear to be transposed in PointCloud (shape: {self.location.shape})")
                self.location = self.location.T
        
        self.count = self.location.shape[0]
        
        if intensity is None:
            self.intensity = np.zeros(self.count)
        else:
            # Reshape intensity if needed
            intensity_array = np.asarray(intensity)
            if len(intensity_array.shape) > 1:
                intensity_array = intensity_array.flatten()
            
            # Truncate or pad to match point count
            if len(intensity_array) > self.count:
                self.intensity = intensity_array[:self.count]
            elif len(intensity_array) < self.count:
                self.intensity = np.pad(intensity_array, (0, self.count - len(intensity_array)))
            else:
                self.intensity = intensity_array
            
    def __len__(self):
        return self.count


def bld_match_via_dcpr(root_dir: str, oar_name: Optional[str] = None, 
                      grid_average: float = 0.05, count_threshold: int = 5000, 
                         max_iterations: int = 30) -> None:
    """
    Perform deformable registration and BLD analysis for contour comparison.
    
    Args:
        root_dir: Base directory containing BLD data
        oar_name: Optional name of specific OAR to process (default: process all OARs)
        grid_average: Grid size for point cloud downsampling (default: 0.05)
        count_threshold: Maximum point count threshold (default: 5000)
        max_iterations: Maximum number of iterations for registration (default: 30)
    """
    # Import garbage collection
    import gc
    
    # Initialize output directories
    root_dir_result = os.path.join(root_dir, 'BiasesResult')
    os.makedirs(root_dir_result, exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'RmseResult'), exist_ok=True)
    
    # Setup logging
    output_txt = os.path.join(root_dir_result, 'BLDMatchViaDCPR_Log_output.txt')
    setup_logging(output_txt)
    
    # Determine operation mode
    if oar_name is None:
        # Batch mode - find all reference templates
        search_pattern = os.path.join(root_dir, '_Ref', '*-Ref.mat')
        mat_files = glob.glob(search_pattern)
        
        if not mat_files:
            logging.error(f"No reference template files found in {os.path.join(root_dir, '_Ref')}")
            return
            
        recursive_flag = True
        recursive_times = len(mat_files)
        logging.info('**********************Batch OARs mode**********************')
        
        # Extract OAR names from filenames
        oars = []
        for mat_file in mat_files:
            filename = os.path.basename(mat_file)
            parts = filename.split('_')
            if len(parts) == 2:
                oars.append(parts[0])
            elif len(parts) >= 5:
                oars.append(f"{parts[0]}_{parts[1]}_{parts[2]}")
            else:
                oars.append(f"{parts[0]}_{parts[1]}")
    else:
        # Single OAR mode
        recursive_flag = False
        recursive_times = 1
        oars = [oar_name]
        logging.info(f'**********************Single OAR mode: {oar_name}**********************')
    
    # Process OARs in chunks
    oar_chunk_size = 1  # Process 2 OARs at a time
    for oar_chunk_start in range(0, recursive_times, oar_chunk_size):
        oar_chunk_end = min(oar_chunk_start + oar_chunk_size, recursive_times)
        logging.info(f"Processing OAR chunk {oar_chunk_start//oar_chunk_size + 1}: OARs {oar_chunk_start+1}-{oar_chunk_end}")
        
        # Process each OAR in this chunk
        for ii in range(oar_chunk_start, oar_chunk_end):
            current_oar = oars[ii]
            logging.info(f'===================={current_oar}====================')
            
            try:
                # Find all BLD files for this OAR
                bld_found_files = []
                dirs = glob.glob(os.path.join(root_dir, f"{current_oar}_*"))
                for d in dirs:
                    if os.path.isfile(d):
                        bld_found_files.append(d)
                
                if not bld_found_files:
                    logging.warning(f"No BLD files found for {current_oar}")
                    continue
                
                # Initialize result files
                result_file = os.path.join(root_dir_result, f'_{current_oar}_Result.csv')
                result_file_std = os.path.join(root_dir_result, f'_{current_oar}-std_Result.csv')
                result_file_rmse = os.path.join(root_dir, 'RmseResult', f'rmse_{current_oar}_Result.csv')
                
                # Create directories for detected errors and aligned data
                detect_error_dir = os.path.join(root_dir, 'DetectedError')
                after_dir_dir = os.path.join(root_dir, 'DIRMatchdata')
                os.makedirs(detect_error_dir, exist_ok=True)
                os.makedirs(after_dir_dir, exist_ok=True)
                
                with open(result_file_rmse, 'w') as fp:
                    fp.write('tstfname, rmse(cm)\n')
                    
                    # Load and process Template points
                    pattern = f"{current_oar}*-Ref.mat"
                    temp_file = glob.glob(os.path.join(root_dir, '_Ref', pattern))
                    if not temp_file:
                        logging.error(f"No template file found for {current_oar}")
                        continue
                        
                    template_data = load_mat_file(temp_file[0])
                    template_pts = template_data.get('refpts', np.array([]))
                    
                    if template_pts.size == 0:
                        logging.error(f"No reference points found in template file for {current_oar}")
                        continue
                        
                    # Check template_pts shape and transpose if needed
                    if template_pts.shape[0] == 3 and template_pts.shape[1] > 3:
                        logging.info(f"Transposing template points from shape {template_pts.shape}")
                        template_pts = template_pts.T  # Transpose to get points as rows
                    
                    # Align to center of mass
                    template_pts = align_com(template_pts)
                    
                    # Initialize point cloud
                    template_intensity = np.zeros(template_pts.shape[0])  # Number of points
                    template_pts_cloud = PointCloud(template_pts, template_intensity)
                    
                    # Create a base template cloud that won't be modified
                    template_pts_base = template_pts.copy()
                    template_intensity_base = np.zeros(template_pts_base.shape[0])
                    
                    # Check point count and downsample if needed
                    point_count = template_pts_cloud.count
                    original_grid_average = grid_average
                    
                    if point_count > count_threshold:
                        logging.info(f'The OAR points number is {point_count}, larger than threshold {count_threshold}')
                        logging.info('-----------Downsample-------------')
                        
                        # Need to downsample
                        flag = False
                        while point_count > count_threshold:
                            template_pts_cloud_downsampled = downsample_point_cloud(template_pts_cloud, grid_average)
                            point_count = template_pts_cloud_downsampled.count
                            
                            if point_count > count_threshold:
                                grid_average += 0.001
                            else:
                                flag = False
                                logging.info(f'The downsample OAR points number is {point_count}')
                        
                        # Store downsampled template base information
                        template_pts_base_downsampled = template_pts_cloud_downsampled.location.copy()
                    else:
                        logging.info(f'The OAR points number is {point_count}')
                        flag = True
                        template_pts_cloud_downsampled = template_pts_cloud
                        template_pts_base_downsampled = template_pts_base
                    
                    # Perform DIR matching
                    logging.info('-------------------DIR-------------------------')
                    intensity_temp = []
                    
                    # Add periodic garbage collection during processing
                    files_processed = 0
                    
                    # Process BLD files in smaller chunks
                    bld_chunk_size = 1
                    total_bld_files = len(bld_found_files)  # Add total count of files

                    for bld_chunk_start in range(0, total_bld_files, bld_chunk_size):
                        bld_chunk_end = min(bld_chunk_start + bld_chunk_size, total_bld_files)
                        logging.info(f"Processing BLD chunk {bld_chunk_start//bld_chunk_size + 1}: files {bld_chunk_start+1}-{bld_chunk_end}")
                        
                        # Process BLD files in this chunk
                        for i in range(bld_chunk_start, bld_chunk_end):
                            bld_file = bld_found_files[i]
                            logging.info(f"-> Processing case {i+1}/{total_bld_files}: {os.path.basename(bld_file)} - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                            try:
                                # Load BLD data
                                bld_single = load_mat_file(bld_file)
                                
                                # Check and ensure correct shape for refptswithbld
                                ref_pts_with_bld = bld_single.get('refptswithbld', np.array([]))
                                ref_pts = bld_single.get('refpts', np.array([]))
                                
                                if ref_pts_with_bld.size == 0 and ref_pts.size > 0:
                                    logging.warning(f"No refptswithbld found in {bld_file}, creating from refpts")
                                    
                                    if ref_pts.shape[0] == 3 and ref_pts.shape[1] > 3:
                                        ref_pts = ref_pts.T
                                    
                                    intensity = np.zeros((ref_pts.shape[0], 1))
                                    ref_pts_with_bld = np.hstack((ref_pts, intensity))
                                
                                # Check if refptswithbld needs to be transposed
                                if ref_pts_with_bld.shape[0] == 4 and ref_pts_with_bld.shape[1] > 4:
                                    logging.info(f"Transposing refptswithbld from shape {ref_pts_with_bld.shape}")
                                    ref_pts_with_bld = ref_pts_with_bld.T
                                
                                # Align points to center of mass
                                if ref_pts.size > 0:
                                    if ref_pts.shape[0] == 3 and ref_pts.shape[1] > 3:
                                        ref_pts = ref_pts.T
                                    bld_single['refpts'] = align_com(ref_pts)
                                
                                # Align refptswithbld coordinates
                                aligned_coords = align_com(ref_pts_with_bld[:, 0:3])
                                ref_pts_with_bld[:, 0:3] = aligned_coords
                                
                                # Create point cloud for this case
                                eval_pts_cloud = PointCloud(ref_pts_with_bld[:, 0:3], ref_pts_with_bld[:, 3])
                                
                                # For each iteration, create a fresh copy of the template point cloud
                                if flag:
                                    template_pts_cloud_this_case = PointCloud(
                                        template_pts_base.copy(),
                                        template_intensity_base.copy()
                                    )
                                    eval_pts_cloud_downsampled = eval_pts_cloud
                                else:
                                    template_pts_cloud_this_case = PointCloud(
                                        template_pts_base_downsampled.copy(),
                                        np.zeros(template_pts_base_downsampled.shape[0])
                                    )
                                    eval_pts_cloud_downsampled = downsample_point_cloud(eval_pts_cloud, original_grid_average)
                                
                                # Ensure both point clouds have the same shape for comparison
                                src_shape = eval_pts_cloud_downsampled.location.shape
                                tgt_shape = template_pts_cloud_this_case.location.shape
                                
                                if src_shape == tgt_shape and np.array_equal(eval_pts_cloud_downsampled.location, template_pts_cloud_this_case.location):
                                    logging.info('Point clouds are identical. No registration needed.')
                                    rmse = 0
                                    
                                    # Direct copy of intensity values
                                    template_pts_cloud_this_case.intensity = eval_pts_cloud_downsampled.intensity.copy()
                                    rmse = np.nan
                                else:
                                    logging.info(f'Source cloud: {eval_pts_cloud_downsampled.count} points, {eval_pts_cloud_downsampled.location.shape[1]} dimensions')
                                    logging.info(f'Target cloud: {template_pts_cloud_this_case.count} points, {template_pts_cloud_this_case.location.shape[1]} dimensions')
                                    
                                    # Perform registration with error handling
                                    try:
                                        reg = DeformableRegistration(
                                            X=template_pts_cloud_this_case.location,
                                            Y=eval_pts_cloud_downsampled.location,
                                            alpha=2,  # Regularization weight
                                            beta=2,     # Width of Gaussian kernel
                                            max_iterations=max_iterations,     
                                            tolerance=1e-5  
                                        )
                                        
                                        # Perform registration
                                        transformed_moving, registration_params = reg.register()
                                        
                                        # Calculate RMSE
                                        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(template_pts_cloud_this_case.location)
                                        distances, _ = nbrs.kneighbors(transformed_moving)
                                        rmse = np.sqrt(np.mean(distances**2))
                                        logging.info(f"Registration RMSE: {rmse:.6f}")
                                        
                                        # Create a mapping to transfer intensity values
                                        nbrs_transformed = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(transformed_moving)
                                        distances, indices = nbrs_transformed.kneighbors(template_pts_cloud_this_case.location)

                                        # Transfer intensity values from registered points to template points
                                        for j in range(len(template_pts_cloud_this_case.location)):
                                            # Check if indices is empty or invalid (matching MATLAB behavior)
                                            if len(indices) == 0 or j >= len(indices) or len(indices[j]) == 0:
                                                # Default to first point as fallback (same as MATLAB)
                                                template_pts_cloud_this_case.intensity[j] = eval_pts_cloud_downsampled.intensity[0]
                                            else:
                                                # Transfer intensity from nearest point
                                                template_pts_cloud_this_case.intensity[j] = eval_pts_cloud_downsampled.intensity[indices[j][0]]
                                    except Exception as e:
                                        logging.error(f"Registration failed: {str(e)}")
                                        rmse = -1  # Use -1 to indicate registration failure
                                        
                                del ref_pts_with_bld, ref_mask, tst_mask  # Explicitly delete large objects
                                gc.collect()
                                QApplication.processEvents()
                                # Write RMSE to file
                                fp.write(f'{bld_file},{rmse}\n')
                                fp.flush()  # Ensure writing after each case
                                
                                # Important: Create a COPY of the intensity array for this case
                                intensity_temp.append(template_pts_cloud_this_case.intensity.copy())
                                
                            except Exception as e:
                                logging.error(f"Error processing BLD file {bld_file}: {str(e)}")
                                logging.error(traceback.format_exc())
                            
                            # Force garbage collection after each BLD file
                            files_processed += 1
                            gc.collect()
                            QApplication.processEvents()
                        
                        # Force memory cleanup after each chunk
                        gc.collect()
                        QApplication.processEvents()
                        
                        # Optional: Log memory usage after chunk
                        try:
                            import psutil
                            process = psutil.Process(os.getpid())
                            memory_mb = process.memory_info().rss / (1024 * 1024)
                            logging.info(f"Memory usage after processing chunk: {memory_mb:.1f} MB")
                        except ImportError:
                            pass
                    
                    # Process results - determine the template location to use for output
                    template_location = template_pts_base if flag else template_pts_base_downsampled
                    
                    # Free memory before processing results
                    gc.collect()
                    QApplication.processEvents()
                    
                    try:
                        # Convert list of intensity arrays to a 2D array for easier processing
                        intensity_matrix = np.column_stack(intensity_temp)
                        
                        # Calculate mean and standard deviation
                        template_mean_intensity = np.nanmean(intensity_matrix, axis=1)
                        template_std_intensity = np.nanstd(intensity_matrix, axis=1, ddof=0)
                        
                        # Free some memory before continuing
                        del intensity_matrix
                        gc.collect()
                        
                        # Create result arrays
                        template_pts_matched = np.column_stack((
                            template_location,
                            template_mean_intensity
                        ))
                        
                        template_pts_matched_std = np.column_stack((
                            template_location,
                            template_std_intensity
                        ))
                        
                        # Calculate tolerance bounds
                        tolerance_low = template_mean_intensity - 1.96 * template_std_intensity
                        tolerance_high = template_mean_intensity + 1.96 * template_std_intensity
                        tolerance_low_99ci = template_mean_intensity - 2.576 * template_std_intensity
                        tolerance_high_99ci = template_mean_intensity + 2.576 * template_std_intensity
                        
                        # Error detection
                        errorindex_99ci = []
                        errorindex_95ci = []
                        errorindex_95ci_01percent = []
                        
                        out_of_range_indices = {}
                        out_of_range_indices_99ci = {}
                        
                        # Process in small chunks to avoid memory issues
                        for j in range(0, len(intensity_temp), 5):
                            chunk_end = min(j+5, len(intensity_temp))
                            
                            for i in range(j, chunk_end):
                                curr_intensity = intensity_temp[i]
                                
                                # Find points outside 95% CI
                                out_range = np.where((curr_intensity < tolerance_low) | (curr_intensity > tolerance_high))[0]
                                out_of_range_indices[i] = out_range
                                
                                # Find points outside 99% CI
                                out_range_99ci = np.where((curr_intensity < tolerance_low_99ci) | (curr_intensity > tolerance_high_99ci))[0]
                                out_of_range_indices_99ci[i] = out_range_99ci
                                
                                # Check if enough points are outside CI
                                if len(out_range_99ci) > 0.01 * len(curr_intensity):
                                    errorindex_99ci.append(i)
                                    
                                if len(out_range) > 0.001 * len(curr_intensity):
                                    errorindex_95ci_01percent.append(i)
                                    
                                if len(out_range) > 0.01 * len(curr_intensity):
                                    errorindex_95ci.append(i)
                            
                            # Force GC after each chunk
                            gc.collect()
                            QApplication.processEvents()
                        
                        # Combine error indices
                        errorindex = list(set(errorindex_99ci + errorindex_95ci))
                        
                        # Error detection results
                        logging.info('====================Error Detection====================')
                        
                        # Center points for visualization
                        template_pts_matched[:, 0:3] = template_pts_matched[:, 0:3] - np.mean(template_pts_matched[:, 0:3], axis=0)
                        
                        # Process detected errors in chunks
                        for error_chunk_start in range(0, len(errorindex), 5):
                            error_chunk_end = min(error_chunk_start + 5, len(errorindex))
                            
                            for i in errorindex[error_chunk_start:error_chunk_end]:
                                error_file = os.path.basename(bld_found_files[i])
                                name, _ = os.path.splitext(error_file)
                                
                                percentage_incorrect = (len(out_of_range_indices[i]) / len(intensity_temp[i])) * 100
                                
                                if i in errorindex_99ci:
                                    reason = '> 1% of number of points outside the 99% CI range'
                                    tag = '99CI'
                                elif i in errorindex_95ci:
                                    reason = '> 1% of number of points outside the 95% CI range'
                                    tag = '95CI'
                                else:
                                    reason = 'Unknown'
                                    tag = 'unknown'
                                
                                logging.info(f'Possible Error: {name}; Number of incorrect points: {len(out_of_range_indices[i])} ({percentage_incorrect:.2f}%); Reason: {reason}')
                                
                                # Save error data
                                modified_name = os.path.join(detect_error_dir, f'_DetectError_{name}_{tag}.csv')
                                error_data = np.column_stack((
                                    template_pts_matched[:, 0:3],
                                    intensity_temp[i]
                                ))
                                np.savetxt(modified_name, error_data, delimiter=',')
                            
                            # Force GC after each chunk
                            gc.collect()
                            QApplication.processEvents()
                        
                        # Save DIR match data in chunks
                        afterdir_foldername = after_dir_dir
                        for j in range(0, len(intensity_temp), 5):
                            chunk_end = min(j+5, len(intensity_temp))
                            
                            for i in range(j, chunk_end):
                                original_name = os.path.basename(bld_found_files[i])
                                name, _ = os.path.splitext(original_name)
                                afterdir_path = os.path.join(afterdir_foldername, f'{name}.csv')
                                afterdir_data = np.column_stack((
                                    template_pts_matched[:, 0:3],
                                    intensity_temp[i]
                                ))
                                np.savetxt(afterdir_path, afterdir_data, delimiter=',')
                            
                            # Force GC after each chunk
                            gc.collect()
                            QApplication.processEvents()
                        
                        # Save final results
                        template_pts_matched_std[:, 0:3] = template_pts_matched_std[:, 0:3] - np.mean(template_pts_matched_std[:, 0:3], axis=0)
                        np.savetxt(result_file, template_pts_matched, delimiter=',')
                        np.savetxt(result_file_std, template_pts_matched_std, delimiter=',')
                    
                    except Exception as e:
                        logging.error(f"Error processing results: {str(e)}")
                        logging.error(traceback.format_exc())
                    
                    # Final GC after processing all BLD files
                    gc.collect()
                    QApplication.processEvents()
                
                logging.info(f'===================={current_oar} Done====================')
            
            except Exception as e:
                logging.error(f"Error processing OAR {current_oar}: {str(e)}")
                logging.error(traceback.format_exc())
            
            # Force GC after each OAR
            gc.collect()
            QApplication.processEvents()
        
        # Force GC after each OAR chunk
        gc.collect()
        QApplication.processEvents()      
    
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


def load_mat_file(filepath: str) -> Dict[str, np.ndarray]:
    """
    Load a MAT file and return its variables as a dictionary.
    
    Args:
        filepath: Path to the MAT file
        
    Returns:
        Dictionary of variable names and values
    """
    result = {}
    
    try:
        try:
            # First attempt: try with h5py (HDF5-based MAT files - version 7.3 and newer)
            with h5py.File(filepath, 'r') as f:
                # Get all dataset names
                def get_all_datasets(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        # Store dataset in the result dictionary
                        data = np.array(obj)
                        # Handle MATLAB's column-major order vs. Python's row-major order
                        if data.ndim > 1:
                            data = data.transpose()
                        result[name] = data
                
                # Visit all groups/datasets recursively
                f.visititems(get_all_datasets)
                
                # Special handling for 'refpts' and 'refptswithbld'
                if 'refpts' not in result and '/refpts' in f:
                    result['refpts'] = np.array(f['/refpts']).transpose()
                
                if 'refptswithbld' not in result and '/refptswithbld' in f:
                    result['refptswithbld'] = np.array(f['/refptswithbld']).transpose()
                
                logging.info(f"Loaded MAT file using h5py: {filepath}")
                
        except (IOError, OSError, ValueError) as hdf_error:
            # Second attempt: try scipy.io.loadmat (older MAT file formats)
            logging.info(f"H5py loading failed ({str(hdf_error)}), trying scipy.io...")
            import scipy.io as sio
            mat_dict = sio.loadmat(filepath)
            
            # Copy variables from scipy's loadmat result
            for key in mat_dict:
                # Skip meta variables that start with '__'
                if not key.startswith('__'):
                    result[key] = mat_dict[key]
                    
            logging.info(f"Loaded MAT file using scipy.io: {filepath}")
    
    except Exception as e:
        logging.error(f"Error loading MAT file {filepath}: {str(e)}")
        # One more attempt: custom low-level approach
        try:
            logging.info("Attempting custom MAT file parsing...")
            import scipy.io as sio
            # Try loading with squeeze_me=True and struct_as_record=False
            result = sio.loadmat(filepath, squeeze_me=True, struct_as_record=False)
            logging.info("Custom MAT file parsing succeeded")
        except Exception as e3:
            logging.error(f"All loading methods failed: {str(e3)}")
            # Return empty dictionary with expected fields
            result = {'refpts': np.zeros((0, 3)), 'refptswithbld': np.zeros((0, 4))}
    
    # Validate expected fields
    if 'refpts' not in result:
        logging.warning(f"Field 'refpts' not found in {filepath}")
        result['refpts'] = np.zeros((0, 3))
        
    if 'refptswithbld' not in result:
        logging.warning(f"Field 'refptswithbld' not found in {filepath}")
        # If we have refpts but not refptswithbld, create it
        if 'refpts' in result and result['refpts'].shape[0] > 0:
            # Create with zero intensity values
            pts = result['refpts']
            intensity = np.zeros((pts.shape[0], 1))
            result['refptswithbld'] = np.hstack((pts, intensity))
        else:
            result['refptswithbld'] = np.zeros((0, 4))
    
    # Ensure arrays have correct shape
    if result['refpts'].ndim == 1:
        result['refpts'] = result['refpts'].reshape(1, -1)
    if result['refptswithbld'].ndim == 1:
        result['refptswithbld'] = result['refptswithbld'].reshape(1, -1)
        
    return result


def align_com(points: np.ndarray) -> np.ndarray:
    """
    Align points to their center of mass.
    
    Args:
        points: Array of points (Nx3 or 3xN)
        
    Returns:
        Aligned points with same shape as input
    """
    # Check if points are likely transposed (3 rows, many columns)
    if points.shape[0] == 3 and points.shape[1] > 3:
        logging.warning(f"Points appear to be transposed in align_com (shape: {points.shape})")
        # Transpose, align, then transpose back
        points_t = points.T
        com = np.mean(points_t, axis=0)
        aligned_points_t = points_t - com
        return aligned_points_t.T
    
    # Calculate center of mass
    com = np.mean(points, axis=0)
    
    # Subtract center of mass from all points
    aligned_points = points - com
    
    return aligned_points


def downsample_point_cloud(point_cloud: PointCloud, grid_size: float) -> PointCloud:
    """
    Downsample a point cloud using a grid-based approach.
    
    Args:
        point_cloud: Input point cloud
        grid_size: Size of grid cells for downsampling
        
    Returns:
        Downsampled point cloud
    """
    points = point_cloud.location
    intensity = point_cloud.intensity
    
    # Check if points array is empty
    if points.shape[0] == 0:
        logging.warning("Empty point cloud provided for downsampling")
        return PointCloud(np.zeros((0, points.shape[1])), np.array([]))
    
    # Create grid indices for each point
    grid_indices = np.floor(points / grid_size).astype(int)
    
    # Get the dimensionality of the point cloud
    ndim = points.shape[1]
    
    # Create a unique key for each grid cell based on dimensionality
    if ndim == 3:
        # 3D points
        grid_keys = grid_indices[:, 0] * 1000000 + grid_indices[:, 1] * 1000 + grid_indices[:, 2]
    elif ndim == 2:
        # 2D points
        grid_keys = grid_indices[:, 0] * 1000 + grid_indices[:, 1]
    else:
        # Arbitrary dimensions - use string hashing
        grid_keys = np.zeros(len(grid_indices), dtype=np.int64)
        for i, idx in enumerate(grid_indices):
            grid_keys[i] = hash(tuple(idx))
    
    # Find unique grid cells
    unique_keys, inverse_indices = np.unique(grid_keys, return_inverse=True)
    
    # Initialize downsampled points and intensities
    downsampled_points = np.zeros((len(unique_keys), ndim))
    downsampled_intensity = np.zeros(len(unique_keys))
    
    # Aggregate points and intensities in each grid cell
    for i, key in enumerate(unique_keys):
        cell_mask = (grid_keys == key)
        cell_points = points[cell_mask]
        cell_intensity = intensity[cell_mask]
        
        # Calculate mean position and intensity
        downsampled_points[i] = np.mean(cell_points, axis=0)
        downsampled_intensity[i] = np.mean(cell_intensity)
    
    return PointCloud(downsampled_points, downsampled_intensity)


def visualize_registration_results(source_cloud: PointCloud, target_cloud: PointCloud, 
                          transformed_cloud: PointCloud, rmse: float, title: str = None):
    """
    Visualize registration results in 3D.
    
    Args:
        source_cloud: Original source point cloud 
        target_cloud: Target point cloud
        transformed_cloud: Registered source point cloud
        rmse: Root mean square error of registration
        title: Optional title for the plot
    """
        # Disable visualization during GUI operation
    if 'QApplication' in sys.modules:
        return None
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    ax.scatter(source_cloud.location[:, 0], source_cloud.location[:, 1], source_cloud.location[:, 2], 
               c='red', s=10, label='Source', alpha=0.5)
    ax.scatter(target_cloud.location[:, 0], target_cloud.location[:, 1], target_cloud.location[:, 2], 
               c='blue', s=10, label='Target', alpha=0.5)
    ax.scatter(transformed_cloud.location[:, 0], transformed_cloud.location[:, 1], transformed_cloud.location[:, 2], 
               c='green', s=10, label='Transformed', alpha=0.5)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    if title:
        ax.set_title(f"{title} (RMSE: {rmse:.4f})")
    else:
        ax.set_title(f"Registration Results (RMSE: {rmse:.4f})")
    
    ax.legend()
    plt.tight_layout()
    
    return fig


if __name__ == "__main__":
    # Example usage
    bld_match_via_dcpr(
        r'...\output',  # Root directory
        'Brainstem',          # Optional: specific OAR
        0.005,               # Grid size for downsampling
        5000,                # Point count threshold,    
        max_iterations=30 
    )
