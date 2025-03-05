"""
DICOM processing utilities for E-SAFE.

This module provides utilities for working with DICOM files and structures.
"""

import numpy as np
import pydicom
import os
import cv2
from matplotlib.path import Path


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