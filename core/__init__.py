"""
Core functionality for E-SAFE (Evaluation System for Auto-segmentation Fidelity and Error).

This package contains modules for processing medical images and performing 
Bidirectional Local Distance (BLD) analysis.
"""

# Import key functionality to make it available at package level
from .filter_patient_st_name import filter_patient_st_name
from .org_study_list import org_study_list
from .bld_batch import bld_batch, get_available_oars
from .bld_match_via_dcpr import bld_match_via_dcpr
from .f_bld_visualization import f_bld_visualization