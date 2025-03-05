"""
Template Selection tab for E-SAFE GUI.

This module implements the Template Selection tab for the E-SAFE GUI.
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QLineEdit, QFileDialog, QGroupBox, QComboBox, QSplitter,
    QTableWidget, QTableWidgetItem, QHeaderView, QRadioButton,
    QMessageBox,QTextEdit
)
from PyQt5.QtCore import Qt

from gui.widgets import FileSelectionWidget
from utils.visualization import MatplotlibCanvas, setup_3d_axes


class TemplateSelectionTab(QWidget):
    """
    Tab for selecting template contours.
    
    This tab allows the user to select a template contour from a set of
    reference contours.
    """
    
    def __init__(self, main_window):
        """
        Initialize the Template Selection tab.
        
        Args:
            main_window: The main window instance
        """
        super().__init__(main_window)
        
        self.main_window = main_window
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Create a split view
        splitter = QSplitter(Qt.Horizontal)
        
        # --- Left side - Template selection ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # BLD directory
        bld_dir_group = QGroupBox("BLD Data Directory")
        bld_dir_layout = QHBoxLayout(bld_dir_group)
        
        self.bld_dir_widget = FileSelectionWidget(
            "", "Directory with BLD data from previous step",
            "directory", "", "Select BLD Data Directory"
        )
        bld_dir_layout.addWidget(self.bld_dir_widget)
        left_layout.addWidget(bld_dir_group)
        
        # Output directory
        output_dir_group = QGroupBox("Template Output Directory (..DLBoutput/_Ref)")
        output_dir_layout = QHBoxLayout(output_dir_group)
        
        self.output_dir_widget = FileSelectionWidget(
            "", "Template output directory",
            "directory", "", "Select Template Output Directory"
        )
        self.output_dir_widget.set_path("./output/_Ref/")
        output_dir_layout.addWidget(self.output_dir_widget)
        left_layout.addWidget(output_dir_group)
        
        # Structure selection
        structure_group = QGroupBox("Structure Selection")
        structure_layout = QVBoxLayout(structure_group)
        
        self.structure_combo = QComboBox()
        self.structure_combo.setMinimumHeight(30)
        self.structure_combo.addItem("Select a structure")
        self.structure_combo.currentIndexChanged.connect(self.structure_selected)
        
        refresh_structures_btn = QPushButton("Refresh Structures")
        refresh_structures_btn.clicked.connect(self.refresh_structures)
        
        structure_layout.addWidget(QLabel("Select Structure:"))
        structure_layout.addWidget(self.structure_combo)
        structure_layout.addWidget(refresh_structures_btn)
        left_layout.addWidget(structure_group)
        
        # Case selection
        case_group = QGroupBox("Case Selection")
        case_layout = QVBoxLayout(case_group)
        
        self.case_list_widget = QTableWidget()
        self.case_list_widget.setColumnCount(3)
        self.case_list_widget.setHorizontalHeaderLabels(["Patient ID", "File", "Select"])
        self.case_list_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        case_layout.addWidget(QLabel("Select a case to use as template:"))
        case_layout.addWidget(self.case_list_widget)
        
        # Button layout for template operations
        template_buttons_layout = QHBoxLayout()
        
        # Preview contour button
        preview_template_btn = QPushButton("Preview Contour")
        preview_template_btn.clicked.connect(self.preview_template_contour)
        
        # Set template button
        select_template_btn = QPushButton("Set Selected Case as Template")
        select_template_btn.clicked.connect(self.set_template)
        
        template_buttons_layout.addWidget(preview_template_btn)
        template_buttons_layout.addWidget(select_template_btn)
        
        case_layout.addLayout(template_buttons_layout)
        
        left_layout.addWidget(case_group)
        
        # --- Right side - Preview ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        preview_group = QGroupBox("Template Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        self.template_canvas = MatplotlibCanvas(parent=self,width=5, height=4, dpi=100)
        preview_layout.addWidget(self.template_canvas)
        
        self.template_info_text = QTextEdit(self)
        self.template_info_text.setReadOnly(True)
        self.template_info_text.setMaximumHeight(150)
        preview_layout.addWidget(self.template_info_text)
        
        right_layout.addWidget(preview_group)
        
        # Add widgets to splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([400, 600])
        
        # Add splitter to main layout
        layout.addWidget(splitter)
        
    def set_bld_dir(self, bld_dir):
        """
        Set the BLD directory from another tab.
        
        Args:
            bld_dir: The BLD directory path
        """
        if bld_dir:
            self.bld_dir_widget.set_path(bld_dir)
            
            # Set default output directory path including _Ref
            root_dir = os.path.dirname(bld_dir)
            default_ref_dir = os.path.join(root_dir, '_Ref')
            
            # Create the _Ref directory if it doesn't exist
            try:
                os.makedirs(default_ref_dir, exist_ok=True)
                self.main_window.log_message(f"Created template directory: {default_ref_dir}", "INFO")
            except Exception as e:
                self.main_window.log_message(f"Could not create template directory: {str(e)}", "WARNING")
                
            # Set default output directory
            self.output_dir_widget.set_path(default_ref_dir)
            
            # Refresh structures
            self.refresh_structures()
    
    def refresh_structures(self):
        """Refresh the structure list from BLD directory."""
        bld_dir = self.bld_dir_widget.get_path()
        
        if not bld_dir or not os.path.isdir(bld_dir):
            QMessageBox.warning(self, "Warning", "Please select a valid BLD data directory")
            return
            
        # Clear existing items
        self.structure_combo.clear()
        self.structure_combo.addItem("Select a structure")
        
        # Find all unique structure names
        structures = set()
        
        for file in os.listdir(bld_dir):
            if file.endswith('.mat'):
                try:
                    # Remove .mat extension
                    file_base = file[:-4]
                    
                    # Split the filename
                    parts = file_base.split('_')
                    
                    # Find the pattern: 
                    # 1. Look for numeric or alphanumeric ID at the end
                    # 2. Prioritize longer IDs like 001, abc123
                    
                    # Try to extract patient ID
                    patient_id = ''
                    structure_parts = []
                    
                    for j in range(len(parts)-1, -1, -1):
                        # Check if part is a numeric or alphanumeric ID
                        if parts[j].isdigit() or (parts[j].isalnum() and not parts[j].isalpha()):
                            patient_id = parts[j]
                            structure_parts = parts[:j]
                            break
                    
                    # If no numeric ID found, use the last part
                    if not patient_id:
                        patient_id = parts[-1]
                        structure_parts = parts[:-1]
                    
                    # Reconstruct full structure name
                    structure_name = '_'.join(structure_parts)
                    
                    if structure_name:  # Only add non-empty structure names
                        structures.add(structure_name)
                
                except Exception as e:
                    # Log any unexpected errors during processing
                    self.main_window.log_message(f"Error processing file {file}: {e}", "WARNING")
        
        # Add to combo box
        for structure in sorted(structures):
            self.structure_combo.addItem(structure)
            
        self.main_window.log_message(f"Found {len(structures)} structures")
    
    def structure_selected(self):
        """Called when a structure is selected."""
        structure = self.structure_combo.currentText()
        
        if structure == "Select a structure":
            return
            
        bld_dir = self.bld_dir_widget.get_path()
        
        if not bld_dir:
            return
            
        # Find all cases for this structure
        self.case_list_widget.setRowCount(0)
        
        case_files = []
        for file in os.listdir(bld_dir):
            if file.startswith(structure + '_') and file.endswith('.mat'):
                case_files.append(file)
        
        # Add to table
        self.case_list_widget.setRowCount(len(case_files))
        
        for i, file in enumerate(case_files):
            # More robust patient ID and full structure name extraction
            try:
                # Remove .mat extension
                file_base = file[:-4]
                
                # Split the filename
                parts = file_base.split('_')
                
                # Try to extract patient ID
                patient_id = ''
                structure_parts = []
                
                for j in range(len(parts)-1, -1, -1):
                    # Check if part is a numeric or alphanumeric ID
                    if parts[j].isdigit() or (parts[j].isalnum() and not parts[j].isalpha()):
                        patient_id = parts[j]
                        structure_parts = parts[:j]
                        break
                
                # If no numeric ID found, use the last part
                if not patient_id:
                    patient_id = parts[-1]
                    structure_parts = parts[:-1]
                
                # Set table items
                self.case_list_widget.setItem(i, 0, QTableWidgetItem(patient_id))
                self.case_list_widget.setItem(i, 1, QTableWidgetItem(file))
                
                radio_btn = QRadioButton()
                self.case_list_widget.setCellWidget(i, 2, radio_btn)
            
            except Exception as e:
                # Log any unexpected errors during processing
                self.main_window.log_message(f"Error processing file {file}: {e}", "WARNING")
    
    def preview_template_contour(self):
        """Preview the template contour using raw data with COM alignment."""
        structure = self.structure_combo.currentText()
        
        if structure == "Select a structure":
            QMessageBox.warning(self, "Warning", "Please select a structure first.")
            return
            
        bld_dir = self.bld_dir_widget.get_path()
        
        if not bld_dir or not os.path.isdir(bld_dir):
            QMessageBox.warning(self, "Warning", "Please select a valid BLD data directory.")
            return
        
        # Find selected case
        selected_file = None
        patient_id = None
        for i in range(self.case_list_widget.rowCount()):
            radio_widget = self.case_list_widget.cellWidget(i, 2)
            if isinstance(radio_widget, QRadioButton) and radio_widget.isChecked():
                selected_file = self.case_list_widget.item(i, 1).text()
                patient_id = self.case_list_widget.item(i, 0).text()
                break
        
        if not selected_file:
            QMessageBox.warning(self, "Warning", "Please select a case to preview.")
            return
        
        try:
            # Load the MAT file
            with h5py.File(os.path.join(bld_dir, selected_file), 'r') as f:
                if 'refptswithbld' in f:
                    points = f['refptswithbld'][:]
                elif 'refpts' in f:
                    points = f['refpts'][:]
                else:
                    self.main_window.log_message("No reference points found in file", "ERROR")
                    return
                
            if points.shape[1] > 3:
                points = points[:, :3]
            
            # Align points to center of mass
            com = np.mean(points, axis=0)
            points_aligned = points - com
            
            # Clear previous plot
            self.template_canvas.axes.clear()
            
            # Plot the aligned point cloud
            scatter = self.template_canvas.axes.scatter(
                points_aligned[:, 0], 
                points_aligned[:, 1], 
                points_aligned[:, 2], 
                c='b', 
                s=30,
                alpha=0.7,
                edgecolors='darkblue'
            )
            
            # Calculate ranges for each axis
            x_range = points_aligned[:, 0].max() - points_aligned[:, 0].min()
            y_range = points_aligned[:, 1].max() - points_aligned[:, 1].min()  
            z_range = points_aligned[:, 2].max() - points_aligned[:, 2].min()
            
            # Set up 3D coordinate system with anatomical symbols
            setup_3d_axes(self.template_canvas.axes, x_range, y_range, z_range)
            
            # Set labels and title
            self.template_canvas.axes.set_xlabel('X (cm)')  
            self.template_canvas.axes.set_ylabel('Y (cm)')
            self.template_canvas.axes.set_zlabel('Z (cm)') 
            self.template_canvas.axes.set_title(f"{structure} - {patient_id} (COM Aligned)")
            # Set aspect ratio to match the actual data proportions
            self.template_canvas.axes.set_box_aspect((x_range, y_range, z_range))
            
            # Set view angle
            self.template_canvas.axes.view_init(-155, 151)
            
            # Update canvas
            self.template_canvas.draw()
            
            # Update info text
            self.template_info_text.clear()
            self.template_info_text.append(f"Structure: {structure}") 
            self.template_info_text.append(f"Patient ID: {patient_id}")
            self.template_info_text.append(f"Total points: {len(points)}")
            self.template_info_text.append(f"COM: ({com[0]:.2f}, {com[1]:.2f}, {com[2]:.2f})")
            self.template_info_text.append(f"X range (COM aligned): {points_aligned[:, 0].min():.2f} to {points_aligned[:, 0].max():.2f} (span: {x_range:.2f})")  
            self.template_info_text.append(f"Y range (COM aligned): {points_aligned[:, 1].min():.2f} to {points_aligned[:, 1].max():.2f} (span: {y_range:.2f})")
            self.template_info_text.append(f"Z range (COM aligned): {points_aligned[:, 2].min():.2f} to {points_aligned[:, 2].max():.2f} (span: {z_range:.2f})") 
            
        except Exception as e:
            self.main_window.log_message(f"Error previewing template contour: {str(e)}", "ERROR")

    def set_template(self):
        """Set the selected case as template."""
        structure = self.structure_combo.currentText()
        
        if structure == "Select a structure":
            QMessageBox.warning(self, "Warning", "Please select a structure first.")
            return
            
        bld_dir = self.bld_dir_widget.get_path()
        output_dir = self.output_dir_widget.get_path()
        
        if not bld_dir or not os.path.isdir(bld_dir):
            QMessageBox.warning(self, "Warning", "Please select a valid BLD data directory.")
            return
        
        if not output_dir:
            QMessageBox.warning(self, "Warning", "Please select an output directory.")
            return
            
        # Find selected case
        selected_file = None
        patient_id = None
        for i in range(self.case_list_widget.rowCount()):
            radio_widget = self.case_list_widget.cellWidget(i, 2)
            if isinstance(radio_widget, QRadioButton) and radio_widget.isChecked():
                selected_file = self.case_list_widget.item(i, 1).text()
                patient_id = self.case_list_widget.item(i, 0).text()
                break
        
        if not selected_file:
            QMessageBox.warning(self, "Warning", "Please select a case to set as template.")
            return
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
            
        # Copy the file to the template directory
        import shutil
        src_file = os.path.join(bld_dir, selected_file)
        dst_file = os.path.join(output_dir, f"{structure}_{patient_id}-Ref.mat")
        dst_file = dst_file.replace('\\', '/')
        try:
            shutil.copy2(src_file, dst_file)
            
            # Success dialog
            QMessageBox.information(self, "Success", f"Template set successfully:\n{dst_file}")
            
            # Set directory for registration tab
            if hasattr(self.main_window.registration_analysis_tab, 'set_root_dir'):
                # Set to parent directory of BLD directory
                root_dir = os.path.dirname(bld_dir)
                self.main_window.registration_analysis_tab.set_root_dir(root_dir)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error setting template: {str(e)}")