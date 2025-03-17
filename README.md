# DPCR-BLD
Deformable Point Clould Registration-Based Bidirectional Local Distance (DPCR-BLD): A Methodology for Systematic Evaluation and Visualization of Local Disagreements in Clinical Auto-Contouring

## Authors

- Jingwei Duan, Ph.D.
- Quan Chen, Ph.D.


## Overview

This repository provides a comprehensive toolkit for systematically evaluating local disagreements in auto-segmentation. The DPCR-BLD methodology enables detailed analysis of contour differences, making it particularly valuable for validating and improving auto-contouring systems in clinical settings.
![Figure2_Workflow](https://github.com/user-attachments/assets/21e8317b-9595-4104-a952-0a4c33c3bc2e)

## Features


- Automated geometric metrics calculation including:
  - Dice similarity coefficient
  - Hausdorff distance
  - Surface Dice
  - Center of mass distance
  - Mean surface distance
- DR-BLD provides a mechanism to spatially identify local contour differences between two contour sets. 
  ![image](https://github.com/user-attachments/assets/867fb05e-a321-43b4-8321-b7406109016b)

- It can be used for contour outlier detection. 
  ![image](https://github.com/user-attachments/assets/779217b2-eff1-4a6b-8789-1a79cfbd5d88)

## Prerequisites

-A folder containing DICOM RTSTRUCT files (RS*.dcm)

Files must be valid DICOM RT Structure Set objects
Each file should contain the contours you want to analyze

- MATLAB R2019b or later
- Python 3.7+ with the following packages:
  - matplotlib
  - numpy
  - argparse
- Required MATLAB toolboxes:
  - Image Processing Toolbox
  - Computer Vision Toolbox
  - Statistics and Machine Learning Toolbox
    
## Key Functions
- `filterPatientStName.m`: Structure set organization
- `orgStudyList.m`: Structure name matching
- `BLD_Batch.m`: Main function for batch processing contour comparisons
- `BLDMatchViaDCPR.m`: Main function for deformable registration, analysis, and outlier detection
- `colormapMultiFigmmRawData.py`: Visualization tools for systematic results
- `f_BLD_visualization.m`: Visualization tools for individual cases


## Usage

### Basic Workflow
0. **Creating the Lookup Table**

Different institutions and physicians may use varying names for the same anatomical structures. The workflow includes tools to help you identify and organize these variations:

First, use the structure name filtering tool to identify all structure names in your dataset:
```matlab
% Automatically scan and extract all unique structure names from your DICOM files
filterPatientStName('input/folder/', 'stnames.csv');
```
This command will:
- Recursively search through all DICOM files in the input folder
- Extract all unique structure names
- Create 'stnames.csv' containing a list of all found structure names
- Help you identify variations of the same structure across different files

Once you have this comprehensive list, create a lookup table to map these variations:

1. Create a new CSV file (e.g., `structure_lookup.csv`)
2. Each row represents one anatomical structure
3. The first column should contain the standardized name
4. Additional columns contain alternative names for the same structure

Example lookup table structure:

| |  | |   |
|--------------|--------------|---------------|---------------|
| Brainstem    | Brain_Stem   | BrainStem     | Brain stem    |
| Parotid_L    | Lt_Parotid   | Parotid Lt    | L Parotid     |
| Parotid_R    | Rt_Parotid   | Parotid Rt    | R Parotid     |
| SpinalCord   | Spinal_Cord  | Cord          |               |

![image](https://github.com/user-attachments/assets/8ceec0cb-77a2-4ee0-a0ab-6dab7f3091d4)

Guidelines for the lookup table:
- Place the AI/auto-segmented structure names in the first column (these are typically consistent)
- Add manual contour variations found in 'stnames.csv' in subsequent columns
- Leave cells empty if there aren't more alternatives
- Maintain case sensitivity as used in your structure sets
- Include underscores and special characters exactly as they appear in the structure names


1. **Organize Structure Sets**

After creating your lookup table, organize your structure sets by matching structure names:

```matlab
% Process reference contours 
orgStudyList('../input/reference_data/', 'structure_lookup.csv', 'reflist.csv');

% Process test/AI contours
orgStudyList('../input/test_data/', 'structure_lookup.csv', 'testlist.csv');
```

This step will:
- Match structure names using your lookup table
- Create organized lists of matched structures
- Handle variations in structure naming
- Generate CSV files for next step


2. **Calculate BLD Metrics**

Use BLD_Batch to compute Bidirectional Local Distance metrics between reference and test contours:

```matlab
% Compare reference and test contours
BLD_Batch('reflist.csv',         % Reference contour list
          'testlist.csv',        % Test contour list
          'results.csv',         % Output metrics file
          './output/BLD/',       % Output directory for BLD data
          '');                   % Optional Windows path prefix
```

Output structure:
```
output/
└── BLD/
    ├── OARname_MRN1.mat    # BLD data for case 1
    ├── OARname_MRN2.mat    # BLD data for case 2
 └── results.csv         # Summary metrics
```

The BLD calculation:
- Computes bidirectional distances between contour pairs
- Creates MAT files needed for subsequent analysis
- Provides comprehensive quality metrics including:
  - Dice similarity coefficient
  - Hausdorff distance
  - Mean surface distance
  - Surface Dice


3. **Template Contour Selection**

For each organ at risk (OAR), a clinically-approved segmentation **must be selected** as the template contour. This is a crucial step that requires careful review of contour representativeness by clinical experts.

Setup instructions:
```
1. Create a directory for template contours:
   mkdir ./output/_Ref/

2. Place the selected template contour files:
   - File naming format: [OARName]_[MRN]-Ref.mat
   - Example: Brainstem_12345-Ref.mat

Directory structure:
output/
└── _Ref/
    ├── Brainstem_12345-Ref.mat    # Template for Brainstem
    ├── Parotid_L_12345-Ref.mat    # Template for Left Parotid
    └── SpinalCord_12345-Ref.mat   # Template for Spinal Cord
```

Important considerations for template selection:
- Choose contours that have been clinically validated
- Ensure the selected contour represents typical anatomy
- Verify the contour follows institutional guidelines

4. **Deformable Registration and BLD Analysis**

After template selection, the next step is to perform deformable registration of BLD data from each reference contour to the selected template contour using Coherent Point Drift (CPD) registration.
This process enables:

Systematic Local Disagreement Analysis:

Maps local contour differences onto the template surface
Generates point-wise disagreement distributions
Provides comprehensive visualization of disagreement patterns

Statistical Outlier Detection:

Identifies major outliers ( more than 1% of the total number of points fell outside the 99% CI range)
Detects minor outliers (more than 1% of the total number of points fell outside the 95% CI range)
Highlights regions of systematic disagreement

```matlab
% Perform deformable registration and BLD analysis
BLDMatchViaDCPR(rootDir, OARname, gridAverage, countthreshold);
```

Parameters:
- `rootDir`: Base directory containing BLD data (where BLD_Batch output is stored)
- `OARname`: (Optional) Name of specific OAR to process (e.g., 'Brainstem')
  - If omitted, processes all OARs found in the directory
- `gridAverage`: (Optional) Grid size for point cloud downsampling (default: 0.05)
- `countthreshold`: (Optional) Maximum point count threshold (default: 5000)

Directory structure requirements:
```
output/
├── _Ref/
│   └── OARname_MRN-Ref.mat    # Template contour
├── BiasesResult/          # Registration results
├── RmseResult/            # Registration error metrics
├── DetectedError/         # Identified discrepancies
└── DIRMatchdata/          # Matched contour data
```

Example usage:
```matlab
% Process single OAR
BLDMatchViaDCPR('./output/', 'Brainstem', 0.05, 5000);

% Process all OARs
BLDMatchViaDCPR('./output/');
```

The function will:
1. Load the template contour from _Ref directory
2. Register each reference contour to the template using CPD
3. Map BLD values from reference to template space
4. Generate comprehensive analysis outputs including:
   - Point-wise disagreement maps
   - Registration quality metrics
   - Statistical summaries
   - Error detection reports
  
5. **Visualization**

The DR-BLD toolkit includes Python-based visualization tools for comprehensive 3D medical data analysis. These tools provide specialized visualization of disagreements data with customizable colormaps and 3D coordinate systems.
a.**Local Disagreement Analysis**
```python
colormapMultiFigmmRawData.py

# Process mean BLD files
process_directory(
    directory="path/to/data/BiasesResult",
    pattern="_Brainstem_Result*.csv"
)
```
Features:
- Diverging colormap (matplotlib.cm.coolwarm)
- Symmetric range based on 2nd and 98th percentiles
- Visualizes areas of systematic disagreement 

b. **Standard Deviation Analysis**
```python
colormapMultiFigmmRawData.py

# Process standard deviation BLD files
process_directory(
    directory="path/to/data/BiasesResult",
    pattern="_Brainstem-std_Result*.csv"
)
```
Features:
- Purple colormap (matplotlib.cm.Purples)
- Symmetric range based on 2nd and 98th percentiles
- Highlights variation in contour agreement

c. **Statistical Outlier Visualization**
```python
colormapMultiFigmmRawData.py

# Process detected outlier files
process_directory(
    directory="path/to/data/DetectedError",
    pattern="_DetectError_Brainstem*.csv"
)
```
Features:
- Diverging colormap (matplotlib.cm.coolwarm)
- Individual case analysis beyond confidence intervals
- Visualizes both major (99% CI) and minor (95% CI) outliers
- Highlights specific regions of significant disagreement
  
### Directory Structure
```
visualization_output/
├── BiasesResult/
│   ├── _Brainstem_Result.csv    # Mean visualization
│   └── _Brainstem-std_Result.csv    # Standard deviation disagreement visualization
└── DetectedError/
    ├── _DetectError_Brainstem_Case1_99CI.csv    # Major outlier visualization
    └── _DetectError_Brainstem_Case2_95CI.csv     # Minor outlier visualization
```

### Key Features
- Interactive 3D visualization
- Customizable colormaps for different analysis types
- Batch processing capabilities

### Usage Example
```python
# Complete visualization workflow
from medical_visualization import process_directory

# Process systematic disagreements
process_directory(
    directory="./BiasesResult",
    pattern="_*.csv"
)

# Process statistical outliers
process_directory(
    directory="./DetectedError",
    pattern="_DetectError_*.csv"
)
```

### Advanced Usage

#### Case-by-Case Visualization

For detailed analysis of individual cases, use f_BLD_visualization to compare specific reference and template contours:

```matlab
% Visualize individual case comparison
f_BLD_visualization(referenceFile, templateFile, countthreshold, pythonpath);
```
Parameters:
- `referenceFile`: Path to .mat file containing reference contour data
- `templateFile`: Path to OAR_MRN-Ref.mat file containing template contour data
- `countthreshold`: (Optional) Maximum point count before downsampling (default: 5000)
- `pythonpath`: (Optional) Path to Python executable for colormap generation

Example usage:
```matlab
% Basic usage with default parameters
f_BLD_visualization('./DIRMatchdata/case1.mat', ...
                   './output/_Ref/Brainstem_MRN-Ref.mat');

% Custom parameters
f_BLD_visualization('./DIRMatchdata/case1.mat', ...
                   './output/_Ref/Brainstem_MRN-Ref.mat', ...
                   3000, ...
                   'C:/Python37/python.exe');
```
**Function Features:**
- Point cloud alignment and registration
- Automatic downsampling for large datasets
- Interactive 3D visualization of:
  - Reference vs. template point clouds
  - Pre- and post-registration comparisons
  - Local disagreement patterns
- Custom colormap generation for visualization



## Output Files

The toolkit generates several types of output files:

- CSV files with quantitative metrics
- MAT files containing intermediate results
- Visualization plots for quality assurance
- Log files for process tracking

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.


## Citation

If you find this toolkit is useful and use this in your research, please cite:
```

```

## Support

For questions and support, please open an issue in the GitHub repository.

## Acknowledgments

Special thanks to contributors and researchers who have helped test and improve this toolkit.

