# DR-BLD
Deformable Registration-Based Bidirectional Local Distance (DR-BLD): A Methodology for Systematic Evaluation and Visualization of Local Disagreements in Clinical Auto-Contouring
# Medical Image Contour Analysis Toolkit

A comprehensive toolkit for systematically evaluate local disagreements in auto-segmentation.

## Features

- Bidirectional Local Distance (BLD) calculation between reference and test contours
- Point cloud registration and comparison using Coherent Point Drift (CPD)
- Structure set name matching and organization
- Automated quality metrics calculation including:
  - Dice similarity coefficient
  - Hausdorff distance
  - Surface Dice
  - Center of mass distance
  - Mean surface distance

## Prerequisites

- MATLAB R2019b or later
- Python 3.7+ with the following packages:
  - matplotlib
  - numpy
  - argparse
- Required MATLAB toolboxes:
  - Image Processing Toolbox
  - Computer Vision Toolbox
  - Statistics and Machine Learning Toolbox

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/contour-analysis-toolkit.git
```

2. Add the toolkit directory to your MATLAB path:
```matlab
addpath(genpath('/path/to/contour-analysis-toolkit'))
```

3. Ensure Python is properly configured in MATLAB:
```matlab
pyversion % Should show Python 3.7 or later
```

## Usage

### Basic Workflow

1. **Organize Structure Sets**
```matlab
% Create structure name lookup table
filterPatientStName('input/folder/', 'stnames.csv');

% Match structure names
orgStudyList('input/folder/', 'lookup.csv', 'matched_list.csv');
```

2. **Calculate BLD Metrics**
```matlab
% Compare reference and test contours
BLD_Batch('reflist.csv', 'testlist.csv', 'results.csv', './output/', '');
```

3. **Visualize Results**
```matlab
% Generate visualization of contour differences
f_BLD_visualization('reference.mat', 'template.mat', 5000, 'python');
```

### Advanced Usage

For complex analysis pipelines, use the BLDMatchViaDCPR function:
```matlab
% Perform registration and analysis
BLDMatchViaDCPR('rootDir/', 'OARname', 0.05, 5000);
```

## Key Functions

- `BLD_Batch.m`: Main function for batch processing contour comparisons
- `BLDMatchViaDCPR.m`: Advanced registration and analysis
- `f_BLD_visualization.m`: Visualization tools for results
- `filterPatientStName.m`: Structure set organization
- `orgStudyList.m`: Structure name matching

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

## Authors

- Jingwei Duan, Ph.D.
- Quan Chen, Ph.D.

## Citation

If you use this toolkit in your research, please cite:
```
@software{medical_contour_analysis,
  author = {Duan, Jingwei and Chen, Quan},
  title = {Medical Image Contour Analysis Toolkit},
  year = {2025},
  version = {1.0}
}
```

## Support

For questions and support, please open an issue in the GitHub repository.

## Acknowledgments

Special thanks to contributors and researchers who have helped test and improve this toolkit.
