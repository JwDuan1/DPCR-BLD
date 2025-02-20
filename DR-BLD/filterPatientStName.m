%% filterPatientStName - Extract RTSTRUCT information from DICOM files
% This function recursively searches through patient folders to find RTSTRUCT 
% DICOM files and extracts relevant structure information into a CSV file.
%
% Syntax:
%   filterPatientStName(inrootfolder, outcsv)
%
% Inputs:
%   inrootfolder - String, root directory containing patient folders
%   outcsv      - String, path to output CSV file
%
% Example:
%   filterPatientStName('/data/HNSCC-3DCT-RT/', 'output_structures.csv')
%
% Authors: 
%   Jingwei Duan, Ph.D.
%   Quan Chen, Ph.D.
%
% Date: February 2025
% Version: 1.0
% License: MIT License
%
% Copyright (c) 2025 Jingwei Duan, Quan Chen
%
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.

function filterPatientStName(inrootfolder, outcsv)
    % Input validation
    validateattributes(inrootfolder, {'char', 'string'}, {'nonempty'});
    validateattributes(outcsv, {'char', 'string'}, {'nonempty'});
    
    % Get directory listing excluding '.' and '..'
    dirList = dir(inrootfolder);
    dirList = dirList(~ismember({dirList.name}, {'.', '..'}));
    
    % Open output CSV file
    fileID = fopen(outcsv, 'w');
    if fileID == -1
        error('Failed to open output file: %s', outcsv);
    end
    
    % Write CSV header
    fprintf(fileID, 'startfolder,structureName,studydate,StudyInstanceUID,ImageSeriesInstanceUID');
    fprintf(fileID, '\n');
    
    % Process each patient folder
    for i = 1:numel(dirList)
        patientName = dirList(i).name;
        
        % Skip if not a directory
        if dirList(i).isdir == 0
            disp(['Processing patient: ', patientName]);
            patientFolder = fullfile(inrootfolder, patientName);
            processPatientFolder(patientFolder, fileID);
        end
    end
    
    % Close output file
    fclose(fileID);
end

%% processPatientFolder - Process individual patient folder
% Recursively processes a patient folder to find and analyze RTSTRUCT files
%
% Inputs:
%   folderPath - String, path to the folder to process
%   fileID     - File identifier for output CSV
%
% Returns:
%   returnval  - Boolean, true if processing was successful

function [returnval] = processPatientFolder(folderPath, fileID)
    % Get directory listing
    dirContents = dir(folderPath);
    
    if dirContents(1).isdir == 0
        % Process single file
        try
            % Read DICOM info with error handling
            info = dicominfo(folderPath, 'UseVRHeuristic', false);
            
            % Check if file is an RTSTRUCT
            if strcmp(info.Modality, 'RTSTRUCT')
                processRTSTRUCT(info, folderPath, fileID);
            end
            
            returnval = true;
        catch ME
            warning('Error processing file %s: %s', folderPath, ME.message);
            returnval = false;
        end
    else
        % Recursively process subdirectory
        newPath = fullfile(folderPath, '/');
        returnval = processPatientFolder(newPath, fileID);
    end
end

%% processRTSTRUCT - Process RTSTRUCT DICOM file
% Extracts and writes structure information from RTSTRUCT file to CSV
%
% Inputs:
%   info       - DICOM info structure
%   filePath   - String, path to the DICOM file
%   fileID     - File identifier for output CSV

function processRTSTRUCT(info, filePath, fileID)
    % Get structure set name (prefer StructureSetName over StructureSetLabel)
    structureSetName = '';
    if isfield(info, 'StructureSetName')
        structureSetName = info.StructureSetName;
    elseif isfield(info, 'StructureSetLabel')
        structureSetName = info.StructureSetLabel;
    end
    
    % Write basic file information
    fprintf(fileID, '%s,%s,%s,%s,%s,', ...
        filePath, ...
        structureSetName, ...
        info.StudyDate, ...
        info.StudyInstanceUID, ...
        info.SeriesInstanceUID);
    
    % Process ROI names
    if isfield(info, 'StructureSetROISequence')
        roiFields = fieldnames(info.StructureSetROISequence);
        numStructures = length(roiFields);
        
        % Initialize cell array for ROI names
        roiNames = cell(1, numStructures);
        
        % Extract ROI names
        for k = 1:numStructures
            roiNames{k} = info.StructureSetROISequence.(roiFields{k}).ROIName;
            fprintf(fileID, '%s,', roiNames{k});
        end
    end
    
    % End the CSV line
    fprintf(fileID, '\n');
end