%% orgStudyList - Organize and match DICOM structure names with lookup table
% This function reads DICOM RTSTRUCT files and matches structure names with 
% a provided lookup table, creating a CSV report of matched structures.
%
% Syntax:
%   orgStudyList(rootfolder, lookupfile, listfile)
%
% Inputs:
%   rootfolder  - String, directory containing RTSTRUCT DICOM files
%   lookupfile  - String, path to CSV file containing structure name lookup table
%   listfile    - String, output CSV file path for matched structures
%
% Example:
%   orgStudyList('path/to/RTSTRUCT/', 'lookup.csv', 'matched_list.csv')
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

function orgStudyList(rootfolder, lookupfile, listfile)
    % Input validation
    validateattributes(rootfolder, {'char', 'string'}, {'nonempty'});
    validateattributes(lookupfile, {'char', 'string'}, {'nonempty'});
    validateattributes(listfile, {'char', 'string'}, {'nonempty'});
    
    % Get directory listing excluding '.' and '..'
    fileList = dir(rootfolder);
    fileList = fileList(~ismember({fileList.name}, {'.', '..'}));
    
    % Read lookup table
    lookupTable = readtable(lookupfile, 'delimiter', ',', 'ReadVariableNames', false);
    [numStructures, numAliases] = size(lookupTable);
    
    % Open output file
    fileID = fopen(listfile, 'w');
    if fileID == -1
        error('Failed to open output file: %s', listfile);
    end
    
    try
        % Write CSV header
        writeHeader(fileID, lookupTable);
        
        % Process each RTSTRUCT file
        processFiles(fileID, fileList, lookupTable, numStructures, numAliases);
        
    catch ME
        fclose(fileID);
        rethrow(ME);
    end
    
    % Close output file
    fclose(fileID);
end

function writeHeader(fileID, lookupTable)
    % Write base headers
    fprintf(fileID, 'Fname,StudyInstanceUID,ImageSeriesInstanceUID,RTSTRUCTInstanceUID');
    
    % Write structure name headers from lookup table
    for i = 1:size(lookupTable, 1)
        fprintf(fileID, ',%s', lookupTable{i,1}{1});
    end
    fprintf(fileID, '\n');
end

function processFiles(fileID, fileList, lookupTable, numStructures, numAliases)
    for i = 1:numel(fileList)
        disp(['Processing: ', fileList(i).name]);
        
        try
            % Construct full file path
            fullPath = fullfile(fileList(i).folder, fileList(i).name);
            
            % Read DICOM info
            roi = dicominfo(fullPath, 'UseVRHeuristic', false);
            
            % Extract ROI names
            roiNames = extractROINames(roi);
            
            % Write file information
            writeFileInfo(fileID, fullPath, roi);
            
            % Match and write structure names
            matchStructures(fileID, roiNames, lookupTable, numStructures, numAliases);
            
        catch ME
            warning('Error processing file %s: %s', fileList(i).name, ME.message);
            continue;
        end
    end
end

function roiNames = extractROINames(roi)
    % Extract ROI names from DICOM structure
    roiFields = fieldnames(roi.StructureSetROISequence);
    numStructures = length(roiFields);
    roiNames = cell(1, numStructures);
    
    for k = 1:numStructures
        roiNames{k} = roi.StructureSetROISequence.(roiFields{k}).ROIName;
    end
end

function writeFileInfo(fileID, fullPath, roi)
    % Write basic file information to CSV
    fprintf(fileID, '%s,%s,%s,%s', ...
        fullPath, ...
        roi.StudyInstanceUID, ...
        roi.ReferencedFrameOfReferenceSequence.Item_1.RTReferencedStudySequence.Item_1.RTReferencedSeriesSequence.Item_1.SeriesInstanceUID, ...
        roi.SeriesInstanceUID);
end

function matchStructures(fileID, roiNames, lookupTable, numStructures, numAliases)
    % Match ROI names with lookup table entries
    for i = 1:numStructures
        found = false;
        
        for j = 1:numAliases
            name1 = lookupTable{i,j}{1};
            
            if isempty(name1)
                break;
            end
            
            % Check for matching structure name
            matchIdx = find(strcmpi(roiNames, name1), 1);
            if ~isempty(matchIdx)
                fprintf(fileID, ',%s', roiNames{matchIdx});
                found = true;
                break;
            end
        end
        
        if ~found
            fprintf(fileID, ', ');
        end
    end
    fprintf(fileID, '\n');
end

%% dicomRecursiveFinder - Recursively find DICOM files of specified modality
% This function recursively searches directories for DICOM files of a specific
% modality.
%
% Inputs:
%   infolder - String, path to search directory
%   ModTag   - String, DICOM modality to search for (e.g., 'RTSTRUCT')
%
% Returns:
%   flist    - Array of file information structures

function [flist] = dicomRecursiveFinder(infolder, ModTag)
    % Get directory listing excluding '.' and '..'
    dirList = dir(infolder);
    dirList = dirList(~ismember({dirList.name}, {'.', '..'}));
    flist = [];
    
    % Process each item in directory
    for i = 1:numel(dirList)
        if dirList(i).isdir
            % Recursively process subdirectories
            subPath = fullfile(dirList(i).folder, dirList(i).name);
            flistTemp = dicomRecursiveFinder(subPath, ModTag);
            flist = [flist, flistTemp]; %#ok<AGROW>
        else
            % Check if file is DICOM of specified modality
            try
                filePath = fullfile(dirList(i).folder, dirList(i).name);
                info = dicominfo(filePath, 'UseVRHeuristic', false);
                
                if strcmpi(info.Modality, ModTag)
                    flist = [flist, dirList(i)]; %#ok<AGROW>
                end
            catch ME
                warning('Error reading file %s: %s', dirList(i).name, ME.message);
            end
        end
    end
end