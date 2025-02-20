%% BLDMatchViaDCPR - Boundary-based Local Distance Match via Deformable CPD Registration
% This function performs point cloud registration and error analysis for medical 
% structure boundaries using Coherent Point Drift (CPD) registration.
%
% Authors: 
%   Jingwei Duan, Ph.D.@ duan.jingwei01@gmail.com
%   Quan Chen, Ph.D.
%
% Date: February 2025
% Version: 1.0
% License: MIT License
%
% Syntax:
%   BLDMatchViaDCPR(rootDir, OARname, gridAverage, countthreshold)
%
% Inputs:
%   rootDir        - Base directory containing BLD data
%   OARname        - (Optional) Name of specific OAR to process. If omitted, processes all OARs
%   gridAverage    - (Optional) Grid size for point cloud downsampling (default: 0.05)
%   countthreshold - (Optional) Maximum point count threshold (default: 5000)
%
% Example:
%   BLDMatchViaDCPR('path/to/data/', 'Stomach', 0.05, 5000)
%   BLDMatchViaDCPR('path/to/data/')


function BLDMatchViaDCPR(rootDir, OARname, gridAverage, countthreshold)
    % Initialize output directories
    rootDirResult = [rootDir 'BiasesResult\'];
    mkdir(rootDirResult);
    mkdir([rootDir 'RmseResult\']);
    outputtxt = [rootDir 'BiasesResult\','BLDMatchViaDCPR_Log_output.txt'];
    diary(outputtxt);

    % Set default parameters
    if nargin < 4
        countthreshold = 5000;
    end
    if nargin < 3
        gridAverage = 0.05;
    end

    % Determine operation mode
    if nargin < 2
        searchPattern = fullfile(rootDir, '\_Ref', '*-Ref.mat');
        matFiles = dir(searchPattern);
        recrusiveflag = 1;
        recrusivetimes = length(matFiles);
        disp('**********************Batch OARs mode**********************')
    else
        recrusiveflag = 0;
        recrusivetimes = 1;
        disp('**********************Single OAR mode**********************')
    end

    % Initialize OAR array
    if recrusiveflag == 0
        OAR{1} = OARname;
    else
        for i = 1:recrusivetimes
            parts = split((matFiles(i).name), '_');
            if length(parts) == 2
                OAR{i} = parts{1};
            elseif length(parts) == 5
                OAR{i} = [parts{1} '_' parts{2} '_' parts{3}];
            else
                OAR{i} = [parts{1} '_' parts{2}];
            end
        end
    end

    % Process each OAR
    for ii = 1:recrusivetimes
        BLDfoundFiles = {};
        disp(['====================' OAR{ii} '===================='])
        dirs = dir(fullfile(rootDir, [OAR{ii} '_*']));
        
        % Initialize result files
        resultfile = [rootDirResult '_' OAR{ii} '_Result.csv'];
        resultfile_std = [rootDirResult '_' OAR{ii} '-std' '_Result.csv'];
        resultfile_rmse = [rootDir 'RmseResult\' 'rmse_' OAR{ii} '_Result.csv'];
        fp = fopen(resultfile_rmse, 'w');
        fprintf(fp, 'tstfname, rmse(cm)\n');

        % Get BLD files
        for k = 1:length(dirs)
            BLDfoundFiles{end+1} = fullfile(dirs(k).folder, dirs(k).name);
        end

        % Load and process Template points
        pattern = sprintf([OAR{ii} '*-Ref.mat'], ii);
        tempfile = dir(fullfile([rootDir, '_Ref\'], pattern));
        Templatepts = load([tempfile.folder '\'  tempfile.name]);
        Templatepts = Templatepts.refpts;
        Templatepts = AlignCOM(Templatepts);
        
        % Initialize point cloud
        temp = zeros(length(Templatepts),1);
        TemplateptsCloud = pointCloud(Templatepts, Intensity=temp);
        
        % Check point count and downsample if needed
        pointCount = TemplateptsCloud.Count;
        if pointCount - countthreshold > 0
            disp(['The OAR points number is ', num2str(TemplateptsCloud.Count), ' larger than threshold ', num2str(countthreshold)]);
            disp('-----------Downsample-------------')
        else
            disp(['The OAR points number is ', num2str(TemplateptsCloud.Count)]);
            flag = 1;
        end
    
        while pointCount - countthreshold > 0
            TemplateptsCloudDowmsampled = pcdownsample(TemplateptsCloud, 'gridAverage', gridAverage);
            pointCount = TemplateptsCloudDowmsampled.Count;
            if pointCount > countthreshold
                gridAverage = gridAverage + 0.001;
            else
                flag = 0;
                disp(['The downsample OAR points number is ', num2str(TemplateptsCloudDowmsampled.Count)]);
            end
        end
        
        % Perform DIR matching
        disp(['-------------------DIR-------------------------']);
        for i = 1:length(dirs)
            fprintf('%s - %s\n', dirs(i).name, datestr(now));
            BLDsingle = load(BLDfoundFiles{i});
            
            % Align points
            BLDsingle.refpts = AlignCOM(BLDsingle.refpts);
            BLDsingle.refptswithbld(:,1:3) = AlignCOM(BLDsingle.refptswithbld(:,1:3));
            
            % Create point cloud
            evalptsCloud = pointCloud(BLDsingle.refptswithbld(:,1:3), Intensity=BLDsingle.refptswithbld(:,4));
            
            % Handle downsampling
            if flag == 1
                evalptsCloudDowmsampled = evalptsCloud;
                refptsCloudDowmsampled = TemplateptsCloud;
            else
                evalptsCloudDowmsampled = pcdownsample(evalptsCloud, 'gridAverage', gridAverage);
                refptsCloudDowmsampled = TemplateptsCloudDowmsampled;
            end
            
            % Perform registration
            if isequal(evalptsCloudDowmsampled.Location, refptsCloudDowmsampled.Location)
                disp('Point clouds are identical. No registration needed.');
                tform = rigid3d(eye(3), [0 0 0]);
                evalmovingReg = evalptsCloudDowmsampled;
                rmse = 0;
            else
                [tform, evalmovingReg, rmse] = pcregistercpd(evalptsCloudDowmsampled, refptsCloudDowmsampled, 'MaxIterations', 30);
            end

            % Record RMSE
            fprintf('Root Mean Square Error (RMSE): %.4f cm\n', rmse);
            fprintf(fp, '%s,%f\n', BLDfoundFiles{i}, rmse);
            
            % Find nearest neighbors
            for j = 1:refptsCloudDowmsampled.Count
                [indices,dists] = findNearestNeighbors(evalmovingReg, refptsCloudDowmsampled.Location(j,:), 1);
                if isempty(indices)
                    indices = 1;
                end
                refptsCloudDowmsampled.Intensity(j) = evalmovingReg.Intensity(indices);
            end
            
            intensitytemp{i} = refptsCloudDowmsampled.Intensity;
        end

        % Process results
        if flag == 1
            TemplateptsCloud.Intensity = mean(cell2mat(intensitytemp), 2);
            TemplateptsCloud_stdev = std(cell2mat(intensitytemp), 0, 2);
            Templatepts_matched = [TemplateptsCloud.Location TemplateptsCloud.Intensity];
            Templatepts_matched_std = [TemplateptsCloud.Location TemplateptsCloud_stdev];
        else
            TemplateptsCloudDowmsampled.Intensity = mean(cell2mat(intensitytemp), 2);
            TemplateptsCloud_stdev = std(cell2mat(intensitytemp), 0, 2);
            Templatepts_matched = [TemplateptsCloudDowmsampled.Location TemplateptsCloudDowmsampled.Intensity];
            Templatepts_matched_std = [TemplateptsCloudDowmsampled.Location TemplateptsCloud_stdev];
        end

        % Calculate tolerance bounds
        ToleranceLow = Templatepts_matched(:,4) - 1.96*Templatepts_matched_std(:,4);
        ToleranceHigh = Templatepts_matched(:,4) + 1.96*Templatepts_matched_std(:,4);
        ToleranceLow_99CI = Templatepts_matched(:,4) - 2.576*Templatepts_matched_std(:,4);
        ToleranceHigh_99CI = Templatepts_matched(:,4) + 2.576*Templatepts_matched_std(:,4);

        % Error detection
        errorindex = [];
        errorindex_99CI = [];
        errorindex_95CI = [];
        errorindex_95CI_01precent = [];
        
        for i = 1:length(intensitytemp)
            currIntensity = intensitytemp{i};
            outOfRangeIndices{i} = find(currIntensity < ToleranceLow | currIntensity > ToleranceHigh);
            outOfRangeIndices_99CI{i} = find(currIntensity < ToleranceLow_99CI | currIntensity > ToleranceHigh_99CI);

            if length(outOfRangeIndices_99CI{i}) > 0.01 * length(currIntensity)
                errorindex_99CI = [errorindex_99CI, i];
            end
            if length(outOfRangeIndices{i}) > 0.001 * length(currIntensity)
                errorindex_95CI_01precent = [errorindex_95CI_01precent, i];
            end
            if length(outOfRangeIndices{i}) > 0.01 * length(currIntensity)
                errorindex_95CI = [errorindex_95CI, i];
            end
        end

        % Process and save error detection results
        errorindex = unique([errorindex_99CI, errorindex_95CI]);
        disp(['====================' 'Error Detection' '===================='])
        Templatepts_matched(:,1:3) = Templatepts_matched(:,1:3) - mean(Templatepts_matched(:,1:3));
        
        DetectError_foldername = [rootDir 'DetectedError\'];
        mkdir(DetectError_foldername);
        
        for i = 1:length(errorindex)
            errorindex_i = errorindex(i);
            original_name = dirs(errorindex_i).name;
            [~, name, ext] = fileparts(original_name);
            
            percentageIncorrect = (length(outOfRangeIndices{errorindex_i}) / length(currIntensity)) * 100;
            
            if ismember(errorindex_i, errorindex_99CI)
                reason = '> 1% of number of points outside the 99% CI range';
                tag = '99CI';
            elseif ismember(errorindex_i, errorindex_95CI)
                reason = '> 1% of number of points outside the 95% CI range';
                tag = '95CI';
            end
            
            disp(sprintf('Possible Error: %s; Number of incorrect points: %d (%.2f%%); Reason: %s', ...
                name, length(outOfRangeIndices{errorindex_i}), percentageIncorrect, reason));
            
            modified_name = [DetectError_foldername '_DetectError_' name '_' tag '.csv'];
            error4dpoints = [Templatepts_matched(:,1:3), intensitytemp{errorindex_i}];
            csvwrite(modified_name, error4dpoints);
        end

        % Save DIR match data
        afterDIR_foldername = [rootDir 'DIRMatchdata\'];
        mkdir(afterDIR_foldername);
        for i = 1:length(intensitytemp)
            original_name = dirs(i).name;
            [~, name, ext] = fileparts(original_name);
            afterDIR_path = [afterDIR_foldername name '.csv'];
            afterDIR_data = [Templatepts_matched(:,1:3), intensitytemp{i}];
            csvwrite(afterDIR_path, afterDIR_data);
        end

        % Save final results
        Templatepts_matched_std(:,1:3) = Templatepts_matched_std(:,1:3) - mean(Templatepts_matched_std(:,1:3));
        csvwrite(resultfile, Templatepts_matched);
        csvwrite(resultfile_std, Templatepts_matched_std);
        disp(['====================' OAR{ii} ' Done' '===================='])
        clear intensitytemp
        
        diary off;
        fclose(fp);
    end
end