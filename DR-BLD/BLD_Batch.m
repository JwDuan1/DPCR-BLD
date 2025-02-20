%% BLD_Batch - Evaluate Medical Image Segmentation Performance
% This function compares test and reference DICOM structure sets and calculates
% various metrics including Dice score, Hausdorff distance, and surface Dice.
% Also, it generate the BLD calculation on Ref-test pair
%
% Syntax:
%   BLD_Batch(reflist, tstlist, resultfile, bldpath, winsuffix)
%
% Inputs:
%   reflist    - String, path to CSV file containing reference (e.g., Manual) structure list 
%   tstlist    - String, path to CSV file containing test (e.g., AI) structure list
%   resultfile - String, path to output CSV file for metrics
%   bldpath    - String, path to save intermediate results
%   winsuffix  - String, optional prefix for Windows path (default: '')
% 
% Example usage:
% BLD_Batch('reflist.csv', 'autolist.csv', 'results.csv', './output/', '')

% Authors: 
%   Jingwei Duan, Ph.D.@ duan.jingwei01@gmail.com
%   Quan Chen, Ph.D.
%
% Date: February 2025
% Version: 1.0
% License: MIT License
%
% Copyright (c) 2025 Jingwei Duan, Quan Chen

function BLD_Batch(reflist, tstlist, resultfile, bldpath, winsuffix)

if nargin < 5
    winsuffix = '';
end

% Create output directory
mkdir(bldpath);

% Initialize logging
logFile = [bldpath 'bld_batch_log.txt'];
logFp = fopen(logFile, 'w');
if logFp == -1
    error('Failed to create log file: %s', logFile);
end

try
    dualLog(logFp, '=== Starting BLD_Batch processing ===');
    dualLog(logFp, 'Reference list: %s', reflist);
    dualLog(logFp, 'Test list: %s', tstlist);
    dualLog(logFp, 'Result file: %s', resultfile);
    
    % Read input tables
    dualLog(logFp, 'Reading input tables...');
    reftable = readtable(reflist, 'delimiter', ',');
    tsttable = readtable(tstlist, 'delimiter', ',');
    dualLog(logFp, 'Successfully loaded input tables');
    
    % Get table dimensions
    reftablesize = size(reftable);
    tsttablesize = size(tsttable);
    numnonstrs = 4;
    numstrs = tsttablesize(2) - numnonstrs;
    
    % Initialize metric arrays
    dice = zeros(tsttablesize(1), numstrs);
    inter = zeros(1, numstrs);
    refonly = zeros(1, numstrs);
    tstonly = zeros(1, numstrs);
    union = zeros(1, numstrs);
    meanD = zeros(1, numstrs);
    hd = zeros(numstrs, 11);
    surfaceDice = zeros(numstrs, 12);
    centerdistance = zeros(1, numstrs);
    tstNumObjects = zeros(1, numstrs);
    refNumObjects = zeros(1, numstrs);
    
    % Open results file
    fp = fopen(resultfile, 'w');
    if fp == -1
        error('Failed to open result file: %s', resultfile);
    end
    fprintf(fp, 'tstfname,reffname');

    % Define surface Dice thresholds
    surfaceDiceThresholdinCM = 0.05:0.05:0.6;
    numsurfaceDiceThresholdinCM = numel(surfaceDiceThresholdinCM);

    % Initialize metric labels
    indlabel = {'dice', 'recall', 'precision'};
    for i = 90:100 
        indlabel{end+1} = ['hausdorff' num2str(i) '(cm)'];
    end
    indlabel{end+1} = 'meanSurDist(cm)';
    for i = 1:numsurfaceDiceThresholdinCM 
        indlabel{end+1} = ['SurfaceDice' num2str(surfaceDiceThresholdinCM(i)*10) 'mm'];
    end
    labelmore = {'tstNumObjects','refNumObjects','COMDistance(cm)'};
    for i = 1:numel(labelmore) 
        indlabel{end+1} = labelmore{i};
    end

    % Write header
    inds = numel(indlabel);
    for i = 1:numstrs
        stname = tsttable.Properties.VariableNames{numnonstrs+i};
        for j = 1:inds
            fprintf(fp, ',%s', [stname, '_', indlabel{j}]);
        end
    end
    fprintf(fp, '\n');

    % Process each case
    dualLog(logFp, 'Beginning case processing (%d total cases)', tsttablesize(1));
    for i = 1:tsttablesize(1)
        fname = tsttable{i,1}{1};
        dualLog(logFp, '-> Processing case %d/%d: %s', i, tsttablesize(1), fname);
        
        % Find matching reference
        found = 0;
        for j = 1:reftablesize(1)
            if strcmp(reftable{j,2}{1}, tsttable{i,2}{1})
                found = j;
                break;
            end
        end
        if ~found
            dualLog(logFp, 'WARNING: No matching reference found for: %s', fname);
            continue;
        end
        
        % Load DICOM info
        reffname = reftable{found,1}{1};
        try
            % Construct DICOM paths
            if nargin >= 5
                tstDicomPath = [winsuffix fname(6:end)];
                refDicomPath = [winsuffix reffname(6:end)];
            else
                tstDicomPath = fname;
                refDicomPath = reffname;
            end
            
            dualLog(logFp, '   Loading test DICOM: %s', tstDicomPath);
            dualLog(logFp, '   Loading reference DICOM: %s', refDicomPath);
            
            infotst = dicominfo(tstDicomPath, 'UseVRHeuristic', false);
            inforef = dicominfo(refDicomPath, 'UseVRHeuristic', false);
            
            dualLog(logFp, '   Successfully loaded both DICOM files');
        catch ME
            dualLog(logFp, 'ERROR: Failed to load DICOM data - %s', ME.message);
            continue;
        end
        
        fprintf(fp, '%s, %s ', fname, reffname);
        casePaths{i} = fname;

        % Process each structure
        for k = (numnonstrs+1):tsttablesize(2)
            idx = k - numnonstrs;
            stname = tsttable.Properties.VariableNames{k};
            dualLog(logFp, '   Processing structure: %s', stname);
            
            % Get structure data
            structtst = [];
            structref = [];
            if iscell(tsttable{i,k}) && iscell(reftable{found,k})
                tstname = tsttable{i,k}{1};
                refname = reftable{found,k}{1};
                structtst = getStructbyName(infotst, tstname);
                structref = getStructbyName(inforef, refname);
            end

            % Handle empty structures
            if isempty(structref) || isempty(fieldnames(structref))
                dualLog(logFp, '      Empty reference structure');
                inter(idx) = 0;
                refonly(idx) = 0;
                tstonly(idx) = 1;
                union(idx) = 1;
                meanD(idx) = NaN;
                hd(idx,:) = NaN;
                centerdistance(idx) = NaN;
                tstNumObjects(idx) = NaN;
                refNumObjects(idx) = NaN;
                surfaceDice(idx,:) = NaN;
                continue;
            end

            if isempty(structtst) || isempty(fieldnames(structtst))
                dualLog(logFp, '      Empty test structure');
                inter(idx) = 0;
                refonly(idx) = 1;
                tstonly(idx) = 0;
                union(idx) = 1;
                meanD(idx) = NaN;
                hd(idx,:) = NaN;
                centerdistance(idx) = NaN;
                tstNumObjects(idx) = NaN;
                refNumObjects(idx) = NaN;
                surfaceDice(idx,:) = NaN;
                continue;
            end

            % Determine structure type and compute metrics
            flag = determineStructureType(stname);
            % dualLog(logFp, '      Computing metrics (type: %s)', flag);

            % Compute overlap metrics
            [inter(idx), refonly(idx), tstonly(idx), union(idx), ...
             hd(idx,:), meanD(idx), surfaceDice(idx,:), ...
             refNumObjects(idx), tstNumObjects(idx), ...
             centerdistance(idx)] = computeOverlap(structref, structtst, ...
             flag, surfaceDiceThresholdinCM, bldpath, stname, inforef.PatientID);

            dualLog(logFp, '      Metrics computed successfully');
        end

        % Calculate final metrics
        if any(2*inter./(inter*2+refonly+tstonly) == 0)
            dice(i,:) = NaN;
        else
            dice(i,:) = 2*inter./(inter*2+refonly+tstonly);
        end
        tpr(i,:) = inter./(inter+refonly);
        ppv(i,:) = inter./(inter+tstonly);
        
        % Store Hausdorff metrics
        for j = 1:11
            hdName = sprintf('hd%ds', j+89);
            eval([hdName '(i,:) = hd(:,' num2str(j) ')'';']);
        end

        meands(i,:) = meanD;
        
        % Store surface Dice metrics
        for j = 1:12
            eval(['surfdices' num2str(j) '(i,:) = surfaceDice(:,' num2str(j) ')'';']);
        end

        % Store other metrics
        tstNumObjectsoutput(i,:) = tstNumObjects;
        refNumObjectsoutput(i,:) = refNumObjects;
        centerdistanceoutput(i,:) = centerdistance;

        % Write results
        for ii = 1:numstrs
            fprintf(fp, repmat(',%f', 1, numel(indlabel)), ...
                dice(i,ii), tpr(i,ii), ppv(i,ii), ...
                hd90s(i,ii), hd91s(i,ii), hd92s(i,ii), hd93s(i,ii), ...
                hd94s(i,ii), hd95s(i,ii), hd96s(i,ii), hd97s(i,ii), ...
                hd98s(i,ii), hd99s(i,ii), hd100s(i,ii), meands(i,ii), ...
                surfdices1(i,ii), surfdices2(i,ii), surfdices3(i,ii), ...
                surfdices4(i,ii), surfdices5(i,ii), surfdices6(i,ii), ...
                surfdices7(i,ii), surfdices8(i,ii), surfdices9(i,ii), ...
                surfdices10(i,ii), surfdices11(i,ii), surfdices12(i,ii), ...
                tstNumObjectsoutput(i,ii), refNumObjectsoutput(i,ii), ...
                centerdistanceoutput(i,ii));
        end
        fprintf(fp, '\n');
        dualLog(logFp, '   Case processing completed');
    end

    fclose(fp);
    dualLog(logFp, '=== Processing completed successfully ===');

catch ME
    dualLog(logFp, 'ERROR: %s', ME.message);
    dualLog(logFp, 'Stack trace: %s', getReport(ME));
    if exist('fp', 'var') && fp ~= -1
        fclose(fp);
    end
    rethrow(ME);
end

fclose(logFp);
end

function dualLog(fp, format, varargin)
    % Write timestamped message to both log file and command window
    timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS.FFF');
    message = sprintf(format, varargin{:});
    
    % Write to log file
    fprintf(fp, '[%s] %s\n', timestamp, message);
    
    % Write to command window
    fprintf('[%s] %s\n', timestamp, message);
end

function flag = determineStructureType(stname)
    % Determine structure type based on name
    if strcmpi(stname, 'SpinalCord1') || strcmpi(stname, 'Rectum1') || ...
       strcmpi(stname, 'FemoralHead_R') || strcmpi(stname, 'FemoralHead_L') || ...
       strcmpi(stname, 'SpinalCanal') || strcmpi(stname, 'Aorta') || ...
       strcmpi(stname, 'V_Venacava_I')
        flag = 'overlap';
    elseif strcmpi(stname, 'Esophagus')
        flag = 'refonly';
    else
        flag = 'default';
    end
end

function [inter, refonly, tstonly, union, meanD, hd, metrics] = initializeEmptyMetrics(k, numnonstrs)
    % Initialize empty metrics when structure is missing
    inter(k-numnonstrs) = 0;
    refonly(k-numnonstrs) = 1;
    tstonly(k-numnonstrs) = 0;
    union(k-numnonstrs) = 1;
    meanD(k-numnonstrs) = NaN;
    hd(k-numnonstrs,:) = NaN;
    metrics = struct('centerdistance', NaN, ...
                    'tstNumObjects', NaN, ...
                    'refNumObjects', NaN, ...
                    'surfaceDice', NaN);
end

% function [inter, refonly, tstonly, union, hd, meanD, surfaceDice, dpbyAng,MaxDistAxial,MaxDistSagittal,MaxDistCoronal,MissingSliceNumAxial,MissingSliceNumSagittal,MissingSliceNumCoronal,OutptsAxial,OutptsSagittal,OutptsCoronal,AxialOutpointSlice,CoronalOutpointSlice,SagittalOutpointSlice]=computeOverlap(structref, structtst,flag, surfaceDiceThresholdinCM)
function [inter, refonly, tstonly, union, hd, meanD, surfaceDice,refNumObjects,tstNumObjects,centerdistance]=computeOverlap(structref, structtst,flag, surfaceDiceThresholdinCM,bldpath,stname,MRN)


[refmins, refmaxs, refslicethickness, structref]=analyStruct(structref);
[tstmins, tstmaxs, tstslicethickness, structtst]=analyStruct(structtst);
numSDSCThresholdinCM = length(surfaceDiceThresholdinCM);
if nargin<3
    flag='default';
end
allmins=min(refmins, tstmins);% the min value for all single values
allmaxs=max(refmaxs, tstmaxs);% the max value for all single values


if(strcmpi(flag, 'refonly'))% if flag = 'refonly'
    allmins(3)=refmins(3);
    allmaxs(3)=refmaxs(3);
elseif(strcmpi(flag, 'overlap'))% if flag = 'overlap'
        allmins(3)=max(refmins(3), tstmins(3));
        allmaxs(3)=min(refmaxs(3), tstmaxs(3));
end

% setup the pixel dimension
header.x_pixdim=1/10;
header.y_pixdim=1/10;
header.z_pixdim=max([refslicethickness, tstslicethickness])/10;
if(header.z_pixdim==0)
    header.z_pixdim=0.1;
end

% setup the start point
header.x_start=allmins(1)/10; % x, y, z
header.y_start=allmins(2)/10;
header.z_start=-allmaxs(3)/10;

% setup the coordinate dimension
header.x_dim=ceil((allmaxs(1)-allmins(1))/10/header.x_pixdim)+2;
header.y_dim=ceil((allmaxs(2)-allmins(2))/10/header.y_pixdim)+2;
header.z_dim=ceil((allmaxs(3)-allmins(3))/10/header.z_pixdim)+1;% why Z +1 while X,y +2 % Zdimension = allmax-allmins at z directions +1

%reference contour mask
refmask=computeMaskfromStruct(structref, header, 'HFS');% unit changed, the pixdim = 0.1 (x,y) or 0.25(z). refmask xdim*ydim*zdim
[Bref, ~]=bwboundaries2Dstackfast(refmask);% Binary ref mask. [#ofpoints,3]
refpts=[];
tstpts=[];
for k=1:length(Bref)
    refpts=[refpts; Bref{k}]; % cell to array
end

% test contour mask
tstmask=computeMaskfromStruct(structtst, header, 'HFS');
[Btst, ~]=bwboundaries2Dstackfast(tstmask);% Binary test mask
if(isempty(Btst))
    hd=ones(1,11)*NaN;
    inter=NaN;
    union=NaN;
    tstonly=NaN;
    refonly=NaN;
    meanD=NaN;
    return;
end
for k=1:length(Btst)
    tstpts=[tstpts; Btst{k}];
end

% # of tst and ref points
numrefpts=size(refpts,1);
numtstpts=size(tstpts,1);

rawrefpts=refpts;
rawtstpts=tstpts;
refpts=refpts.*(ones(numrefpts,1)*[header.x_pixdim, header.y_pixdim, header.z_pixdim]);
tstpts=tstpts.*(ones(numtstpts,1)*[header.x_pixdim, header.y_pixdim, header.z_pixdim]);

[hd, meanD,~, dp, dq]=HausdorffDistPctile(refpts, tstpts,[90:100]);% output dp always be positive

ccref=bwconncomp(refmask);
cctst=bwconncomp(tstmask);
refNumObjects=ccref.NumObjects;
tstNumObjects=cctst.NumObjects;




refcnter=findMassCenter(refmask);
refmaskcenter=refcnter.*[header.x_pixdim, header.y_pixdim, header.z_pixdim];
tstcnter=findMassCenter(tstmask);
tstmaskcenter=tstcnter.*[header.x_pixdim, header.y_pixdim, header.z_pixdim];


ptsref = refpts;
tempsize=size(ptsref);
ptsref(tempsize(1)+1,:)=refmaskcenter;

ptstst = tstpts;
tempsize=size(ptstst);
ptstst(tempsize(1)+1,:)=tstmaskcenter;

centerdistance = pdist2(tstmaskcenter,refmaskcenter, 'euclidean');


in = refinsidetstpts(tstmask,rawrefpts);

bldtemp_total = bld(refpts,tstpts);

temp_refpts_size=size(refpts, 1);
% bldtemp_total_empty = zeros(temp_refpts_size, 1);
bldtemp_total(in == 0)=-1*bldtemp_total(in == 0);

refptswithbld = [refpts,bldtemp_total];




for ii=1:numSDSCThresholdinCM
    
    surfaceDice(ii)=(sum(dp<surfaceDiceThresholdinCM(ii))+sum(dq<surfaceDiceThresholdinCM(ii)))/(numel(dp)+numel(dq));
    APLdq(ii) = 0;
    for i =1: numel(dq)
        if dq(i)>surfaceDiceThresholdinCM(ii)
            APLdq(ii) = APLdq(ii) + dq(i);
        end
    end

end

hd95=hd(6);
hd98=hd(9);
bitMask=zeros([header.x_dim, header.y_dim, header.z_dim], 'uint8');
bitMask=bitset(bitMask, 1, refmask);
bitMask=bitset(bitMask, 2, tstmask);
inter=sum(bitMask(:)==3);
refonly=sum(bitMask(:)==1);
tstonly=sum(bitMask(:)==2);
union=sum(bitMask(:)>0);


output =[bldpath stname '_' MRN '.mat'];
% save (output,'refptswithbld','refmask','header');
save (output,'refptswithbld','refmask','tstmask','header','tstpts','refpts','rawtstpts','rawrefpts');

flag_visualization='NonVisualization';
if flag_visualization == 'NonVisualization'
    refptswithbldCOM=refptswithbld;
    refptswithbldCOM(:,1:3)=refptswithbldCOM(:,1:3)-mean(refptswithbldCOM(:,1:3));
    resultfile = [bldpath 'Visualize_' stname '_' MRN '.csv'];
    % csvwrite(resultfile,refptswithbldCOM);

    resultfileref = [bldpath 'Visualize_' stname '_' MRN '_ref' '.csv'];
    resultfiletst = [bldpath 'Visualize_' stname '_' MRN '_tst' '.csv'];
    tstptsCOM = tstpts;
    refptsCOM = refpts;
    tstptsCOM(:,1:3)=tstptsCOM(:,1:3)-mean(tstptsCOM(:,1:3));
    refptsCOM(:,1:3)=refptsCOM(:,1:3)-mean(refptsCOM(:,1:3));
    % csvwrite(resultfileref,refpts);
    % csvwrite(resultfiletst,tstpts);
end

end



function [centroid,MinDistance]=CentoridCenterDistance(mask,pts,header)
%refcentorid
centroid=findMassCenter(mask);
centroid=centroid.*[header.x_pixdim, header.y_pixdim, header.z_pixdim];
[~,~,~,~,MinDistance] = HausdorffDistPctile(pts, centroid,[90:100]);
end

function [centroid,MinDistance]=CentoridCenterDistance2D(mask,pts,header)
%refcentorid
centroid=findMassCenter2D(mask);
centroid=centroid.*[header.x_pixdim, header.y_pixdim];
centroid(3)=pts(1,3);
[~,~,~,~,MinDistance] = HausdorffDistPctile(pts, centroid,[90:100]);
end



function [mins, maxs, slicethickness, instruct]=analyStruct(instruct)
    temp=fieldnames(instruct);

    slicez(length(temp))=0; % the length of z direction reflects on length (temp) where temp = fildeneamrs (instructs) - the number of instruct
    for i=1:length(temp)
        if(isempty(instruct.(temp{i})))
            instruct=rmfield(instruct, temp{i});% remove the file if this slice is empty
        else
        numpoints=instruct.(temp{i}).NumberOfContourPoints; % number of contour point in this slice
        pts=zeros(3,numpoints); % 3 colums, numpoints Rows
        pts(:)=instruct.(temp{i}).ContourData(:);
        if(numpoints>1)
            tmpmins=min(pts');
            tmpmaxs=max(pts');
        else
            tmpmins=pts';
            tmpmaxs=pts';
        end
        slicez(i)=tmpmins(3);
        if(i<2)
            mins=tmpmins;
            maxs=tmpmaxs;
        else
            mins=min(mins, tmpmins);
            maxs=max(maxs, tmpmaxs);
        end
        end
    end
    if(~isempty(temp)&&isfield(instruct.Item_1, 'ContourSlabThickness'))
            slicethickness=instruct.(temp{i}).ContourSlabThickness;
    else if(~isfield(instruct.Item_1, 'ContourSlabThickness')&& length(temp)>1)
            slicethickness=median(diff(sort(unique(slicez), 'ascend')));
        else
            slicethickness=0;
        end
    end
end

function cnter=findMassCenter(refmask)
[x, y, z] = meshgrid(1:size(refmask, 2), 1:size(refmask, 1), 1:size(refmask, 3));
weightedx = x .* refmask;
weightedy = y .* refmask;
weightedz = z .* refmask;
cnter = [single(sum(weightedy(:))) / single(sum(refmask(:))), single(sum(weightedx(:))) / single(sum(refmask(:))), single(sum(weightedz(:))) / single(sum(refmask(:)))] ;
end







function cnter=findMassCenter2D(refmaskslice)
[x, y] = meshgrid(1:size(refmaskslice, 2), 1:size(refmaskslice, 1));
weightedx = x .* refmaskslice;
weightedy = y .* refmaskslice;
cnter = [single(sum(weightedy(:))) / single(sum(refmaskslice(:))), single(sum(weightedx(:))) / single(sum(refmaskslice(:)))] ;

end


function[angle]=computeAng2D(refptsslice, refmaskslicecenter)
dx=(refptsslice(:,1)-refmaskslicecenter(1));
dy=(refptsslice(:,2)-refmaskslicecenter(2));
r=sqrt(dx.^2+dy.^2);
angle=atan2(dy,dx);
end
