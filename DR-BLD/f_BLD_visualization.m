% Main script showing example usage
% Example:
% f_BLD_visualization('path/to/reference.mat', 'path/to/Tempalte.mat')

function f_BLD_visualization(referenceFile, TemplateFile,countthreshold,pythonpath)
    % F_BLD_VISUALIZATION Visualizes and compares point clouds for medical imaging analysis
    %
    % This function performs point cloud visualization and comparison between reference
    % and Template contours.
    % It includes functionalities for point cloud alignment, downsampling, registration,
    % and visualization of disagreements between contours.
    %
    % Inputs:
    %   referenceFile  - Path to .mat file containing reference data
    %   TemplateFile    - Path to .mat file containing Template data
    %   countthreshold - (Optional) Maximum point count before downsampling (default: 5000)
    %   pythonpath    - (Optional) Path to Python executable for colormap generation
    
    % Set default parameters if not provided
    if(nargin<3)
       countthreshold=5000;  % Default point count threshold
    end     
    
    if(nargin<4)
        pythonpath='"c:\Program Files (x86)\Microsoft Visual Studio\Shared\Python37_64\python.exe"';
    end
    
    % Load input data files
    data = load(referenceFile);
    Templatedata = load(TemplateFile);
    
    % Load colormap for visualization
    % Uses Python's matplotlib coolwarm colormap for better visualization of differences
    coolwarm = getPyPlot_cMap('coolwarm', 256, false, pythonpath);
    
    % Extract and process reference points
    refpts = data.refpts;  % Original reference points
    refptswithbld = data.refptswithbld;  % Reference points with BLD (Boundary Line Distance) values
    
    % Calculate center of mass for alignment
    refptscom = mean(refpts, 1);  % Mean along rows for each column
    
    % Align reference points to their center of mass for consistent positioning
    refpts = AlignCOM(refpts);
    
    % Process test points
    tstpts = data.tstpts;
    % Align test points to the same coordinate system as reference points
    tstpts = tstpts - repmat(refptscom, size(tstpts, 1), 1);
    
    % Process Template points
    Templatepts = Templatedata.refpts;
    Templatepts = AlignCOM(Templatepts);  % Align to center of mass

    % Create point cloud objects for visualization
    % Reference point cloud
    refptsCloud = pointCloud(refpts);
    refptsCloud.Color = 'blue';
    
    % Reference points with BLD information
    refptswithbldCloud = pointCloud(refpts);
    refptswithbldCloud.Intensity = 10 * refptswithbld(:,4);  % Scale BLD values for visualization
    
    % Test point cloud
    tstptsCloud = pointCloud(tstpts);
    tstptsCloud.Color = 'red';

    % Template point cloud
    temp = zeros(length(Templatepts),1);
    TemplateptsCloud = pointCloud(Templatepts, 'Intensity', temp);
    TemplateptsCloud.Color = 'green';

    % Get initial point count for downsampling check
    pointCount=TemplateptsCloud.Count;

    % Downsampling process for large point clouds
    gridAverage = 0.05;  % Initial grid size for downsampling
    if pointCount- countthreshold >0
        % Display original point count and start downsampling
        disp(['The OAR points number is ', num2str(TemplateptsCloud.Count), ' larger than threshold ', num2str(countthreshold)]);
        disp('-----------Downsample-------------')
        
        % Visualize original point clouds before downsampling
        figure;
        pcshowpair(refptsCloud, tstptsCloud, 'MarkerSize', 30, 'ColorSource', "Color", 'BackgroundColor', "white", 'VerticalAxisDir', "Down");
        hold on;
        h1 = plot(nan, nan, 'o', 'MarkerSize', 10, 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'b');
        h2 = plot(nan, nan, 'o', 'MarkerSize', 10, 'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'r');
        h3 = plot(nan, nan, 'o', 'MarkerSize', 10, 'MarkerEdgeColor', 'g', 'MarkerFaceColor', 'g');
        legend([h1, h2], {'Reference point cloud - NoDownsample', 'Test point cloud - NoDownsample'}, 'TextColor', 'black', 'FontSize', 14, 'Location', 'southoutside');
        hold off;
        set(gca, 'FontSize', 18);
    else
        disp(['The OAR points number is ', num2str(TemplateptsCloud.Count)]);
        flag=1;
    end

    % Iterative downsampling until point count is below threshold
    while pointCount- countthreshold >0
        % Downsample all point clouds consistently
        TemplateptsCloud = pcdownsample(TemplateptsCloud,'gridAverage',gridAverage);
        pointCount = TemplateptsCloud.Count;
        if pointCount > countthreshold
            gridAverage = gridAverage +0.001;  % Gradually increase grid size
        else
            flag=0;
            disp(['The downsample OAR points number is ', num2str(TemplateptsCloud.Count)]);
        end
        % Downsample other point clouds with same grid size
        tstptsCloud= pcdownsample(tstptsCloud,'gridAverage',gridAverage);
        refptswithbldCloud= pcdownsample(refptswithbldCloud,'gridAverage',gridAverage);
        refptsCloud= pcdownsample(refptsCloud,'gridAverage',gridAverage);
    end  
    
    % Perform non-rigid registration between point clouds
    evalptsCloudDownsampled = refptswithbldCloud;
    [tform, evalmovingReg,rmse] = pcregistercpd(evalptsCloudDownsampled, TemplateptsCloud, 'Transform', "Nonrigid", 'MaxIterations', 30);
    disp(['Registration RMSE: ', num2str(rmse)]);  % Display registration error with label
    movingTransformed = pctransform(evalptsCloudDownsampled, tform);
    evalmovingReg.Color = 'blue';

    % Transfer intensity values from registered points to Template points
    for j = 1:TemplateptsCloud.Count
        [indices, ~] = findNearestNeighbors(evalmovingReg, TemplateptsCloud.Location(j,:), 1);
        TemplateptsCloud.Intensity(j) = evalmovingReg.Intensity(indices);
    end
    
    % Visualization Section
    % 1. Reference vs Test Point Clouds
    figure('Name', 'Point Cloud Comparison');
    title('Reference vs Test Point Cloud Comparison', 'FontSize', 14);
    pcshowpair(refptsCloud, tstptsCloud, 'MarkerSize', 30, 'ColorSource', "Color", 'BackgroundColor', "white", 'VerticalAxisDir', "Down");
    hold on;
    h1 = plot(nan, nan, 'o', 'MarkerSize', 10, 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'b');
    h2 = plot(nan, nan, 'o', 'MarkerSize', 10, 'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'r');
    h3 = plot(nan, nan, 'o', 'MarkerSize', 10, 'MarkerEdgeColor', 'g', 'MarkerFaceColor', 'g');
    legend([h1, h2], {'Reference point cloud', 'Test point cloud'}, 'TextColor', 'black', 'FontSize', 14, 'Location', 'southoutside');
    hold off;
    set(gca, 'FontSize', 18);

    % 2. Reference vs Template Point Clouds
    figure('Name', 'Template Comparison');
    title('Reference vs Template Point Cloud Comparison', 'FontSize', 14);
    pcshowpair(refptsCloud, TemplateptsCloud, 'MarkerSize', 40, 'ColorSource', "Color", 'BackgroundColor', "white", 'VerticalAxisDir', "Down");
    hold on;
    h1 = plot(nan, nan, 'o', 'MarkerSize', 10, 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'b');
    h2 = plot(nan, nan, 'o', 'MarkerSize', 10, 'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'r');
    h3 = plot(nan, nan, 'o', 'MarkerSize', 10, 'MarkerEdgeColor', 'g', 'MarkerFaceColor', 'g');
    legend([h1, h3],{'Reference point cloud - example case', 'Template point cloud'}, 'TextColor', 'black', 'FontSize', 14, 'Location', 'southoutside');
    hold off;
    set(gca, 'FontSize', 18);

    % 3. Registered Point Clouds
    figure('Name', 'Registration Results');
    title('Registration Result: Reference vs Template', 'FontSize', 14);
    pcshowpair(evalmovingReg, TemplateptsCloud, 'MarkerSize', 40, 'ColorSource', "Color", 'BackgroundColor', "white", 'VerticalAxisDir', "Down");
    hold on;
    h1 = plot(nan, nan, 'o', 'MarkerSize', 10, 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'b');
    h2 = plot(nan, nan, 'o', 'MarkerSize', 10, 'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'r');
    h3 = plot(nan, nan, 'o', 'MarkerSize', 10, 'MarkerEdgeColor', 'g', 'MarkerFaceColor', 'g');
    legend([h1, h3],{'Reference point cloud - example case', 'Template point cloud'}, 'TextColor', 'black', 'FontSize', 14, 'Location', 'southoutside');
    hold off;
    set(gca, 'FontSize', 18);

    % 4. Visualize disagreements on reference contour
    figure('Name', 'Reference Disagreements');
    title('Disagreements on Reference Contour', 'FontSize', 14);
    pcshow(refptswithbldCloud, 'MarkerSize', 30, 'ColorSource', "Intensity", 'BackgroundColor', "white", 'VerticalAxisDir', "Down");
    colormap(coolwarm);  % Use matplotlib's coolwarm colormap
    global_max_abs_offset = max(abs(max(refptswithbldCloud.Intensity)), abs(min(refptswithbldCloud.Intensity)));
    caxis([-global_max_abs_offset, global_max_abs_offset]);  % Symmetric color scaling
    colorbar('Color', [1 1 1]);
    ylabel(colorbar, 'Disagreements (mm)');
    set(gca, 'FontSize', 18);
    
    % 5. Visualize disagreements on Template contour
    figure('Name', 'Template Disagreements');
    title('Disagreements on Template Contour', 'FontSize', 14);
    pcshow(TemplateptsCloud, 'MarkerSize', 30, 'ColorSource', "Intensity", 'BackgroundColor', "white", 'VerticalAxisDir', "Down");
    colormap(coolwarm);  % Use matplotlib's coolwarm colormap
    global_max_abs_offset = max(abs(max(TemplateptsCloud.Intensity)), abs(min(TemplateptsCloud.Intensity)));
    caxis([-global_max_abs_offset, global_max_abs_offset]);  % Symmetric color scaling
    colorbar('Color', [1 1 1]);
    ylabel(colorbar, 'Disagreements (mm)');
    set(gca, 'FontSize', 18);
end