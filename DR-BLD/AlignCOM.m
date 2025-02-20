function [point_COM,point_original] = AlignCOM(input_matrix)
    % AlignCOM - Aligns points with their center of mass
    % Input: matrix with x,y,z coordinates in columns
    % Output: 
    %   point_original: original points
    %   point_COM: points aligned to center of mass
    
    % Store original points
    point_original = input_matrix;

    % Calculate center of mass (mean of each coordinate)
    com = mean(input_matrix, 1);  % Calculate mean along rows for each column
    
    % Subtract COM from all points to center them
    point_COM = input_matrix - repmat(com, size(input_matrix, 1), 1);
end