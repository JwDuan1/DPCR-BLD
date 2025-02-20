function in = refinsidetstpts(tstmask,rawrefpts)
% if ref inside pts point 

[rows, cols, ~] = size(tstmask); % Get the dimensions of tstmask
zeroLayer = zeros(rows, cols);   % Create a single layer of zeros
tstmask = cat(3, zeroLayer, tstmask, zeroLayer); % Add zero layers to beginning and end
rawrefpts(:,3)=rawrefpts(:,3)+1;

fv_tst = isosurface(tstmask, 0.999);  % Create the patch object
fv_tst.faces = fliplr(fv_tst.faces);    % Ensure normals point OUT
rawrefpts_xy_exchanges = [rawrefpts(:, 2),rawrefpts(:, 1),rawrefpts(:, 3)];
% rawrefpts_xy_exchanges =rawrefpts;
refintst = inpolyhedron(fv_tst, rawrefpts_xy_exchanges);
in=refintst;

% 
% figure, hold on, view(3)        % Display the result
% patch(fv_tst,'FaceColor','g','FaceAlpha',0.1,'LineStyle','-.')
% plot3(rawrefpts(in,2),rawrefpts(in,1),rawrefpts(in,3),'bo','MarkerFaceColor','b')
% plot3(rawrefpts(~in,2),rawrefpts(~in,1),rawrefpts(~in,3),'ro')

if length(in) ~= size(rawrefpts,1)
    disp('Error! code: refinsidetstpts');
end

end

