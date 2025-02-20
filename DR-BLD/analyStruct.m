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
    if(length(temp)>0&&isfield(instruct.Item_1, 'ContourSlabThickness'))
            slicethickness=instruct.(temp{i}).ContourSlabThickness;
    else if(~isfield(instruct.Item_1, 'ContourSlabThickness')&& length(temp)>1)
            slicethickness=median(diff(sort(unique(slicez), 'ascend')));
        else
            slicethickness=0;
        end
    end
end
