function saveMask2RTStruct(inRS, outRS, ptvmask, volheader,ROIName)
global newROIname
infoRS=dicominfo(inRS);
numROIs=length(fieldnames(infoRS.StructureSetROISequence));
roinumber=numROIs+1;  % make sure not collide with existing.  improve later
refframeUID=infoRS.StructureSetROISequence.Item_1.ReferencedFrameOfReferenceUID;
infoRS.StructureSetROISequence.(['Item_', num2str(roinumber)])= ...
    struct('ROINumber', roinumber, 'ReferencedFrameOfReferenceUID', refframeUID, ...
    'ROIName', newROIname, 'ROIGenerationAlgorithm', '');

contourCount=0;
imgseq=infoRS.ROIContourSequence.Item_1.ContourSequence.Item_1.ContourImageSequence;

for i=1:volheader.z_dim
    if(sum(sum(ptvmask(:,:,i)))<1) continue; end
    boundary=bwboundaries(ptvmask(:,:,i)');
    zc=-((i-1)*volheader.z_pixdim+volheader.z_start)*10;  %0.5???
 
    if(~isempty(boundary))
        for j=1:numel(boundary)
            ContourData=[];
            contourCount=contourCount+1;
            tempbound = boundary{j};
            subbound = [tempbound(:,2),tempbound(:,1)];
            xc=((subbound(:,1)-1)*volheader.x_pixdim+volheader.x_start)*10;
            yc=((subbound(:,2)-1)*volheader.y_pixdim+volheader.y_start)*10;
            ContourData(:,1)=xc;
            ContourData(:,2)=yc;
            ContourData(:,3)=zc;
            ContourData=ContourData';
            ContourData=ContourData(:);
            NumCPts = size(ContourData,1)/3;
            ContourGeometricType = 'CLOSED_PLANAR';
            ContourSequence.(['Item_', num2str(contourCount)])= ...
                struct('ContourImageSequence', imgseq, 'ContourGeometricType', ContourGeometricType, ...
                'NumberOfContourPoints', NumCPts, 'ContourData', ContourData);
        end
    end

end

infoRS.ROIContourSequence.(['Item_', num2str(roinumber)])= ...
    struct('ROIDisplayColor', [200,0,0], 'ContourSequence', ContourSequence, 'ReferencedROINumber', roinumber);
infoRS.StructureSetLabel = 'STAPLE';
infoRS.SeriesDescription = 'STAPLE';
infoRS.StructureSetROISequence.(['Item_', num2str(roinumber)]).ROIGenerationAlgorithm= 'STAPLE';
infoRS.StructureSetROISequence.(['Item_', num2str(roinumber)]).ROIName=ROIName;
dicomwrite([], outRS, infoRS, 'CreateMode', 'copy');
end
