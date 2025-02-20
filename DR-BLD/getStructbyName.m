function struct=getStructbyName(roi, name)
temp1=fieldnames(roi.StructureSetROISequence);
temp=fieldnames(roi.ROIContourSequence); % roi = dicominfo(tstdcm)
numstrsc=length(temp1);
numstrsc1=length(temp);
roiNames{numstrsc}=[];
roiNums(numstrsc)=0;
foundc=0;
struct=[];
for kc=1:numstrsc
     if(strcmpi(roi.StructureSetROISequence.(temp1{kc}).ROIName,name))
         foundc=1;
         foundroinum=roi.StructureSetROISequence.(temp1{kc}).ROINumber;
         break;
     end
end
if (foundc)
    for jc=1:numstrsc1
        if(isfield(roi.ROIContourSequence.(temp{jc}), 'ReferencedROINumber')&&(foundroinum==roi.ROIContourSequence.(temp{jc}).ReferencedROINumber)&&isfield(roi.ROIContourSequence.(temp{jc}), 'ContourSequence'))
            struct=roi.ROIContourSequence.(temp{jc}).ContourSequence;
            break
        end
    end
end
end