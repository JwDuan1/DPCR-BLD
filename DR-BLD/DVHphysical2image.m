function [pti]=DVHphysical2image(ptp, header, pos)

if(strcmp(pos, 'HFS'))
pti(3,:)=round(-1*(ptp(3,:)/10+header.z_start)/header.z_pixdim)+1;
pti(1,:)=round( (ptp(1,:)/10-header.x_start)/header.x_pixdim)+1;
pti(2,:)=round( (ptp(2,:)/10-header.y_start)/header.y_pixdim)+1;
end
if(strcmp(pos, 'HFP'))
pti(3,:)=round(-1*(ptp(3,:)/10+header.z_start)/header.z_pixdim)+1;
pti(1,:)=round(header.x_dim-(ptp(1,:)/10-header.x_start)/header.x_pixdim);
pti(2,:)=round(header.y_dim-(ptp(2,:)/10-header.y_start)/header.y_pixdim);
end
if(strcmp(pos, 'FFP'))
pti(3,:)=round(-1*(ptp(3,:)/10+header.z_start)/header.z_pixdim)+1;
pti(1,:)=round( (ptp(1,:)/10-header.x_start)/header.x_pixdim)+1;
pti(2,:)=round(header.y_dim-(ptp(2,:)/10-header.y_start)/header.y_pixdim);
end
if(strcmp(pos, 'FFS'))
pti(3,:)=round(-1*((ptp(3,:)/10+header.z_start))/header.z_pixdim)+1;
pti(1,:)=round((-1*ptp(1,:)/10-header.x_start)/header.x_pixdim)+1;
pti(2,:)=round( (ptp(2,:)/10-header.y_start)/header.y_pixdim)+1;
end
end

