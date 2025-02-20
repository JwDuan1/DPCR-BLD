function [mask]=computeMaskfromStruct(struct1, header, pos)
temp=fieldnames(struct1);
numcurves=length(temp);

mask=zeros([header.x_dim, header.y_dim, header.z_dim]);

prevz=-100000;
for i=1:numcurves
    isxor=0;
    numpoints=struct1.(temp{i}).NumberOfContourPoints;

    pts=zeros(3,numpoints);
    pts(:)=struct1.(temp{i}).ContourData(:);
    if numpoints<3
        prevz=pts(3);
        continue;
    end
    if(abs(pts(3)-prevz)<1e-5)
        isxor=1;
    end
    
    prevz=pts(3);
    pts=DVHphysical2image(pts, header, pos);
    
    if pts(3)<1 || pts(3)>header.z_dim
        continue;
    end
    if(isxor)
        mask(:,:,pts(3))=xor(mask(:,:,pts(3)),poly2mask(pts(2,:), pts(1,:), header.x_dim, header.y_dim));
    else
        mask(:,:,pts(3))=poly2mask(pts(2,:), pts(1,:), header.x_dim, header.y_dim);
    end
end

end

% function [mask]=computeMaskfromStruct(struct1, doseheader, header, pos)
% temp=fieldnames(struct1);
% numcurves=length(temp);
% m=doseheader.x_dim; n=doseheader.y_dim; p=doseheader.z_dim;
% pad=([header.x_dim, header.y_dim, header.z_dim]-[m,n,p])*0.5;
% mask=zeros([m,n,p]);
% 
% for i=1:numcurves
%     numpoints=struct1.(temp{i}).NumberOfContourPoints;
%     if numpoints<3
%         continue;
%     end
%     pts=zeros(3,numpoints);
%     pts(:)=struct1.(temp{i}).ContourData(:);
%     
%     pts=DVHphysical2image(pts, header, pos);
%     mask2d=poly2mask(pts(1,:), pts(2,:), header.x_dim, header.y_dim);
% 
%     
%     
%     sliceidx=pts(3)+pad(3);
%     if sliceidx<1 || sliceidx>header.z_dim
%         continue;
%     end
%     mask(:,:,pts(3)+pad(3))=mask2d(pad(1)+(1:m),pad(2)+(1:n));
% end
% 
% end
