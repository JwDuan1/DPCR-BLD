function [ B, C ] = bwboundaries2Dstackfast( BW )
CC=bwconncomp(BW);
C=labelmatrix(CC);
B=cell(max(max(max(C))),1);
% now find all the perimeters of each component in C.
for i=1:max(max(max(C)))
      tempim=bsxfun(@eq,C,i); %make binary image where the pixels of object i are true
      %make a ROI to speed up this process
      zproj=sum(tempim,3);
      xmin=find(sum(zproj,2)>0,1,'first');
      xmax=find(sum(zproj,2)>0,1,'last');
      ymin=find(sum(zproj,1)>0,1,'first');
      ymax=find(sum(zproj,1)>0,1,'last');
      planes=sum(sum(tempim,1),2);
      planes=planes(:);
      zmin=find(planes>0,1,'first');
      zmax=find(planes>0,1,'last');
      % now find the perimeter of object i
      test=bwperim(tempim(xmin:xmax,ymin:ymax,zmin:zmax));
      BW=false(size(tempim));
      BW(xmin:xmax,ymin:ymax,zmin:zmax)=test;
      % and create the coordinates of the border
      [x,y,z]=ind2sub(size(BW),find(BW));
      B{i}=[x y z];
end