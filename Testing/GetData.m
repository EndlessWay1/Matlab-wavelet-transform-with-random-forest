function [dataImg,dataLabel] = GetData()
%GETDATA Summary of this function goes here
%   Detailed explanation goes here
[imgORIGA, labelORIGA] = GetImagesORIGA();

idx = (labelORIGA == 0);

negIdx = randsample(find(idx), sum(~idx));

newIdx = [negIdx; find(~idx)];
newIdx = newIdx(randperm(length(newIdx)));

newImgORIGA = imgORIGA(newIdx);
newLabelORIGA = labelORIGA(newIdx);


% [imgG1020, labelG1020] = GetImagesG1020();
% idx = (labelG1020 == 0);
% 
% negIdx = randsample(find(idx), sum(~idx));
% 
% newIdx = [negIdx; find(~idx)];
% newIdx = newIdx(randperm(length(newIdx)));
% 
% newImgG1020 = imgG1020(newIdx);
% newLabelG1020 = labelG1020(newIdx);

dataImg = [newImgORIGA];
dataLabel = [newLabelORIGA];

sIdx = randperm(length(dataLabel));

dataImg = dataImg(sIdx);
dataLabel = dataLabel(sIdx);


end

