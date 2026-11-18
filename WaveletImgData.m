
function [HL1, LH1, HH1, LL2, HL2, LH2, HH2] = WaveletImgData(dataImg)
%WAVELETIMGDATA Summary of this function goes here
%   Detailed explanation goes here

n = size(dataImg, 1);
HL1 = cell(n, 1);LH1 = cell(n, 1);HH1 = cell(n, 1);
LL2 = cell(n, 1);HL2 = cell(n, 1);LH2 = cell(n, 1);HH2 = cell(n, 1);

for i= 1:n
    [HL1i, LH1i, HH1i, LL2i, HL2i, LH2i, HH2i] = WaveletConvertLvl2(dataImg{i});
    HL1{i} = HL1i;
    LH1{i} = LH1i;
    HH1{i} = HH1i;
    LL2{i} = LL2i;
    HL2{i} = HL2i;
    LH2{i} = LH2i;
    HH2{i} = HH2i;
end

end

