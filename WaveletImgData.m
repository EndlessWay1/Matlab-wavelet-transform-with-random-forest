
function subband_fold = WaveletImgData(dataImg)
%WAVELETIMGDATA Summary of this function goes here
%   Detailed explanation goes here

    jumlah_citra = numel(dataImg);

    subband_fold = cell(jumlah_citra, 1);

    % Melakukan ekstraksi fitur pada seluruh citra
    for i = 1:jumlah_citra
        % Menyimpan 7 subband dalam bentuk matriks
        subband = WaveletDecomp(dataImg{i});
        subband_fold{i} = subband;
    end

end

