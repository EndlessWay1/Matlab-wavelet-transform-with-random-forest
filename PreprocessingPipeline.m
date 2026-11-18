clear all; clc;

[data, label] = GetData();

% Convert RGB images to grayscale
data = cellfun(@(img) im2gray(img), data, ...
               'UniformOutput', false);

% Extract 7 wavelet subbands
[HL1, LH1, HH1, LL2, HL2, LH2, HH2] = WaveletImgData(data);

% Extract GLCM features
HL1GLCM = GrayDataExtraction(HL1);
LH1GLCM = GrayDataExtraction(LH1);
HH1GLCM = GrayDataExtraction(HH1);

LL2GLCM = GrayDataExtraction(LL2);
HL2GLCM = GrayDataExtraction(HL2);
LH2GLCM = GrayDataExtraction(LH2);
HH2GLCM = GrayDataExtraction(HH2);

% Combine all subband features
featureTables = {
    HL1GLCM, LH1GLCM, HH1GLCM, ...
    LL2GLCM, HL2GLCM, LH2GLCM, HH2GLCM
};

subbandNames = {'HL1', 'LH1', 'HH1', ...
                'LL2', 'HL2', 'LH2', 'HH2'};

% Add subband prefixes to column names
for i = 1:length(featureTables)

    T = featureTables{i};

    T.Properties.VariableNames = ...
        strcat(subbandNames{i}, '_', T.Properties.VariableNames);

    featureTables{i} = T;

end

% Combine horizontally
finalFeatures = [featureTables{:}];

% Add classification labels
finalFeatures.Label = label(:);

% Export CSV
writetable(finalFeatures, 'GLCM_Wavelet_Features.csv');

fprintf('CSV successfully exported!\n');
fprintf('Total images: %d\n', height(finalFeatures));
fprintf('Total features: %d\n', width(finalFeatures)-1);