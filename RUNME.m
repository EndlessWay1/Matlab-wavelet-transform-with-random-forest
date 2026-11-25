clear; clc; close all;

rng(42);

%% Dataset Path
dataset_origa = "ORIGA/*.jpg";
dataset_g1020 = "G1020_Cleaned/*.jpg";

origa_list = 'OrigaList.csv';
g1020_list = 'G1020.csv';

% Getting dataset already resized into 256 x 256
[ORIGA_img, ORIGA_label] = GetImages(dataset_origa, origa_list);
[G1020_img, G1020_label] = GetImages(dataset_g1020, g1020_list);

jumlah_origa = size(ORIGA_img, 1);
jumlah_g1020 = size(G1020_img, 1);

fprintf('Found %i images in dataset ORIGA\n', jumlah_origa);
fprintf('Found %i images in dataset G1020\n', jumlah_g1020);


%% INFO of G1020
fprintf('Class Distribution of G1020:\n')
tabulate(G1020_label);

%% INFO of ORIGA
fprintf('Class Distribution of ORIGA:\n')
tabulate(ORIGA_label);


%% Equalized the dataset

% We take the normal only from the ORIGA dataset because of its high
% quality images

ORIGA_idx = ORIGA_label == 1;
G1020_idx = G1020_label == 1;

ORIGA_sick = ORIGA_img(ORIGA_idx);
G1020_sick = G1020_img(G1020_idx);
ORIGA_healty = ORIGA_img(~ORIGA_idx);

SICKLY_img = [ORIGA_sick;G1020_sick];
SICKLY_label = ones(size(SICKLY_img, 1), 1);

rand_idx = randperm(size(ORIGA_healty, 1));
ORIGA_healty = ORIGA_healty(rand_idx(1:size(SICKLY_label, 1)));

IMAGE_DATA = [ORIGA_healty; SICKLY_img];
IMAGE_LABEL = [zeros(size(ORIGA_healty, 1), 1); SICKLY_label];

rand_idx = randperm(size(IMAGE_LABEL, 1));
IMAGE_DATA = IMAGE_DATA(rand_idx);
IMAGE_LABEL = IMAGE_LABEL(rand_idx);

tabulate(IMAGE_LABEL);
fprintf("Total Data: %i\n", length(IMAGE_LABEL));

%% Preprocessing

% Grayscale conversion
IMAGE_DATA = cellfun(@(img) im2gray(img), IMAGE_DATA, ...
               'UniformOutput', false);

figure;
tiledlayout(1, 5, 'TileSpacing', 'compact', 'Padding', 'compact');

for i = 1:5
    nexttile;
    imshow(IMAGE_DATA{i});
    title(['Class ', num2str(IMAGE_LABEL(i))]);
end

%% StratifiedKFold

c = cvpartition(IMAGE_LABEL,"KFold", 5, "Stratify", true);

K_train_img = cell(c.NumTestSets, 1);
K_test_img = cell(c.NumTestSets, 1);
K_train_label = cell(c.NumTestSets, 1);
K_test_label = cell(c.NumTestSets, 1);

for i = 1:c.NumTestSets

    fprintf("Fold: %i\n", i);
    trainIdx = training(c, i);
    testIdx = test(c, i);
    
    train_img = IMAGE_DATA(trainIdx);
    train_label = IMAGE_LABEL(trainIdx);

    [Aug_Img, AugLabel] = TambahAugmentasi(train_img, train_label, 100);

    combined_img = [train_img; Aug_Img];
    combined_label = [train_label; AugLabel];
    train_randomize_idx = randperm(length(Aug_Img) + length(train_img));
    K_train_img{i} = combined_img(train_randomize_idx);
    K_train_label{i} = combined_label(train_randomize_idx);

    K_test_label{i} = IMAGE_LABEL(testIdx);
    K_test_img{i} = IMAGE_DATA(testIdx);

    fprintf("Jumlah data training awal: %i\n", length(train_img));
    fprintf("Jumlah gambar augmentasi: %i\n", length(Aug_Img));
    fprintf("Jumlah data training akhir: %i\n", length(K_train_img{i}));
    fprintf("Jumlah data testing: %i\n", length(K_test_img{i}));

    fprintf("Augmentasi label 0: %i\n", sum(AugLabel == 0));
    fprintf("Augmentasi label 1: %i\n", sum(AugLabel == 1));
    fprintf('\n');
end

%% 5 Random Images After Augmentation

% Mengambil Fold pertama
fold = 1;

trainIdx = training(c, fold);

train_img = IMAGE_DATA(trainIdx);
train_label = IMAGE_LABEL(trainIdx);

% Melakukan augmentasi
[Aug_Img, AugLabel, Before_Img] = TambahAugmentasi(train_img, train_label, 100);
randomIdx = randperm(length(Aug_Img), 5);

figure('Name', 'Before and After Augmentation', ...
       'NumberTitle', 'off', ...
       'Color', 'white', ...
       'Position', [100 100 1000 900]);

for i = 1:5

    subplot(5, 2, (i-1)*2 + 1);
    imshow(Before_Img{randomIdx(i)});
    title("Before - Label " + AugLabel(randomIdx(i)));

    subplot(5, 2, (i-1)*2 + 2);
    imshow(Aug_Img{randomIdx(i)});
    title("After - Label " + AugLabel(randomIdx(i)));

end

drawnow;
%% Discrete Wavelet Transform Subband Extraction

% Menentukan level dekomposisi
dwtLevel = 2;


% Menyimpan 7 subband DWT dalam bentuk matriks
K_subband_dwt_train = cell(c.NumTestSets, 1);
K_subband_dwt_test = cell(c.NumTestSets, 1);


% Melakukan ekstraksi fitur pada setiap fold
for fold = 1:c.NumTestSets

    subband_fold = WaveletImgData(K_train_img{fold});
    % Menyimpan hasil ekstraksi setiap fold
    K_subband_dwt_train{fold} = subband_fold;

    % Menampilkan hasil ekstraksi setiap fold
    fprintf("\nHasil Ekstraksi DWT Training Fold %d\n", fold);
    fprintf("Jumlah citra: %d\n", length(K_train_img{fold}));
    fprintf("Jumlah subband per citra: %d\n", numel(subband_fold{1}));


    subband_fold = WaveletImgData(K_test_img{fold});
    % Menyimpan hasil ekstraksi setiap fold
    K_subband_dwt_test{fold} = subband_fold;

    % Menampilkan hasil ekstraksi setiap fold
    fprintf("\nHasil Ekstraksi DWT Testing Fold %d\n", fold);
    fprintf("Jumlah citra: %d\n", length(K_test_img{fold}));
    fprintf("Jumlah subband per citra: %d\n", numel(subband_fold{1}));
    
end

fprintf("\nEkstraksi fitur DWT selesai!\n");

%% Wavelet GLCM Extraction Features
FeatureExtracted = {'Dissimilarity','Correlation','Homogeneity','Contrast','Energy'};
FeatureLength = numel(FeatureExtracted);
offsets = [0 1; -1 1; -1 0; -1 -1];
OffsetLength = size(offsets, 1);
SubbandNames = {'LL2','LH2','HL2','HH2','LH1','HL1','HH1'};
glcmLevels = 256;
nSub = 3*dwtLevel + 1;
featPerSub = FeatureLength * OffsetLength;

[colIdx, rowIdx] = meshgrid(1:glcmLevels, 1:glcmLevels);
absDiff = abs(rowIdx - colIdx);

K_glcm_train = cell(c.NumTestSets, 1);
K_glcm_test = cell(c.NumTestSets, 1);


for i = 1:c.NumTestSets
    fprintf("\nFold: %i\n", i);
    fprintf("Hasil Ekstraksi GLCM Training: ");

    
    subband_cell = K_subband_dwt_train{i};
    train_length = length(subband_cell);
    train_glcm_res = zeros(train_length, featPerSub * nSub);
    
    for j = 1:train_length
        for k = 1:nSub
            sb = mat2gray(subband_cell{j}{k});          % scale to [0,1]
            GLCM = graycomatrix(sb, 'Offset', offsets, 'NumLevels', glcmLevels, ...
                                'GrayLimits', [0 1], 'Symmetric', true);

            props = graycoprops(GLCM, FeatureExtracted(2:end));

            dissim = zeros(1, OffsetLength);
            for m = 1:OffsetLength
                P = double(GLCM(:,:,m));
                P = P ./ sum(P(:));
                dissim(m) = sum(sum(P .* absDiff));
            end

            base = (k-1) * featPerSub;
            train_glcm_res(j, base + (1:featPerSub)) = ...
                [dissim, props.Correlation, props.Homogeneity, props.Contrast, props.Energy];
        end
    end
    K_glcm_train{i} = train_glcm_res;
    fprintf("✅\n");


    fprintf("Hasil Ekstraksi GLCM Testing: ");

    subband_cell = K_subband_dwt_test{i};
    test_length = length(subband_cell);
    test_glcm_res = zeros(test_length, featPerSub * nSub);

    for j = 1:test_length
        for k = 1:nSub
            sb = mat2gray(subband_cell{j}{k});          % scale to [0,1]
            GLCM = graycomatrix(sb, 'Offset', offsets, 'NumLevels', glcmLevels, ...
                                'GrayLimits', [0 1], 'Symmetric', true);

            props = graycoprops(GLCM, FeatureExtracted(2:end));

            dissim = zeros(1, OffsetLength);
            for m = 1:OffsetLength
                P = double(GLCM(:,:,m));
                P = P ./ sum(P(:));
                dissim(m) = sum(sum(P .* absDiff));
            end

            base = (k-1) * featPerSub;
            test_glcm_res(j, base + (1:featPerSub)) = ...
                [dissim, props.Correlation, props.Homogeneity, props.Contrast, props.Energy];
        end
    end
    K_glcm_test{i} = test_glcm_res;

    fprintf("✅\n");
    
end

%% PCA EXTRACTION

pca = PCA(0.95);

K_PCA_train = cell(c.NumTestSets, 1);
K_PCA_test = cell(c.NumTestSets, 1);

for i = 1:c.NumTestSets

    fprintf("\nFold: %i\n", i);
    
    X_train = K_glcm_train{i};
    pca = pca.fit(X_train);
    Z_train = pca.transform(X_train);
    K_PCA_train{i} = Z_train;
    
    X_test = K_glcm_test{i};
    Z_test = pca.transform(X_test);
    K_PCA_test{i} = Z_test;

    pca.toCSV(Z_train, K_train_label{i}, sprintf("train/GLAUCOMA_PCA_WAVELET_DATA_K%d.csv", i));
    pca.toCSV(Z_test, K_test_label{i}, sprintf("test/GLAUCOMA_PCA_WAVELET_DATA_K%d.csv", i));
    
    fprintf("PCA REPORT:\n");
    pca.report();
    

end

%% RUN MODEL SIMULATION (K-FOLD RANDOM FOREST ON PCA FEATURES)
K = c.NumTestSets;
classNames = {'Normal (0)', 'Glaucoma (1)'};   % change to match your labels

metricNames = {'Accuracy','Precision','Recall','F1','Specificity'};
results     = zeros(K, numel(metricNames));
trainTime   = zeros(K, 1);
C_total     = zeros(2, 2);
yTrueAll    = [];
yPredAll    = [];

fprintf("\n========== K-FOLD RANDOM FOREST (K = %d) ==========\n", K);

for i = 1:K
    fprintf("\n---------- Fold %d / %d ----------\n", i, K);

    % PCA features and labels for this fold
    x_train = K_PCA_train{i};
    x_test  = K_PCA_test{i};
    y_train = double(K_train_label{i}(:));
    y_test  = double(K_test_label{i}(:));

    % Guard against the row mismatch seen earlier
    assert(size(x_train,1) == numel(y_train), ...
        "Fold %d: train features (%d rows) and labels (%d) differ.", ...
        i, size(x_train,1), numel(y_train));
    assert(size(x_test,1) == numel(y_test), ...
        "Fold %d: test features (%d rows) and labels (%d) differ.", ...
        i, size(x_test,1), numel(y_test));

    fprintf("Training samples : %d\n", size(x_train,1));
    fprintf("Testing samples  : %d\n", size(x_test,1));
    fprintf("PCA features     : %d\n", size(x_train,2));

    % Train Random Forest (same arguments as your example)
    rng(42 + i);                         % reproducible, different per fold
    model = RandomForest(200, 10, 10, 10);
    tic;
    model = model.fit(x_train, y_train);
    trainTime(i) = toc;

    % Predict on the held-out fold
    y_pred = model.predict(x_test);
    y_pred = double(y_pred(:));

    % Confusion matrix, class order: 0 then 1
    C  = confusionmat(y_test, y_pred, 'Order', [0 1]);
    TN = C(1,1);  FP = C(1,2);
    FN = C(2,1);  TP = C(2,2);

    % Metrics (max(...,1) avoids 0/0 when a class is never predicted)
    accuracy    = (TP + TN) / sum(C(:));
    precision   = TP / max(TP + FP, 1);
    recall      = TP / max(TP + FN, 1);
    f1Score     = 2 * precision * recall / max(precision + recall, eps);
    specificity = TN / max(TN + FP, 1);

    results(i, :) = [accuracy, precision, recall, f1Score, specificity];

    % Accumulate for pooled evaluation
    C_total  = C_total + C;
    yTrueAll = [yTrueAll; y_test];
    yPredAll = [yPredAll; y_pred];

    fprintf("Train time   : %.2f s\n", trainTime(i));
    fprintf("Accuracy     : %.4f (%.2f%%)\n", accuracy, accuracy*100);
    fprintf("Precision    : %.4f\n", precision);
    fprintf("Recall       : %.4f\n", recall);
    fprintf("F1-Score     : %.4f\n", f1Score);
    fprintf("Specificity  : %.4f\n", specificity);
    fprintf("Confusion matrix [TN FP; FN TP]:\n");
    disp(C);
end

%% K-FOLD SUMMARY
resultsTable = array2table(results, 'VariableNames', metricNames);
resultsTable.TrainTime_s = trainTime;
resultsTable.Fold = (1:K)';
resultsTable = movevars(resultsTable, 'Fold', 'Before', 1);

fprintf("\n========== PER-FOLD RESULTS ==========\n");
disp(resultsTable);

fprintf("\n========== MEAN ± STD ACROSS %d FOLDS ==========\n", K);
for m = 1:numel(metricNames)
    fprintf("%-12s: %.4f ± %.4f\n", metricNames{m}, ...
        mean(results(:,m)), std(results(:,m)));
end
fprintf("%-12s: %.2f s total\n", "Train time", sum(trainTime));

fprintf("\n========== POOLED CONFUSION MATRIX (ALL FOLDS) ==========\n");
disp(C_total);

%% VISUALIZATION
figure;
confusionchart(C_total, classNames);
title(sprintf("Random Forest Pooled Confusion Matrix - PCA Features (%d-Fold)", K));

figure;
bar(results);
xlabel("Fold");
ylabel("Score");
legend(metricNames, 'Location', 'southoutside', 'Orientation', 'horizontal');
title("Random Forest Metrics per Fold");
ylim([0 1.05]);
grid on;

writetable(resultsTable, "RF_PCA_KFold_Results.csv");