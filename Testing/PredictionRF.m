clc;
clear;
close all;

% Load PCA training and testing datasets
trainData = readtable("PCA_Train_GLCM_Wavelet_Features.csv");
testData  = readtable("PCA_Test_GLCM_Wavelet_Features.csv");

% Training features and labels
x_train = trainData{:, 1:end-1};
y_train = trainData{:, end};

% Testing features and labels
x_test = testData{:, 1:end-1};
y_test = testData{:, end};

% Display dataset information
fprintf('Training samples: %d\n', size(x_train, 1));
fprintf('Testing samples : %d\n', size(x_test, 1));
fprintf('PCA features   : %d\n', size(x_train, 2));

disp("Training data preview:");
disp(head(trainData));

rng(42);
% Train Random Forest
model = RandomForest(200, 10, 10, 10);

disp("Running");
model = model.fit(x_train, y_train);
disp("Done Training");

% Prediction
y_pred = model.predict(x_test);

% Confusion Matrix
% Class order: 0 = Benign, 1 = Malignant
C = confusionmat(y_test, y_pred, 'Order', [0 1]);

TN = C(1,1);
FP = C(1,2);
FN = C(2,1);
TP = C(2,2);

% Evaluation Metrics

% Accuracy
accuracy = (TP + TN) / sum(C(:));

% Precision
precision = TP / (TP + FP);

% Recall / Sensitivity
recall = TP / (TP + FN);

% F1-Score
f1Score = 2 * (precision * recall) / (precision + recall);

% Specificity
specificity = TN / (TN + FP);

% Display Results
fprintf('\n========== RANDOM FOREST RESULTS ==========\n');

fprintf('Accuracy     : %.4f (%.2f%%)\n', ...
    accuracy, accuracy * 100);

fprintf('Precision    : %.4f\n', precision);
fprintf('Recall       : %.4f\n', recall);
fprintf('F1-Score     : %.4f\n', f1Score);
fprintf('Specificity  : %.4f\n', specificity);

fprintf('\n========== CONFUSION MATRIX ==========\n');
disp(C);

% Confusion Matrix Visualization
figure;

confusionchart(C, {'Benign (0)', 'Malignant (1)'});

title('Random Forest Confusion Matrix - PCA Features');