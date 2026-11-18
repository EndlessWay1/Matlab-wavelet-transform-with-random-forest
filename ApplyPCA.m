clear all; clc; close all;

% Load dataset
data = readtable("GLCM_Wavelet_Features.csv");


y = data{:, end};
x = data{:, 1:end-1};

% Dataset size
[n, p] = size(x);


% Verify dimensions
fprintf('Number of samples: %d\n', size(x, 1));
fprintf('Number of features: %d\n', size(x, 2));

% Check class distribution
tabulate(y);

% Calc the columnwise mean
myu = mean(x);

Z = zeros(n, p);

for i = 1:n
    Z(i, :) = x(i, :) - myu;
end

S = cov(Z);

Zr = zeros(n, p);
for i = 1:n
    for j = 1:p
        Zr(i,j) = Z(i,j)/sqrt(S(j,j));
    end
end

Zcov = cov(Zr);
[V, A] = eig(Zcov);

V = V(:, end:-1:1);
% Proporsi kumulatif
eVal = diag(A);
eVal = eVal(end:-1:1);

Total = sum(eVal);
percent = eVal/Total;

cumulativePercentage = cumsum(percent);
percentage = 0.95; % Retain 95% variance

lengthOfData = find(cumulativePercentage >= percentage, 1);

VKomp = V(:, 1:lengthOfData);

W = Zr * VKomp;

fprintf('\nOriginal number of features: %d\n', p);
fprintf('Remaining features after removing constant columns: %d\n', size(Zr, 2));
fprintf('Selected principal components: %d\n', lengthOfData);
fprintf('Retained variance: %.2f%%\n', ...
    cumulativePercentage(lengthOfData) * 100);

fprintf('Reduced dataset dimensions: %d x %d\n', size(W, 1), size(W, 2));


figure;
subplot(2,1,1);

plot(1:length(eVal), eVal, '-o', ...
    'LineWidth', 1.5, 'MarkerSize', 5);

xlabel('Principal Component');
ylabel('Eigenvalue');
title('PCA Eigenvalues (Scree Plot)');
grid on;


subplot(2,1,2);

plot(1:length(cumulativePercentage), ...
    cumulativePercentage * 100, '-o', ...
    'LineWidth', 1.5, 'MarkerSize', 5);

hold on;

yline(percentage * 100, '--r', '95% Threshold');

xline(lengthOfData, '--g', ...
    sprintf('%d Components', lengthOfData));

xlabel('Number of Principal Components');
ylabel('Cumulative Explained Variance (%)');
title('Cumulative Explained Variance');
ylim([0 105]);
grid on;

hold off;

sgtitle('PCA Analysis');


% Add original class labels
reducedData = array2table(W);

% Rename principal component columns
reducedData.Properties.VariableNames = ...
    compose('PC%d', 1:lengthOfData);

reducedData.Label = y;

% Export to CSV
writetable(reducedData, 'PCA_GLCM_Wavelet_Features.csv');

fprintf('\nPCA dataset successfully saved!\n');