clear; clc;

allFiles = dir('ORIGA/Images_Cropped/*.jpg');
labels = readtable('ORIGA/OrigaList.csv');

reshapeSize = [448, 448, 3];
dataSize = length(allFiles);

imageCell = cell(dataSize, 1);
imageLabel = zeros(dataSize, 1);

for i = 1:dataSize
    try
        % Read image
        img = imread(fullfile(allFiles(i).folder, allFiles(i).name));

        % Resize image
        img = imresize(img, reshapeSize(1:2));

        % Ensure image has 3 channels
        if size(img, 3) == 1
            img = repmat(img, [1, 1, 3]);
        end

        % Find corresponding label
        idx = strcmp(string(labels{:, 2}), allFiles(i).name);

        if any(idx)
            imageLabel(i) = labels{find(idx, 1), 5};
        else
            continue;
        end

        % Store image
        imageCell{i} = img;

    catch ME
        fprintf('Skipping %s: %s\n', allFiles(i).name, ME.message);
    end
end

disp("Done Reading");
% Remove empty entries
validIdx = ~cellfun(@isempty, imageCell);
imageCell = imageCell(validIdx);
imageLabel = imageLabel(validIdx);

% Display images
figure;

numImages = length(imageCell);

infectedCell = imageCell(imageLabel == 1);
uninfectedCell = imageCell(imageLabel ~= 1);

np = size(infectedCell, 1);
nn = size(uninfectedCell, 1);

for i = 1:min(10, np)
    subplot(4, 5, i);
    imshow(infectedCell{i});
    title(sprintf('Label: Positive'));
end

for i = 1:min(10, nn)
    subplot(4, 5, i + 10);
    imshow(uninfectedCell{i});
    title(sprintf('Label: Negative'));
end

orientation = [0, pi/4, pi/2, pi*3/4];
lambdas = 12;
sigma = 0.56*lambdas;
psi = 0;
gamma = 0.5;

M = permute(cat(4, imageCell{:}), [4, 1, 2, 3]);

[feat, featureNames] = GaborFilterTransform(M, orientation, sigma, lambdas, psi, gamma);

dataTable = array2table(feat, 'VariableNames', featureNames);
dataTable.Label = imageLabel(:);
writetable(dataTable, 'transformGaborData.csv');