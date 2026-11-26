function [img,label] = GetImages(data_path, csv_path)

    allFiles = dir(data_path);
    labels = readtable(csv_path);

    reshapeSize = [256, 256, 3];
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
            idx = strcmp(string(labels{:, 1}), allFiles(i).name);

            if any(idx)
                imageLabel(i) = labels{find(idx, 1), 2};
            else
                continue;
            end

            % Store image
            imageCell{i} = img;

        catch ME
            fprintf('Skipping %s: %s\n', allFiles(i).name, ME.message);
        end
    end
    img = imageCell;
    label = imageLabel;
end

