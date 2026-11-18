function stats = GrayDataExtraction(Subands)

    n = length(Subands);

    offsets = [
         0  1;
        -1  1;
        -1  0;
        -1 -1
    ];

    numlevels = 256;

    % Preallocate feature table
    featureNames = {'contrast', 'correlation', ...
                    'energy', 'homogeneity', 'entropy', 'dissimilarity'};

    directions = {'0', '45', '90', '135'};

    columnNames = {};

    for f = 1:length(featureNames)
        for d = 1:length(directions)
            columnNames{end+1} = ...
                [featureNames{f}, '_', directions{d}];
        end
    end

    features = zeros(n, length(columnNames));

    for i = 1:n

        GLCM = graycomatrix(Subands{i}, ...
            'NumLevels', numlevels, ...
            'Offset', offsets);

        % Extract features
        result = GLCM_Features(GLCM);

        % Convert struct features into a row
        row = [];

        for f = 1:length(featureNames)
            row = [row; result.(featureNames{f})(:)];
        end

        features(i, :) = row';

    end

    % Convert to table
    stats = array2table(features, ...
        'VariableNames', columnNames);

end