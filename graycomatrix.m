
function GLCM = graycomatrix(I, varargin)
% GRAYCOMATRIX Custom Gray-Level Co-occurrence Matrix
%
% Usage:
%   GLCM = graycomatrix(I, 'NumLevels', 8, 'Offset', offsets)
%
% Offset format: [row_offset, column_offset]
%
% Optional parameters:
%   NumLevels : Number of gray levels (default 8)
%   Offset    : Spatial offsets (default [0 1])
%   GrayLimits: [minimum maximum] intensity range
%   Symmetric : Make GLCM symmetric (default false)

    % Default parameters
    numLevels = 8;
    offsets = [0 1];
    grayLimits = [];
    symmetric = false;

    % Parse arguments
    for k = 1:2:length(varargin)
        switch lower(varargin{k})
            case 'numlevels'
                numLevels = varargin{k+1};

            case 'offset'
                offsets = varargin{k+1};

            case 'graylimits'
                grayLimits = varargin{k+1};

            case 'symmetric'
                symmetric = varargin{k+1};

            otherwise
                error('Unknown parameter: %s', varargin{k});
        end
    end

    % Validate input
    if ~ismatrix(I)
        error('Input image must be a 2D grayscale image.');
    end

    if numLevels < 1 || numLevels ~= floor(numLevels)
        error('NumLevels must be a positive integer.');
    end

    I = double(I);

    % Determine gray limits
    if isempty(grayLimits)
       grayLimits = [min(I(:)), max(I(:))]; 
    end

    low = grayLimits(1);
    high = grayLimits(2);

    % Quantize grayscale values
    if high == low
        Q = ones(size(I));
    else
        Q = floor((I - low) ./ (high - low) * numLevels) + 1;
        Q = max(1, min(numLevels, Q));
    end

    [rows, cols] = size(Q);
    numOffsets = size(offsets, 1);

    % Preallocate GLCM
    GLCM = zeros(numLevels, numLevels, numOffsets);

    % Calculate GLCM for each offset
    for k = 1:numOffsets

        dr = offsets(k, 1);
        dc = offsets(k, 2);

        % Valid pixel coordinates
        r1 = max(1, 1-dr):min(rows, rows-dr);
        c1 = max(1, 1-dc):min(cols, cols-dc);

        if isempty(r1) || isempty(c1)
            continue;
        end

        % Extract neighboring pixels
        A = Q(r1, c1);
        B = Q(r1+dr, c1+dc);

        % Count gray-level pairs
        counts = accumarray( ...
            [A(:), B(:)], ...
            1, ...
            [numLevels, numLevels]);

        if symmetric
            counts = counts + counts';
        end

        GLCM(:,:,k) = counts;

    end

end

