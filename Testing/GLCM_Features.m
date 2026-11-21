
function stats = GLCM_Features(GLCM)
% GLCM_FEATURES Extract statistical features from GLCM
%
% Features:
%   Contrast
%   Correlation
%   Energy
%   Homogeneity
%   Entropy

% Dissimilarity

    [numLevels, ~, numOffsets] = size(GLCM);

    % Gray-level coordinate matrices
    [i, j] = ndgrid(1:numLevels, 1:numLevels);

    % Preallocate
    stats = struct();

    stats.contrast = zeros(numOffsets, 1);
    stats.correlation = zeros(numOffsets, 1);
    stats.energy = zeros(numOffsets, 1);
    stats.homogeneity = zeros(numOffsets, 1);
    stats.entropy = zeros(numOffsets, 1);
    stats.dissimilarity = zeros(numOffsets, 1);

    for k = 1:numOffsets

        % Normalize GLCM into probabilities
        P = double(GLCM(:,:,k));
        total = sum(P(:));

        if total == 0
            continue;
        end

        P = P ./ total;

        % Marginal probabilities
        px = sum(P, 2);
        py = sum(P, 1);

        % Means
        mux = sum((1:numLevels)' .* px);
        muy = sum((1:numLevels) .* py);

        % Standard deviations
        sigmax = sqrt(sum(((1:numLevels)' - mux).^2 .* px));
        sigmay = sqrt(sum(((1:numLevels) - muy).^2 .* py));

        % Contrast
        stats.contrast(k) = sum(sum((i-j).^2 .* P));

        % Energy
        stats.energy(k) = sum(P(:).^2);

        % Homogeneity
        stats.homogeneity(k) = sum(sum(P ./ (1 + abs(i-j))));

        % Entropy
        p = P(P > 0);
        stats.entropy(k) = -sum(p .* log2(p));

        % Correlation
        if sigmax > 0 && sigmay > 0
            stats.correlation(k) = ...
                sum(sum(((i-mux).*(j-muy).*P))) ...
                / (sigmax * sigmay);
        else
            stats.correlation(k) = 1;
        end
        
        % Dissimilarity
        stats.dissimilarity(k) = sum(sum(abs(i-j).*P));
        
    end

end
