function [features, featureNames] = GaborFilterTransform( ...
    images, orientations, sigma, lambda, psi, gamma)

    nImages = size(images, 1);
    nOrientations = numel(orientations);
    
    statNames   = {'Mean', 'Var', 'Skew',"Energy", 'Entropy'};
    nS = numel(statNames);

    features = zeros(nImages, nOrientations * nS);
    
    for n = 1:nImages

        % 1. Image preprocessing
        img = squeeze(images(n, :, :, :));
        img = im2double(img);

        % Convert RGB image to grayscale
        if ndims(img) == 3
            img = rgb2gray(img);
        end

        % 2. Apply Gabor filters

        col = 1;

        for o = 1:nOrientations

            theta = orientations(o);

            % Generate Gabor kernel
            gb = GaborFilter( ...
                sigma, theta, lambda, psi, gamma, 3);

            % Normalize filter energy
            gb = gb / (norm(gb(:), 2) + eps);

            % Apply Gabor convolution
            response = conv2(img, gb, 'same');
            r = double(response(:));

            % Mean
            mu = mean(r);

            % Variance
            centered = r - mu;
            variance = mean(centered.^2);

            % Skewness
            if variance < eps
                skew = 0;
            else
                skew = mean(centered.^3) / ...
                    (variance^(3/2));
            end

            % Energy
            energy = sum(r.^2);

             % Entropy (Shannon entropy)
            counts = histcounts(r, 256);

            p = counts / sum(counts);

            % Remove zero probabilities
            p = p(p > 0);

            entropyValue = -sum(p .* log2(p));

            % 4. Store features

            features(n, col:col+4) = ...
                [mu, variance, skew, energy, entropyValue];

            col = col + 5;

        end
   
    end
    featureNames = cell(1, size(features, 2));
    col = 1;
    for o = 1:nOrientations
        for s = 1:nS
            featureNames{col} = sprintf('Gabor_T%d_%s', ...
                round(rad2deg(orientations(o))), statNames{s});
            col = col + 1;
        end
    end
end

