classdef PCA
    %PCA Principal component analysis with z-score standardization.
    %   pca = PCA(0.95);  pca = pca.fit(xTrain);  Z = pca.transform(xTest);

    properties
        percentage              % target cumulative variance (0-1)
        originalFeatures        % number of columns in training data
        validFeatures           % logical mask of non-constant columns
        cumulativePercentage
        myu
        sigma
        eigVec
        eigVal
        lengthOfData            % number of selected components
    end

    methods
        function obj = PCA(percentage)
            if percentage <= 0 || percentage > 1
                error('percentage must be in (0, 1], e.g. 0.95 for 95%%.');
            end
            obj.percentage = percentage;
        end

        function obj = fit(obj, xTrain)
            obj.originalFeatures = size(xTrain, 2);

            % Statistics from training data only
            obj.myu    = mean(xTrain, 1);
            obj.sigma  = std(xTrain, 0, 1);

            % Identify non-constant features BEFORE dividing
            obj.validFeatures = obj.sigma > 0;

            % Standardize valid features only
            v = obj.validFeatures;
            ZTrain = (xTrain(:, v) - obj.myu(v)) ./ obj.sigma(v);

            % Covariance of standardized data (= correlation matrix)
            Zcov = cov(ZTrain);

            % Eigendecomposition, sorted descending
            [V, A] = eig(Zcov);
            [eVal, order] = sort(diag(A), 'descend');
            eVal = max(eVal, 0);            % clip numerical noise
            obj.eigVal = eVal;
            obj.eigVec = V(:, order);

            % Explained variance
            obj.cumulativePercentage = cumsum(eVal / sum(eVal));
            k = find(obj.cumulativePercentage >= obj.percentage, 1);
            if isempty(k)
                k = numel(eVal);
            end
            obj.lengthOfData = k;
        end

        function Z = transform(obj, data)
            v = obj.validFeatures;
            ZData = (data(:, v) - obj.myu(v)) ./ obj.sigma(v);
            Z = ZData * obj.eigVec(:, 1:obj.lengthOfData);
        end

        function report(obj)
            fprintf('\nOriginal number of features: %d\n', obj.originalFeatures);
            fprintf('Remaining features after removing constant columns: %d\n', ...
                sum(obj.validFeatures));
            fprintf('Selected principal components: %d\n', obj.lengthOfData);
            fprintf('Retained variance: %.2f%%\n', ...
                obj.cumulativePercentage(obj.lengthOfData) * 100);

            figure;

            subplot(2,1,1);
            plot(1:numel(obj.eigVal), obj.eigVal, '-o', ...
                'LineWidth', 1.5, 'MarkerSize', 5);
            xlabel('Principal Component');
            ylabel('Eigenvalue');
            title('PCA Eigenvalues (Training Data)');
            grid on;

            subplot(2,1,2);
            plot(1:numel(obj.cumulativePercentage), ...
                obj.cumulativePercentage * 100, '-o', ...
                'LineWidth', 1.5, 'MarkerSize', 5);
            hold on;
            yline(obj.percentage * 100, '--r', ...
                sprintf('%g%% Threshold', obj.percentage * 100));
            xline(obj.lengthOfData, '--g', ...
                sprintf('%d Components', obj.lengthOfData));
            xlabel('Number of Principal Components');
            ylabel('Cumulative Explained Variance (%)');
            title('Cumulative Explained Variance');
            grid on;
            hold off;
        end

        function toCSV(obj, data, labels, dataPath)
            if size(labels, 1) ~= size(data, 1)
                error('toCSV:sizeMismatch', ...
                    'Data has %d rows but labels has size %s.', ...
                    size(data, 1), mat2str(size(labels)));
            end
            folder = fileparts(dataPath);
            if ~isempty(folder) && ~isfolder(folder)
                mkdir(folder);
            end
            T = array2table(data, ...
                'VariableNames', compose('PC%d', 1:size(data, 2)));
            T.Label = labels;
            writetable(T, dataPath);
        end
    end
end