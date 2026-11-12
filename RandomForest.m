classdef RandomForest
    %RANDOMFOREST Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        n_estimator
        max_depth
        min_sample_split
        n_features
        trees
    end
    
    methods
        function obj = RandomForest(n_estimator, max_depth, min_sample_split, n_features)
            %RANDOMFOREST Construct an instance of this class
            %   Detailed explanation goes here
            obj.n_estimator = n_estimator;
            obj.max_depth = max_depth;
            obj.min_sample_split = min_sample_split;
            obj.n_features = n_features;
            obj.trees = cell(n_estimator, 1);
        end
        
        function obj = fit(obj, x,y)
            min_sample = obj.min_sample_split;
            max_depths = obj.max_depth;
            feat = obj.n_features;
            for i=1:obj.n_estimator
                tree = DecisionTree(min_sample, max_depths, feat);
                
                [x_sample, y_sample] = obj.BoostrapSamples(x, y);
                tree = tree.fit(x_sample, y_sample);
                obj.trees{i} = tree;
            end
        
        end
       
        function y = predict(obj, x)
            n_samples = size(x, 1);
            n_trees = numel(obj.trees);

            pred = zeros(n_samples, n_trees);

            for i = 1:n_trees
                pred(:, i) = obj.trees{i}.predict(x);
            end

            y = mode(pred, 2);
        end
    end
   methods(Static)
       function [x_s, y_s] = BoostrapSamples(x, y)
            n_samples = size(x, 1);
            idx = randsample(n_samples, n_samples, true);
            x_s = x(idx, :);
            y_s = y(idx, :);
       end
    end
end

