
classdef DecisionTree
    %DECISIONTREE Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        minSampleSplit
        maxDepth
        n_features
        root
    end
    
    methods
        function obj = DecisionTree(minSampleSplit, maxDepth, n_feat)
            %DECISIONTREE Construct an instance of this class
            %   Detailed explanation goes here
            obj.minSampleSplit = minSampleSplit;
            obj.maxDepth = maxDepth;
            obj.n_features = n_feat;
            obj.root = [];
        end
        
        function y = predict(obj, x)
            [s, ~] = size(x);
            res = zeros(s, 1);
            for i = 1:s
                res(i) = obj.traverseTree(x(i, :), obj.root);
            end
            y = res;
        end
        
        function y = traverseTree(obj, x, node)
            if (node.isLeafNode())
                y = node.value;
                return;
            end
            
            if (x(1, node.feature) < node.threshold)
                y = obj.traverseTree(x, node.left);
            else
                y = obj.traverseTree(x, node.right);
            end
        end
        

        function gain = infoGainFunc(obj, x_col, y, threshold)

            left = x_col < threshold;
            right = ~left;

            if ~any(left) || ~any(right)
                gain = -Inf;
                return;
            end

            % Parent Gini impurity
            p = mean(y == 1);
            parentGini = 1 - p^2 - (1-p)^2;

            % Left child Gini impurity
            pLeft = mean(y(left) == 1);
            giniLeft = 1 - pLeft^2 - (1-pLeft)^2;

            % Right child Gini impurity
            pRight = mean(y(right) == 1);
            giniRight = 1 - pRight^2 - (1-pRight)^2;

            % Weighted child impurity
            n = numel(y);

            weightedGini = ...
                (sum(left)/n) * giniLeft + ...
                (sum(right)/n) * giniRight;

            % Gini gain
            gain = parentGini - weightedGini;

        end
        
        function [bestFeat, bestThreshold] = bestSplit(obj, x, y, feat_idx)
            bestGain = -Inf;
            bestFeat = NaN;
            bestThreshold = NaN;

            for i = 1:numel(feat_idx)
                feature = feat_idx(i);

                x_col = x(:, feature);

                values = unique(x_col);

                if numel(values) < 2
                    continue;
                end

                thresholds = (values(1:end-1) + values(2:end)) / 2;
                for k = 1:numel(thresholds)
                    gain = obj.infoGainFunc(x_col, y, thresholds(k));

                    if gain > bestGain
                        bestGain = gain;
                        bestFeat = feature;
                        bestThreshold = thresholds(k);
                    end
                end
            end
        end
        
        
        
        function root = growTree(obj, x, y, depth)
            [s, p] = size(x);
            y_lab = sum(y == 1);

%             check stop critiria (base case)
            if (depth >= obj.maxDepth || y_lab == s || 0 == y_lab || s < obj.minSampleSplit)
                if y_lab > s - y_lab
                    leaf_value = 1;
                else
                    leaf_value = 0;
                end
                root = Node(leaf_value);
                return;
            end
                
            n_feat = min(p, obj.n_features);

            feat_idx = randperm(p, n_feat);

            % Find best split
            [bestFeat, bestThreshold] = obj.bestSplit(x, y, feat_idx);
            if isnan(bestFeat) || isnan(bestThreshold)
                leaf_value = double(y_lab > s - y_lab);
                root = Node(leaf_value);
                return;
            end

            % Create child nodes
            left_idx = find(x(:, bestFeat) < bestThreshold);
            right_idx = find(x(:, bestFeat) >= bestThreshold);

            % Prevent empty children
            if isempty(left_idx) || isempty(right_idx)
                leaf_value = double(y_lab > s - y_lab);
                root = Node(leaf_value);
                return;
            end

            left = obj.growTree(x(left_idx,:), y(left_idx), depth + 1);
            right = obj.growTree(x(right_idx,:), y(right_idx), depth + 1);
            root = Node(NaN, bestFeat, bestThreshold, left, right);
        end
        
        function obj = fit(obj, x, y)
            y = y(:);
            [~, p] = size(x);
%             check if n is in bound
            if isnan(obj.n_features)
                obj.n_features = p;
            else
                obj.n_features = min(p, obj.n_features);
            end
            
            obj.root = obj.growTree(x, y, 0);
            
        end
    end
end

