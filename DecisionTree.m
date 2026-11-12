
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
%             Using gini impurity calc scheme
            left  = x_col < threshold;
            right = ~left;
            if sum(left) == 0 || sum(right) == 0
                gain = Inf; return;
            end
            s  = numel(y);
            ls = sum(left);   rs = sum(right);
            pl = sum(y(left))  / ls;
            pr = sum(y(right)) / rs;
            gain = (ls/s)*(1 - pl^2 - (1-pl)^2) + (rs/s)*(1 - pr^2 - (1-pr)^2);
        end
        
        function [bestFeat,bestThreshold] = bestSplit(obj, x, y, feat_idx)
            bestGain = Inf;
            split_idx = NaN;
            splitThreshold = NaN;
            
            n = numel(feat_idx);
            for i = 1:n
                x_col = x(:, feat_idx(i));
                values = unique(x_col);
                threshold = (values(1:end-1) + values(2:end)) / 2;
                [m,~] = size(threshold);
                for k = 1:m
                    gain = obj.infoGainFunc(x_col, y, threshold(k));
                    
                    if gain < bestGain 
                        bestGain = gain;
                        split_idx = feat_idx(i);
                        splitThreshold = threshold(k);
                    end
                end
            end
            
            bestFeat = split_idx;
            bestThreshold = splitThreshold;
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
            %               find best split
            [bestFeat, bestThreshold] = obj.bestSplit(x, y, feat_idx);
            if isnan(bestFeat) || isnan(bestThreshold)
                leaf_value = double(y_lab > s - y_lab);
                root = Node(leaf_value);
                return;
            end
%               create child nodes
            left_idx = find(x(:, bestFeat) < bestThreshold);
            right_idx = find(x(:, bestFeat) >= bestThreshold);
            
            left = obj.growTree(x(left_idx, :), y(left_idx), depth + 1);
            right = obj.growTree(x(right_idx, :), y(right_idx), depth + 1);
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

