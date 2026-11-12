
classdef Node
    %Node Summary of this class goes here
    %   Detailed explanation goes here

    properties
        feature;
        threshold;
        left;
        right;
        value;
    end

    methods
        function obj = Node(value, features, threshold, left, right)
            %TREE Construct an instance of this class
            %   Detailed explanation goes here
            if nargin < 2
                features = NaN;
            end
            if nargin < 3
                threshold = NaN;
            end
            if nargin < 4
                left = [];
            end
            if nargin < 5
                right = [];
            end
            obj.feature = features;
            obj.threshold = threshold;
            obj.left = left;
            obj.right = right;
            obj.value = value;
        end

        function bool = isLeafNode(obj)
            bool = ~isnan(obj.value);
        end
    end
end
