
function [x_train, y_train, x_test, y_test] = train_test_split(x, y, split)

n = size(x, 1);

truthValue = (y == 1);
posVal = sum(truthValue);
negVal = n - posVal;

trainPosVal = floor(posVal * split);
testPosVal = posVal - trainPosVal;

trainNegVal = floor(negVal * split);
testNegVal = negVal - trainNegVal;

Pos = x(truthValue, :);
Neg = x(~truthValue, :);

% Positive samples
logicalPosVec = [true(trainPosVal, 1); false(testPosVal, 1)];
logicalPosVec = logicalPosVec(randperm(posVal));

x_trainPos = Pos(logicalPosVec, :);
y_trainPos = ones(trainPosVal, 1);

x_testPos = Pos(~logicalPosVec, :);
y_testPos = ones(testPosVal, 1);

% Negative samples
logicalNegVec = [true(trainNegVal, 1); false(testNegVal, 1)];
logicalNegVec = logicalNegVec(randperm(negVal));

x_trainNeg = Neg(logicalNegVec, :);
y_trainNeg = zeros(trainNegVal, 1);

x_testNeg = Neg(~logicalNegVec, :);
y_testNeg = zeros(testNegVal, 1);

% Combine training samples
logicalTrain = randperm(trainPosVal + trainNegVal);

x_train = [x_trainPos; x_trainNeg];
y_train = [y_trainPos; y_trainNeg];

x_train = x_train(logicalTrain, :);
y_train = y_train(logicalTrain);

% Combine testing samples
logicalTest = randperm(testPosVal + testNegVal);

x_test = [x_testPos; x_testNeg];
y_test = [y_testPos; y_testNeg];

x_test = x_test(logicalTest, :);
y_test = y_test(logicalTest);

end
