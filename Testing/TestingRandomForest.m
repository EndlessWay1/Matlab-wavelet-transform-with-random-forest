clc;
clear;

data = readtable("breastCancerDataset.csv");

% Binary target: M = 1, B = 0
y = strcmp(string(data{:, 2}), 'M');

% Feature matrix: columns 3 onward
x = data{:, 3:end};

% Display the first few rows
disp(head(data));

% Cross varidation (train: 70%, test: 30%)
cv = cvpartition(size(data,1),'HoldOut',0.3);
idx = cv.test;
% Separate to training and test data
dataTrain = data(~idx,:);
dataTest  = data(idx,:);
x_train = dataTrain{:, 3:end};
y_train = strcmp(string(dataTrain{:, 2}), 'M');

x_test = dataTest{:, 3:end};
y_test = strcmp(string(dataTest{:, 2}), 'M');


model = RandomForest(100, 64, 32,32);
disp("Running");
model = model.fit(x_train, y_train);
disp("Done Training");
y_pred = model.predict(x_test);
accuracy = sum(y_pred == y_test)/length(y_test);