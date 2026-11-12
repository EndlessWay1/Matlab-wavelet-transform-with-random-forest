clc;
clear;

data = readtable("transformGaborData.csv");

y = data{:, end};

x = data{:, 1:(end - 1)};

[x_train, y_train, x_test, y_test] = train_test_split(x, y, 0.8);

rng(42);

model = RandomForest(100, 64, 5, 20);
disp("Running");
model = model.fit(x_train, y_train);
disp("Done Training");
y_pred = model.predict(x_test);

baseline_test = max(mean(y_test==1), mean(y_test~=1));

% what is it predicting?
C = confusionmat(y_test, y_pred);
recall_pos = C(2,2) / sum(C(2,:));
precision_pos = C(2,2) / sum(C(:,2));
fprintf("Predicted positive: %d of %d\n", sum(y_pred==1), numel(y_pred));


cv = cvpartition(y, 'KFold', 5);
aucs = zeros(cv.NumTestSets,1);
for k = 1:cv.NumTestSets
    tr = training(cv,k); te = test(cv,k);
    mdl = TreeBagger(200, x(tr,:), y(tr), 'Method','classification', ...
                     'Prior','Uniform', 'MinLeafSize',5);
    [~, score] = predict(mdl, x(te,:));
    [~,~,~,aucs(k)] = perfcurve(y(te), score(:,2), 1);
end
fprintf("AUC: %.3f +/- %.3f\n", mean(aucs), std(aucs));

classNames = ["Normal", "Glacoma"];

figure;
cm = confusionchart(C, classNames);

