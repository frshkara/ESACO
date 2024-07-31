function [dataTrain, dataTrainClass, dataTest, dataTestClass] = SplitDataset(dataset)

instances = dataset(:, 1:end-1);
classes = dataset(:, end);

max_class = max(classes);
classes(classes==0) = max_class + 1;
classes(classes==-1) = max_class + 2;

% Cross varidation (train: 60%, test: 40%)
cv = cvpartition(size(instances,1),'HoldOut',0.4);
idx = cv.test;

% Separate to training and test data
dataTrain = instances(~idx,:);
dataTest  = instances(idx,:);
dataTrainClass = classes(~idx,1);
dataTestClass = classes(idx,1);

end

