function [dataTrainLabel] = MissDatasetLabel1(dataTrainLabel, keepRatio)

if keepRatio > 1 || keepRatio < 0
    error('MissRatio should varies between [0,...,1]');
end

[nIns, ~] = size(dataTrainLabel);

labeledIdx = randperm(nIns, floor( (keepRatio) * nIns ));
unlabeledIdx = setdiff(1:nIns, labeledIdx);

dataTrainLabel(unlabeledIdx) = nan;

% labeled = [dataTrain(labeledIdx, :), dataTrainLabel(labeledIdx, :)];
% unlabeled = [dataTrain(unlabeledIdx, :), repelem(nan, length(unlabeledIdx))'];

end

