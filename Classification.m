function Result = Classification( method, dataTrain,  dataTrainClass, dataTest, dataTestClass)

% dataTrainClass = cellstr(num2str(dataTrainClass));
% dataTestClass = cellstr(num2str(dataTestClass));

switch method
    case 'svm'
        model = fitcecoc(dataTrain, dataTrainClass);
    case 'knn'
        model = fitcknn(dataTrain, dataTrainClass, 'NumNeighbors', 5);
    case 'dtree'
        model = fitctree(dataTrain, dataTrainClass);
    case 'naiveBayed'
        model = fitcnb(dataTrain, dataTrainClass);
    otherwise
        error('invalid classification method');
end


predictions = predict(model,dataTest);
[~,Result]= confusion.getMatrix(dataTestClass,predictions);

end


function EVAL = Evaluate(ACTUAL,PREDICTED)
% This fucntion evaluates the performance of a classification model by 
% calculating the common performance measures: Accuracy, Sensitivity, 
% Specificity, Precision, Recall, F-Measure, G-mean.
% Input: ACTUAL = Column matrix with actual class labels of the training
%                 examples
%        PREDICTED = Column matrix with predicted class labels by the
%                    classification model
% Output: EVAL = Row matrix with all the performance measures
idx = (ACTUAL()==1);
p = length(ACTUAL(idx));
n = length(ACTUAL(~idx));
N = p+n;
tp = sum(ACTUAL(idx)==PREDICTED(idx));
tn = sum(ACTUAL(~idx)==PREDICTED(~idx));
fp = n-tn;
fn = p-tp;
tp_rate = tp/p;
tn_rate = tn/n;
accuracy = (tp+tn)/N;
sensitivity = tp_rate;
specificity = tn_rate;
precision = tp/(tp+fp);
recall = sensitivity;
f_measure = 2*((precision*recall)/(precision + recall));
gmean = sqrt(tp_rate*tn_rate);
EVAL = [accuracy sensitivity specificity precision recall f_measure gmean];
end

function [Precision,Recall,Fmeasure] = Amin(dataTestClass, predictions)

a=(unique(dataTestClass));
actual=(dataTestClass);
pr=(predictions);
m=size(a,1);
de=zeros(m,m);
for i=1:m
    for j=1:m
        de(i,j)=sum(pr==a(i)&actual==a(j));
    end
end
% [de,order]= confusionmat(dataTestClass,predictions)
Recall=zeros(m,1);
Precision=zeros(m,1);
for i=1:m
    Recall(i,1)=de(i,i)/sum(de(:,i));
    Precision(i,1)=de(i,i)/sum(de(i,:));
end
Precision(isnan(Precision))=0;
Recall=mean(Recall);
Precision=mean(Precision);
Fmeasure=(2*Recall*Precision)/(Recall+Precision);

end


