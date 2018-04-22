%SVM
filename = 'orientation_transformed.csv';
M = csvread(filename);
M = unique(M,'rows');


% X = M(:,2:13)
% Y = M(:,1)
[m,n] = size(M);
n_features = 12;

trainRatio = 0.8;
validationRatio = 0.2;
testRatio = 0;



[trainInd,validationInd,~] = dividerand(m,trainRatio,validationRatio,testRatio);
 

train = M(trainInd,:);
test = M(testInd,:);
validation = M(validationInd,:);


XTrain = train(:,2:n_features+1);
YTrain = train(:,1);

XValidation = validation(:,2:n_features+1);
YValidation = validation(:,1);


% Mdl = fitrsvm(XTrain,YTrain,'KernelFunction','gaussian','Standardize',true,'Optimizehyperparameters','auto',... 'HyperparameterOptimizationOptions',struct);
% Mdl_optimized = fitrsvm(XTrain,YTrain,'KernelFunction','gaussian','Standardize',true,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('AcquisitionFunctionName', 'expected-improvement-plus', 'MaxObjectiveEvaluations', 60))
% predicted = predict(Mdl_optimized,XValidation)
