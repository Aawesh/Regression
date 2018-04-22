filename = 'orientation_transformed.csv';
M = csvread(filename);
M = unique(M,'rows');

[m,n] = size(M);
n_features = 12;

trainRatio = 0.7;
validationRatio = 0.3;
testRatio = 0;
window_size = 90;



[trainInd,validationInd,~] = dividerand(m,trainRatio,validationRatio,testRatio);

% size(trainInd)
% size(validationInd)
% size(testInd)

train = M(trainInd,:);
%test = M(testInd,:);
validation = M(validationInd,:);


XTrain = train(:,2:n_features+1);
YTrain = train(:,1);

XValidation = validation(:,2:n_features+1);
YValidation = validation(:,1);

XTest = test(:,2:n_features+1);
YTest = test(:,1);

num_4d_rows = floor(size(XTrain,1)/window_size)*window_size;
XTrain = XTrain(1:(num_4d_rows),:);
XTrain = reshape(XTrain,window_size,n_features,1,[]);

n = num_4d_rows/90;
start = 1;
for i = 1:n
    YTrain(i) = mode(YTrain(start:start+90-1));
    start = start+90;
end
YTrain(n+1:end) = [];


num_4d_rows = floor(size(XValidation,1)/window_size)*window_size;
XValidation = XValidation(1:(num_4d_rows),:);
XValidation = reshape(XValidation,window_size,n_features,1,[]);

n = num_4d_rows/90;
start = 1;
for i = 1: n
    YValidation(i) = mode(YValidation(start:start+90-1));
    start = start+90;
end
YValidation(n+1:end) = [];

layers = [
    imageInputLayer([window_size n_features 1])

    convolution2dLayer(6,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    averagePooling2dLayer(2,'Stride',2)

    convolution2dLayer(6,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    averagePooling2dLayer(2,'Stride',2)
  
    convolution2dLayer(6,64,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(6,64,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    dropoutLayer(0.2)
    fullyConnectedLayer(1000)
    fullyConnectedLayer(1)
    regressionLayer];

miniBatchSize  = 10;
validationFrequency = floor(numel(YTrain)/miniBatchSize);

options = trainingOptions('sgdm',...
    'MiniBatchSize',miniBatchSize,...
    'MaxEpochs',30,...
    'InitialLearnRate',1e-5,...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.1,...
    'LearnRateDropPeriod',20,...
    'Shuffle','every-epoch',...
    'validationData',{XValidation,YValidation},...
    'validationFrequency',validationFrequency,...
    'validationPatience',Inf,...
    'Plots','training-progress',...
    'Verbose',false);

net = trainNetwork(XTrain,YTrain,layers,options);
net.Layers

YPredicted = predict(net,XValidation);

predictionError = YValidation - YPredicted;

thr = 0.5;
numCorrect = sum(abs(predictionError) < thr);
numValidationData = numel(YValidation);

accuracy = numCorrect/numValidationData

squares = predictionError.^2;
rmse = sqrt(mean(squares))
