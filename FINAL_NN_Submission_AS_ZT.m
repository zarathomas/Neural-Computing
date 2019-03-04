%% Neural Network Implementation 
%Conduct Bayesian Optimisation on 3 different types of neural network 
%This code borrows heavily from the following tutorial: https://uk.mathworks.com/help/nnet/examples/deep-learning-using-bayesian-optimization.html
%Import csv data 
%Note: Each row represents an observation. The first column is the label that shows the correct digit for each sample in the dataset
%In the remaining columns, a row represents a 28 x 28 image of a handwritten digit.
data = csvread('train.csv',1,0);

%Transpose data 
targets = data(:,1);
targets = categorical(targets); 
inputs = data(:,2:end);  % the rest of the columns are predictors
n_inputs = size(inputs,1);
formatted_inputs = zeros(28,28,1,n_inputs);
for i = 1:n_inputs
    digit = reshape(inputs(i,:), [28,28]);
    formatted_inputs(:,:,1,i) = digit;
end

trainLabels = targets(1:23520, :);
trainImages = formatted_inputs(:,:,:,1:23520);
valLabels = targets(23521:29400, :);
valImages = formatted_inputs(:,:,:,23521:29400);
testImages = formatted_inputs(:,:,:,29401:end); 
testLabels = targets(29401:end, :);

%%Set the range of variables for optimisations
optimVars = [
    optimizableVariable('InitialLearnRate',[1e-3 5e-2],'Transform','log')
    optimizableVariable('Momentum',[0.8 0.95])
    optimizableVariable('L2Regularization',[1e-10 1e-2],'Transform','log')];

% Perform Bayesian Optimization
% Create the objective function for the Bayesian optimizer.
clear functions
ObjFcn = makeObjFcn(trainImages,trainLabels,valImages,valLabels);

% 
% Perform Bayesian optimization by minimizing the classification error on
% the validation set. 
diary;
BayesObject = bayesopt(ObjFcn,optimVars,'OutputFcn', @assignInBase,'MaxObj',30,'MaxTime',2*60*60,'IsObjectiveDeterministic',false,'UseParallel',false);
diary off; 

% Evaluate Final Network
% Load the best network found in the optimization and its validation
% accuracy.

bestIdx = BayesObject.IndexOfMinimumTrace(end);
fileName = BayesObject.UserDataTrace{bestIdx};
load(fileName);

[predictedLabels,probs] = classify(trainedNet,testImages);
testAccuracy = mean(predictedLabels == testLabels);
testError = 1 - testAccuracy;

testAccuracy
testError
valError

% Calculate the confusion matrix for the test data and display it as a
% heatmap. The highest confusion is between cats and dogs.
figure
[cmat,classNames] = confusionmat(testLabels,predictedLabels);
h = heatmap(classNames,classNames,cmat);
xlabel('Predicted Class');
ylabel('True Class');
title('Confusion Matrix');

%% SVM Implementation 
% Load Data Into Matlab 
df = csvread('train.csv', 1, 0);                  % read train.csv

% Data Preparation 
n = size(df, 1);                    % number of samples in the dataset
rng(1);                             % for reproducibility
cvpt = cvpartition(n,'Holdout',0.3);   % hold out 1/3 of the dataset
Train = df(training(cvpt),:);
Test = df(test(cvpt),:);

TrainingTargets = Train(:,1); 
TrainingInputs = Train(:,2:end); 
TestTargets = Test(:,1); 
TestInputs = Test(:,2:end);

% Train SVM Classifier - Hyperparameter Optimization 
% Fitcecoc is used for multiclass SVM models. The values for the
% hyperparameters (Box Constraint and KernelScale) are replaced with the
% optimised values in the hyperparameter optimization ouput. 
t = templateSVM('KernelFunction','polynomial','PolynomialOrder',1,'BoxConstraint',1,'KernelScale',1 )


tStart = tic; % used to record the time of the hyperparameter optimization 
rng default
Mdl = fitcecoc(TrainingInputs,TrainingTargets,'OptimizeHyperparameters',t, ...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus','ShowPlots','true','MaxObjectiveEvaluations',30))
tElapsed = toc(tStart);
tElapsed

%% Train SVM Classifier - Grid Search  
tStart = tic;
rng default % Used for Reproducibility of results 
pool = parpool; % Invoke workers for parallelization 
t = templateSVM('KernelFunction','gaussian','BoxConstraint',972.62, 'KernelScale', 34.489)
options = statset('UseParallel',true); %Used for Parallelization of work
Mdl = fitcecoc(TrainingInputs,TrainingTargets,'Learners',t, 'Options',options);
tElapsed = toc(tStart);
tElapsed
% Evaluate Classifier
% Make class predictions using the test features.
predictedLabels = predict(Mdl, TestInputs);
% Tabulate the results usi  ng a confusion matrix.
confMat = confusionmat(TestTargets, predictedLabels);

filename1 = 'ConfusionMatrix_gaussian.xlsx'; % Saves Confusion Matrix as CSV file
filename2 = 'TimeElapsed_gaussian.xlsx';  % Saves Elapsed Time as CSV file
xlswrite(filename1,confMat,1)
xlswrite(filename2,tElapsed,1)




