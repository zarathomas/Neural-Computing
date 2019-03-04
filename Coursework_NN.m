%% Load Data Into Matlab 
df = csvread('train.csv', 1, 0);                  % read train.csv
%% Data Preparation 
n = size(df, 1);                    % number of samples in the dataset
rng(1);                             % for reproducibility
cvpt = cvpartition(n,'Holdout',0.3);   % hold out 1/3 of the dataset
Train = df(training(cvpt),:);
Test = df(test(cvpt),:);

TrainingTargets = Train(:,1); 
TrainingInputs = Train(:,2:end); 
TestTargets = Test(:,1); 
TestInputs = Test(:,2:end);

%% 

% Train SVM Classifier 
t = templateSVM('KernelFunction','linear','BoxConstraint',1 )
%
tStart = tic;
Mdl = fitcecoc(TrainingInputs,TrainingTargets,'Learners',t);
tElapsed = toc(tStart);
tElapsed
% Evaluate Classifier
% Make class predictions using the test features.
predictedLabels = predict(Mdl, TestInputs);
% Tabulate the results using a confusion matrix.
confMat = confusionmat(TestTargets, predictedLabels);


%% Train SVM Classifier 
t = templateSVM('KernelFunction','linear','BoxConstraint',10 )
%
tStart = tic;
Mdl = fitcecoc(TrainingInputs,TrainingTargets,'Learners',t);
tElapsed = toc(tStart);
tElapsed
% Evaluate Classifier
% Make class predictions using the test features.
predictedLabels = predict(Mdl, TestInputs);
% Tabulate the results using a confusion matrix.
confMat = confusionmat(TestTargets, predictedLabels);
confMat
3
%
filename1 = 'ConfusionMatrix.xlsx';
filename2 = 'TimeElapsed.xlsx';
xlswrite(filename1,confMat,3)
xlswrite(filename2,tElapsed,3)

%% Train SVM Classifier 
t = templateSVM('KernelFunction','linear','BoxConstraint',100 )
%
tStart = tic;
Mdl = fitcecoc(TrainingInputs,TrainingTargets,'Learners',t);
tElapsed = toc(tStart);
tElapsed
% Evaluate Classifier
% Make class predictions using the test features.
predictedLabels = predict(Mdl, TestInputs);
% Tabulate the results usi  ng a confusion matrix.
confMat = confusionmat(TestTargets, predictedLabels);
confMat
4
%%
filename1 = 'ConfusionMatrix.xlsx';
filename2 = 'TimeElapsed.xlsx';
xlswrite(filename1,confMat,4)
xlswrite(filename2,tElapsed,4)

%% Train SVM Classifier 
t = templateSVM('KernelFunction','polynomial','PolynomialOrder',2,'BoxConstraint',0.39361, 'KernelScale',6.9524)
%
tStart = tic;
Mdl = fitcecoc(TrainingInputs,TrainingTargets,'Learners',t);
tElapsed = toc(tStart);
tElapsed
% Evaluate Classifier
% Make class predictions using the test features.
predictedLabels = predict(Mdl, TestInputs);
% Tabulate the results usi  ng a confusion matrix.
confMat = confusionmat(TestTargets, predictedLabels);
confMat
5
filename1 = 'ConfusionMatrix_gaussian2.xlsx';
filename2 = 'TimeElapsed_gaussian2.xlsx';
xlswrite(filename1,confMat,1)
xlswrite(filename2,tElapsed,1)

%% Train SVM Classifier 
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
%%
% Evaluate Classifier
% Make class predictions using the test features.
predictedLabels = predict(Mdl, TestInputs);
% Tabulate the results usi  ng a confusion matrix.
confMat = confusionmat(TestTargets, predictedLabels);
confMat

%%

%% Train SVM Classifier 

tStart = tic;
rng default
pool = parpool; % Invoke workers
t = templateSVM('KernelFunction','gaussian','BoxConstraint',972.62, 'KernelScale', 34.489)
options = statset('UseParallel',true);
Mdl = fitcecoc(TrainingInputs,TrainingTargets,'Learners',t, 'Options',options);
tElapsed = toc(tStart);
tElapsed
% Evaluate Classifier
% Make class predictions using the test features.
predictedLabels = predict(Mdl, TestInputs);
% Tabulate the results usi  ng a confusion matrix.
confMat = confusionmat(TestTargets, predictedLabels);
confMat
5

filename1 = 'ConfusionMatrix_gaussian2.xlsx';
filename2 = 'TimeElapsed_gaussian2.xlsx';
xlswrite(filename1,confMat,1)
xlswrite(filename2,tElapsed,1)
