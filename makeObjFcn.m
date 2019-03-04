%This code borrows heavily from the following tutorial: https://uk.mathworks.com/help/nnet/examples/deep-learning-using-bayesian-optimization.html
function ObjFcn = makeObjFcn(trainImages,trainLabels,valImages,valLabels)
ObjFcn = @valErrorFun;
    function [valError,cons,fileName] = valErrorFun(optVars)
        imageSize = [28 28 1];  %Define three different types of neural network architecture;
                                %this needs to be manually set throughout
                                %the function 
        numClasses = numel(unique(trainLabels));
        layers = [
              imageInputLayer(imageSize)
              fullyConnectedLayer(numClasses)
              softmaxLayer
              classificationLayer];  
          
        layers2 = [
              imageInputLayer(imageSize)
              fullyConnectedLayer(800)
              reluLayer %Originally reluLayer
              fullyConnectedLayer(numClasses)
              softmaxLayer
              classificationLayer];
      
        layers3 = [
              imageInputLayer(imageSize)
              fullyConnectedLayer(1000)
              reluLayer %Originally reluLayer
              fullyConnectedLayer(numClasses)
              softmaxLayer
              classificationLayer];    
    
     
        layers_CNN = [
        imageInputLayer([28 28 1])
        convolution2dLayer(5,20)
        reluLayer
        maxPooling2dLayer(2,'Stride',2)
        fullyConnectedLayer(10)
        softmaxLayer
        classificationLayer];

       
        %%
        % Specify options for network training. 
        miniBatchSize = 128;
        numValidationsPerEpoch = 20;
        validationFrequency = floor(size(trainImages,4)/miniBatchSize/numValidationsPerEpoch);
        %options = trainingOptions('sgdm','InitialLearnRate',1e-3,'Momentum',optVars.Momentum,'MaxEpochs',100,'MiniBatchSize',miniBatchSize,'L2Regularization',1e-10,'Shuffle','every-epoch','Verbose',false,'Plots','training-progress','ValidationData', {valImages,valLabels},'ValidationPatience',4,'ValidationFrequency',validationFrequency);
        options = trainingOptions('sgdm','InitialLearnRate',optVars.InitialLearnRate,'Momentum',optVars.Momentum,'MaxEpochs',1000,'MiniBatchSize',miniBatchSize,'L2Regularization',optVars.L2Regularization,'Shuffle','every-epoch','Verbose',false,'Plots','training-progress','ValidationData', {valImages,valLabels},'ValidationPatience',4,'ValidationFrequency',validationFrequency);

        %%
        % Train the network and plot the training progress during training.
        % Close the plot after training finishes.
        trainedNet = trainNetwork(trainImages,trainLabels ,layers_CNN,options); %Specify which layers option
        close(findall(groot,'Tag','NNET_CNN_TRAININGPLOT_FIGURE'))
        %%
        % After training stops, lower the learning rate by a factor of 10
        % and continue training. 
        options = trainingOptions('sgdm','InitialLearnRate',optVars.InitialLearnRate/10,'Momentum',optVars.Momentum,'MaxEpochs',1000,'MiniBatchSize',miniBatchSize,'L2Regularization',optVars.L2Regularization,'Shuffle','every-epoch','Verbose',false,'Plots','training-progress','ValidationData',{valImages,valLabels},'ValidationPatience',4,'ValidationFrequency',validationFrequency);
        %options = trainingOptions('sgdm','InitialLearnRate',1e-3,'Momentum',optVars.Momentum,'MaxEpochs',100,'MiniBatchSize',miniBatchSize,'L2Regularization',1e-10,'Shuffle','every-epoch','Verbose',false,'Plots','training-progress','ValidationData',{valImages,valLabels},'ValidationPatience',4,'ValidationFrequency',validationFrequency);
        trainedNet = trainNetwork(trainImages,trainLabels,layers_CNN, options);
        close
        
        %%
        % Evaluate the trained network on the validation set, and calculate
        % the validation error.
        predictedLabels = classify(trainedNet,valImages);
        valAccuracy = mean(predictedLabels == valLabels);
        valError = 1 - valAccuracy;
        
        %%
        % Create a file name containing the validation error, and save the
        % network, validation error, and training options to disk. 
        fileName = num2str(valError,10) + ".mat";
        save(fileName,'trainedNet','valError','options')
        cons = [];
        
        
    end
end
