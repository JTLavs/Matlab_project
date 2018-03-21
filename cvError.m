%% Function to perform K-fold Cross Validation on a training/test set using
%  a passed in classifier function.

%  the function returns the missclassification error for the given 
%  classifier

%  trainingData -  a set of labelled training data
%  kFolds - the number of partitions to make in each set
%  classifier - a reference to a classifier object.
% 
%
%% - Dan
function cvErr = cvError(trainingData, trainingLabels, kFolds, classifier)
    indices = crossvalind('KFold', size(trainingData,1), kFolds);
    
  
    cvErr = [];
    for i = 1:kFolds
        testData = trainingData(indices == i, :);
        testLabels = trainingLabels(indices == i, :);
        
        trainData = trainingData(indices ~= i, :);
        trainLabels = trainingLabels(indices ~= i, :);
        
        classifier.train(trainData, trainLabels);
        predictions = [];
        for j = 1:size(testData,1)
            newClass = classifier.test(testData(j,:))
            predictions = [predictions; newClass]
        end
      
        numCorrect = (testLabels == predictions);
        numCorrect = size(numCorrect(numCorrect == 1),1);
        cvErr = [cvErr; 1 - (numCorrect /size(testData,1))];
        
    end
    
    cvErr = mean (cvErr);
end