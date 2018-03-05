%% Function to perform K-fold Cross Validation on a training/test set using
%  a passed in classifier function.

%  the function returns the missclassification error for the given 
%  classifier

%  trainingData -  a set of labelled training data
%  kFolds - the number of partitions to make in each set
%  classifier - a function handle to reference the classifier
%%
function cvErr = cvError(trainingData, kFolds, classifier)
    
end