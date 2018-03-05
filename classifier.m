classdef classifier
   
    properties
        TrainingFunction    % Function to train the classifier
        TestingFunction     % Function to test new data against the model
    end
    
    methods
        function obj=classifier(trainingFunction, testingFunction)
            obj.TrainingFunction = trainingFunction;
            obj.TestingFunction = testingFunction;
        end
        
        function train(obj, trainingData, trainingLabels)
            obj.Model = obj.TrainingFunction(trainingData, trainingLabels)
        end
        
    end
end