classdef classifier < handle
   
    properties
        TrainingFunction    % Function to train the classifier
        TestingFunction     % Function to test new data against the model
        Model               % Output of training
    end
    
    methods
        function obj=classifier(trainingFunction, testingFunction)
            obj.TrainingFunction = trainingFunction;
            obj.TestingFunction = testingFunction;
        end
        
        function train(obj, trainingData, trainingLabels)
            obj.Model = obj.TrainingFunction(trainingData, trainingLabels);
        end
        
        function prediction = test(obj, new_item)
            % TODO: Add functionality to allow optional parameters such as
            %       k-value, kernel etc.
            prediction = obj.TestingFunction(new_item, obj.Model);
        end
        
    end
end