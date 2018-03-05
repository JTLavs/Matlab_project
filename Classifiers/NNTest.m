function prediction = NNTest(testImage, modelNN)
    closest = Inf;
    for x = 1:size(modelNN.neighbours,1)
         Ed = EuclideanDistance(modelNN.neighbours(x,:), testImage);
         if Ed < closest
            closest = Ed;
            predictionLabel = modelNN.labels(x,:);
         end
    end
    prediction = predictionLabel;
end

