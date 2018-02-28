function prediction = KNNTesting( testImage, modelNN, K )
    Ed = [];
    for x = 1:size(modelNN.neighbours,1)
         Ed = [Ed EuclideanDistance(modelNN.neighbours(x,:), testImage)];
    end
    
    [B,I] = sort(Ed, 'ascend');
    neighboursindexes = I(1:K);
    
    prediction = mode(modelNN.labels(neighboursindexes));
end

