function prediction = KNNTest( testImage, modelNN, K )
    Ed = [];
    for x = 1:size(modelNN.neighbours,1)
         Ed = [Ed EuclideanDistance(modelNN.neighbours(x,:), testImage)];
    end
    
    [B,I] = sort(Ed, 'ascend');
    
    if nargin < 3
        K = 3;
    end
    
    neighboursindexes = I(1:K);
    
    prediction = mode(modelNN.labels(neighboursindexes));
end

