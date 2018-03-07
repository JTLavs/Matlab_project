function err = getErrorRate(trainedClassifier, features, labels, hogImage)

    for i=1:size(features, 1)
       hog = hog_feature_vector(hogImage);
       pred = [pred; trainedClassifier.test(hog)];
    end
    
    numCorrect = (labels == pred);
    numCorrect = size(numCorrect(numCorrect == 1),1);
    err = 1 - (numCorrect /size(testData,1));
        
    
end