function err = getErrorRate(trainedClassifier, testData, testLabels)

    %% Setup masks for edge extraction
 

    maskA = ones(3);
    maskA(:,1) = maskA(:,1) -2;
    maskA(:,2) = maskA(:,2) -1;

    maskB = ones(3);
    maskB(1,:) = maskB(1,:) -2;
    maskB(2,:) = maskB(2,:) -1;
    pred = [];
    
    for i=1:size(testData, 1)

       Im = reshape(testData(i,:),160,96);
       [ImEdEx, ImIhor, ImIver] =  edgeExtraction(Im,maskA, maskB);
       hog = hog_feature_vector(ImEdEx);

       pred = [pred; trainedClassifier.test(hog)];

    end
    
    numCorrect = (testLabels == pred);
    numCorrect = size(numCorrect(numCorrect == 1),1);
    err = 1 - (numCorrect /size(testData,1));
        
    
end