function err = getErrorRate(trainedClassifier, features, labels)

    pred = [];
    for i=1:size(features, 1)
 
       
       pred = [pred; trainedClassifier.test(features(i,:))];
    end
    
    numCorrect = (labels == pred);
    numCorrect = size(numCorrect(numCorrect == 1),1);
    err = 1 - (numCorrect /size(features,1));
        
    
end