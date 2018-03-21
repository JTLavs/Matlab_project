function struct = trainClassifier(classifier, labels, hogEdEx, hogHor, hogVer, hogIm)

    classifier.train(hogEdEx, labels);
    struct.hogEdEx = classifier;
    classifier.train(hogHor, labels);
    struct.hogHor = classifier;
    classifier.train(hogVer, labels);
    struct.hogVer = classifier;
    classifier.train(hogIm, labels);
    struct.hogIm = classifier;
    
end