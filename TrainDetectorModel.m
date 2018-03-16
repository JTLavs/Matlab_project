function detectorModel = TrainDetectorModel()

    TRAINING_DATASET_PATH = 'Data/pedestrian_train.cdataset';
    knn = classifier(@KNNTrain,  @KNNTest);
    svm = classifier(@SVMTrain, @SVMTest);
    nn = classifier(@NNTrain, @NNTest);
    
    %% Load training images
    [training_images, training_labels] = loadPedestrianDatabase(TRAINING_DATASET_PATH, 10);
    pedestrians = find(training_labels == 1);
    others = find(training_labels == -1);


    training_images= [training_images(pedestrians,:); training_images(others,:)];
    training_labels = [training_labels(pedestrians,:); training_labels(others,:)];
    
    %% Setup masks for edge extraction

    maskA = ones(3);
    maskA(:,1) = maskA(:,1) -2;
    maskA(:,2) = maskA(:,2) -1;

    maskB = ones(3);
    maskB(1,:) = maskB(1,:) -2;
    maskB(2,:) = maskB(2,:) -1;

    %% Extract hog feature vectors for training images
    [tr_hogEdEx, tr_hogHor, tr_hogVer, tr_hogIm] = getHogFeatures(training_images, maskA, maskB);
    
    %% Train classifier models

    knn.train(tr_hogEdEx, training_labels);
    svm.train(tr_hogEdEx, training_labels);
    nn.train(tr_hogEdEx, training_labels);
    
  
    detectorModel = struct;
    detectorModel.knn = knn;
    detectorModel.nn = nn;
    detectorModel.svm = svm;

    

end