function detectorModel = TrainDetectorModel()

    TRAINING_DATASET_PATH = 'Data/pedestrian_train.cdataset';
    TEST_DATASET_PATH = 'Data/pedestrian_test.cdataset';
    knn = classifier(@KNNTrain,  @KNNTest);
    svm = classifier(@SVMTrain, @SVMTest);
    nn = classifier(@NNTrain, @NNTest);
    
    %% Load training images
    [training_images, training_labels] = loadPedestrianDatabase(TRAINING_DATASET_PATH, 1);
    pedestrians = find(training_labels == 1);
    others = find(training_labels == -1);


    training_images= [training_images(pedestrians,:); training_images(others,:)];
    training_labels = [training_labels(pedestrians,:); training_labels(others,:)];
    
    %% Load tets images
     [test_images, test_labels] = loadPedestrianDatabase(TEST_DATASET_PATH, 1);
    pedestrians = find(test_labels == 1);
    others = find(test_labels == -1);


    test_images= [test_images(pedestrians,:); test_images(others,:)];
    test_labels = [test_labels(pedestrians,:); test_labels(others,:)];
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

   
    svm.train(tr_hogEdEx, training_labels);
   
    
 
    detectorModel = struct;
    detectorModel.knn = knn;
    detectorModel.nn = nn;
    detectorModel.svm = svm;

    %% Get Test features
    [te_hogEdEx, te_hogHor, te_hogVer, te_hogIm] = getHogFeatures(test_images, maskA, maskB);
   
    %% GET ERROR

    getErrorRate(detectorModel.ans.svm, te_hogEdEx, test_labels);

end