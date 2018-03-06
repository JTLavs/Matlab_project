%% Clear Environment
close all
clear all
%%
%% Add path variables
addpath ..\Data
addpath ..\Classifiers
addpath ..\SVM-KM
%%
%% Set up globals
TRAINING_DATASET_PATH = 'pedestrian_train.cdataset';
TEST_DATASET_PATH = 'pedestrian_test.cdataset';

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
training_features = [];

cvErr = cvError(training_images, training_labels, 5, nn);

maskA = ones(3);
maskA(:,1) = maskA(:,1) -2;
maskA(:,2) = maskA(:,2) -1;

maskB = ones(3);
maskB(1,:) = maskB(1,:) -2;
maskB(2,:) = maskB(2,:) -1;
%% Extract hog feature vectors for training images

for i=1:size(training_images, 1)
    Im = reshape(training_images(i,:),160,96);
    %ImBrightness =  brightEnchance(Im,50);
    [ImEdEx, ImIhor, ImIver] =  edgeExtraction(Im,maskA, maskB);
    
    hog = hog_feature_vector(Im);
    training_features = [training_features; training_images(i,:)]; 
end

%% Train classifier models

%knn.train(training_features, training_labels)
%svm.train(training_features, training_labels)
%nn.train(training_features, training_labels)

%% Load test images
[test_images, test_labels] = loadPedestrianDatabase(TEST_DATASET_PATH, 10);
pedestrians = find(test_labels == 1);
others = find(test_labels == -1);

test_images= [test_images(pedestrians,:); test_images(others,:)]; 
test_labels= [test_labels(pedestrians); test_labels(others)];

<<<<<<< HEAD
%% Extract hog feature vectors for test images
test_features = [];
for i=1:size(test_images, 1)

   Im = reshape(test_images(i,:),160,96);
   [ImEdEx, ImIhor, ImIver] =  edgeExtraction(Im,maskA, maskB);
   hog = hog_feature_vector(Im);
   
   %test_features = [test_features; hog];
   % TODO: classify each test image here using hog vector
   %classificationResult(i,:) = svm.test(test_images(i,:));
end
%%
%comparison = (test_labels==classificationResult);
%Accuracy = sum(comparison)/length(comparison)
%ErrRate = 1 - Accuracy
cvErr
 
err = getErrorRate(svm, test_images, test_labels)
