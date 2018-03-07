%% Clear Environment
close all
clear all
%%
%% Add path variables
addpath Data
addpath Classifiers
addpath SVM-KM
%%
%% Set up globals
TRAINING_DATASET_PATH = '../Data/pedestrian_train.cdataset';
TEST_DATASET_PATH = '../Data/pedestrian_test.cdataset';

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

maskA = ones(3);
maskA(:,1) = maskA(:,1) -2;
maskA(:,2) = maskA(:,2) -1;

maskB = ones(3);
maskB(1,:) = maskB(1,:) -2;
maskB(2,:) = maskB(2,:) -1;


%% Extract hog feature vectors for training images
[tr_hogEdEx, tr_hogHor, tr_hogVer, tr_hogIm] = getHogFeatures(training_images, maskA, maskB);

%% Cross Validation

edEx_nnCV = cvError(tr_hogEdEx, training_labels, 5, nn);
edEx_knnCV = cvError(tr_hogEdEx, training_labels, 5, knn);

hor_nnCV = cvError(tr_hogHor, training_labels, 5, nn);
hor_knnCV = cvError(tr_hogHor, training_labels, 5, knn);

ver_nnCV = cvError(tr_hogVer, training_labels, 5, nn);
ver_knnCV = cvError(tr_hogVer, training_labels, 5, knn);

hogIm_nnCV = cvError(tr_hogIm, training_labels, 5, nn);
hogIm_knnCV = cvError(tr_hogIm, training_labels, 5, knn);

images_nnCV = cvError(training_images, training_labels, 5, nn);
images_knnCV = cvError(training_images, training_labels, 5, knn);


%% Train classifier models

knn.train(training_features, training_labels);
svm.train(training_features, training_labels);
nn.train(training_features, training_labels);


%% Load test images
[test_images, test_labels] = loadPedestrianDatabase(TEST_DATASET_PATH, 10);
pedestrians = find(test_labels == 1);
others = find(test_labels == -1);

test_images= [test_images(pedestrians,:); test_images(others,:)]; 
test_labels= [test_labels(pedestrians); test_labels(others)];

%% Test Error

