clear all
%% ========================================================================
%
%                       config
%
%  ========================================================================
%% Load Trained Model
%  This model contains trained classifiers for NN, KNN and SVM trained with
%  HogEdEx - this is optional
model = load('detectorModel.mat');
model = model.detectorModel;


%% Setup masks for edge extraction

maskA = ones(3);
maskA(:,1) = maskA(:,1) -2;
maskA(:,2) = maskA(:,2) -1;

maskB = ones(3);
maskB(1,:) = maskB(1,:) -2;
maskB(2,:) = maskB(2,:) -1;
%% ========================================================================
%
%                       training
%
%  ========================================================================
%  Here you will find sections to load the training set at sampling rate 10
%  or 1.

%% Load Train Data - 1500 entries
% Preferred method is to load in training data and train the classifiers
% with all HOGs
[train_images_full, train_labels_full] = loadPedestrianDatabase('Data/pedestrian_train.cdataset',1);
pedestrians = find(train_labels_full == 1);
others = find(train_labels_full == -1);


train_images_full= [train_images_full(pedestrians,:); train_images_full(others,:)];
train_labels_full = [train_labels_full(pedestrians,:); train_labels_full(others,:)];

%% Load Train Data - 150 entries
[train_images_small, train_labels_small] = loadPedestrianDatabase('Data/pedestrian_train.cdataset',10);
pedestrians = find(train_labels_small == 1);
others = find(train_labels_small == -1);


train_images_small= [train_images_small(pedestrians,:); train_images_small(others,:)];
train_labels_small = [train_labels_small(pedestrians,:); train_labels_small(others,:)];

%% Extract hog features - for FULL training image set
[hogEdEx, hogHor, hogVer, hogIm] = getHogFeatures(train_images_full, maskA, maskB)

HOG_train_full = struct;
HOG_train_full.hogEdEx = hogEdEx;
HOG_train_full.hogHor = hogHor;
HOG_train_full.hogVer = hogVer;
HOG_train_full.hogIm = hogIm;

%% Extract hog features - for SMALL training image set
[hogEdEx, hogHor, hogVer, hogIm] = getHogFeatures(train_images_small, maskA, maskB)

HOG_train_small = struct;
HOG_train_small.hogEdEx = hogEdEx;
HOG_train_small.hogHor = hogHor;
HOG_train_small.hogVer = hogVer;
HOG_train_small.hogIm = hogIm;

%% Train classifiers and create model

% Setup classifier objects
knn = classifier(@KNNTrain,  @KNNTest);
svm = classifier(@SVMTrain, @SVMTest);
nn = classifier(@NNTrain, @NNTest);

model = struct;

%% Train SVM
model.svm = trainClassifier(svm, train_labels_small,...
    HOG_train_small.hogEdEx,...
    HOG_train_small.hogHor,...
    HOG_train_small.hogVer,...
    HOG_train_small.hogIm);

%% Train KNN
model.knn = trainClassifier(knn, train_labels_small,...
    HOG_train_small.hogEdEx,...
    HOG_train_small.hogHor,...
    HOG_train_small.hogVer,...
    HOG_train_small.hogIm);

%% Train NN
model.nn = trainClassifier(nn, train_labels_small,...
    HOG_train_small.hogEdEx,...
    HOG_train_small.hogHor,...
    HOG_train_small.hogVer,...
    HOG_train_small.hogIm);

%% ========================================================================
%
%                       testing
%
%  ========================================================================

%% Load Test Data - 1500 entries
[test_images_full, test_labels_full] = loadPedestrianDatabase('Data/pedestrian_test.cdataset',1);
pedestrians = find(test_labels_full == 1);
others = find(test_labels_full == -1);


test_images_full= [test_images_full(pedestrians,:); test_images_full(others,:)];
test_labels_full = [test_labels_full(pedestrians,:); test_labels_full(others,:)];
%% Load Test Data - 150 entries
[test_images_small, test_labels_small] = loadPedestrianDatabase('Data/pedestrian_test.cdataset',10);
pedestrians = find(test_labels_small == 1);
others = find(test_labels_small == -1);


test_images_small= [test_images_small(pedestrians,:); test_images_small(others,:)];
test_labels_small = [test_labels_small(pedestrians,:); test_labels_small(others,:)];

%% Extract hog features - for full image set
[hogEdEx, hogHor, hogVer, hogIm] = getHogFeatures(test_images_full, maskA, maskB)

HOG_test_full = struct;
HOG_test_full.hogEdEx = hogEdEx;
HOG_test_full.hogHor = hogHor;
HOG_test_full.hogVer = hogVer;
HOG_test_full.hogIm = hogIm;

%% Extract hog features - for small image set
[hogEdEx, hogHor, hogVer, hogIm] = getHogFeatures(test_images_small, maskA, maskB)

HOG_test_small = struct;
HOG_test_small.hogEdEx = hogEdEx;
HOG_test_small.hogHor = hogHor;
HOG_test_small.hogVer = hogVer;
HOG_test_small.hogIm = hogIm;


%% TEST 150 Images with HOG ED EX
knnPredictions = [];
svmPredictions = [];
nnPredictions = [];
for i=1:150
    knnPredictions = [knnPredictions; model.knn.hogEdEx.test(HOG_test_small.hogEdEx(i, :))];
    svmPredictions = [svmPredictions; model.svm.hogEdEx.test(HOG_test_small.hogEdEx(i, :))];
    nnPredictions = [nnPredictions; model.nn.hogEdEx.test(HOG_test_small.hogEdEx(i, :))];
end

test_results.svm.hogEdEx.predictions = svmPredictions;
test_results.nn.hogEdEx.predictions = nnPredictions;
test_results.knn.hogEdEx.predictions = knnPredictions;

test_results.svm.hogEdEx.acc = size(test_results.svm.hogEdEx.predictions(test_results.svm.hogEdEx.predictions == test_labels_small),1) / size(test_labels_small,1);
test_results.knn.hogEdEx.acc = size(test_results.knn.hogEdEx.predictions(test_results.knn.hogEdEx.predictions == test_labels_small),1) / size(test_labels_small,1);
test_results.nn.hogEdEx.acc = size(test_results.nn.hogEdEx.predictions(test_results.nn.hogEdEx.predictions == test_labels_small),1) / size(test_labels_small,1);

%% CV 150 Images with HOG EdEx
test_results.knn.hogEdEx.cvAcc = 1 - cvError(HOG_train_small.hogEdEx, train_labels_small, 3, knn);
test_results.svm.hogEdEx.cvAcc = 1 - cvError(HOG_train_small.hogEdEx, train_labels_small, 3, svm);
test_results.nn.hogEdEx.cvAcc = 1 - cvError(HOG_train_small.hogEdEx, train_labels_small, 3, nn);
%% Get ROC curves
[x,y] = perfcurve(test_labels_small, test_results.svm.predictions, 1);
test_results.svm.roc.x = x;
test_results.svm.roc.y = y;

[x,y] = perfcurve(test_labels_small, test_results.nn.predictions, 1);
test_results.nn.roc.x = x;
test_results.nn.roc.y = y;

[x,y] = perfcurve(test_labels_small, test_results.knn.predictions, 1);
test_results.knn.roc.x = x;
test_results.knn.roc.y = y;

figure
subplot(4,4,1)
plot(test_results.svm.roc.x, test_results.svm.roc.y);
subplot(4,4,2)
plot(test_results.nn.roc.x, test_results.nn.roc.y);
subplot(4,4,3)
plot(test_results.knn.roc.x, test_results.knn.roc.y);

