%% ========================================================================
%Add paths
addpath .\Classifiers

addpath(genpath('.\Data'))

addpath export_fig
addpath .\SVM-KM
addpath '.\Feature Extraction'
%  ========================================================================
%  This model contains trained classifiers for NN, KNN and SVM trained with
%  HogEdEx - this is optional
%% Setup masks for edge extraction, we tried two masks.

% maskA = [1 , 0; 0 , -1];
% maskB = [0, 1 ; -1, 0];

maskA = ones(3);
maskA(:,1) = maskA(:,1) - 2;
maskA(:,2) = maskA(:,2) - 1;

maskB = ones(3);
maskB(:,1) = maskB(:,1) - 2;
maskB(:,2) = maskB(:,2) - 1;

%% Load pre-trained SVM model
svm = load('final_svm_model');
svm = svm.svm;
%% ========================================================================
%
%                       training
%
%  ========================================================================
%  Here you will find sections to load the training set at sampling rate 10 (150 imgs)
%  or 1 (1500 imgs).

% Load Train Data - 1500 entries
% Preferred method is to load in training data and train the classifiers
% with all HOGs
[train_images_full, train_labels_full] = loadPedestrianDatabase('Data/pedestrian_train.cdataset',1);
pedestrians = find(train_labels_full == 1);
others = find(train_labels_full == -1);
% 

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

model_small = struct;

%% Train SVM
model_small.svm = trainClassifier(svm, train_labels_small,...
    HOG_train_small.hogEdEx,...
    HOG_train_small.hogHor,...
    HOG_train_small.hogVer,...
    HOG_train_small.hogIm);

%% Train KNN
model_small.knn = trainClassifier(knn, train_labels_small,...
    HOG_train_small.hogEdEx,...
    HOG_train_small.hogHor,...
    HOG_train_small.hogVer,...
    HOG_train_small.hogIm);

%% Train NN
model_small.nn = trainClassifier(nn, train_labels_small,...
    HOG_train_small.hogEdEx,...
    HOG_train_small.hogHor,...
    HOG_train_small.hogVer,...
    HOG_train_small.hogIm);

%
%
%% TRAIN 1500
%
%
%
%% Train SVM
%USE SvmTrain to change kernel and C values
model_large.svm = trainClassifier(svm, train_labels_full,...
    HOG_train_full.hogEdEx,...
    HOG_train_full.hogHor,...
    HOG_train_full.hogVer,...
    HOG_train_full.hogIm);

%% Train KNN
model_large.knn = trainClassifier(knn, train_labels_full,...
    HOG_train_full.hogEdEx,...
    HOG_train_full.hogHor,...
    HOG_train_full.hogVer,...
    HOG_train_full.hogIm);

%% Train NN
model_large.nn = trainClassifier(nn, train_labels_full,...
    HOG_train_full.hogEdEx,...
    HOG_train_full.hogHor,...
    HOG_train_full.hogVer,...
    HOG_train_full.hogIm);

%% For main demo we will load in our best feature hog
%bestHogs = load('hogs.mat')


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
%Use KNNTest to change K values
knnPredictions = [];
svmPredictions = [];
nnPredictions = [];
for i=1:150
    knnPredictions = [knnPredictions; model_small.knn.hogEdEx.test(HOG_test_small.hogEdEx(i, :))];
    svmPredictions = [svmPredictions; model_small.svm.hogEdEx.test(HOG_test_small.hogEdEx(i, :))];
    nnPredictions = [nnPredictions; model_small.nn.hogEdEx.test(HOG_test_small.hogEdEx(i, :))];
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


%% TEST 1500 Images with HOG ED EX
knnPredictions = [];
svmPredictions = [];
nnPredictions = [];
for i=1:1500
    knnPredictions = [knnPredictions; model_large.knn.hogEdEx.test(HOG_test_full.hogEdEx(i, :))];
    %svmPredictions = [svmPredictions; model_large.svm.hogEdEx.test(HOG_test_full.hogEdEx(i, :))];
    nnPredictions = [nnPredictions; model_large.nn.hogEdEx.test(HOG_test_full.hogEdEx(i, :))];
end

%test_results.svm.hogEdEx.predictions = svmPredictions;
%test_results.nn.hogEdEx.predictions = nnPredictions;
%test_results.knn.hogEdEx.predictions = knnPredictions;

%test_results.svm.hogEdEx.acc = size(test_results.svm.hogEdEx.predictions(test_results.svm.hogEdEx.predictions == test_labels_full),1) / size(test_labels_full,1);
%test_results.knn.hogEdEx.acc = size(test_results.knn.hogEdEx.predictions(test_results.knn.hogEdEx.predictions == test_labels_full),1) / size(test_labels_full,1);
%test_results.nn.hogEdEx.acc = size(test_results.nn.hogEdEx.predictions(test_results.nn.hogEdEx.predictions == test_labels_full),1) / size(test_labels_full,1);

%% CV 1500 Images with HOG EdEx
test_results.knn.hogEdEx.cvAcc = 1 - cvError(HOG_train_full.hogEdEx, train_labels_full, 3, knn);
test_results.svm.hogEdEx.cvAcc = 1 - cvError(HOG_train_full.hogEdEx, train_labels_full, 3, svm);
test_results.nn.hogEdEx.cvAcc = 1 - cvError(HOG_train_full.hogEdEx, train_labels_full, 3, nn);

%% Confusion MAT
c = confusionmat(train_labels_full, knnPredictions)
tn = c(1,1);
tp = c(2,2);
fp = c(1,2);
fn = c(2,1);

acc = (tp+tn)/size(train_labels_full,1);
% How often does it predict yes? = sensitivity
sensitivity = tp/(fn + tp);
% When it predicts yes, how often is it correct? = precision
precision = tp/(fp + tp);

% How often does the yes condition actually occur in our sample?
prevalence = (fn + tp)/size(train_labels_full,1);
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

%%show curves
figure
subplot(4,4,1)
plot(test_results.svm.roc.x, test_results.svm.roc.y);
subplot(4,4,2)
plot(test_results.nn.roc.x, test_results.nn.roc.y);
subplot(4,4,3)
plot(test_results.knn.roc.x, test_results.knn.roc.y);


%%
knn.train(hogs, labels);
%% Sliding Window
%Loads all the pedestrain images for the video
peds = getPedImgs();
%Pass in model you want to use along with index of image in peds array.
%%
for i =90:size(peds,1)
   SlidingWindow(peds(i, :), svm); 
end


