close all
clear all
addpath .\SVM-KM\

[t_images, labels] = loadPedestrianDatabase('pedestrian_train.cdataset', 10);
pedestrians = find(labels == 1);
others = find(labels == -1);

t_images= [t_images(pedestrians,:); t_images(others,:)];
features = [];

maskA = ones(3);
maskA(:,1) = maskA(:,1) -2;
maskA(:,2) = maskA(:,2) -1;

maskB = ones(3);
maskB(1,:) = maskB(1,:) -2;
maskB(2,:) = maskB(2,:) -1;

figure
for i=1:size(t_images, 1)
    Im = reshape(t_images(i,:),160,96);
    %ImBrightness =  brightEnchance(Im,50);
    [ImEdEx, ImIhor, ImIver] =  edgeExtraction(Im,maskA, maskB);
    
    hog = hog_feature_vector(ImIhor);
    features = [features; hog]; 
end

%     Im = reshape(t_images(1,:),160,96);
%     
%     
%     hog = hog_feature_vector(ImEdEx);
%     subplot(2,3,1), imshow(uint8(Im))
%     subplot(2,3,2), showHog(hog, [160 96])
%     subplot(2,3,3), imshow(uint8(ImEdEx))
%     subplot(2,3,4), imshow(uint8(ImIver))
%     subplot(2,3,5), imshow(uint8(ImIhor))

%%SVM with HOG
%model = SVMtraining(features, labels);

%%SVM with full images
%model = SVMtraining(t_images, labels);

%%Nearest neighbour classification with HOG
model = NNtraining(t_images, labels);

%%Nearest neighbour classification with full images
%model = NNtraining(t_images, labels);

[tst_images, tst_labels] = loadPedestrianDatabase('pedestrian_test.cdataset', 10);
pedestrians = find(tst_labels == 1);
others = find(tst_labels == -1);

tst_images= [tst_images(pedestrians,:); tst_images(others,:)]; 
tst_labels= [tst_labels(pedestrians); tst_labels(others)];
for i=1:size(tst_images, 1)

   Im = reshape(tst_images(i,:),160,96);
   [ImEdEx, ImIhor, ImIver] =  edgeExtraction(Im,maskA, maskB);
   hog = hog_feature_vector(ImIhor);
%    
%    %SVMTesing with hog
%    %classificationResult(i,1) = SVMTesting(hog, model);
%    
%    %%NNTesting with hog
     %classificationResult(i,1) = NNTesting(hog, model);
%    
%    %KNNTesing with hog
     classificationResult(i,1) = KNNTesting(Im, model, 11);
%    
end

comparison = (tst_labels==classificationResult);
Accuracy = sum(comparison)/length(comparison)