%% Setup
addpath .\SVM-KM\
addpath .\Classifiers\
addpath .\Data\
load('detectorModel.mat') ;

%% The current END for this is at the very end 
%loop through the file of pedestrain - image by image
path = 'Data\pedestrian\';
file = fullfile(path,'image*.jpg');
d = dir(file);
for k=1:numel(d)
    I=fullfile(path,d(k).name);
    %imshow(I)
%end %unsure of bug when printing looping it through this whole section,
     %works if the loop is ended here just showing images before BB
    
   
 %%
    %Open testing image and convert to gary scale
    %I=imread('Data\pedestrian/image_00000308.jpg');
    I=double(I);
    %I = rgb2gray(I);
    
    
    %samplingX=round(size(I,1)/numberRows);
    windowWidth = round(96 * 0.6);
    windowHeight = round(160 * 0.6);
    %samplingY=round(size(I,2)/numberColumns);
    
    
    pedCounter=0;
    
    %Implementation of a simplified slidding window
    % we will be accumulating all the predictions in this variable
    predictedFullImage=[];
    BBs = [];
    
    %% Setup masks for edge extraction
    
    maskA = ones(3);
    maskA(:,1) = maskA(:,1) -2;
    maskA(:,2) = maskA(:,2) -1;
    
    maskB = ones(3);
    maskB(1,:) = maskB(1,:) -2;
    maskB(2,:) = maskB(2,:) -1;
    
    %% Create image pyramid
    p1 = impyramid(I,'reduce');
    p2 = impyramid(p1, 'reduce');
    p3 = impyramid(p2,'reduce');
    
    %% Iteration
    for pyramid = 1:4
        
        pyramidImage = [];
        % terrible code but  matlab doesn't allow array of arrays without 4d
        % matrix - and thats far too confusing
        if(pyramid == 1)
            pyramidImage = I;
        elseif(pyramid == 2)
            pyramidImage = p1;
        elseif(pyramid ==3)
            pyramidImage = p2;
        elseif(pyramid ==4)
            pyramidImage = p3;
        end
        
        
        for r=1:windowHeight:size(pyramidImage,1)
            predictedRow=[];
            
            for c= 1:windowWidth:size(pyramidImage,2)
                
                if (c+windowWidth-1 <= size(pyramidImage,2)) && (r+windowHeight-1 <= size(pyramidImage,1))
                    
                    %we crop the full image to the sliding window size
                    image = pyramidImage(r:r+windowHeight-1, c:c+windowWidth-1);
                    
                    % Resize to 160*96 because the training set images were this
                    % size
                    image = imresize(image,[160 96]);
                    
                    %extract edges
                    [ImEdEx, ImIhor, ImIver] =  edgeExtraction(image,maskA, maskB);
                    
                    % Get hog
                    hogEdEx = hog_feature_vector(ImEdEx);
                    
                    prediction =  detectorModel.knn.test(hogEdEx);
                    
                    
                    if prediction == 1
                        pedCounter = pedCounter+1;
                        BB = [r c windowHeight windowWidth];
                        BBs = [BBs; BB];
                    end
                    
                    %predictedRow=[predictedRow prediction];
                end
            end
            
            
        end
    end
    
    
    imshow(uint8(I)), hold on
    
    for k=1:size(BBs)
        rectangle('Position', [BBs(k,2) BBs(k,1) BBs(k,4), BBs(k,3)])
    end
    
end
%% Evaluation
%Groundtruth
% solutionTruth = [7 2 1 0 4 1 4 9 5 9;...
%                  0 6 9 0 1 5 9 7 3 4;...
%                  9 6 6 5 4 0 7 4 0 1;...
%                  3 1 3 4 7 2 7 1 2 1;...
%                  1 7 4 2 3 5 1 2 4 4];

%comparison = (predictedFullImage==solutionTruth);
%Accuracy = sum(sum(comparison))/ (size(comparison,1)*size(comparison,2))
