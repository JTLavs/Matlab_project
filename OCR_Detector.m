clear all
close all
addpath .\SVM-KM\
detectorModel = load('detectorModel.mat') ;

%Open testing image and convert to gary scale
I=imread('pedestrian/image_00000099.jpg');


%For simplifying this task, we will assume we know that there are 5 rows of
%sigits, with 10 numbers each
%numberRows=5;
%numberColumns=10;

%Using previous infromation, we can calculate how tall and wide is each
%digit
%samplingX=round(size(I,1)/numberRows);
samplingX = 96;
samplingY = 160;
%samplingY=round(size(I,2)/numberColumns);

figure
pedCounter=0;

%Implementation of a simplified slidding window
% we will be accumulating all the predictions in this variable
predictedFullImage=[];
BBs = [];
%for each digit within the image, 
for r=1:samplingX:size(I,1)
    predictedRow=[];
    
    for c= 1:samplingY:size(I,2)
        
        if (c+samplingY-1 <= size(I,2)) && (r+samplingX-1 <= size(I,1))
  
        %we crop the digit
        digitIm = I(r:r+samplingX-1, c:c+samplingY-1);
        
        % we convert it into doubles from 0 to 1 and invert them (rememebr that in the training set, the digitd were white on black background)
        %digitIm = mat2gray(digitIm);
        %digitIm = imcomplement(digitIm);
        
        %need to  invert image here
        
        %All training examples were 28x28. To have any chance, we need to
        %resample them into a 28x28 imaGE
        digitIm = imresize(digitIm,[160 96]);
        
        %We display teh individually segmented digits
        %subplot(numberRows,numberColumns,digitCounter)

        %digitIm = preprocessDigit(digitIm);
        %we reshape the digit into a vector
        
        %digitVector = reshape(digitIm, 1, 15360);
        
        %error that x and xsup should have the same no of columns.
        hog = hog_feature_vector(digitIm);
        prediction =  KNNTesting(hog, detectorModel.model, 15);  
        prediction
        
        if prediction == 1
            pedCounter = pedCounter+1;
            BB = [r c samplingX samplingY];
            BBs = [BBs; BB];
        end
        
        %predictedRow=[predictedRow prediction];
        end
    end
    
    %predictedFullImage=[predictedFullImage; predictedRow];
    imshow(I), hold on
    
    for k=1:size(BBs)
        rectangle('Position', [BBs(k,1) BBs(k,2) BBs(k,3), BBs(k,4)])
    end
end

%predictedFullImage

%% Evaluation
%Groundtruth
% solutionTruth = [7 2 1 0 4 1 4 9 5 9;...
%                  0 6 9 0 1 5 9 7 3 4;...
%                  9 6 6 5 4 0 7 4 0 1;...
%                  3 1 3 4 7 2 7 1 2 1;...
%                  1 7 4 2 3 5 1 2 4 4];
             
%comparison = (predictedFullImage==solutionTruth);
%Accuracy = sum(sum(comparison))/ (size(comparison,1)*size(comparison,2))
