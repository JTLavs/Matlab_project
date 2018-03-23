
   
nnd = imageDatastore('./Data/gsImages','IncludeSubfolders',true,'LabelSource','foldernames');

%%
[trainCNN, valCNN] = splitEachLabel(nnd, 400, 'randomize')
%%
layers = [
    imageInputLayer([160 96])
    
    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,64,'Padding',1)
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
]

options = trainingOptions('sgdm',...
    'MaxEpochs',3, ...
    'ValidationData',valCNN,...
    'ValidationFrequency',30,...
    'Verbose',false,...
    'Plots','training-progress');

net = trainNetwork(trainCNN, layers, options);
