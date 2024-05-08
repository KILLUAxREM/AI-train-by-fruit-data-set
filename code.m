close all
% Clear the workspace
clear
%%
% Clear the command window
clc
%%
imds = imageDatastore('train/', ...
'IncludeSubfolders',true, ...
'LabelSource','foldernames');
labelCount = countEachLabel(imds);
%%

% Check the dimension of an input image
img = readimage(imds,1); 
size(img)
%%
% Split and display images
[imdsTrain,imdsValidation, imdsTest] = splitEachLabel(imds, 0.7, 0.15, 0.15);

%%
numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages,16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsTrain, idx(i));
    imshow(I)
end
%%
% Load pre-trained network
net = alexnet;

% Analyze network 
analyzeNetwork(net)
%%
% Define inputsize
inputSize = [227 227 3];

inputSize(1)= net.Layers(1).InputSize(1); 

%%

% Replace final layers 
layersTransfer = net.Layers(1:end-3);

% Transfer layers to new classification
numClasses = numel(categories(imdsTrain.Labels));
%%

layers = [ 
    layersTransfer 
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20) 
    softmaxLayer 
    classificationLayer];
%%
pixelRange = [-30 30]; 
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange); 
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);
%%
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
%%
options = trainingOptions('sgdm', ...
    'MiniBatchSize',350, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');
%%
netTransfer = trainNetwork(augimdsTrain,layers,options);
%%
[YPred,scores] = classify(netTransfer,augimdsValidation);
%%
idx = randperm(numel(imdsValidation.Files),4); 
figure 
for i = 1:4 
    subplot(2,2,i) 
    I = readimage(imdsValidation,idx(i)); 
    imshow(I) 
    label = YPred(idx(i)); 
    title(string(label)); 
end

%%
% Evaluate accuracy on the validation set
YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation);

% Test the trained network
augimdsTest = augmentedImageDatastore(inputSize(1:2), imdsTest);
YPredTest = classify(netTransfer, augimdsTest);
YTest = imdsTest.Labels;
testAccuracy = mean(YPredTest == YTest);
disp(['Test Accuracy: ' num2str(testAccuracy)]);
%%
plotconfusion(YTest,YPredTest)