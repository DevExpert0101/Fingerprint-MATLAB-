clear; clc; close all;

layers = [
        imageInputLayer([64 64 1])
        convolution2dLayer(3,32,'Padding','same')
        layerNormalizationLayer
        reluLayer
        convolution2dLayer(3,32,'Padding','same')
        layerNormalizationLayer
        reluLayer
        convolution2dLayer(3,64,'Padding','same')
        layerNormalizationLayer
        reluLayer
        maxPooling2dLayer(3, 'Stride', 2)
        convolution2dLayer(3,64,'Padding','same')
        layerNormalizationLayer
        reluLayer
        convolution2dLayer(3,64,'Padding','same')
        layerNormalizationLayer
        reluLayer
        maxPooling2dLayer(2, 'Stride', 2)
        convolution2dLayer(3,384,'Padding','same')
        layerNormalizationLayer
        reluLayer
        convolution2dLayer(3,384,'Padding','same')
        layerNormalizationLayer
        reluLayer
        convolution2dLayer(3,256,'Padding','same')
        layerNormalizationLayer
        reluLayer
        maxPooling2dLayer(2, 'Stride', 2)
        fullyConnectedLayer(2048)
        fullyConnectedLayer(2048)
        fullyConnectedLayer(2)
        softmaxLayer
        classificationLayer]
 
 single_images = dir('Data\train\Single');
 for i=3:length(single_images)
    image = imread(['Data\train\Single\' single_images(i).name]);
    image = imresize(image,[64 64]);
    imwrite(image,['Data\my_train\Single\' 's' num2str(i-2) '_' num2str(0) '.bmp']);
    for j=1:72
        rimage = imrotate(image, 5*j,'nearest','crop');
        imwrite(rimage,['Data\my_train\Single\' 's' num2str(i-2) '_' num2str(j) '.bmp']);
    end
 end
 
 overlapped_images = dir('Data\train\Overlapped');
 for i=3:length(overlapped_images)
    image = imread(['Data\train\Overlapped\' overlapped_images(i).name]);
    image = imresize(image,[64 64]);
    imwrite(image,['Data\my_train\Overlapped\' 's' num2str(i-2) '_' num2str(0) '.bmp']);
    for j=1:72
        rimage = imrotate(image, 5*j,'bilinear','crop');
        imwrite(rimage,['Data\my_train\Overlapped\' 's' num2str(i-2) '_' num2str(j) '.bmp']);
    end
 end
 
    
 imds = imageDatastore(fullfile('Data','my_train',{'Single','Overlapped'}),'LabelSource','foldernames');
 inputSize = [64 64];
 imds.ReadFcn = @(loc)imresize(imread(loc),inputSize);
 
 [imds_train, imds_test] = splitEachLabel(imds, 0.6)
 fileNums = numel(imds.Files);
 
my_imds = imageDatastore(fullfile('Data','train',{'Single'}));
my_imds.ReadFcn = @(loc)imresize(imread(loc),inputSize);
% trainning
initialLearnRate = 0.00001;
learnRateSchedule = 'piecewise';
learnRateDropFactor = 0.1;
learnRateDropPeriod = 30;
momentum = 0.9;
l2Regularization = 0.001; % Weight decay
batchSize = 64;
epochs = 10;
shuffle = 'every-epoch';
checkpointPath = '.\CheckPoints';
patchesPerImage = 512;

options = trainingOptions('sgdm', ...
    'InitialLearnRate', initialLearnRate, ...
    'Momentum', momentum, ...
    'LearnRateSchedule', learnRateSchedule, ...
    'LearnRateDropFactor', learnRateDropFactor, ...
    'LearnRateDropPeriod', learnRateDropPeriod, ...
    'MiniBatchSize', batchSize, ...
    'MaxEpochs', epochs, ...
    'L2Regularization', l2Regularization, ...
    'Shuffle', shuffle, ...
    'Plots', 'training-progress', ...
    'Verbose', 1, ...
    'VerboseFrequency', floor(fileNums*patchesPerImage/batchSize), ...   
    'ValidationFrequency', floor(fileNums*patchesPerImage/batchSize/4), ...
    'ValidationPatience', Inf, ...
    'ExecutionEnvironment', 'gpu');

trainNet = trainNetwork(imds, layers, options);



output = classify(trainNet, my_imds)
