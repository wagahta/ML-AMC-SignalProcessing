% amc_project.m
% Unique ML Project: Automatic Modulation Classification using 2D CNN on IQ Signals
% Author: Valva
% Date: August 19, 2025
% Description: Generates synthetic modulated signals, trains a 2D CNN for classification,
%              with custom SNR augmentation. Optimized for MATLAB Online.
% Requirements: MATLAB Deep Learning Toolbox
% Usage: Run end-to-end. Outputs: Trained net, accuracy plot, confusion matrix.

clear; close all; clc;

%% Step 1: Check Toolbox Availability
if ~license('test', 'neural_network_toolbox')
    error('Deep Learning Toolbox is required. Please ensure it is installed.');
end
% Communications Toolbox check (optional, we'll implement fallbacks)
hasCommToolbox = license('test', 'communication_toolbox');

%% Step 2: Define Parameters
% Modulation types
modTypes = {'BPSK', 'QPSK', '8PSK', '16QAM', '64QAM'}; % Define as cell array of strings
modTypesCat = categorical(modTypes); % Categorical for assignments
numModTypes = length(modTypes);  % 5 classes

% Signal parameters (reduced for MATLAB Online)
numSamplesPerSignal = 128;       % Shorter signal length
numTrainSignalsPerMod = 500;     % Reduced for performance
numTestSignalsPerMod = 100;      % Reduced for performance
snrRange = -5:5:20;              % SNR levels in dB

% Calculate total signals
totalTrainSignals = numTrainSignalsPerMod * numModTypes;
totalTestSignals = numTestSignalsPerMod * numModTypes;

%% Step 3: Generate Synthetic Data (No Communications Toolbox Dependency)
% Preallocate arrays
trainData = zeros(numSamplesPerSignal, 2, 1, totalTrainSignals); % [Length, IQ, 1, NumSignals]
trainLabels = categorical(repmat({''}, totalTrainSignals, 1), modTypes); % Initialize with empty strings
testData = zeros(numSamplesPerSignal, 2, 1, totalTestSignals);
testLabels = categorical(repmat({''}, totalTestSignals, 1), modTypes);

% Generate data
idxTrain = 1;
idxTest = 1;
for i = 1:numModTypes
    mod = modTypes{i}; % Use string for modulation
    for j = 1:numTrainSignalsPerMod
        sig = generateModSignal(mod, numSamplesPerSignal);
        snr = snrRange(randi(length(snrRange)));
        noisySig = custom_awgn(sig, snr);
        trainData(:, :, 1, idxTrain) = [real(noisySig), imag(noisySig)];
        trainLabels(idxTrain) = modTypesCat(i); % Assign exact category
        idxTrain = idxTrain + 1;
    end
    for j = 1:numTestSignalsPerMod
        sig = generateModSignal(mod, numSamplesPerSignal);
        snr = snrRange(randi(length(snrRange)));
        noisySig = custom_awgn(sig, snr);
        testData(:, :, 1, idxTest) = [real(noisySig), imag(noisySig)];
        testLabels(idxTest) = modTypesCat(i); % Assign exact category
        idxTest = idxTest + 1;
    end
end

% Verify labels
uniqueTrainLabels = categories(trainLabels);
uniqueTestLabels = categories(testLabels);
fprintf('Unique training labels: %s\n', strjoin(uniqueTrainLabels, ', '));
fprintf('Unique test labels: %s\n', strjoin(uniqueTestLabels, ', '));
if length(uniqueTrainLabels) ~= numModTypes || length(uniqueTestLabels) ~= numModTypes
    error('Label mismatch: Expected %d classes, got %d (train) and %d (test).', ...
        numModTypes, length(uniqueTrainLabels), length(uniqueTestLabels));
end

% Save dataset for GitHub
save('data/amc_dataset.mat', 'trainData', 'trainLabels', 'testData', 'testLabels');

%% Step 4: Define 2D CNN Architecture (Simplified for MATLAB Online)
% Layers for analysis (without classification layer)
layersForAnalysis = [
    imageInputLayer([numSamplesPerSignal, 2, 1], 'Name', 'input')  % [Length, IQ, 1]
    convolution2dLayer([5, 2], 32, 'Padding', 'same', 'Name', 'conv1')  % 32 filters
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer([2, 1], 'Stride', [2, 1], 'Name', 'pool1')
    
    convolution2dLayer([3, 2], 64, 'Padding', 'same', 'Name', 'conv2')  % 64 filters
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer([2, 1], 'Stride', [2, 1], 'Name', 'pool2')
    
    fullyConnectedLayer(128, 'Name', 'fc1')
    reluLayer('Name', 'relu3')
    dropoutLayer(0.5, 'Name', 'dropout')
    
    fullyConnectedLayer(numModTypes, 'Name', 'fc2')
    softmaxLayer('Name', 'softmax')
];

% Layers for training (with classification layer)
layersForTraining = [
    layersForAnalysis
    classificationLayer('Name', 'output')
];

% Verify network (skip if analysis fails)
try
    analyzeNetwork(layersForAnalysis);
catch e
    warning('Network analysis skipped: %s', e.message);
end

%% Step 5: Train the Network
options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.01, ...
    'MaxEpochs', 10, ...  % Reduced for MATLAB Online
    'Shuffle', 'every-epoch', ...
    'MiniBatchSize', 32, ...  % Smaller batch
    'ValidationData', {testData, testLabels}, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% Train (handle potential memory issues)
try
    net = trainNetwork(trainData, trainLabels, layersForTraining, options);
catch e
    fprintf('Training error: %s\n', e.message);
    fprintf('Trying with smaller batch size...\n');
    options.MiniBatchSize = 16;
    net = trainNetwork(trainData, trainLabels, layersForTraining, options);
end

%% Step 6: Evaluate and Visualize
% Classify test data
predictedLabels = classify(net, testData);

% Accuracy
accuracy = sum(predictedLabels == testLabels) / totalTestSignals;
fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);

% Confusion matrix
figure;
cm = confusionchart(testLabels, predictedLabels);
cm.Title = 'Confusion Matrix for AMC';
saveas(gcf, 'results/confusion_matrix.png');

% Plot example signals
figure;
for i = 1:5
    subplot(5, 1, i);
    plot(testData(:, 1, 1, i), 'b'); hold on;
    plot(testData(:, 2, 1, i), 'r');
    title(sprintf('Example %s Signal', modTypes{i}));
    legend('I', 'Q');
end
saveas(gcf, 'results/example_signals.png');

%% Step 7: Save Results for GitHub
save('results/trained_amc_net.mat', 'net');
fprintf('Results saved. Ready to upload to GitHub in "ML-AMC-SignalProcessing".\n');

%% Local Functions (Moved to End for MATLAB Online Compatibility)
% Custom modulation functions (fallback if no Communications Toolbox)
function sig = custom_bpsk(symbols)
    sig = 2 * (symbols - 0.5);  % Map 0,1 to -1,1
end

function sig = custom_qpsk(symbols)
    angles = pi/4 + (symbols * pi/2);  % 0,1,2,3 -> pi/4, 3pi/4, 5pi/4, 7pi/4
    sig = exp(1j * angles);
end

function sig = custom_8psk(symbols)
    angles = (symbols * pi/4);  % 0 to 7 -> 0, pi/4, ..., 7pi/4
    sig = exp(1j * angles);
end

function sig = custom_16qam(symbols)
    I = mod(symbols, 4) - 1.5;  % -1.5, -0.5, 0.5, 1.5
    Q = floor(symbols/4) - 1.5;
    sig = (I + 1j * Q) / sqrt(10);  % Normalize energy
end

function sig = custom_64qam(symbols)
    I = mod(symbols, 8) - 3.5;  % -3.5 to 3.5
    Q = floor(symbols/8) - 3.5;
    sig = (I + 1j * Q) / sqrt(42);  % Normalize energy
end

function noisy = custom_awgn(sig, snr_db)
    snr_linear = 10^(snr_db/10);
    noise_power = 1 / snr_linear;  % Signal power = 1 (normalized)
    noise = sqrt(noise_power/2) * (randn(size(sig)) + 1j * randn(size(sig)));
    noisy = sig + noise;
end

% Generate modulated signals
function sig = generateModSignal(modType, numSamples)
    symbols = randi([0, getM(modType)-1], numSamples, 1);
    persistent hasCommToolbox;  % Persistent to avoid re-checking
    if isempty(hasCommToolbox)
        hasCommToolbox = license('test', 'communication_toolbox');
    end
    if hasCommToolbox
        switch modType
            case 'BPSK'
                sig = pskmod(symbols, 2);
            case 'QPSK'
                sig = pskmod(symbols, 4);
            case '8PSK'
                sig = pskmod(symbols, 8);
            case '16QAM'
                sig = qammod(symbols, 16);
            case '64QAM'
                sig = qammod(symbols, 64);
        end
    else
        switch modType
            case 'BPSK'
                sig = custom_bpsk(symbols);
            case 'QPSK'
                sig = custom_qpsk(symbols);
            case '8PSK'
                sig = custom_8psk(symbols);
            case '16QAM'
                sig = custom_16qam(symbols);
            case '64QAM'
                sig = custom_64qam(symbols);
        end
    end
end

% Modulation order
function M = getM(modType)
    switch modType
        case 'BPSK', M = 2;
        case 'QPSK', M = 4;
        case '8PSK', M = 8;
        case '16QAM', M = 16;
        case '64QAM', M = 64;
    end

end
