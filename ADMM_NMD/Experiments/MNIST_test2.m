close all; clear all; clc;

% Load MNIST digit (e.g., '9')
digit = 9;
[XTrain, TTrain, ~] = digitTrain4DArrayData;
XTrain = squeeze(XTrain);           % remove singleton 3rd dim
labels = double(TTrain);            % convert to numeric
idx = find(labels == digit, 1);     % get index of a digit
img = double(XTrain(:,:,idx));

% Normalize image
X = img / max(img(:));
[m, n] = size(X);

% Noise levels to test
noise_levels = [0.00, 0.05, 0.10, 0.15, 0.20];  % % of noise
num_levels = length(noise_levels);

% Store reconstructions
X_noisy_list = cell(1, num_levels);
X_recon_fro_list = cell(1, num_levels);
X_recon_l1_list  = cell(1, num_levels);

% ADMM parameters
r =5;
param.maxiter = inf;
param.maxtime =5;
param.display=0;
param.updateWH='closed form';
% Loop over noise levels
for i = 1:num_levels
    noise_ratio = noise_levels(i);
    num_outliers = round(noise_ratio * numel(X));
    
    % X_noisy = X;
    % if noise_ratio > 0
    %     indices = randperm(numel(X), num_outliers);
    %     X_noisy(indices) = rand(1, num_outliers);  % random noise
    % end
    X_noisy = imnoise(X, 'salt & pepper', noise_levels(i)); 
    
    % Save noisy version
    X_noisy_list{i} = X_noisy;
    
    % --- ReLU + Frobenius ---
    model = 'ReLU';
    error_measure = 'Frobenius';
    results_fro = admm_nmd(X_noisy,r,model,error_measure,param);
    X_recon_fro = max(0, results_fro.W * results_fro.H);
    X_recon_fro_list{i} = X_recon_fro;

    % --- ReLU + L1 ---
    error_measure = 'L1';
    results_L1 = admm_nmd(X_noisy,r,model,error_measure,param);
    X_recon_l1 = max(0, results_L1.W * results_L1.H);
    X_recon_l1_list{i} = X_recon_l1;
end

% --- Plot results in 3 rows and 5 columns ---
figure('Position', [100, 100, 1400, 700]);

for i = 1:num_levels
    % Row 1: Original or Noisy
    subplot(3, num_levels, i);
    if noise_levels(i) == 0
        imshow(X); title('Original');
    else
        imshow(X_noisy_list{i}); title(sprintf('%.0f%% Noise', noise_levels(i)*100));
    end

    % Row 2: ReLU + Frobenius
    subplot(3, num_levels, i + num_levels);
    imshow(X_recon_fro_list{i});
    title('ReLU + Frobenius');

    % Row 3: ReLU + L1
    subplot(3, num_levels, i + 2*num_levels);
    imshow(X_recon_l1_list{i});
    title('ReLU + L1');
end
