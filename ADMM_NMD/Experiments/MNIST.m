clear all;close all; clc;
Y=load('mnist_all.mat');
%w1=1:10:500; %Number of images for each digit
w1=1:50;
X=[Y.train0(w1,:);Y.train1(w1,:);Y.train2(w1,:);Y.train3(w1,:);Y.train4(w1,:);...
    Y.train5(w1,:);Y.train6(w1,:);Y.train7(w1,:);Y.train8(w1,:);Y.train9(w1,:)];
X=double(X);
X = X / max(X(:));
[m,n]=size(X);
%%
% ADMM parameters
r =32;
param.maxiter = inf;
param.maxtime =30;
param.display=0;
param.updateWH='closed form';
% Noise levels to test
noise_levels = [0.00, 0.05, 0.10, 0.15, 0.20];  % % of noise
num_levels = length(noise_levels);

% Loop over noise levels
for i = 1:num_levels
    noise_ratio = noise_levels(i);
    num_outliers = round(noise_ratio * numel(X));

    % X_noisy = X;
    % if noise_ratio > 0
    %     indices = randperm(numel(X), num_outliers);
    %    X_noisy(indices) = rand(1, num_outliers);  % random noise
    % 
    % end
    X_noisy = imnoise(X, 'salt & pepper', noise_levels(i));
    % Save noisy version
    X_noisy_list{i} = X_noisy;

    % --- ReLU + Frobenius ---
    model = 'ReLU';
    error_measure = 'Frobenius';
    results_fro(i) = admm_nmd(X_noisy,r,model,error_measure,param);
    X_recon_fro{i} = min(1,max(0, results_fro(i).W * results_fro(i).H));

    % --- ReLU + L1 ---
    error_measure = 'L1';
    results_L1(i) = admm_nmd(X_noisy,r,model,error_measure,param);
    X_recon_l1{i} = min(1,max(0, results_L1(i).W * results_L1(i).H));

     % --- MinMax + L1 ---
     model='MinMax';
     error_measure = 'L1';
     % Initialize with solution of ReLU + L1
     param.a=0;
     param.b=1;
     % param.W0=results_L1(i).W;
     % param.H0=results_L1(i).H;
     % param.lambda=results_L1(i).lambda;
     % param.T = results_L1(i).T;

     
    results_minmax_L1(i) = admm_nmd(X_noisy,r,model,error_measure,param);
    X_recon_minmax_l1{i} = min(param.b,max(param.a, results_minmax_L1(i).W * results_minmax_L1(i).H));

   
end

%% ===== Visualization: One Digit Across Noise Levels and Reconstructions =====
digit_to_show = 8;        % 0..9
k_within_digit = 5;       % 1..numel(w1)

num_per_digit = numel(w1);
row_idx = digit_to_show * num_per_digit + k_within_digit;   % 1-based indexing

figure('Color','w','Name','MNIST: Noisy vs Reconstructions across noise levels');
set(gcf, 'Position', [100 100 1900 800]);  % wide layout for clarity

% Text styling
titleArgs = {'Interpreter','latex','FontSize',12,'FontWeight','bold'};

% Use a tiled layout with more left padding
t = tiledlayout(4, num_levels, 'Padding','loose', 'TileSpacing','compact');

for i = 1:num_levels
    % ----- Row 1: Noisy input -----
    nexttile(i);
    im_noisy = reshape(X_noisy_list{i}(row_idx, :), 28, 28)';
    imagesc(im_noisy); axis image off; colormap(gray);
    title(sprintf('Noise %d\\%%', round(100*noise_levels(i))), titleArgs{:});

    % ----- Row 2: ReLU + Frobenius -----
    nexttile(num_levels + i);
    im_fro = reshape(X_recon_fro{i}(row_idx, :), 28, 28)';
    imagesc(im_fro); axis image off; colormap(gray);

    % ----- Row 3: ReLU + L1 -----
    nexttile(2*num_levels + i);
    im_l1 = reshape(X_recon_l1{i}(row_idx, :), 28, 28)';
    imagesc(im_l1); axis image off; colormap(gray);

    % ----- Row 4: MinMax + L1 -----
    nexttile(3*num_levels + i);
    im_minmax_l1 = reshape(X_recon_minmax_l1{i}(row_idx, :), 28, 28)';
    imagesc(im_minmax_l1); axis image off; colormap(gray);
end

% --- Add row labels manually (always visible) ---
annotation('textbox', [0.02 0.78 0.1 0.1], 'String', '\textbf{Noisy input}', ...
    'Interpreter','latex','FontSize',14,'FontWeight','bold','EdgeColor','none',...
    'HorizontalAlignment','center','VerticalAlignment','middle');

annotation('textbox', [0.02 0.57 0.1 0.1], 'String', '\textbf{ReLU + Fro}', ...
    'Interpreter','latex','FontSize',14,'FontWeight','bold','EdgeColor','none',...
    'HorizontalAlignment','center','VerticalAlignment','middle');

annotation('textbox', [0.02 0.36 0.1 0.1], 'String', '\textbf{ReLU + L1}', ...
    'Interpreter','latex','FontSize',14,'FontWeight','bold','EdgeColor','none',...
    'HorizontalAlignment','center','VerticalAlignment','middle');

annotation('textbox', [0.02 0.15 0.1 0.1], 'String', '\textbf{MinMax + L1}', ...
    'Interpreter','latex','FontSize',14,'FontWeight','bold','EdgeColor','none',...
    'HorizontalAlignment','center','VerticalAlignment','middle');

% --- LaTeX-formatted super title ---
sgtitle(sprintf('\\textbf{Digit } %d \\ \\textbf{Across Noise Levels}', ...
    digit_to_show), ...
    'Interpreter','latex','FontSize',18,'FontWeight','bold');

% --- Export high-quality figure for Overleaf ---
exportgraphics(gcf, 'MNIST_gradient_descent_figure.pdf', ...
    'ContentType','image', 'BackgroundColor','white', 'Resolution',300);



%% ===== Error tables (Frobenius norms vs. clean ground truth) =====
X_clean     = X;
normX_clean = norm(X_clean,'fro');% Ground truth (clean, noise-free)
x_clean_row = X_clean(row_idx, :);      % Selected sample’s clean row
normX_clean_row = norm(x_clean_row,'fro');

% Preallocate per-sample errors (row_idx) and full-matrix errors
err_noisy_row    = zeros(1, num_levels);
err_fro_row      = zeros(1, num_levels);
err_l1_row       = zeros(1, num_levels);
err_minmax_row   = zeros(1, num_levels);

err_noisy_full   = zeros(1, num_levels);
err_fro_full     = zeros(1, num_levels);
err_l1_full      = zeros(1, num_levels);
err_minmax_full  = zeros(1, num_levels);

for i = 1:num_levels
    % --- Per-sample (selected digit instance) errors ---
    err_noisy_row(i)   = norm(X_noisy_list{i}(row_idx, :)     - x_clean_row, 'fro')/normX_clean_row;
    err_fro_row(i)     = norm(X_recon_fro{i}(row_idx, :)      - x_clean_row, 'fro')/ normX_clean_row;
    err_l1_row(i)      = norm(X_recon_l1{i}(row_idx, :)       - x_clean_row, 'fro')/normX_clean_row;
    err_minmax_row(i)  = norm(X_recon_minmax_l1{i}(row_idx, :)- x_clean_row, 'fro')/normX_clean_row;

    % --- Full-matrix errors (aggregate over all samples) ---
    err_noisy_full(i)  = norm(X_noisy_list{i}     - X_clean, 'fro')/normX_clean;
    err_fro_full(i)    = norm(X_recon_fro{i}      - X_clean, 'fro')/normX_clean;
    err_l1_full(i)     = norm(X_recon_l1{i}       - X_clean, 'fro')/normX_clean;
    err_minmax_full(i) = norm(X_recon_minmax_l1{i}- X_clean, 'fro')/normX_clean;
end

% ---------- Per-sample table ----------
T_row = table( ...
    round(100*noise_levels(:)), ...
    err_noisy_row(:), ...
    err_fro_row(:), ...
    err_l1_row(:), ...
    err_minmax_row(:), ...
    'VariableNames', {'NoisePct','Noisy_vs_GT','ReLU_Fro_vs_GT','ReLU_L1_vs_GT','MinMax_L1_vs_GT'});

disp('Frobenius errors for the SELECTED sample (row_idx):');
disp(T_row);

% ---------- Full-matrix table ----------
T_full = table( ...
    round(100*noise_levels(:)), ...
    err_noisy_full(:), ...
    err_fro_full(:), ...
    err_l1_full(:), ...
    err_minmax_full(:), ...
    'VariableNames', {'NoisePct','Noisy_vs_GT','ReLU_Fro_vs_GT','ReLU_L1_vs_GT','MinMax_L1_vs_GT'});

disp('Frobenius errors for the ENTIRE dataset (full matrices):');
disp(T_full);
