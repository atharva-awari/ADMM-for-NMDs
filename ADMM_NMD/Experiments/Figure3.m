clear all; clc;
load MITatharva.mat

noise_level=1;
X_noisy = poissrnd(X*noise_level)/noise_level;
X_noisy(X_noisy==0)=eps;

%% ADMM parameters MinMax+KL
model = 'MinMax';
error_measure = 'KL divergence';
param.maxiter = 3000;
param.maxtime = 10;
[m,n] = size(X);
r = 4;
param.a=0.5;
param.b=1;
param.display = 0;
results_minmax_kl = admm_nmd(X_noisy,r,model,error_measure,param);
X_recon_minmax_kl = min(param.b,max(param.a,results_minmax_kl.W*results_minmax_kl.H));

%% ADMM parameters MinMax+Fro
model = 'MinMax';
error_measure = 'Frobenius';
param.maxiter = 3000;
param.maxtime = 10;
param.a=0.5;
param.b=1;
results_minmax_fro = admm_nmd(X_noisy,r,model,error_measure,param);
X_recon_minmax_fro = min(param.b,max(param.a,results_minmax_fro.W*results_minmax_fro.H));

%% ADMM parameters MinMax+L1
model = 'MinMax';
error_measure = 'L1';
param.maxiter = 3000;
param.maxtime = 10;
param.a=0.5;
param.b=1;
results_minmax_l1 = admm_nmd(X_noisy,r,model,error_measure,param);
X_recon_minmax_l1 = min(param.b,max(param.a,results_minmax_l1.W*results_minmax_l1.H));


%% ADMM parameters ReLU+KL
model = 'ReLU';
error_measure = 'KL divergence';
param.maxiter = 3000;
param.maxtime = 10;
results_relu_kl = admm_nmd(X_noisy,r,model,error_measure,param);
X_recon_relu_kl = max(0,results_relu_kl.W*results_relu_kl.H);

%% Reconstruction
figure;
subplot(3,2,1);
imshow(1-X);
title('Clean image');

subplot(3,2,2);
imshow(1-X_noisy);
title('Noisy Image');

subplot(3,2,3);
imshow(1-X_recon_minmax_kl);
title('Low-rank recon MinMax+KL');

subplot(3,2,4);
imshow(1-X_recon_minmax_fro);
title('Low-rank recon MinMax+Fro');

subplot(3,2,5);
imshow(1-X_recon_minmax_l1);
title('Low-rank recon MinMax+L1');

subplot(3,2,6);
imshow(1-X_recon_relu_kl);
title('Low-rank recon ReLU+KL');



%% Relative errors
fprintf('Relative error wrt ground truth - MinMax + KL = %f \n', norm(X - X_recon_minmax_kl,'fro') / norm(X,'fro'));

fprintf('Relative error wrt ground truth - MinMax + Fro =  %f\n',  norm(X - X_recon_minmax_fro, 'fro' ) / norm(X,'fro'));

fprintf('Relative error wrt ground truth - MinMax + L1 = %f \n', norm(X - X_recon_minmax_l1,'fro') / norm(X,'fro'));

fprintf('Relative error wrt ground truth - ReLU+ KL  =  %f\n',  norm(X - X_recon_relu_kl, 'fro' ) / norm(X,'fro'));

