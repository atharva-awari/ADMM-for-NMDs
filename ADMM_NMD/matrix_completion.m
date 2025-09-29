clear all; clc;
clear;

m=361;n=2429;
r_true = 30;       % true rank used to generate data
r = 2;            % factorization rank you want to fit
param.p_obs = 0.8;      % fraction of observed entries
sigma_noise = 0.0; % noise std

%1) Generate synthetic low-rank nonnegative matrix X = U * V'
% U = rand(m, r_true);    % nonnegative
% V = rand(n, r_true);    % nonnegative
% X = U * V';             % m x n (nonnegative)
load CBCL.mat;
% 1) Normalize by maximum entry
X_norm = X ./ max(X(:));
% 2) Binarize by rounding
X_bin = round(X_norm);
X = X_bin;

% Parameters
model = 'ReLU';
error_measure = 'Frobenius';
param.maxiter = 1000000;
param.maxtime = 20;
param.task = 'Matrix completion';
M= rand(m,n) < param.p_obs;  % binary mask (1 observed, 0 missing)
param.mask=M;
param.a=0; param.b=1;
% 2) Observed matrix with noise (only meaningful where M==1)
B = zeros(m,n);
B(M) = X(M) + sigma_noise * randn(sum(M(:)),1); % noisy observed entries
B(B < 0) = 0;  % keep nonnegativity
param.B = B;


results = admm_nmd(X,r, model, error_measure,param);





