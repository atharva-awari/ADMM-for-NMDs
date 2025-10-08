%% Run ADMM-NMD on synthetic data
%% You can choose the model:
% Nonlinear function = ReLU, CSF, MinMax, or Modulus
% Error measure = Frobenius,L1 (Componentwise L1 norm), or KL divergence
close all; clear all; clc;
model = 'ReLU';
error_measure = 'Frobenius';
%% Define parameters of the algorithm
param.maxiter = inf;
param.maxtime = 10;
% If the user wants to use MinMax model, define the desired interval [a,b]
%param.a=-1;
%param.b = 1;
%% Generate synthetic input matrix
% Size of the input matrix and rank
m =200;
n=150;
r=30;
% Random factors to generate data matrix X
W0 = randn(m,r);
H0 = randn(r,n);
% Generating synthetic data matrix X
switch model
    case 'ReLU'
        X = max(0,W0*H0);
    case 'CSF'
        X = (W0*H0).^2;
    case 'MinMax'
        X = min(param.b, max(param.a, W0*H0));
    case 'Modulus'
        X = abs(W0*H0);
end

%% Run ADMM algorithm
results = admm_nmd(X,r, model, error_measure,param);
