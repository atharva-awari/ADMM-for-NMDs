close all; clear all; clc;
%Model = ReLU / CSF / MinMax / Modulus
%error_measure = Frobenius, L1, KL divergence
model = 'ReLU';
error_measure = 'Frobenius';

%% Define parameters if needed
param.maxiter = inf;
param.maxtime = 5;

%%
m =100;
n=70;
r=15;

W0 = randn(m,r);
H0 = randn(r,n);

switch model
    case 'ReLU'
        X = max(0,W0*H0);
        %load CBCL.mat;
    case 'CSF'
        X = (W0*H0).^2;

    case 'MinMax'
        X = min(1, max(eps, W0*H0));
        
    case 'Modulus'
        X = abs(W0*H0);
 end


[T, W, H, residual, objective, rho] = admm_nmd(X,r, model, error_measure,param);
