close all; clear all; clc;

%Model = ReLU / CSF / MinMax / Modulus
%error_measure = Frobenius, L1, KL divergence
model = 'MinMax';
error_measure = 'KL divergence';

%rng(2025)

m =242;
n=22;
r=10;

%Select interval if you're using MinMax model [a,b]
options.a = eps;
options.b = 1;


W0 = randn(m,r);
H0 = randn(r,n);
switch model
    case 'ReLU'
        X = max(0,W0*H0);
        %load CBCL.mat;
    case 'CSF'
        X = (W0*H0).^2;

    case 'MinMax'
        %X = min(options.b, max(options.a, W0*H0));
        load heart.mat;

    case 'Modulus'
        X = abs(W0*H0);
end


% Initialising Theta and W,H using SVD(X)
 % [U,S,V] = svds(X,r);
 % Theta = U*S*V';
 % W = -U;
 % H = -S*V';
 %Theta = zeros(m,n);

W = randn(m,r);
H = randn(r,n);


Theta = randn(m,n);
f = find(X>0);
Theta(f) = max(eps,Theta(f));
X(X==0)=eps;


%Parameters input
rho = 0.5;
maxiter = inf;
maxtime = 15;

%Lagrange Multiplier
lambda = randn(m,n);


[Theta, W, H, residual, objective] = admm_nmd(X, rho, maxiter, maxtime, W, H, Theta, lambda, model, error_measure, options);
