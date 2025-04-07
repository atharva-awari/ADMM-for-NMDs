clear all; clc;

%Model = ReLU / CSF / MinMax / Modulus
%error_measure = Frobenius, L1, KL divergence
model = 'MinMax';
error_measure = 'Frobenius';

m =20;
n=10;
r=5;

%Select interval if you're using MinMax model [a,b]
options.a = -1;
options.b = 1;


W0 = randn(m,r);
H0 = randn(r,n);
switch model
    case 'ReLU'
        X = max(0,W0*H0);

    case 'CSF'
        X = (W0*H0).^2;

    case 'MinMax'
        X = min(options.b, max(options.a, W0*H0));

    case 'Modulus'
        X = abs(W0*H0);
end


%Initialising Theta and W,H using SVD(X)
[U,S,V] = svds(X,r);
Theta = U*S*V';

W = U;
H = S*V';


%Parameters input
rho = 5;
maxiter = 10000;
maxtime = 5;

%Lagrange Multiplier
lambda = zeros(m,n);


[Theta, W, H, residual, objective] = admm_nmd(X, rho, maxiter, maxtime, W, H, Theta, lambda, model, error_measure, options);
