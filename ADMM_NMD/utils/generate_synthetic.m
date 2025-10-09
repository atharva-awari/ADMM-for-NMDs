function [X] = generate_synthetic(m,n,r, model,param)
%Generating random low-rank factors
W0=randn(m,r);
H0=randn(r,n);
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
