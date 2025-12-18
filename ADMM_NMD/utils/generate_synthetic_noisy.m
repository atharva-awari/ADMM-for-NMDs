function [X, Xnoisy] = generate_synthetic_noisy(m,n,r, model,error_measure,param)
%Generating random low-rank factors
W0=rand(m,r);
H0=rand(r,n);
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

%% Adding noise
switch error_measure
    case 'Frobenius'
        N = randn(m,n);
        Xnoisy = X + 0.2*(norm(X,'fro') / norm(N,'fro')).* N ;
        Xnoisy = max(0,Xnoisy);

    case 'KL divergence'
        noise_level=10;
        Xnoisy = poissrnd(X*noise_level)/noise_level;
       
    case 'L1'
        b = 0.1;                      % noise scale
        U = rand(size(X)) - 0.5;      % uniform(-0.5, 0.5)
        noise = -b .* sign(U) .* log(1 - 2*abs(U));
        Xnoisy = X + noise;

end