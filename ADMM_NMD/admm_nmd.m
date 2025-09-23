function [T, W, H, residual, objective, rho] = admm_nmd(X,r,model, error_measure, param)


% Computes an approximate solution of the following non linear matrix
% decomposition problem (NMD)
%
%       min_{W,H, T} d(X , f(T))  s.t. T = WH
%
% using Alternating Direction method of Multipliers. With this flexible
% framework, the user can choose the nonlinear model and the loss function
% as per their requirements. 

%****** Input ******
%   X       : m-by-n matrix.
%           For ReLU: the matrix should be nonnegative and sparse.
%           For CSF : the matrix should be nonnegative. 
%           For MinMax: The entries of the matrix should lie in the
%           interval [a,b]. The interval can be specified by the user. 
%           For Modulus: the matrix should be nonnegative.

%   r       : scalar, desired approximation rank
%   param   : structure, containing the parameters of the algorithm .
%       .Theta0 = initialization of the variable Theta (default: randn).
%       .W0     = initialization of the variable W (default: randn).
%       .H0     = initialization of the variable H (default: randn).
%       .lambda = initialization of the Lagrange Multiplier (default: randn).
%       .maxiter= maximum number of iterations (default: 1000).
%       .tol    = tolerance on the relative error (default: 1.e-9).
%       .tolerr = tolerance on 10 successive errors (err(i+1)-err(i-10))(default: 1.e-10)
%       .maxtime= time limit (default: 10 seconds).
%       For using MinMax model, specify the interval [a,b] (default: [0,1].
%       .a      = initial point of the interval.
%       .b      = end point of the interval.
% ****** Output ******
%   T       : m-by-n matrix, approximate solution of
%            min_{W,H, T} d(X , f(T))  s.t. T = WH
%   W       : low-rank factor W of dimension m x r. 
%   H       : low-rank factor H of dimension r x n.
%           such that X ~ f(WH)

% objective : vector containing evolution of relative error along
%             iterations d(X, f(WH)) / norm(X).
% residual : vector containing evolution of residual error along
%             iterations || T - WH||_F / ||T||_F.

%% How to add other models with a new nonlinearity and loss function
% In this framework, we have added 12 different combinations of nonlinear
% functions and loss-functions. However, to add new nonlinearity or loss
% function, solve the one-dimensional subproblems for each entry of T and
% add it to update_T.m file. 
%%
[m,n]=size(X); 
 if nargin < 5
    param = [];
 end
 if ~isfield(param,'W0') || ~isfield(param,'H0')
     param.W0=randn(m,r); param.H0=randn(r,n);  
 end
 
if ~isfield(param,'maxiter')
    param.maxiter = 1000; 
end
if ~isfield(param,'tol')
    param.tol = 1.e-9; 
end
if ~isfield(param,'tolerr')
    param.tolerr = 1e-10;
end
if ~isfield(param,'maxtime')
    param.maxtime = 10; 
end
if ~isfield(param,'Theta')
    param.T = randn(m,n);
end
if ~isfield(param,'lambda')
    param.lambda = randn(m,n);
end
if ~isfield(param,'a')
    param.a = eps;
end
if ~isfield(param,'b')
    param.b = 1;
end


cpu0=tic;
compute_obj=0;


%Define the initial variables
W=param.W0; H=param.H0; T = param.T;  lambda = param.lambda; maxiter= param.maxiter; maxtime = param.maxtime;

% In the case of KL-divergence, the initial T must be positive wherever X
% is positive. 
if strcmpi(error_measure, 'KL divergence')
 f = find(X>0);
 T(f) = max(eps, T(f));
 X(X==0)=eps;
else 
    T=param.T;
end


k=1;
start_obj=tic;
compute_obj=compute_obj+toc(start_obj);
runtime=toc(cpu0)-compute_obj;
savetime(k)=runtime;

%% Scaling W and H 

    for i = 1:r
        norm_w = norm(W(:, i), 2);
        norm_h = norm(H(i, :), 2);

        % Avoid division by zero
        if norm_w == 0 || norm_h == 0
            continue;
        end

        W(:, i) = (W(:, i) / norm_w) * norm_h;
        H(i, :) = (H(i, :) / norm_h) * norm_w;
    end 

rho(k) =1 ;

[residual(k), objective(k)] = compute_residual_objective(X, W, H, T, model, error_measure, param);


while k < maxiter && runtime<maxtime
     
    
    Ht=H';
    HHt=H*Ht;
    LW=norm(HHt);
    W=W+1/(rho(k)*LW)*(lambda*Ht-rho(k)*(W*HHt-T*Ht));


    Wt=W';
    WtW=Wt*W;
    LH=norm(WtW); 
    H=H+1/(rho(k)*LH)*(Wt*lambda-rho(k)*(WtW*H-Wt*T));

    A = W*H;
    T_prev = T; 
    [T] = update_T(X,rho(k),T,lambda,A, model, error_measure, param);

    lambda = lambda + rho(k)*(T - A);

    
    
%% Adaptive update for the penalty paramater rho
 mu = 10;
t_incr = 2;
t_decr = 2;

p_residual = norm(T - W*H, 'fro');
S = rho(k)*W'*(T - T_prev);
d_residual = norm(S, 'fro');


if p_residual > mu*d_residual
    rho(k+1) = t_incr*rho(k);
    
elseif d_residual > mu*p_residual
    rho(k+1) = rho(k)/t_decr;

else
    rho(k+1) =rho(k);
end
%%    
    
    k=k+1;

    start_obj=tic;
    compute_obj=compute_obj+toc(start_obj);
    runtime=toc(cpu0)-compute_obj;
    savetime(k)=runtime;

    [residual(k), objective(k)] = compute_residual_objective(X, W, H, T, model, error_measure, param);


end

%Plotting objective and residual

subplot(1,2,1)
semilogy(objective,'b','LineWidth',1.5);hold on; 
hold on

xlabel('Iterations');
ylabel('Objective');

subplot(1,2,2)
semilogy(residual,'r','LineWidth',1.5);hold on; 

ylabel('Residual');
xlabel('Iterations'); 