function results = admm_nmd(X,r,model, error_measure, param)
% Computes an approximate solution of the following non linear matrix
% decomposition problem (NMD)
%
%       min_{W,H, T} d(X , f(T))  s.t. T = WH
%
% using Alternating Direction method of Multipliers. With this flexible
% framework, the user can choose the nonlinear model and the loss function
% as per their requirements.

%****** Input ******
%   X          : m-by-n matrix.
%   r          : scalar, desired approximation rank
%   model      : Select nonlinearity among: ReLU, CSF, MinMax, Modulus
% error_measure: Select the loss function among: Frobenius, L1, KL divergence

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
%       .mask   = mask for missing entries, if any.
%                  param.mask has the same size as X, and
%                  param.mask(i,j)=0 if (i,j) is missing,
%                  param.mask(i,j)=1 if (i,j) is observed.
%       .validmask  = must be a subset of .mask,
%                       and X(param.validmask) are actually known.
%                       This allows to validate the factorization. The user
%                       can put entries in the mask that are actually known
%                       to validate that the model can recover them. This
%                       is computed via the RMSE.
%                       By default, param.validmask=param.mask
%       .p_obs  = percentage of observed entries in matrix completion.
%       .B      = corrupted matrix with noise and missing entries.
%       .display= displays plots of relative error and residual
%                  (default = 1 means plots are drawn, 0 = no plots)

% ****** Output ******
%   T       : m-by-n matrix, approximate solution of
%            min_{W,H, T} d(X , f(T))  s.t. T = WH
%   W       : low-rank factor W of dimension m x r.
%   H       : low-rank factor H of dimension r x n.
%           such that X ~ f(WH)
% If no mask is provided, the task is not matrix completion, we compute and display:
% objective : vector containing evolution of relative error along
%             iterations d(X, f(WH)) / norm(X).
% residual : vector containing evolution of residual error along
%             iterations || T - WH||_F / ||T||_F.
% If mask is provided, the task is matrix completion, we compute and display:
%rmse_test : RMSE on test data
%rmse_train: RMSE on train data

%% How to add other models with a new nonlinearity and loss function
% In this framework, we have added 12 different combinations of nonlinear
% functions and loss-functions. However, to add new nonlinearity or loss
% function, solve the one-dimensional subproblems for each entry of T and
% add it to update_T.m file.
%%
addpath('./utils');
%% Checking that the input matrix follows the desired model
if strcmpi(model,'ReLU')
    if min(X(:)) < 0
        warning('The input matrix contains negative entries, not ideal for ReLU-NMD.');
    end
end
if strcmpi(model,'CSF')
    if min(X(:)) < 0
        warning('The input matrix contains negative entries, not ideal for CSF.');
    end
end
if strcmpi(model,'MinMax')
    if min(X(:)) < param.a && max(X(:)) > param.b
        warning('The input matrix contains entries which do not lie in the interval [a,b], not ideal for MinMax model.');
    end
end
if strcmpi(model,'Modulus')
    if min(X(:)) < 0
        warning('The input matrix contains negative entries, not ideal for Modulus.');
    end
end
%% Defining default setting for parameters
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
if ~isfield(param,'mask')
    param.mask=[];
end
if ~isfield(param,'validmask')
    param.validmask=param.mask;
end
if ~isfield(param,'display')
    param.display = 1;
end
cpu0=tic;
timedisplay = 0.1; cntdisp = 0;
compute_obj=0;
%Define the initial variables
W=param.W0;
H=param.H0;
T = param.T;
lambda = param.lambda;
maxiter= param.maxiter;
maxtime = param.maxtime;
% In the case of KL-divergence, the initial T must be positive wherever X is positive.
if strcmpi(error_measure, 'KL divergence')
    f = find(X>0);
    T(f) = max(eps, T(f));
    X(X==0)=eps;
else
    T=param.T;
end
if isfield(param, 'mask') && ~isempty(param.mask)
    disp('This is a matrix completion task');
    disp('RMSE computation for missing entries will be performed. Display of iterations and RMSE on test data:');
    M = param.mask;
    if isfield(param,'B')
        B = param.B;
    else
        warning('param.B, the matrix with corrupted entries is not provided. Using X as the data matrix.');
        B = X;        % fallback to X
    end
else
    disp('Full data factorization mode detected.');
    disp('Display of iteration number and relative error in %:');
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
% Setting initial value for penalty paramter rho
rho(k) =1 ;
A = W*H;
% Computing initial RMSE/Relative error
if isfield(param, 'mask') && ~isempty(param.mask)
    [rmse_test(k),rmse_train(k)]= rmse_comp(X, A, model, param);
else
    [residual(k), objective(k)] = compute_residual_objective(X, A, T, model, error_measure, param);
end
%% Running the main loop of ADMM for NMD
while k < maxiter && runtime<maxtime
    %Updating W (model independent update)
    Ht=H';
    HHt=H*Ht;
    LW=norm(HHt);
    W=W+1/(rho(k)*LW)*(lambda*Ht-rho(k)*(W*HHt-T*Ht));
    %Updating H (model independent update)
    Wt=W';
    WtW=Wt*W;
    LH=norm(WtW);
    H=H+1/(rho(k)*LH)*(Wt*lambda-rho(k)*(WtW*H-Wt*T));
    %Define A=WH for simplicity
    A = W*H;
    T_prev = T;
    % Updating T, depending on the task of full factorization or matrix completion
    if isfield(param, 'mask') && ~isempty(param.mask)
        [T] = update_T(B,rho(k),T,lambda,A, model, error_measure, param);
        T(M==0) = rho(k)* A(M==0) - lambda(M==0) ./ rho(k) ;
    else
        [T] = update_T(X,rho(k),T,lambda,A, model, error_measure, param);
    end
    % Updating the Lagrange variable lambda (model independent)
    lambda = lambda + rho(k)*(T - A);
    %% Adaptive update for the penalty paramater rho
    mu = 10;
    t_incr = 2;
    t_decr = 2;
    p_residual = norm(T - A, 'fro');
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
    %% Calculation of RMSE (for matrix completion) or the Relative errors (for full factorization)
    if isfield(param, 'mask') && ~isempty(param.mask)
        [rmse_test(k),rmse_train(k)]= rmse_comp(X, A, model, param);
    else
        [residual(k), objective(k)] = compute_residual_objective(X, A , T, model, error_measure, param);
    end
    %%
    if param.display == 1
        if toc(cpu0) >= timedisplay
            if isfield(param, 'mask') && ~isempty(param.mask)
                fprintf('%2.0f:%2.3f - ',k,rmse_test(k));
            else
                fprintf('%2.0f:%2.3f - ',k,100*objective(k));
            end
            timedisplay = min(2*timedisplay,timedisplay+5);
            cntdisp = cntdisp+1;
            if mod(cntdisp,5) == 0
                fprintf('\n');
            end
        end
    end
end

%% Results
% Making sure optional variables exist
if ~exist('rmse_train','var'), rmse_train = []; end
if ~exist('rmse_test','var'),  rmse_test  = []; end
if ~exist('residual','var'), residual = []; end
if ~exist('objective','var'),  objective  = []; end
if ~exist('rho','var'),        rho        = []; end
if ~exist('iter','var'),       iter       = NaN; end
if ~exist('status_msg','var'), status_msg = 'unknown'; end
% Build results struct
results = struct();
results.T           = T;
results.W           = W;
results.H           = H;
results.residual    = residual;
results.objective   = objective;
results.rmse_train  = rmse_train;
results.rmse_test   = rmse_test;
results.rho         = rho;
% Keep the parameters actually used
results.param = param;

%% Plotting results
if param.display==1
    if isfield(param, 'mask') && ~isempty(param.mask)
        figure('Name','RMSE Plots','NumberTitle','off');
        % --- RMSE on Test Data ---
        subplot(1,2,1);
        plot(rmse_test, 'b-', 'LineWidth', 2); grid on;
        xlabel('Iterations','Interpreter','latex','FontSize',16);
        ylabel('RMSE on Test Data','Interpreter','latex','FontSize',16);
        title('Test RMSE','Interpreter','latex','FontSize',18);
        set(gca,'FontSize',14,'LineWidth',1);
        set(gca,'TickLabelInterpreter','latex');
        % --- RMSE on Train Data ---
        subplot(1,2,2);
        plot(rmse_train, 'r-', 'LineWidth', 2); grid on;
        xlabel('Iterations','Interpreter','latex','FontSize',16);
        ylabel('RMSE on Train Data','Interpreter','latex','FontSize',16);
        title('Train RMSE','Interpreter','latex','FontSize',18);
        set(gca,'FontSize',14,'LineWidth',1);
        set(gca,'TickLabelInterpreter','latex');
        % --- Overall figure title ---
        sgtitle('RMSE Evolution During ADMM Iterations','Interpreter','latex','FontSize',18);
    else
        subplot(1,2,1);
        semilogy(objective,'b-','LineWidth',2); grid on;
        xlabel('Iterations','Interpreter','latex','FontSize',16);
        if strcmpi(error_measure, 'KL divergence')
            ylabel('Normalized KL:$KL(X,\hat{X})/KL(X,\bar{X})$', 'Interpreter','latex', 'FontSize',16,'FontWeight','bold');
        else
            ylabel('Relative error: $d(X,f(WH))/d(X,0)$','Interpreter','latex','FontSize',16, 'FontWeight','bold');
        end
        title('ADMM Objective','Interpreter','latex','FontSize',18);
        set(gca,'FontSize',14,'LineWidth',1);
        set(gca,'TickLabelInterpreter','latex');
        % --- Residual plot ---
        subplot(1,2,2);
        semilogy(residual,'r-','LineWidth',2); grid on;
        xlabel('Iterations','Interpreter','latex','FontSize',16);
        ylabel('Residual: $\|T-WH\|_F/\|T\|_F$','Interpreter','latex','FontSize',16);
        title('ADMM Residual','Interpreter','latex','FontSize',18);
        set(gca,'FontSize',14,'LineWidth',1);
        set(gca,'TickLabelInterpreter','latex');
        sgtitle('ADMM evolution of error','Interpreter','latex','FontSize',18);
    end
end