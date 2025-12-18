%% Experiment: ADMM vs WLRA under different missing ratios
clear; clc;

%% Load data
load CBCL.mat;   % assumes X is in CBCL.mat

%% Model and Error Measure for ADMM
model    = 'MinMax';      % 'ReLU', 'CSF', 'MinMax', or 'Modulus'
error_measure = 'Frobenius';   % 'Frobenius','L1', or 'KL divergence'

%% Parameters of ADMM algorithm
param.maxiter   = 100;
param.maxtime   = 20;
param.synthetic = true;
param.a         = 0; 
param.b         = 1;      % interval [a,b] for MinMax model
param.display   = 0;

%% WLRA options
options.rank   = 5;
options.maxiter = 100;

%% Factorization rank
r = 5;  % factorization rank

%% Basic info
[m,n] = size(X);
rng(0);    % for reproducibility

%% Missing ratios to test
p_miss_list = [0, 0.05,0.10, 0.2, 0.5, 0.8,];   % 0% missing, 20%, 50%, 80%
n_cases     = numel(p_miss_list);

%% Storage for results
rmse_train_admm = zeros(n_cases,1);
rmse_test_admm  = zeros(n_cases,1);
rmse_train_wlra = zeros(n_cases,1);
rmse_test_wlra  = zeros(n_cases,1);

fprintf('=== ADMM vs WLRA on CBCL ===\n');
fprintf('Missing ratio   |  Method  |  RMSE_train   RMSE_test\n');
fprintf('-----------------------------------------------------\n');
Xtrue=X;
for k = 1:n_cases
    model    = 'MinMax';      % 'ReLU', 'CSF', 'MinMax', or 'Modulus'
    error_measure = 'Frobenius'; 
    
    X=Xtrue;
    p_miss = p_miss_list(k);
    p_obs  = 1 - p_miss;

    %% 1) Construct observed mask M_obs
    %if p_miss == 0
        % fully observed case: know all of X
      %  M_obs = true(m,n);
    %else
        % only a subset of entries are observed
       M_obs = rand(m,n) <= p_obs;
    %end

    %% 2) Split M_obs into train (80%) and test (20%), disjoint
    obs_idx = find(M_obs);            % linear indices of observed entries
    n_obs   = numel(obs_idx);

    n_train = round(0.8 * n_obs);
    perm    = randperm(n_obs);
    train_idx = obs_idx(perm(1:n_train));
    test_idx  = obs_idx(perm(n_train+1:end));

    M_train = false(m,n);
    M_test  = false(m,n);
    M_train(train_idx) = true;
    M_test(test_idx)   = true;

    %% ========= ADMM on training mask =========
 
    param.mask  = M_train;
    param.validmask = M_test;
    Xtrain = zeros(m,n);
    Xtrain(M_train) = X(M_train);
    results_admm = admm_nmd(Xtrain, r, model, error_measure, param);

    % RMSE on train and test sets
    A_admm = results_admm.W*results_admm.H;        % predictions (pre-nonlinearity)

    metrics_admm = mc_metrics(X, X, A_admm, model, param);
    fprintf('   %4.0f %%       |  ADMM    |   %8.4f    %8.4f\n', ...
        p_miss*100, metrics_admm.rmse_train, metrics_admm.rmse_validation);

    %% ========= WLRA on training mask =========
    % WLRA takes the mask of observed entries; here we give it training mask
    options.rank = 5;
    options.maxiter=100;
    model = 'WLRA';
    [W,H,e] = WLRA(Xtrain,M_train,options);
    A_wlra = W*H';
  
    metrics_wlra = mc_metrics(X, X, A_wlra, model, param);
    fprintf('   %4.0f %%       |  WLRA    |   %8.4f    %8.4f\n', ...
        p_miss*100, metrics_wlra.rmse_train, metrics_wlra.rmse_validation);

end

fprintf('-----------------------------------------------------\n');


