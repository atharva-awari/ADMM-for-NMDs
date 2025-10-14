close all; clear all; clc;

%% Model and Error Measure
model         = 'ReLU';      %Choose the model among: 'ReLU', 'CSF', 'MinMax', or 'Modulus'
error_measure = 'Frobenius'; %Choose the error measure among: 'Frobenius','L1' (Componentwise L1 norm), or 'KL divergence'

  
%% Define parameters of the algorithm
param.maxiter = inf;
param.maxtime = 5;

param.a=0; param.b = 1; %If the user wants to use MinMax model, define the desired interval [a,b]

%% Data
m      = 200;
n      = 100;
r      = 20;  % factorization rank you want to fit
r_true = 30;  % the true underlying factorization rank of the data
X      = generate_synthetic(m,n,r_true,model,param); %Function generating a synthetic matrix X

%Mask definition
p_obs       = 0.8;                      % fraction of observed entries
M           = rand(m,n) < p_obs;        % binary mask (1 observed, 0 missing)
param.mask  = M;
% The algorithm will create a matrix 'B' with only observed entries of the data matrix X and run ADMM on B. 


%% Run the algorithm
results = admm_nmd(X,r,model,error_measure,param);


