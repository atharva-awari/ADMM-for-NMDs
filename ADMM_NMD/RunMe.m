close all; clear all; clc;
addpath('./utils');
rng(2025);
%% Model and Error Measure
model         = 'MinMax';      %Choose the model among: 'ReLU', 'CSF', 'MinMax', or 'Modulus'
error_measure = 'L1'; %Choose the error measure among: 'Frobenius','L1' (Componentwise L1 norm), or 'KL divergence'

%% Define parameters of the algorithm
param.maxiter = inf;
param.maxtime =5;
param.updateWH='closed form';
param.a=0; param.b = 1;  %For the MinMax model, the user has to define the desired interval [a,b] to generate data in that interval
%param.display = 0;        %If the user does not want any display of the plots

%% Data
m = 200;
n = 100;
r = 30;
X = generate_synthetic(m,n,r,model,param);
%Random initialization
%param.lambda=randn(m,n);
 % param.T = randn(m,n);
 % param.W0=randn(m,r);
 % param.H0=randn(r,n);
%% Run of the algorithm
results = admm_nmd(X,r,model,error_measure,param);