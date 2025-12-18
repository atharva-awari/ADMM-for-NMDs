close all; clear all; clc;
addpath('./utils');
rng(2025);

load CBCL.mat

[m,n] = size(X);
r = 10;

W0 = randn(m,r);
H0 = randn(r,n);
param.maxit=150;
[W,H,W_stop,H_stop,err,iter,savetime] = alternatescheme(X,W0,H0,param);

model = 'ReLU';
error_measure = 'Frobenius';
Z = max(0,W0*H0); 
optscal = sum(sum(Z.*X))/sum(Z(:).^2); 
W0 = optscal*W0;
param.W0=W0;
param.H0=H0;
param.maxiter=15;
param.display = 0;

results = admm_nmd(X, r, model, error_measure, param);

%% --- Side-by-side plots: Error vs Iteration and Error vs Time ---
figure('Color','w');

% -------------------------
% Left: Error vs Iteration
% -------------------------
subplot(1,2,1); hold on; grid on; box on;

p1 = semilogy(err, 'LineWidth', 2);
p2 = semilogy(results.objective, 'LineWidth', 2);

xlabel('Iteration','Interpreter','latex','FontSize',14);
ylabel('Relative error','Interpreter','latex','FontSize',14);
title('Error vs Iteration','Interpreter','latex','FontSize',16);

legend([p1 p2], ...
    {'$\mathrm{Coordinate descent}$', '$\mathrm{ADMM}$'}, ...
    'Interpreter','latex','FontSize',13,'Location','northeast');

set(gca,'FontSize',12,'LineWidth',1);

% --------------------
% Right: Error vs Time
% --------------------
subplot(1,2,2); hold on; grid on; box on;

p3 = semilogy(savetime, err, 'LineWidth', 2);
p4 = semilogy(results.time, results.objective, 'LineWidth', 2);

xlabel('Time (s)','Interpreter','latex','FontSize',14);
ylabel('Relative error','Interpreter','latex','FontSize',14);
title('Error vs Time','Interpreter','latex','FontSize',16);

legend([p3 p4], ...
    {'$\mathrm{Coordinate descent}$', '$\mathrm{ADMM}$'}, ...
    'Interpreter','latex','FontSize',13,'Location','northeast');

set(gca,'FontSize',12,'LineWidth',1);


