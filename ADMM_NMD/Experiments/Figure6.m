close all; clear all; clc;
addpath('./utils');
rng(2025);

%% ================== Experiment Setup ==================
m = 100;
n = 80;
r = 5;

model = 'CSF';   % only CSF
loss_display = {'Frobenius','L1','KL divergence'};
loss_fields  = {'Frobenius','L1','KL'};

param.maxiter   = 100;
param.maxtime   = 20;
param.updateWH  = 'closed form';
param.display   = 0;

num_runs = 1;

% Styles for different losses
styles = {'-o','--^',':*'};   % Frobenius, L1, KL
markEvery = 1;

results_all = struct();


%% ================== Main Loop ==================
figure('Color','w','Name','CSF: Objective vs Iterations');
set(gcf,'Position',[200 200 700 420]);
hold on;

for l = 1:length(loss_display)

    error_measure = loss_display{l};
    loss_field    = loss_fields{l};

    obj_runs = cell(num_runs,1);
    minLen = inf;

    for run = 1:num_runs

        % Reproducible but different initializations
        rng(2025 + 100*l + run);

        % Low-rank factors (fixed across runs for fairness)
        W_true = rand(m, r);
        H_true = rand(r, n);

        % CSF nonlinearity: entrywise square
        X_clean = (W_true * H_true).^2;

        %Random initialization
        param.T0 = rand(m,n);
        param.W0 = rand(m,r);
        param.H0 = rand(r,n);
        % Run ADMM
        results = admm_nmd(X_clean, r, model, error_measure, param);

        obj = results.objective(:);
        obj_runs{run} = obj;
        minLen = min(minLen, numel(obj));
    end

    % Average objectives
    OBJ = zeros(minLen, num_runs);
    for run = 1:num_runs
        OBJ(:,run) = obj_runs{run}(1:minLen);
    end
    obj_mean = mean(OBJ, 2);

    results_all.(loss_field).objective_mean = obj_mean;

    % Plot
    semilogy(obj_mean, styles{l}, ...
        'LineWidth', 2, ...
        'MarkerIndices', 1:markEvery:minLen, ...
        'DisplayName', loss_display{l});
end

%% ================== Plot Formatting ==================
xlabel('\textbf{Iteration}','Interpreter','latex','FontSize',12,'FontWeight','bold');
ylabel('\textbf{Average objective}','Interpreter','latex','FontSize',12,'FontWeight','bold');
title('\textbf{CSF: Average Objective vs Iterations}', ...
    'Interpreter','latex','FontSize',13,'FontWeight','bold');

legend('Location','best','Interpreter','latex');
set(gca,'yscale','log');
grid on; box on;
%hold off;
