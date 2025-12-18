close all; clear all; clc;
addpath('./utils');
rng(2025);

%% ================== Experiment Setup ==================
m = 100;
n = 80;
r = 5;

models = {'ReLU','CSF','MinMax','Modulus'};
loss_display = {'Frobenius','L1','KL divergence'};
loss_fields  = {'Frobenius','L1','KL'};

param.maxiter   = 1000;
param.maxtime   = 20;
param.updateWH  = 'closed form';
param.a         = eps;
param.b         = 1;
param.display   = 0;

num_runs = 10;

% Line/marker styles per model (distinct + readable)
styles = {'-o','--^',':*','-.s'};   % ReLU, CSF, MinMax, Modulus
markEvery = 50;                    % marker spacing (avoid clutter)

results_all = struct();

%% ================== Main Loop ==================
for l = 1:length(loss_display)

    error_measure = loss_display{l};   % passed to ADMM
    loss_field    = loss_fields{l};    % safe struct name

    figure('Color','w','Name',['Convergence - ' error_measure]);
    set(gcf,'Position',[200 200 700 420]);   % compact single plot
    hold on;

    for m_id = 1:length(models)

        model = models{m_id};

        % Accumulate objectives (handle variable length safely)
        obj_runs = cell(num_runs,1);
        minLen = inf;

        for run = 1:num_runs
            % Different seed per run but reproducible overall
            rng(2025 + 1000*l + 100*m_id + run);

            % Generate synthetic data (and/or random init inside your solver)
            X = generate_synthetic(m,n,r,model,param);

            % Run ADMM
            results = admm_nmd(X, r, model, error_measure, param);

            obj = results.objective(:);
            obj_runs{run} = obj;
            minLen = min(minLen, numel(obj));
        end

        % Truncate to common length and average
        OBJ = zeros(minLen, num_runs);
        for run = 1:num_runs
            OBJ(:,run) = obj_runs{run}(1:minLen);
        end
        obj_mean = mean(OBJ, 2);

        % Store
        results_all.(loss_field).(model).objective_mean = obj_mean;

        % Plot mean objective (semilogy)
        semilogy(obj_mean, styles{m_id}, ...
            'LineWidth', 2, ...
            'MarkerIndices', 1:markEvery:minLen, ...
            'DisplayName', model);
    end

    xlabel('\textbf{Iteration}','Interpreter','latex','FontSize',12,'FontWeight','bold');
    ylabel('\textbf{Average objective}','Interpreter','latex','FontSize',12,'FontWeight','bold');
    title(['\textbf{Average objective vs. iterations (', error_measure, ')}'], ...
        'Interpreter','latex','FontSize',13,'FontWeight','bold');

    legend('Location','best','Interpreter','latex');
    set(gca,'yscale','log');
    grid on; box on;
    hold off;
end
