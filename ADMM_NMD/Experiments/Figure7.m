%% rho_ablation_supplementary.m
%  Ablation: Adaptive rho vs Fixed rho on synthetic data
%  All 12 model/loss combinations, matching Section V-A setup
%  m=100, n=80, r=5, 1000 iterations, 10 runs (averaged)
%  Outputs: 12 tikz-ready .txt files + 12 .png figures

close all; clear all; clc;
addpath('./utils');
if ~exist('figures','dir'), mkdir('figures'); end

%% ====================================================================
%  CONFIGURATION (matching paper Section V-A)
%  ====================================================================
nruns   = 10;
m       = 100;
n       = 80;
r       = 5;
maxiter = 1000;
maxtime = 20;  % generous, iteration limit is what matters

rho_values = [0.01, 0.1, 1, 10, 100];
n_configs  = length(rho_values) + 1;  % +1 for adaptive

models         = {'ReLU','MinMax'};
error_measures = {'Frobenius','L1','KL divergence'};

% Plot styles
colors  = {[0.8 0 0.8], [0 0.7 0.7], [0.5 0.5 0.5], [1 0.5 0], [0.6 0.2 0.8], [0 0 1]};
markers = {'v', 'd', 'p', 'h', '>', '^'};
styles  = {'--', '--', '--', '--', '--', '-'};  % dashed for fixed, solid for adaptive

% Legend labels
leg_labels = cell(1, n_configs);
for i = 1:length(rho_values)
    leg_labels{i} = sprintf('Fixed $\\rho=%g$', rho_values(i));
end
leg_labels{end} = 'Adaptive $\rho$';

%% ====================================================================
%  MAIN LOOP
%  ====================================================================
combo = 0;
for mi = 1:length(models)
    model = models{mi};

    for ei = 1:length(error_measures)
        error_measure = error_measures{ei};
        combo = combo + 1;

        fprintf('\n==========================================================\n');
        fprintf('  [%d/12] %s + %s\n', combo, model, error_measure);
        fprintf('==========================================================\n');

        % Storage: objective curves per config, per run
        % We'll accumulate and average
        obj_sum = zeros(maxiter, n_configs);
        obj_count = zeros(maxiter, n_configs);

        for run = 1:nruns
            rng(2024 + run);

            % Generate synthetic data
            param_gen = struct();
            if strcmpi(model, 'MinMax')
                param_gen.a = eps;
                param_gen.b = 1;
            end
            X = generate_synthetic(m, n, r, model, param_gen);
            
            % For KL divergence, ensure no exact zeros
            % if strcmpi(error_measure, 'KL divergence')
            %     X = max(X, eps);
            % end

            % Default initialization (SVD-based, computed once per run)
            % Let admm_nmd compute it from X — but we need the SAME
            % init for all configs. So compute it here and pass it in.
            if strcmpi(model, 'CSF')
                T0 = sqrt(X);
            else
                T0 = X;
            end
            [U0, S0, V0] = svds(T0, r);
            W0 = U0 * sqrt(S0);
            H0 = sqrt(S0) * V0';

            % ----- Run each configuration -----
            for ci = 1:n_configs
                param_admm = struct();
                param_admm.T0 = T0;
                param_admm.W0 = W0;
                param_admm.H0 = H0;
                param_admm.maxiter = maxiter;
                param_admm.maxtime = maxtime;
                param_admm.display = 0;
                if strcmpi(model, 'MinMax')
                    param_admm.a = eps;
                    param_admm.b = 1;
                end

                if ci <= length(rho_values)
                    % Fixed rho
                    param_admm.adaptive_rho = false;
                    param_admm.rho0 = rho_values(ci);
                else
                    % Adaptive rho
                    param_admm.adaptive_rho = true;
                    param_admm.rho0 = 1;
                end

                try
                    results = admm_nmd(X, r, model, error_measure, param_admm);
                    obj = results.objective(:);
                    L = length(obj);
                    obj_sum(1:L, ci) = obj_sum(1:L, ci) + obj;
                    obj_count(1:L, ci) = obj_count(1:L, ci) + 1;
                catch ME
                    fprintf('    WARNING: %s+%s config %d run %d failed: %s\n', ...
                        model, error_measure, ci, run, ME.message);
                end
            end
            fprintf('  Run %d/%d done\n', run, nruns);
        end

        % Compute average (avoid division by zero)
        obj_avg = obj_sum ./ max(obj_count, 1);
        % Trim trailing zeros (find max valid length per config)
        valid_len = zeros(1, n_configs);
        for ci = 1:n_configs
            valid_len(ci) = find(obj_count(:, ci) > 0, 1, 'last');
        end
        max_valid = max(valid_len);

        %% ============================================================
        %  EXPORT .txt for tikz
        %  ============================================================
        % Short names for filenames
        loss_short = error_measure;
        if strcmpi(error_measure, 'KL divergence'), loss_short = 'KL'; end

        txt_file = sprintf('figures/rho_ablation_%s_%s.txt', model, loss_short);
        fid = fopen(txt_file, 'w');
        % Header: y1..y5 for fixed rho, y6 for adaptive
        fprintf(fid, 'y1 y2 y3 y4 y5 y6\n');
        for i = 1:max_valid
            for ci = 1:n_configs
                if i <= valid_len(ci)
                    fprintf(fid, '%.6e ', obj_avg(i, ci));
                else
                    fprintf(fid, '%.6e ', obj_avg(valid_len(ci), ci));
                end
            end
            fprintf(fid, '\n');
        end
        fclose(fid);
        fprintf('  Saved: %s\n', txt_file);

        %% ============================================================
        %  MATLAB FIGURE (quick check)
        %  ============================================================
        figure('Color','w','Name',sprintf('%s + %s', model, error_measure));
        hold on; grid on; box on;
        ph = zeros(n_configs, 1);
        for ci = 1:n_configs
            L = valid_len(ci);
            mk_interval = max(1, floor(L/15));
            ph(ci) = semilogy(1:L, obj_avg(1:L, ci), ...
                styles{ci}, 'LineWidth', 2, 'Color', colors{ci}, ...
                'Marker', markers{ci}, 'MarkerSize', 5, ...
                'MarkerIndices', 1:mk_interval:L);
        end
        xlabel('Iteration','Interpreter','latex','FontSize',14);
        ylabel('Average objective','Interpreter','latex','FontSize',14);
        title(sprintf('%s + %s', model, error_measure),'Interpreter','latex','FontSize',16);
        legend(ph, leg_labels, 'Interpreter','latex','FontSize',9,'Location','northeast');
        set(gca,'FontSize',12,'LineWidth',1);
        saveas(gcf, sprintf('figures/rho_ablation_%s_%s.png', model, loss_short));
        fprintf('  Saved figure: figures/rho_ablation_%s_%s.png\n', model, loss_short);
    end
end

fprintf('\n*** Rho ablation (supplementary) complete. ***\n');
fprintf('*** 12 tikz files saved in figures/ ***\n');