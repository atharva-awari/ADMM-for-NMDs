%% Table3_extended.m
%  Matrix Completion: ADMM vs WLRA vs LMaFit vs RTRMC
%  CBCL dataset, multiple missing ratios, 10 seeds, mean +/- std
%
%  BEFORE RUNNING:
%  1. Download LMaFit:
%     http://lmafit.blogs.rice.edu/
%     Place lmafit_mc_adp.m in this folder or on your MATLAB path.
%
%  2. Download Manopt (for RTRMC):
%     https://www.manopt.org/downloads.html
%     Unzip and run: addpath(genpath('path/to/manopt'))
%
%  Both are standard, well-cited baselines in the matrix completion
%  literature.

close all; clear all; clc;
addpath('./utils');
addpath(genpath('manopt'));
if ~exist('figures','dir'), mkdir('figures'); end

%% ====================================================================
%  CONFIGURATION
%  ====================================================================
nruns       = 10;
r           = 5;         % factorization rank
p_miss_list = [0, 0.05, 0.10, 0.20, 0.50, 0.80];
n_cases     = length(p_miss_list);

% ADMM parameters
admm_maxiter = 100;
admm_maxtime = 20;

% Method flags (set to false if code not available)
run_lmafit = true;
run_rtrmc  = true;

%% ====================================================================
%  CHECK DEPENDENCIES
%  ====================================================================
if run_lmafit && ~exist('lmafit_mc_adp','file')
    warning('lmafit_mc_adp.m not found. Disabling LMaFit. Download from: http://lmafit.blogs.rice.edu/');
    run_lmafit = false;
end
if run_rtrmc && ~exist('fixedrankembeddedfactory','file')
    warning('Manopt not found. Disabling RTRMC. Download from: https://www.manopt.org/');
    run_rtrmc = false;
end

%% ====================================================================
%  LOAD DATA
%  ====================================================================
load('CBCL.mat');
Xtrue = double(X);
Xtrue = Xtrue / max(Xtrue(:));  % normalize to [0,1]
[m, n] = size(Xtrue);
fprintf('Data: CBCL %d x %d, rank %d\n\n', m, n, r);

%% ====================================================================
%  STORAGE
%  ====================================================================
n_methods = 2 + run_lmafit + run_rtrmc;  % ADMM, WLRA, +optional
method_names = {'ADMM', 'WLRA'};
if run_lmafit, method_names{end+1} = 'LMaFit'; end
if run_rtrmc,  method_names{end+1} = 'RTRMC';  end

rmse_test_all  = zeros(n_cases, nruns, n_methods);
rmse_train_all = zeros(n_cases, nruns, n_methods);

%% ====================================================================
%  MAIN EXPERIMENT
%  ====================================================================
for k = 1:n_cases
    p_miss = p_miss_list(k);
    p_obs  = 1 - p_miss;

    fprintf('==========================================================\n');
    fprintf('  Missing ratio: %.0f%%\n', p_miss * 100);
    fprintf('==========================================================\n');

    for run = 1:nruns
        rng(2024 + run);

        % --- Create observation mask ---
        M_obs = rand(m, n) <= p_obs;

        % --- Split observed into 80% train / 20% test ---
        obs_idx = find(M_obs);
        n_obs   = numel(obs_idx);
        n_train = round(0.8 * n_obs);
        perm    = randperm(n_obs);
        train_idx = obs_idx(perm(1:n_train));
        test_idx  = obs_idx(perm(n_train+1:end));

        M_train = false(m, n);
        M_test  = false(m, n);
        M_train(train_idx) = true;
        M_test(test_idx)   = true;

        % Training data (zeros where unobserved)
        Xtrain = zeros(m, n);
        Xtrain(M_train) = Xtrue(M_train);

        mi = 0;  % method index

        % ============================================================
        %  1. ADMM (MinMax + Frobenius)
        % ============================================================
        mi = mi + 1;
        param_admm = struct();
        param_admm.maxiter   = admm_maxiter;
        param_admm.maxtime   = admm_maxtime;
        param_admm.synthetic = true;
        param_admm.a         = 0;
        param_admm.b         = 1;
        param_admm.display   = 0;
        param_admm.mask      = M_train;
        param_admm.validmask = M_test;

        results = admm_nmd(Xtrain, r, 'MinMax', 'Frobenius', param_admm);
        A_admm = results.W * results.H;
        Xhat_admm = min(1, max(0, A_admm));  % apply MinMax

        rmse_train_all(k, run, mi) = sqrt(mean((Xtrue(M_train) - Xhat_admm(M_train)).^2));
        rmse_test_all(k, run, mi)  = sqrt(mean((Xtrue(M_test)  - Xhat_admm(M_test)).^2));

        % ============================================================
        %  2. WLRA
        % ============================================================
        mi = mi + 1;
        options_wlra = struct();
        options_wlra.rank    = r;
        options_wlra.maxiter = 100;
        [W_wlra, H_wlra, ~] = WLRA(Xtrain, M_train, options_wlra);
        A_wlra = W_wlra * H_wlra';

        rmse_train_all(k, run, mi) = sqrt(mean((Xtrue(M_train) - A_wlra(M_train)).^2));
        rmse_test_all(k, run, mi)  = sqrt(mean((Xtrue(M_test)  - A_wlra(M_test)).^2));

        % ============================================================
        %  3. LMaFit (if available)
        % ============================================================
        if run_lmafit
            mi = mi + 1;
            opts_lm = struct();
            opts_lm.tol      = 1e-6;
            opts_lm.maxit    = 100;
            opts_lm.print    = 0;
            opts_lm.est_rank = 0;  % fixed rank (no adaptive rank reduction)
            opts_lm.Zfull    = 1;  % dense mode, no MEX needed

            % LMaFit takes linear indices and values
            known = find(M_train);
            data  = Xtrue(known);
            [X_lm, Y_lm, ~] = lmafit_mc_adp(m, n, r, known, data, opts_lm);
            A_lmafit = X_lm * Y_lm;

            rmse_train_all(k, run, mi) = sqrt(mean((Xtrue(M_train) - A_lmafit(M_train)).^2));
            rmse_test_all(k, run, mi)  = sqrt(mean((Xtrue(M_test)  - A_lmafit(M_test)).^2));
        end

        % ============================================================
        %  4. RTRMC via Manopt (if available)
        % ============================================================
        if run_rtrmc
            mi = mi + 1;

            % Build Manopt problem for fixed-rank matrix completion
            manifold = fixedrankembeddedfactory(m, n, r);

            problem = struct();
            problem.M = manifold;
            problem.cost  = @(X) mc_cost(X, Xtrue, M_train);
            problem.egrad = @(X) mc_egrad(X, Xtrue, M_train);
            

            % Solver options
            opts_manopt = struct();
            opts_manopt.maxiter = 100;
            opts_manopt.verbosity = 0;

            % Solve with trust regions on the manifold
            X0 = manifold.rand();
            [X_sol, ~, ~] = trustregions(problem, X0, opts_manopt);
            A_rtrmc = X_sol.U * X_sol.S * X_sol.V';

            rmse_train_all(k, run, mi) = sqrt(mean((Xtrue(M_train) - A_rtrmc(M_train)).^2));
            rmse_test_all(k, run, mi)  = sqrt(mean((Xtrue(M_test)  - A_rtrmc(M_test)).^2));
        end

        fprintf('  Run %d/%d: ', run, nruns);
        for j = 1:n_methods
            fprintf('%s=%.4f  ', method_names{j}, rmse_test_all(k, run, j));
        end
        fprintf('\n');
    end
end

%% ====================================================================
%  SUMMARY TABLE
%  ====================================================================
fprintf('\n\n================================================================\n');
fprintf('  SUMMARY: RMSE on Test Set (mean +/- std over %d runs)\n', nruns);
fprintf('================================================================\n');
fprintf('Missing %%');
for j = 1:n_methods
    fprintf(' | %15s', method_names{j});
end
fprintf('\n');
fprintf(repmat('-', 1, 12 + 18*n_methods));
fprintf('\n');

for k = 1:n_cases
    fprintf('%6.0f %%  ', p_miss_list(k)*100);
    for j = 1:n_methods
        mu = mean(rmse_test_all(k, :, j));
        sd = std(rmse_test_all(k, :, j));
        fprintf(' | %6.4f +/- %6.4f', mu, sd);
    end
    fprintf('\n');
end

%% ====================================================================
%  EXPORT LaTeX TABLE
%  ====================================================================
latex_file = 'figures/table_matrix_completion.tex';
fid = fopen(latex_file, 'w');
fprintf(fid, '\\begin{table}[h!]\n\\centering\n');
fprintf(fid, '\\caption{Matrix completion on CBCL ($r{=}%d$). Test RMSE (mean $\\pm$ std over %d runs).}\n', r, nruns);
fprintf(fid, '\\label{tab:mc_extended}\n');
fprintf(fid, '\\footnotesize\n');
fprintf(fid, '\\begin{tabular}{c');
for j = 1:n_methods
    fprintf(fid, 'c');
end
fprintf(fid, '}\n\\toprule\n');
fprintf(fid, '\\textbf{Missing \\%%}');
for j = 1:n_methods
    fprintf(fid, ' & \\textbf{%s}', method_names{j});
end
fprintf(fid, ' \\\\\n\\midrule\n');

for k = 1:n_cases
    fprintf(fid, '%d\\%%', round(p_miss_list(k)*100));
    % Find best method for this row
    means = zeros(1, n_methods);
    for j = 1:n_methods
        means(j) = mean(rmse_test_all(k, :, j));
    end
    [~, best] = min(means);
    for j = 1:n_methods
        mu = mean(rmse_test_all(k, :, j));
        sd = std(rmse_test_all(k, :, j));
        if j == best
            fprintf(fid, ' & $\\mathbf{%.4f \\pm %.4f}$', mu, sd);
        else
            fprintf(fid, ' & $%.4f \\pm %.4f$', mu, sd);
        end
    end
    fprintf(fid, ' \\\\\n');
end

fprintf(fid, '\\bottomrule\n\\end{tabular}\n\\end{table}\n');
fclose(fid);
fprintf('\nSaved LaTeX table: %s\n', latex_file);

fprintf('\n*** Matrix completion experiment complete. ***\n');

%% ====================================================================
%  HELPER FUNCTIONS FOR MANOPT (Riemannian Matrix Completion)
%  ====================================================================
function val = mc_cost(X, Xtrue, M_train)
    % Cost: 0.5 * ||M .* (USV' - Xtrue)||_F^2
    A = X.U * X.S * X.V';
    diff = (A - Xtrue) .* M_train;
    val = 0.5 * sum(diff(:).^2);
end

function G = mc_egrad(X, Xtrue, M_train)
    % Euclidean gradient: m x n matrix
    % Manopt converts this to Riemannian gradient internally
    A = X.U * X.S * X.V';
    G = (A - Xtrue) .* M_train;
end
