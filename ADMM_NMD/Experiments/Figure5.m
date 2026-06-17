%% Figure5.m
%  Extended comparison: ADMM vs Coordinate Descent for ReLU-NMD
%  Datasets: CBCL, MNIST
%  Multiple random initializations with mean +/- std reporting
%  Outputs: convergence plots (error vs iter, error vs time) + summary table
%  Also exports .txt files for tikz plots in the paper
%  Saves initializations for PyTorch comparison script

close all; clear all; clc;
addpath('./utils');
if ~exist('figures','dir'), mkdir('figures'); end
if ~exist('figures/inits','dir'), mkdir('figures/inits'); end

%% ====================================================================
%  CONFIGURATION
%  ====================================================================
nruns       = 10;       % Number of random initializations
ranks       = [10, 32]; % Rank for [CBCL, MNIST]
cd_maxiter  = inf;      % Max iterations for Coordinate Descent
admm_maxiter= inf;      % Max iterations for ADMM
admm_maxtime= 10;       % Time limit for ADMM (seconds)
cd_maxtime  = 10;       % Time limit for CD   (seconds)

datasets = {'CBCL', 'MNIST'};

%% ====================================================================
%  MAIN LOOP OVER DATASETS
%  ====================================================================
for d = 1:length(datasets)
    fprintf('\n==========================================================\n');
    fprintf('  Dataset: %s  |  Rank: %d  |  Runs: %d\n', datasets{d}, ranks(d), nruns);
    fprintf('==========================================================\n');

    % Load data
    if strcmpi(datasets{d}, 'CBCL')
        S = load('CBCL.mat');
        fn = fieldnames(S);
        X  = double(S.(fn{1}));
    elseif strcmpi(datasets{d}, 'MNIST')
        Y  = load('mnist_all.mat');
        w1 = 1:50;  % 50 images per digit -> 500 total
        X  = [Y.train0(w1,:); Y.train1(w1,:); Y.train2(w1,:); ...
              Y.train3(w1,:); Y.train4(w1,:); Y.train5(w1,:); ...
              Y.train6(w1,:); Y.train7(w1,:); Y.train8(w1,:); ...
              Y.train9(w1,:)];
        X  = double(X);
        X  = X / max(X(:));  % Normalize to [0,1]
    end
    [m, n] = size(X);
    r = ranks(d);
    fprintf('  Matrix size: %d x %d\n', m, n);

    % Also save X for PyTorch to load (once per dataset)
    save(sprintf('figures/inits/X_%s.mat', datasets{d}), 'X');

    % Storage for summary statistics
    final_err_cd   = zeros(nruns, 1);
    final_err_admm = zeros(nruns, 1);
    final_time_cd  = zeros(nruns, 1);
    final_time_admm= zeros(nruns, 1);

    % Storage for convergence curves
    all_err_cd    = cell(nruns, 1);
    all_time_cd   = cell(nruns, 1);
    all_err_admm  = cell(nruns, 1);
    all_time_admm = cell(nruns, 1);

    for run = 1:nruns
        fprintf('\n--- Run %d/%d ---\n', run, nruns);
        rng(2024 + run);  % Reproducible but different per run

        % ----- Common initialization -----
        W0 = randn(m, r);
        H0 = randn(r, n);

        % ----- Common scaling -----
        Z0 = max(0, W0 * H0);
        optscal = sum(sum(Z0 .* X)) / sum(Z0(:).^2);
        W0_scaled = optscal * W0;

        % ----- Save initialization for PyTorch -----
        save(sprintf('figures/inits/init_%s_run%d.mat', datasets{d}, run), ...
            'W0_scaled', 'H0');

        % ============================================================
        %  Coordinate Descent
        % ============================================================
        param_cd        = struct();
        param_cd.maxit  = cd_maxiter;
        param_cd.time   = cd_maxtime;

        [~, ~, ~, ~, err_cd, ~, time_cd] = ...
            alternatescheme(X, W0_scaled, H0, param_cd);

        all_err_cd{run}  = err_cd(:);
        all_time_cd{run} = time_cd(:);
        final_err_cd(run) = err_cd(end);
        final_time_cd(run) = time_cd(end);
        fprintf('  CD  : final err = %.4f, time = %.2f s\n', err_cd(end), time_cd(end));

        % ============================================================
        %  ADMM
        % ============================================================
        model          = 'ReLU';
        error_measure  = 'Frobenius';

        param_admm          = struct();
        param_admm.W0       = W0_scaled;
        param_admm.H0       = H0;
        param_admm.maxiter  = admm_maxiter;
        param_admm.maxtime  = admm_maxtime;
        param_admm.display  = 0;

        results = admm_nmd(X, r, model, error_measure, param_admm);

        all_err_admm{run}  = results.objective(:);
        all_time_admm{run} = results.time(:);
        final_err_admm(run) = results.objective(end);
        final_time_admm(run) = results.time(end);
        fprintf('  ADMM: final err = %.4f, time = %.2f s\n', ...
            results.objective(end), results.time(end));
    end

    %% ================================================================
    %  SUMMARY TABLE (printed to console + saved)
    %  ================================================================
    fprintf('\n----------------------------------------------------------\n');
    fprintf('  SUMMARY: %s (rank %d, %d runs)\n', datasets{d}, r, nruns);
    fprintf('----------------------------------------------------------\n');
    fprintf('  Method |  Final Error (mean +/- std) | Time (mean +/- std)\n');
    fprintf('  CD     |  %.4f +/- %.4f          | %.2f +/- %.2f s\n', ...
        mean(final_err_cd), std(final_err_cd), ...
        mean(final_time_cd), std(final_time_cd));
    fprintf('  ADMM   |  %.4f +/- %.4f          | %.2f +/- %.2f s\n', ...
        mean(final_err_admm), std(final_err_admm), ...
        mean(final_time_admm), std(final_time_admm));
    fprintf('----------------------------------------------------------\n');

    %% ================================================================
    %  SELECT MEDIAN RUN for convergence plots
    %  ================================================================
    [~, idx_cd]   = sort(final_err_cd);
    [~, idx_admm] = sort(final_err_admm);
    med_cd   = idx_cd(ceil(nruns/2));
    med_admm = idx_admm(ceil(nruns/2));

    err_cd_plot   = all_err_cd{med_cd};
    time_cd_plot  = all_time_cd{med_cd};
    err_admm_plot = all_err_admm{med_admm};
    time_admm_plot= all_time_admm{med_admm};

    %% ================================================================
    %  EXPORT .txt for tikz (iteration-based)
    %  ================================================================
    maxlen = max(length(err_cd_plot), length(err_admm_plot));
    y1 = [err_cd_plot;   err_cd_plot(end)   * ones(maxlen - length(err_cd_plot), 1)];
    y2 = [err_admm_plot; err_admm_plot(end) * ones(maxlen - length(err_admm_plot), 1)];

    iter_file = sprintf('figures/ADMMvsCD_iter_%s.txt', datasets{d});
    fid = fopen(iter_file, 'w');
    fprintf(fid, 'y1 y2\n');
    for i = 1:maxlen
        fprintf(fid, '%.6e %.6e\n', y1(i), y2(i));
    end
    fclose(fid);
    fprintf('  Saved: %s\n', iter_file);

    %% ================================================================
    %  EXPORT .txt for tikz (time-based)
    %  ================================================================
    time_file = sprintf('figures/ADMMvsCD_time_%s.txt', datasets{d});
    fid = fopen(time_file, 'w');
    fprintf(fid, 'x1 y1 x2 y2\n');
    maxlen_t = max(length(time_cd_plot), length(time_admm_plot));
    tc = [time_cd_plot;   time_cd_plot(end)  * ones(maxlen_t - length(time_cd_plot), 1)];
    ec = [err_cd_plot;    err_cd_plot(end)   * ones(maxlen_t - length(err_cd_plot), 1)];
    ta = [time_admm_plot; time_admm_plot(end)* ones(maxlen_t - length(time_admm_plot), 1)];
    ea = [err_admm_plot;  err_admm_plot(end) * ones(maxlen_t - length(err_admm_plot), 1)];
    for i = 1:maxlen_t
        fprintf(fid, '%.6e %.6e %.6e %.6e\n', tc(i), ec(i), ta(i), ea(i));
    end
    fclose(fid);
    fprintf('  Saved: %s\n', time_file);

    %% ================================================================
    %  EXPORT LaTeX table rows
    %  ================================================================
    table_file = sprintf('figures/table_ADMMvsCD_%s.txt', datasets{d});
    fid = fopen(table_file, 'w');
    fprintf(fid, '%% LaTeX table rows for %s (rank %d, %d runs)\n', datasets{d}, r, nruns);
    fprintf(fid, '%% Method & Final Rel. Error & Time (s) & Iterations \\\\\n');
    fprintf(fid, 'CD~\\cite{awari2024coordinate} & $%.4f \\pm %.4f$ & $%.2f \\pm %.2f$ & %d \\\\\n', ...
        mean(final_err_cd), std(final_err_cd), ...
        mean(final_time_cd), std(final_time_cd), length(all_err_cd{med_cd}));
    fprintf(fid, 'ADMM (proposed) & $\\mathbf{%.4f \\pm %.4f}$ & $\\mathbf{%.2f \\pm %.2f}$ & %d \\\\\n', ...
        mean(final_err_admm), std(final_err_admm), ...
        mean(final_time_admm), std(final_time_admm), length(all_err_admm{med_admm}));
    fclose(fid);
    fprintf('  Saved: %s\n', table_file);

    %% ================================================================
    %  MATLAB FIGURES (for quick visual check)
    %  ================================================================
    figure('Color','w','Name',sprintf('ADMM vs CD — %s', datasets{d}));

    % --- Left: Error vs Iteration ---
    subplot(1,2,1); hold on; grid on; box on;
    p1 = semilogy(err_cd_plot,   'r-s', 'LineWidth', 2, 'MarkerSize', 5, ...
        'MarkerIndices', 1:10:length(err_cd_plot));
    p2 = semilogy(err_admm_plot, 'b-^', 'LineWidth', 2, 'MarkerSize', 5, ...
        'MarkerIndices', 1:5:length(err_admm_plot));
    xlabel('Iteration','Interpreter','latex','FontSize',14);
    ylabel('Relative error','Interpreter','latex','FontSize',14);
    title(sprintf('%s — Error vs Iteration', datasets{d}),'Interpreter','latex','FontSize',16);
    legend([p1 p2], {'Coordinate Descent','ADMM'}, ...
        'Interpreter','latex','FontSize',12,'Location','northeast');
    set(gca,'FontSize',12,'LineWidth',1);

    % --- Right: Error vs Time ---
    subplot(1,2,2); hold on; grid on; box on;
    p3 = semilogy(time_cd_plot,   err_cd_plot,   'r-s', 'LineWidth', 2, 'MarkerSize', 5, ...
        'MarkerIndices', 1:10:length(err_cd_plot));
    p4 = semilogy(time_admm_plot, err_admm_plot, 'b-^', 'LineWidth', 2, 'MarkerSize', 5, ...
        'MarkerIndices', 1:5:length(err_admm_plot));
    xlabel('Time (s)','Interpreter','latex','FontSize',14);
    ylabel('Relative error','Interpreter','latex','FontSize',14);
    title(sprintf('%s — Error vs Time', datasets{d}),'Interpreter','latex','FontSize',16);
    legend([p3 p4], {'Coordinate Descent','ADMM'}, ...
        'Interpreter','latex','FontSize',12,'Location','northeast');
    set(gca,'FontSize',12,'LineWidth',1);

    % Save figure
    saveas(gcf, sprintf('figures/ADMMvsCD_%s.png', datasets{d}));
    fprintf('  Saved figure: figures/ADMMvsCD_%s.png\n', datasets{d});

end

fprintf('\n*** All experiments complete. ***\n');
fprintf('*** Initializations saved in figures/inits/ for PyTorch script. ***\n');
