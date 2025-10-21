close all; clear; clc;
addpath('./utils');

%% ===== Experiment grid =====
models         = {'ReLU','CSF','MinMax','Modulus'};
error_measures = {'Frobenius','L1','KL divergence'};
update_methods = {'gradient descent','svd','QR','closed form'};

num_runs = 10;                     % average over 10 runs

%% ===== Problem size & base params =====
m = 200; n = 100; r = 30;

base_param                  = struct();
base_param.maxiter          = inf;
base_param.maxtime          = 10;
base_param.updateWH         = 'gradient descent';  % overwritten in loop
base_param.a                = eps;                 % MinMax interval
base_param.b                = 1;
base_param.display          =0;

rng(42);  % global reproducibility

%% ===== Main loops =====
for im = 1:numel(models)
    model = models{im};

    for ie = 1:numel(error_measures)
        err_name = error_measures{ie};

        % Accumulate series for each method over runs (each cell holds a vector per run)
        obj_runs = cell(1, numel(update_methods));
        res_runs = cell(1, numel(update_methods));
        for iu = 1:numel(update_methods)
            obj_runs{iu} = cell(num_runs,1);
            res_runs{iu} = cell(num_runs,1);
        end

        run_seeds = 1:num_runs;  % same datasets across methods within a run

        for run = 1:num_runs
            rng(run_seeds(run));  % fix dataset for this run

            % Generate synthetic data once per run
            gen_param = base_param;
            if strcmpi(model,'MinMax')
                gen_param.a = base_param.a;
                gen_param.b = base_param.b;
            end
            X = generate_synthetic(m, n, r, model, gen_param);

            % Evaluate each update method on the SAME X
            for iu = 1:numel(update_methods)
                method        = update_methods{iu};
                param         = base_param;
                param.updateWH = method;

                results = admm_nmd(X, r, model, err_name, param);

                % Robustly grab vectors (force column)
                obj = pull_vec(results, 'objective');
                res = pull_vec(results, 'residual');

                obj_runs{iu}{run} = obj;
                res_runs{iu}{run} = res;
            end
        end

        %% ===== Aggregate (mean over runs at each iteration index) =====
        [x_obj, mean_obj] = aggregate_series(obj_runs);
        [x_res, mean_res] = aggregate_series(res_runs);

        %% ===== Plot: two panels side-by-side comparing methods =====
        f = figure('Color','w', 'Name', sprintf('%s + %s (mean over %d runs)', model, err_name, num_runs), ...
                   'Position',[100 100 1200 450]);
        tl = tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

        % --- Left: Objective
        nexttile;
        hold on;
        for iu = 1:numel(update_methods)
            y = mean_obj{iu};
            if all(y > 0)
               semilogy(x_obj{iu}, y, 'LineWidth', 1.5);
            else
                plot(x_obj{iu}, y, 'LineWidth', 1.5);
            end
        end
        grid on;
        xlabel('Iteration'); ylabel('Objective (mean over runs)');
        title('Objective vs. iteration');
        legend(update_methods, 'Location','best'); legend boxoff;

        % --- Right: Residual
        nexttile;
        hold on;
        for iu = 1:numel(update_methods)
            y = mean_res{iu};
           if all(y > 0)
               semilogy(x_res{iu}, y, 'LineWidth', 1.5);
            else
                plot(x_res{iu}, y, 'LineWidth', 1.5);
            end
        end
        grid on;
        xlabel('Iteration'); ylabel('Residual (mean over runs)');
        title('Residual vs. iteration');
        legend(update_methods, 'Location','best'); legend boxoff;

        sgtitle(sprintf('%s + %s — comparison of updateWH (N=%d runs)', model, err_name, num_runs), 'FontWeight','bold');

        % Optional: save the figure
        % outname = sprintf('comp_%s_%s.png', lower(model), regexprep(lower(err_name),'\s+','_'));
        % exportgraphics(f, outname, 'Resolution', 200);
    end
end

%% ===== Helpers =====

function v = pull_vec(results, fieldname)
% Extract a vector field safely and return a column vector.
    v = [];
    if isstruct(results) && isfield(results, fieldname) && ~isempty(results.(fieldname))
        v = results.(fieldname);
    end
    if isempty(v)
        v = NaN;   % ensure something is plotted/aggregated
    end
    v = v(:);     % column
end

function [x_list, mean_list] = aggregate_series(per_method_runs)
% per_method_runs: 1xM cell, each cell is {num_runs x 1} of vectors (different lengths allowed)
% Returns:
%   x_list   : 1xM cell with x = 1:L_max (per method)
%   mean_list: 1xM cell with nanmean over runs at each index (NaN-padded)
    M = numel(per_method_runs);
    x_list    = cell(1,M);
    mean_list = cell(1,M);
    for m = 1:M
        runs = per_method_runs{m};     % cell of vectors
        Ls   = cellfun(@numel, runs);
        Lmax = max(Ls);
        A    = NaN(Lmax, numel(runs));
        for k = 1:numel(runs)
            v = runs{k};
            L = numel(v);
            A(1:L, k) = v;            % NaN-pad the rest
        end
        mean_list{m} = nanmean(A, 2);  % mean over runs at each iteration index
        x_list{m}    = (1:Lmax).';
    end
end
