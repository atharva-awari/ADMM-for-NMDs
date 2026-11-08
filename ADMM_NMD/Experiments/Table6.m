%RUN_LEDM_COMPARISON  ADMM vs coordinate descent (CD) for CSF on linear
% Euclidean distance matrices M(i,j) = (i-j)^2, whose square root rank is 2.
%
% Replicates the structure of Table 7 of Lefebvre, Vandaele & Gillis
% (MLSP 2024), comparing the success rate of ADMM and CD from IDENTICAL
% starting points, and records the time each method needs to reach the
% target relative error.
%
% PROTOCOL (identical for both methods)
%   * starting point: csf_init(M, r, 'random')  -- randn with optimal
%     scaling, as in the paper; ADMM receives W0, H0 and T0 = W0*H0
%   * success: best-so-far relative error ||M-(WH).^2||_F/||M||_F < TARGET
%   * stagnation: best-so-far error not reduced by factor ALPHA over
%     WINDOW iterations (paper: 0.99 over 10), see evaluate_run.m
%   * caps: MAXITER iterations and MAXTIME seconds per run per method
%   * CD stops live on target/stagnation; ADMM runs to its cap and the
%     same rules are applied afterwards to its recorded trajectory, so
%     that admm_nmd.m does not need to be modified. Decisions are
%     identical; only the experiment's wall-clock time is longer.
%
% OUTPUTS (written to OUTDIR)
%   ledm_results.mat     all per-run results (resumable, see below)
%   ledm_table.tex       success rates and median time to target
%   ledm_paired.tex      paired outcome counts at r = 2
%
% RESUMING: results are saved after every (n, r) configuration. If the
% script is interrupted, rerun it and finished configurations are skipped.
%
% REQUIRES on the path: admm_nmd.m (your current version) and its
% dependencies, update_T_new.m, cardanoatharva_vec.m, csf_cd.m,
% csf_init.m, evaluate_run.m.
%
% NOTE on the CSF convention: ADMM uses whichever update_T_new.m is on
% the path (Option A or B). Decide before running; the two are not
% interchangeable mid-experiment.

clear; clc;

%% ------------------------------------------------------------- settings
sizes   = 10:-1:3;        % matrix sizes n (paper: 3..10)
ranks   = [ 2 ];        % r = 1 must fail: it fixes the rank at exactly 2
N       = 100;            % runs per (n, r)
target  = 1e-4;           % success threshold on the relative error
alpha   = 0.99;           % stagnation factor        (paper: 0.99)
window  = 10;             % stagnation window, its   (paper: 10)
maxiter = 1e4;            % iteration cap per run    (paper: 1e4)
maxtime = 2;              % time cap per run per method, seconds
plotR   = 2;              % rank for the paired table and saved trajectories
nkeep   = 300;            % trajectory points kept per run (for plotting)
outdir  = 'ledm_results';

if ~exist(outdir, 'dir'), mkdir(outdir); end
matfile = fullfile(outdir, 'ledm_results.mat');

%% --------------------------------------------------------- load / resume
if exist(matfile, 'file')
    S = load(matfile);
    cur  = [N, target, alpha, window, maxiter, maxtime];
    prev = [S.N, S.target, S.alpha, S.window, S.maxiter, S.maxtime];
    if ~isequal(cur, prev)
        error(['Existing results in %s were produced with different settings.\n' ...
               '  saved   [N target alpha window maxiter maxtime] = [%s]\n' ...
               '  current [N target alpha window maxiter maxtime] = [%s]\n' ...
               'Delete the folder "%s" to start afresh, or restore the saved settings.'], ...
               matfile, num2str(prev), num2str(cur), outdir);
    end
    R = S.R;
    fprintf('Resuming: %d configurations already done with identical settings.\n', numel(R));
else
    R = struct('n', {}, 'r', {}, 'cd', {}, 'admm', {}, 'trajCD', {}, 'trajAD', {});
end
isdone = @(n, r) any(arrayfun(@(s) s.n == n && s.r == r, R));

o_cd = struct('extrapolate', true, 'maxiter', maxiter, 'maxtime', maxtime, ...
              'alpha', alpha, 'target', target);

%% ------------------------------------------------------------- main loop
for n = sizes
    [I, J] = meshgrid(1:n, 1:n);
    M = (I - J).^2;
    for r = ranks
        if isdone(n, r), continue; end
        tcfg = tic;
        cdr = repmat(struct('success', false, 't_hit', NaN, 'it_hit', NaN, 'stop_it', NaN), N, 1);
        adr = cdr;
        trajCD = cell(N, 1);  trajAD = cell(N, 1);

        for run = 1:N
            % identical starting point for both methods
            rng(100000 + 1000*n + 10*r + run);
            [W0, H0] = csf_init(M, r, 'random');

            % ---- CD (stops live on target or stagnation) ----
            [~, ~, out] = csf_cd(M, W0, H0, o_cd);
            cdr(run) = evaluate_run(out.err, out.time, target, alpha, window);

            % ---- ADMM (runs to its cap; rules applied afterwards) ----
            p = struct('W0', W0, 'H0', H0, 'T0', W0*H0, ...
                       'maxiter', maxiter + 1, ...   % admm_nmd does maxiter-1 its
                       'maxtime', maxtime, 'display', 0);
            res = admm_nmd(M, r, 'CSF', 'Frobenius', p);
            adr(run) = evaluate_run(res.objective, res.time, target, alpha, window);

            if r == plotR
                trajCD{run} = thin(out.err, out.time, cdr(run).stop_it, nkeep);
                trajAD{run} = thin(res.objective, res.time, adr(run).stop_it, nkeep);
            end
        end

        R(end+1) = struct('n', n, 'r', r, 'cd', cdr, 'admm', adr, ...
                          'trajCD', {trajCD}, 'trajAD', {trajAD}); %#ok<SAGROW>
        save(matfile, 'R', 'sizes', 'ranks', 'N', 'target', 'alpha', ...
             'window', 'maxiter', 'maxtime');
        fprintf('n = %2d, r = %d:  CD %5.1f%%   ADMM %5.1f%%   (%.0fs)\n', ...
            n, r, 100*mean([cdr.success]), 100*mean([adr.success]), toc(tcfg));
    end
end

%% ----------------------------------------------------------- summarise
fprintf('\n%-9s %-3s %5s %10s %10s %14s %14s\n', 'Size', 'r', 'runs', 'CD', 'ADMM', ...
        'med t CD (ms)', 'med t ADMM (ms)');
fid = fopen(fullfile(outdir, 'ledm_table.tex'), 'w');
fprintf(fid, '%% Matrix size & r & CD & ADMM & median time to target, CD & ADMM (ms)\n');
for n = sizes
    for r = ranks
        s  = R(arrayfun(@(x) x.n == n && x.r == r, R));
        sc = 100*mean([s.cd.success]);   sa = 100*mean([s.admm.success]);
        nr = numel(s.cd);                                % actual runs stored
        mc = 1e3*median_ok([s.cd.t_hit]);  ma = 1e3*median_ok([s.admm.t_hit]);
        fprintf('%2d x %-4d %-3d %5d %9.1f%% %9.1f%% %14s %14s\n', n, n, r, nr, sc, sa, ...
                fmt(mc), fmt(ma));
        fprintf(fid, '$%d\\times %d$ & %d & %.1f\\%% & %.1f\\%% & %s & %s \\\\\n', ...
                n, n, r, sc, sa, fmt(mc), fmt(ma));
    end
end
fclose(fid);

% paired outcomes at r = plotR: same starting points, so tabulate jointly
fprintf('\nPaired outcomes at r = %d (same starting points for both methods):\n', plotR);
fprintf('%-9s %6s %10s %8s %8s\n', 'Size', 'both', 'ADMM only', 'CD only', 'neither');
fid = fopen(fullfile(outdir, 'ledm_paired.tex'), 'w');
fprintf(fid, '%% Matrix size & both & ADMM only & CD only & neither\n');
for n = sizes
    s  = R(arrayfun(@(x) x.n == n && x.r == plotR, R));
    c  = [s.cd.success];  a = [s.admm.success];
    k  = [sum(a & c), sum(a & ~c), sum(~a & c), sum(~a & ~c)];
    fprintf('%2d x %-4d %6d %10d %8d %8d\n', n, n, k);
    fprintf(fid, '$%d\\times %d$ & %d & %d & %d & %d \\\\\n', n, n, k);
end
fclose(fid);

fprintf('\nDone. Outputs in %s/\n', outdir);

%% ============================================================ helpers
function T = thin(err, time, stop_it, nkeep)
% keep the trajectory up to the stopping point, log-spaced to <= nkeep pts
err = err(:);  time = time(:) - time(1);
last = min(numel(err), stop_it + 1);
idx  = unique(round(logspace(0, log10(last), nkeep)));
T = [time(idx), err(idx)];
end

function m = median_ok(t)
t = t(~isnan(t));
if isempty(t), m = NaN; else, m = median(t); end
end

function s = fmt(x)
if isnan(x), s = '--'; else, s = sprintf('%.2f', x); end
end