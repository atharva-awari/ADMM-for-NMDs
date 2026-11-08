function r = evaluate_run(err, time, target, alpha, window)
%EVALUATE_RUN  Apply the SAME stopping rules to any method's trajectory.
%
%   r = evaluate_run(err, time, target, alpha, window)
%
%   err(1)   relative error at the initial point (iteration 0)
%   err(k+1) relative error after iteration k;  time aligned with err.
%
%   A run SUCCEEDS if its best-so-far error falls below TARGET before it
%   is declared stagnated. It STAGNATES at the first multiple of WINDOW
%   iterations at which the best-so-far error has not decreased by a
%   factor ALPHA over the previous WINDOW iterations (paper's rule:
%   alpha = 0.99, window = 10).
%
%   Why best-so-far: coordinate descent is monotone, so for CD this is
%   exactly the paper's rule. ADMM is not monotone; applying the test to
%   the current error would end ADMM runs during a transient rise that
%   could still have succeeded. The running minimum is monotone for any
%   method, so the rule means the same thing for both.
%
%   Times are shifted so that time(1) = 0 for every method.
%
%   Output fields: success, t_hit, it_hit (NaN if no success), stop_it.

err  = err(:);
time = time(:) - time(1);
best = cummin(err);
nIt  = numel(err) - 1;

stopIdx = numel(err);
for it = window:window:nIt
    if best(it + 1) > alpha * best(it + 1 - window)
        stopIdx = it + 1;
        break
    end
end

hit = find(best(1:stopIdx) < target, 1);
r.success = ~isempty(hit);
if r.success
    r.t_hit  = time(hit);
    r.it_hit = hit - 1;
else
    r.t_hit  = NaN;
    r.it_hit = NaN;
end
r.stop_it = stopIdx - 1;
end
