function [U, V, out] = csf_cd(M, U, V, opts)
%CSF_CD  Coordinate descent for the component-wise squared factorization
%
%   [U, V, out] = csf_cd(M, U0, V0, opts)
%
%   Solves   min_{U,V}  || M - (U*V).^2 ||_F^2
%   with U in R^{m x r}, V in R^{r x n}.
%
%   MATLAB port of the coordinate descent (CD) algorithm of
%     J. Lefebvre, A. Vandaele, N. Gillis, "Component-wise squared
%     factorization", IEEE MLSP 2024,
%   following the reference Julia implementation at
%     https://github.com/Riipou/CSF  (Julia/algorithms/CD.jl).
%
%   ---------------------------------------------------------------
%   FIDELITY NOTES -- where this port follows the code, not the paper
%   ---------------------------------------------------------------
%   1. RESTART. The paper's Algorithm 3 describes the beta update but
%      does not mention a restart. The reference code does restart:
%      if an extrapolated step increases the error, the step is
%      discarded and a plain (non-extrapolated) CD update is redone
%      from the previous iterate. This port does the same. Omitting
%      the restart gives a noticeably weaker baseline.
%   2. beta_bar is initialised to 1 once, before the loop.
%   3. Iterates are identical to the reference, but the coefficient
%      sums are vectorised across columns. For fixed U the columns of
%      V are independent, so updating coordinate p of every column at
%      once is exactly equivalent to the reference column-by-column
%      loop. This matters for a fair timing comparison in MATLAB,
%      where scalar loops would penalise CD artificially.
%   4. The previous-iterate error is cached rather than recomputed;
%      it is provably equal to the error at the end of the previous
%      iteration, so iterates are unchanged and CD only runs faster.
%   5. If a column of the fixed factor is identically zero, the
%      reference code divides 0/0 and produces NaN. Here the variable
%      simply keeps its value, since it does not affect the objective.
%
%   ---------------------------------------------------------------
%   OPTIONS  (all optional)
%   ---------------------------------------------------------------
%     opts.maxiter     maximum iterations                [inf]
%     opts.maxtime     time limit in seconds             [inf]
%     opts.extrapolate use the extrapolated variant      [true]
%     opts.alpha       stop if, every 10 iterations, the error has
%                      not decreased by a factor alpha   [inf = off]
%                      (paper uses 0.99 or 0.9999 for exact
%                      factorisation tests)
%     opts.beta1       initial extrapolation parameter   [0.3]
%     opts.eta         beta decrease factor              [1.5]
%     opts.gamma       beta increase factor              [1.05]
%     opts.gammabar    beta_bar increase factor          [1.01]
%     opts.display     print progress                    [false]
%
%   OUTPUT
%     out.err   relative error ||M-(UV).^2||_F/||M||_F per iteration,
%               out.err(1) being the initial error
%     out.time  elapsed time in seconds, aligned with out.err
%     out.iter  number of iterations performed
%
%   See also CSF_INIT.

if nargin < 4, opts = struct(); end
maxiter     = getopt(opts, 'maxiter',     inf);
maxtime     = getopt(opts, 'maxtime',     inf);
extrapolate = getopt(opts, 'extrapolate', true);
alpha       = getopt(opts, 'alpha',       inf);
beta1       = getopt(opts, 'beta1',       0.3);
eta         = getopt(opts, 'eta',         1.5);
gamma       = getopt(opts, 'gamma',       1.05);
gammabar    = getopt(opts, 'gammabar',    1.01);
display     = getopt(opts, 'display',     false);

if ~isfinite(maxiter) && ~isfinite(maxtime) && ~(alpha <= 1)
    error('csf_cd:noStop', ...
        'Set at least one of opts.maxiter, opts.maxtime or opts.alpha.');
end

normM   = norm(M, 'fro');
errfun  = @(U, V) norm(M - (U*V).^2, 'fro');

e0       = errfun(U, V);
out.err  = e0 / normM;
out.time = 0;
prev10   = e0;            % reference for the alpha stopping test

t0 = tic;   % private timer handle: a bare tic would reset the global
            % timer and corrupt any timing the caller is doing
it = 0;

if extrapolate
    % State as in the reference code:
    %   beta     <- beta_bis   (current extrapolation parameter)
    %   betabar  <- beta       (cap on beta)
    %   betaprev <- beta_pred  (previous beta)
    Up = U;  Vp = V;  Utemp = U;  Vtemp = V;
    beta     = beta1;
    betabar  = 1;
    betaprev = beta1;
    errprev  = e0;        % cached error of the previous iterate

    while it < maxiter
        it = it + 1;

        V  = update_V(M,  U,  V + beta*(V - Vp));
        Vp = Vtemp;
        U  = update_V(M', V', (U + beta*(U - Up))')';
        Up = Utemp;
        Utemp = U;  Vtemp = V;

        err = errfun(U, V);

        if err > errprev
            % restart: discard the extrapolated step and redo a plain
            % CD update from the previous iterate (Up, Vp)
            V = update_V(M,  Up, Vp);
            U = update_V(M', V', Up')';
            Utemp = U;  Vtemp = V;
            err = errfun(U, V);
            bcopy    = beta;
            beta     = beta / eta;
            betabar  = betaprev;
            betaprev = bcopy;
        else
            betaprev = beta;
            beta     = min(betabar, gamma*beta);
            betabar  = min(1, gammabar*betabar);
        end
        errprev = err;

        stop = record_and_check(err);
        if stop, break; end
    end
else
    while it < maxiter
        it = it + 1;
        V = update_V(M,  U,  V);
        U = update_V(M', V', U')';
        err = errfun(U, V);
        stop = record_and_check(err);
        if stop, break; end
    end
end

out.iter = it;

    % -----------------------------------------------------------------
    function stop = record_and_check(err)
        % nested: shares out, it, prev10, normM, ... with csf_cd
        out.err(end+1)  = err / normM;   
        out.time(end+1) = toc(t0);     
        stop = false;
        if display && (it == 1 || mod(it, 10) == 0)
            fprintf('  it %5d   rel err %.6e   t %.2fs\n', ...
                it, out.err(end), out.time(end));
        end
        if mod(it, 10) == 0 && alpha <= 1
            if err > alpha * prev10
                stop = true;
            end
            prev10 = err;
        end
        if out.time(end) >= maxtime
            stop = true;
        end
    end
end

% =====================================================================
function V = update_V(M, U, V)
%UPDATE_V  One CD pass over all entries of V with U fixed.
%   Columns of V are independent given U, so coordinate p is updated for
%   every column at once; within each column, coordinates p = 1..r are
%   still visited in order with the residual refreshed after each one,
%   exactly as in the reference implementation.
r = size(U, 2);
R = U * V;                         % current product U*V
for p = 1:r
    a  = U(:, p);
    Rm = R - a * V(p, :);          % product without coordinate p  (d)
    a2 = a.^2;  a3 = a2 .* a;  a4 = a2.^2;
    % derivative of the 1-D quartic: c3 x^3 + c2 x^2 + c1 x + c0
    c3 = 4  * sum(a4);                               % scalar
    c2 = 12 * (a3' * Rm);                            % 1 x n
    c1 = 4  * (3 * (a2' * Rm.^2) - a2' * M);         % 1 x n
    c0 = 4  * (a' * Rm.^3 - a' * (Rm .* M));         % 1 x n
    V(p, :) = cubic_argmin(c3, c2, c1, c0, V(p, :));
    R = Rm + a * V(p, :);
end
end

% =====================================================================
function x = cubic_argmin(a, b, c, d, xold)
%CUBIC_ARGMIN  Minimise g(x) = (a/4)x^4 + (b/3)x^3 + (c/2)x^2 + d x
%   entrywise, by finding the real roots of g'(x) = a x^3 + b x^2 + c x + d
%   (Cardano / trigonometric method) and keeping the one with smallest g.
%   a is a scalar >= 0; b, c, d are row vectors.
if a <= realmin
    x = xold;                      % zero column: variable is irrelevant
    return
end
n = numel(b);
B = b / a;  C = c / a;  D = d / a;
p = C - B.^2 / 3;                  % depressed cubic t^3 + p t + q = 0
q = 2 * B.^3 / 27 - B .* C / 3 + D;
shift = -B / 3;                    % x = t + shift
disc = -(4 * p.^3 + 27 * q.^2);    % > 0: three real roots

T = nan(3, n);

% one real root (Cardano); also covers the double-root case disc == 0,
% returning the simple root, which is the actual extremum of g
one = ~(disc > 0);
if any(one)
    s = sqrt(-disc(one) / 108);    % = sqrt(q^2/4 + p^3/27)
    u = -q(one) / 2 + s;
    v = -q(one) / 2 - s;
    T(1, one) = realcbrt(u) + realcbrt(v);
end

% three real roots (trigonometric method)
three = disc > 0;
if any(three)
    pp  = p(three);  qq = q(three);
    arg = (-qq / 2) .* sqrt(-27 ./ pp.^3);
    arg = min(max(arg, -1), 1);    % guard against rounding outside [-1,1]
    phi = acos(arg);
    m   = 2 * sqrt(-pp / 3);
    T(1, three) = m .* cos( phi            / 3);
    T(2, three) = m .* cos((phi + 2 * pi) / 3);
    T(3, three) = m .* cos((phi + 4 * pi) / 3);
end

X = T + shift;                     % candidate minimisers, 3 x n
G = (a / 4) * X.^4 + (b / 3) .* X.^3 + (c / 2) .* X.^2 + d .* X;
G(isnan(G)) = inf;
[gmin, idx] = min(G, [], 1);
x = X(sub2ind(size(X), idx, 1:n));
bad = ~isfinite(gmin);             % no valid candidate: keep old value
x(bad) = xold(bad);
end

% =====================================================================
function y = realcbrt(x)
%REALCBRT  Real cube root, correct for negative arguments.
y = sign(x) .* abs(x).^(1/3);
end

% =====================================================================
function v = getopt(s, f, default)
if isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = default; end
end
