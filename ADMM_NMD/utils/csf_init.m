function [U, V] = csf_init(M, r, choice)
%CSF_INIT  Initialisation for CSF, matching the reference implementation
%
%   [U, V] = csf_init(M, r, 'random')
%   [U, V] = csf_init(M, r, 'svd')
%
%   Port of init_matrix in Julia/algorithms/functions.jl of
%   https://github.com/Riipou/CSF  (Lefebvre, Vandaele, Gillis 2024).
%
%   'random'  U, V drawn from N(0,1), then scaled by lambda^(1/4) with
%             lambda = <(UV).^2, M> / <(UV).^2, (UV).^2>,
%             the optimal scaling of (UV).^2 towards M (eq. (8) of the
%             paper). As in the reference, if lambda < 0 the draw is
%             repeated from U(0,1); this cannot happen for nonnegative M.
%
%   'svd'     U = U_r*sqrt(S_r), V = sqrt(S_r)*V_r' from the truncated
%             SVD of M itself.
%             NOTE: this is the SVD of M, not of sqrt(M), so (UV).^2
%             initially approximates M.^2 rather than M. This follows
%             the reference code and the paper faithfully. It may help
%             explain why the paper reports random initialisation
%             outperforming SVD initialisation (Table 3).
%
%   IMPORTANT for fair comparisons: CSF is very sensitive to the
%   starting point, so pass the SAME (U, V) to every method being
%   compared, including the ADMM framework.

[m, n] = size(M);
switch lower(choice)
    case 'random'
        U = randn(m, r);  V = randn(r, n);
        A = (U * V).^2;
        lambda = sum(A(:) .* M(:)) / sum(A(:).^2);
        if lambda < 0
            U = rand(m, r);  V = rand(r, n);
            A = (U * V).^2;
            lambda = sum(A(:) .* M(:)) / sum(A(:).^2);
        end
        U = U * lambda^(1/4);
        V = V * lambda^(1/4);
    case 'svd'
        [Us, S, Vs] = svd(M, 'econ');
        s = sqrt(diag(S(1:r, 1:r)));
        U = Us(:, 1:r) .* s';
        V = s .* Vs(:, 1:r)';
    otherwise
        error('csf_init:choice', 'choice must be ''random'' or ''svd''.');
end
end
