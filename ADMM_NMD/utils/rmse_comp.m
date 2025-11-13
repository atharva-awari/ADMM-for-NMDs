%% RMSE computation on test and train data
function [rmse_test, rmse_train, rmse_missing] = rmse_comp(X, A, model, param)
% X: ground truth or observed matrix to compare against (use B if real data)
% A: pre-activation estimate (e.g., WH)
% model: 'ReLU' | 'CSF' | 'MinMax' | 'Modulus'
% param.mask: logical/0-1 mask of observed entries (1=observed)
% param.validmask: logical/0-1 subset of mask used for validation

% ---- Safe defaults ----
if ~isfield(param,'mask') || isempty(param.mask)
    param.mask = true(size(X));             % full-data case
end
if ~isfield(param,'validmask') || isempty(param.validmask)
    param.validmask = param.mask;           % default: “validation = train set”
end

M_mask   = logical(param.mask);
M_valid  = logical(param.validmask);                      
if isequal(M_mask, M_valid)
    % No separate validation mask given
    M_train = M_mask;      % use all observed entries for training
    M_test  = M_mask;      % validation same as training
    %warning('validmask == mask: Using full observed set for both train and test RMSE.');
else
    M_train = M_mask & ~M_valid;
    M_test  = M_valid;
end
M_missing = ~M_mask;

% ---- Model nonlinearity ----
switch model
    case 'ReLU'
        Xhat = max(0, A);
    case 'CSF'
        Xhat = A.^2;
    case 'MinMax'
        a = param.a; b = param.b;
        Xhat = min(b, max(a, A));
    case 'Modulus'
        Xhat = abs(A);
    otherwise
        error('Unknown model: %s', model);
end

% RMSE helpers 
rmse_on = @(M) sqrt( sum( sum( (M .* (X - Xhat).^2) ) ) / max(1, nnz(M)) );

rmse_train   = rmse_on(M_train);
rmse_test    = rmse_on(M_test);
rmse_missing = rmse_on(M_missing);          % for synthetic experiments
end
