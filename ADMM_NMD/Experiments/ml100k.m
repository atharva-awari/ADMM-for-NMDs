clear; clc;

% Load MovieLens-100k triplets [user item rating timestamp]
data = importdata('Datasets/ml-100k/u.data','\t');
data = data(:,1:3);                 % keep [u,i,r]

nr  = size(data,1);
ntr = round(0.8 * nr);              % 80% train

% random split (without replacement)
idx_tr = randsample(nr, ntr, false);
idx_te = setdiff(1:nr, idx_tr)';

dataTr = data(idx_tr,:);
dataTe = data(idx_te,:);

m = max(data(:,1));                  % #users
n = max(data(:,2));                  % #items

% Build dense matrices for train and test
X_tr  = zeros(m,n);
X_te  = zeros(m,n);

for k = 1:size(dataTr,1)
    X_tr(dataTr(k,1), dataTr(k,2)) = dataTr(k,3);
end
for k = 1:size(dataTe,1)
    X_te(dataTe(k,1), dataTe(k,2)) = dataTe(k,3);
end

% Masks
M_tr  = X_tr ~= 0;                   % 1 where training ratings known
M_te  = X_te ~= 0;                   % 1 where test ratings known (disjoint from M_tr)

% Reference matrix of all known entries (train OR test)
%   - use X_ref to compare predictions against the true ratings wherever available
X_ref = zeros(m,n);
X_ref(M_tr) = X_tr(M_tr);
X_ref(M_te) = X_te(M_te);


% ADMM-NMD settings
r = 5;                              % choose a rank
model = 'MinMax';                      % or 'CSF'/'MinMax'/'Modulus'
error_measure = 'Frobenius';


param = struct();
param.mask       = M_tr;             % <-- training mask (observed entries)
param.validmask  = M_te;             % <-- validation mask (held-out test entries) (DISJOINT)
param.display    = 0;
param.maxiter    = inf;
param.maxtime    = 60;
param.updateWH   = 'closed form'; % or 'svd'/'gradient descent'/'QR'
param.a=-1000;param.b=1000;
% Run
results = admm_nmd(X_tr, r, model, error_measure, param);
A = results.W * results.H;        % predictions (pre-nonlinearity)

metrics = mc_metrics([], X_ref, A, model, param);
fprintf('MovieLens100k  |  RMSE_train = %.4f   RMSE_test = %.4f\n', ...
        metrics.rmse_train, metrics.rmse_validation);
