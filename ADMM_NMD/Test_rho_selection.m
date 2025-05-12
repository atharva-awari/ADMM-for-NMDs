close all; clear all; clc;

models = {'ReLU', 'CSF', 'MinMax', 'Modulus'};
errors = { 'KL divergence'};


rhos = [0.5, 1, 5, 10, 100, 500];
ranks = [15, 30];
num_seeds = 10;
maxiter = 1000;
maxtime = 10;
%Select interval if you're using MinMax model [a,b]
options.a = eps;
options.b = 1;

% Preallocate: iterations x model x error x rho x rank
avg_errors = zeros(maxiter, length(models), length(errors), length(rhos), length(ranks));

for r = 1:length(ranks)
    rank_val = ranks(r);
    for m = 1:length(models)
        model = models{m};
        for e = 1:length(errors)
            error_measure = errors{e};
            for rho_idx = 1:length(rhos)
                rho = rhos(rho_idx);
                total_error = zeros(maxiter, num_seeds);

                for seed = 1:num_seeds
                    rng(seed); % consistent init
                    W = randn(100, rank_val); % dimensions can be changed
                    H = randn(rank_val, 70);
                    WH = W * H;

                    % Generate X based on model
                    switch model
                        case 'ReLU'
                            X = max(0, WH);
                        case 'CSF'
                            X = WH .^ 2;
                        case 'MinMax'
                            X = min(options.b, max(options.a, WH));
                        case 'Modulus'
                            X = abs(WH);
                        otherwise
                            error('Unknown model');
                    end


%Initialising Theta and W,H using SVD(X)
[U,S,V] = svds(X,rank_val);
Theta = U*S*V';

W = U;
H = S*V';
 %Theta = randn(100,70);
f = find(X>0);
Theta(f) = max(eps,Theta(f));
 %W = randn(100,rank_val);
 %H = randn(rank_val,70);

%Lagrange Multiplier
lambda = randn(100,70);

                    % Run ADMM and log residuals across iterations
                    [Theta, W, H, residual, objective] = admm_nmd(X, rho, maxiter, maxtime, W, H, Theta, lambda, model, error_measure, options);

                    total_error(:, seed) = objective;
                end

                % Average residual across seeds
                avg_errors(:, m, e, rho_idx, r) = mean(total_error, 2);
            end
        end
    end
end


%% Plotting 

for m = 1:length(models)
    for e = 1:length(errors)
        figure;
        for r = 1:length(ranks)
            subplot(1, 2, r);
            hold on;
            for rho_idx = 1:length(rhos)
                plot(1:maxiter, avg_errors(:, m, e, rho_idx, r), 'LineWidth', 1.5);
            end
            title([models{m} ' + ' errors{e} ' (Rank=' num2str(ranks(r)) ')']);
            xlabel('Iteration');
            ylabel('Objective');
            legend(arrayfun(@(r) ['\rho = ' num2str(r)], rhos, 'UniformOutput', false));
            grid on;
        end
    end
end
