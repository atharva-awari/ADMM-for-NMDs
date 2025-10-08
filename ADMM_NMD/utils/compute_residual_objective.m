function [residual, objective] = compute_residual_objective(X, A, T, model, error_measure, param)
switch model

    case 'ReLU'
        switch error_measure
            case 'Frobenius'
                normX = norm(X , 'fro');
                residual = norm(T - A, 'fro') / norm(T,'fro');
                objective = norm(X - max(0, A), 'fro') / normX;
            case 'L1'
                X_approx = max(0, A);
                residual = norm(T - A, 'fro') / norm(T,'fro');
                objective = sum(abs(X(:) - X_approx(:))) / sum(abs(X(:)));
            case 'KL divergence'
                X_approx = max(0, A);
                % Compute mean of all elements in X
                mu_X = mean(X(:));
                % Get the size of X
                [m, n] = size(X);
                % Construct the estimation matrix X_est
                em = ones(m,1); % Column vector of ones
                en = ones(n,1); % Row vector of ones
                X_est = mu_X * (em * en');
                % Theta = max(eps,Theta);
                X(X == 0) = eps;
                X_approx(X_approx == 0) = eps;
                % Compute KL divergence element-wise
                kl_matrix = (X .* log(X ./ X_approx) - X + X_approx );
                kl_est = (X .* log(X ./ X_est) - X + X_est);
                % Sum over all elements to get total KL divergence
                objective = sum(kl_matrix(:)) / sum(kl_est(:));
                % Compute the residual betweeen Theta and A=WH
                residual = norm(T - A, 'fro') / norm(T,'fro');
        end
    case 'CSF'
        switch error_measure
            case 'Frobenius'
                normX = norm(X , 'fro');
                residual = norm(T - A, 'fro') / norm(T,'fro');
                objective = norm(X - (A).^2, 'fro') / normX;
            case 'L1'
                X_approx = (A).^2;
                residual = norm(T - A, 'fro') / norm(T,'fro');
                objective = sum(abs(X(:) - X_approx(:))) / sum(abs(X(:)));
            case 'KL divergence'
                X_approx = (A).^2;
                % Compute mean of all elements in X
                mu_X = mean(X(:));
                % Get the size of X
                [m, n] = size(X);
                % Construct the estimation matrix X_est
                em = ones(m,1); % Column vector of ones
                en = ones(n,1); % Row vector of ones
                X_est = mu_X * (em * en');
                %Theta = max(eps,Theta);
                X(X == 0) = eps;
                X_approx(X_approx == 0) = eps;
                % Compute KL divergence element-wise
                kl_matrix = (X .* log(X ./ X_approx) - X + X_approx );
                kl_est = (X .* log(X ./ X_est) - X + X_est);
                % Sum over all elements to get total KL divergence
                objective = sum(kl_matrix(:)) / sum(kl_est(:));
                residual = norm(T - A, 'fro') / norm(T,'fro');
        end
    case 'MinMax'
        a = param.a;
        b = param.b;
        switch error_measure
            case 'Frobenius'
                normX = norm(X , 'fro');
                residual = norm(T - A, 'fro') / norm(T,'fro');
                objective = norm(X - min(b, max(a, A)), 'fro') / normX;
            case 'L1'
                X_approx = min(b, max(a, A));
                residual = norm(T - A, 'fro') / norm(T,'fro');
                objective = sum(abs(X(:) - X_approx(:))) / sum(abs(X(:)));
            case 'KL divergence'
                X_approx = min(b, max(a, A));
                % Compute mean of all elements in X
                mu_X = mean(X(:));
                X(X == 0) = eps;
                X_approx(X_approx == 0) = eps;
                % Compute KL divergence element-wise
                kl_matrix = (X .* log(X ./ X_approx) - X + X_approx );
                kl_est = (X .* log(X / mu_X) - X + mu_X);
                % Sum over all elements to get total KL divergence
                objective = sum(kl_matrix(:)) / sum(kl_est(:));
                residual = norm(T - A, 'fro') / norm(T,'fro');
        end
    case 'Modulus'
        switch error_measure
            case 'Frobenius'
                normX = norm(X , 'fro');
                residual = norm(T - A, 'fro') / norm(T,'fro');
                objective = norm(X - abs(A), 'fro') / normX;
            case 'L1'
                X_approx = abs(A);
                residual = norm(T - A, 'fro') / norm(T,'fro');
                objective = sum(abs(X(:) - X_approx(:))) / sum(abs(X(:)));
            case 'KL divergence'
                X_approx = abs(A);
                % Compute mean of all elements in X
                mu_X = mean(X(:));
                % Get the size of X
                [m, n] = size(X);
                % Construct the estimation matrix X_est
                em = ones(m,1); % Column vector of ones
                en = ones(n,1); % Row vector of ones
                X_est = mu_X * (em * en');
                T = max(eps,T);
                X(X == 0) = eps;
                X_approx(X_approx == 0) = eps;
                % Compute KL divergence element-wise
                kl_matrix = (X .* log(X ./ X_approx) - X + X_approx );
                kl_est = (X .* log(X ./ X_est) - X + X_est);
                % Sum over all elements to get total KL divergence
                objective = sum(kl_matrix(:)) / sum(kl_est(:));
                residual = norm(T - A, 'fro') / norm(T,'fro');
        end

end
