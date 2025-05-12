function [residual, objective] = compute_residual_objective(X, W, H, Theta, model, error_measure, options)

switch model

    case 'ReLU'

        switch error_measure

            case 'Frobenius'

                normX = norm(X , 'fro');
                residual = norm(Theta - W*H, 'fro') / norm(Theta,'fro');
                objective = norm(X - max(0, W*H), 'fro') / normX;

            case 'L1'

                X_approx = max(0, W * H);
              A =W*H;
             residual = sum(abs(Theta(:) - A(:))) / sum(abs(Theta(:)));
             objective = sum(abs(X(:) - X_approx(:))) / sum(abs(X(:)));

            case 'KL divergence'

                X_approx = max(0, W * H);
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

              % Compute the residual betweeen Theta and WH
              residual = norm(Theta - W*H, 'fro') / norm(Theta,'fro');

        end

    case 'CSF'

        switch error_measure

            case 'Frobenius'

                normX = norm(X , 'fro');

               residual = norm(Theta - W*H, 'fro') / norm(Theta,'fro');
               objective = norm(X - (W*H).^2, 'fro') / normX;

            case 'L1'
                
                A =W*H;
                X_approx = (W * H).^2;
                residual = sum(abs(Theta(:) - A(:))) / sum(abs(Theta(:)));
                objective = sum(abs(X(:) - X_approx(:))) / sum(abs(X(:)));


            case 'KL divergence'

                X_approx = (W * H).^2;

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

             residual = norm(Theta - W*H, 'fro') / norm(Theta,'fro');

        end

    case 'MinMax'
        a = options.a;
        b = options.b;
        switch error_measure

            case 'Frobenius'

                normX = norm(X , 'fro');

               residual = norm(Theta - W*H, 'fro') / norm(Theta,'fro');
               objective = norm(X - min(b, max(a, W*H)), 'fro') / normX;

            case 'L1'

                X_approx = min(b, max(a, W * H));
                A =W*H;
                residual = sum(abs(Theta(:) - A(:))) / sum(abs(Theta(:)));
                objective = sum(abs(X(:) - X_approx(:))) / sum(abs(X(:)));

            case 'KL divergence'

                X_approx = min(b, max(a, W * H));

                % Compute mean of all elements in X
               mu_X = mean(X(:)); 
    
    
               % Get the size of X
                %[m, n] = size(X);
    
                % Construct the estimation matrix X_est
                %em = ones(m,1); % Column vector of ones
                %en = ones(n,1); % Row vector of ones
                %X_est = mu_X * (em * en');
    



               %Theta = max(eps,Theta);
               X(X == 0) = eps;
               X_approx(X_approx == 0) = eps;

              % Compute KL divergence element-wise
             kl_matrix = (X .* log(X ./ X_approx) - X + X_approx );
             kl_est = (X .* log(X / mu_X) - X + mu_X);
             % Sum over all elements to get total KL divergence
             objective = sum(kl_matrix(:)) / sum(kl_est(:));

             residual = norm(Theta - W*H, 'fro') / norm(Theta,'fro');

        end

    case 'Modulus'

        switch error_measure

            case 'Frobenius'

                normX = norm(X , 'fro');

                residual = norm(Theta - W*H, 'fro') / norm(Theta,'fro');
                objective = norm(X - abs(W*H), 'fro') / normX;

            case 'L1'
                 A =W*H;
                X_approx = abs(W * H);
                residual = sum(abs(Theta(:) - A(:))) / sum(abs(Theta(:)));
                objective = sum(abs(X(:) - X_approx(:))) / sum(abs(X(:)));

            case 'KL divergence'

                X_approx = abs(W * H);

                % Compute mean of all elements in X
                mu_X = mean(X(:)); 
    
    
                % Get the size of X
                [m, n] = size(X);
    
                % Construct the estimation matrix X_est
                em = ones(m,1); % Column vector of ones
                en = ones(n,1); % Row vector of ones
                X_est = mu_X * (em * en');
    



               Theta = max(eps,Theta);
               X(X == 0) = eps;
               X_approx(X_approx == 0) = eps;


 
              % Compute KL divergence element-wise
              kl_matrix = (X .* log(X ./ X_approx) - X + X_approx );
              kl_est = (X .* log(X ./ X_est) - X + X_est);
              % Sum over all elements to get total KL divergence
             objective = sum(kl_matrix(:)) / sum(kl_est(:));
             residual = norm(Theta - W*H, 'fro') / norm(Theta,'fro');

        end

end
