function [X, rho, maxiter, maxtime, W, H, lambda, Theta, A] = get_initial_matrices(model_choice)
    % Ask user for matrix dimensions
    rows = input('Enter the number of rows for matrix X: ');
    cols = input('Enter the number of columns for matrix X: ');

    % Get rank for decomposition
    rank = input('Enter the rank (factorization dimension) for W and H: ');

    % Generate a random input matrix X
    W0 = randn(rows,rank);
    H0 = randn(rank,cols);
    
    switch model_choice
        case 1 
         X = max(0,W0*H0);
    
        case 2
         X = (W0*H0).^2;
    
        case 3
         X = min(1, max(-1, W0*H0));
        
        case 4
         X = abs(W0*H0);
    end

    % Get penalty parameter
    rho = input('Enter the penalty parameter rho: ');

    % Get maximum iterations
    maxiter = input('Enter the maximum number of iterations: ');

    % Get maximum time (seconds)
    maxtime = input('Enter the maximum allowed running time (in seconds): ');

    

    % Generate initial matrices Theta, W,  H using SVD of X for
    % initialization
   
    [U,S,V] = svds(X,rank);
    Theta = U*S*V';

     W = U;
     H = S*V';

     A = W*H;
   
     % Generate multiplicative factor lambda
    lambda = randn(rows,cols);


    
end
