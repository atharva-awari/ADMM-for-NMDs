function [Theta, W, H, residual, obj,savetime]= modulus_KLdivergence()

[X, rho, maxiter, maxtime, W, H, lambda, Theta , A] = get_initial_matrices(4);

epsilon = 0.01;
cpu0=tic;
compute_obj=0;

X_approx = abs(W * H);

% Compute mean of all elements in X
    mu_X = mean(X(:)); 
    mu_Theta = mean(Theta(:));
    
  % Get the size of X
    [m, n] = size(X);
    
  % Construct the estimation matrix X_est
    e = ones(m,1); % Column vector of ones
    X_est = mu_X * (e * e');
    Theta_est = mu_Theta * (e*e'); 


k=1;
start_obj=tic;
Theta = max(epsilon,Theta);
X(X == 0) = epsilon;
X_approx(X_approx == 0) = epsilon;


 
% Compute KL divergence element-wise
    kl_matrix = (X .* log(X ./ X_approx) - X + X_approx );
    kl_est = (X .* log(X ./ X_est) - X + X_est);
    % Sum over all elements to get total KL divergence
    obj(k) = sum(kl_matrix(:)) / sum(kl_est(:));

    % Compute KL divergence element-wise
    kl_matrix1 = (Theta .* log(Theta ./ A) - Theta + A);
    kl_est1 = (Theta .* log(Theta ./ Theta_est) - Theta + Theta_est);
    % Sum over all elements to get total KL divergence
    residual(k) = sum(kl_matrix1(:)) / sum(kl_est1(:));


compute_obj=compute_obj+toc(start_obj);
runtime=toc(cpu0)-compute_obj;
savetime(k)=runtime;

while k < maxiter && runtime<maxtime

    [Theta, W, H, lambda, A] = update_factors(Theta, W, H, lambda,A, rho);

    p = find(X>0);
    q = find(X==0);

    %% Updating Theta entry-wise
    

    
    
Theta(p) = (1 / 2*rho) * ((rho*A(p) - lambda(p) - 1) + sqrt((lambda(p) - rho*A(p) + 1).^2 + 4*rho*X(p)));

t1 = (rho*A - lambda - 1) ./ rho;
t2 = (rho*A - lambda +1) ./ rho;

Y = rho*A - lambda;

y1 = find(Y>1);
q1 = intersect(q,y1);

y2 = find(Y<-1);
q2 = intersect(q,y2);

y3 = find(Y>-1);
y4 = find(Y<1);
y5 = intersect(y3,y4);
q3 = intersect(q,y5); 

Theta(q1) = t1(q1);
Theta(q2) = t2(q2); 
Theta(q3) = 0;


%%
k=k+1; 
start_obj=tic;
X_approx = abs(A);



% Compute mean of all elements in X
    
    mu_Theta = mean(Theta(:));
    
    
  % Construct the estimation matrix X_est
    X_est = mu_X * (e * e');
    Theta_est = mu_Theta * (e*e'); 


 % Compute KL divergence element-wise
    kl_matrix = (X .* log(X ./ X_approx) - X + X_approx );
    kl_est = (X .* log(X ./ X_est) - X + X_est);
    % Sum over all elements to get total KL divergence
    obj(k) = sum(kl_matrix(:)) / sum(kl_est(:));

    % Compute KL divergence element-wise
    kl_matrix1 = (Theta .* log(Theta ./ A) - Theta + A);
    kl_est1 = (Theta .* log(Theta ./ Theta_est) - Theta + Theta_est);
    % Sum over all elements to get total KL divergence
    residual(k) = sum(kl_matrix1(:)) / sum(kl_est1(:));


compute_obj=compute_obj+toc(start_obj);

runtime=toc(cpu0)-compute_obj;
  
savetime(k)=runtime;

end



%Plotting objective and residual

subplot(1,2,1)
semilogy(obj,'b','LineWidth',1.5);hold on; 
hold on

xlabel('Iterations');
ylabel('Objective');

subplot(1,2,2)
semilogy(residual,'r','LineWidth',1.5);hold on; 

ylabel('Residual');
xlabel('Iteration'); 


% Save them individually in the base workspace
    assignin('base', 'X', X);
    assignin('base', 'W', W);
    assignin('base', 'H', H);
    assignin('base', 'Theta', Theta);
    assignin('base', 'obj', obj);
    assignin('base', 'residual', residual);



end