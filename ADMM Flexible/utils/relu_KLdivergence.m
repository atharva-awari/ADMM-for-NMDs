function [Theta, W, H, residual, obj,savetime]= relu_KLdivergence()

[X, rho, maxiter, maxtime, W, H, lambda, Theta , A] = get_initial_matrices();

epsilon = 0.01;
cpu0=tic;
compute_obj=0;

X_approx = max(0, W * H);

k=1;
start_obj=tic;
Theta = max(epsilon,Theta);
X(X == 0) = epsilon;
X_approx(X_approx == 0) = epsilon;


 % Compute KL divergence element-wise
    kl_matrix = X .* log(X ./ X_approx) - X + X_approx;

    % Sum over all elements to get total KL divergence
    obj(k) = sum(kl_matrix(:));

    % Compute KL divergence element-wise
    kl_matrix1 = Theta .* log(Theta ./ A) - Theta + A;

    % Sum over all elements to get total KL divergence
    residual(k) = sum(kl_matrix1(:));

compute_obj=compute_obj+toc(start_obj);
runtime=toc(cpu0)-compute_obj;
savetime(k)=runtime;

while k < maxiter && runtime<maxtime

    [Theta, W, H, lambda, A] = update_factors(Theta, W, H, lambda,A, rho);

    %% Updating Theta entry-wise
    p = find(X>0);
    q = find(X==0);

    
    Theta = max(epsilon,Theta);

    Theta(p) = (1 / 2*rho) * ((rho*A(p) - lambda(p) - 1) + sqrt((lambda(p) - rho*A(p) + 1).^2 + 4*rho*X(p)));

    t1 = (rho*A - lambda + 1) ./ rho;
    t2 = (rho*A - lambda) ./ rho;
    
    Y = rho*A - lambda;
    
    y1 = find(Y>1);
    q1 = intersect(q,y1);
    
    y2 = find(Y<0);
    q2 = intersect(q,y2);
    
    y3 = find(Y>0);
    y4 = find(Y<1);
    y5 = intersect(y3,y4);
    q3 = intersect(q,y5); 

    Theta(q1) = t1(q1);
    Theta(q2) = t2(q2); 
    Theta(q3) = 0;


k=k+1; 
start_obj=tic;
X_approx = max(0,A);


% Compute KL divergence element-wise
    kl_matrix = X .* log(X ./ X_approx) - X + X_approx;

    % Sum over all elements to get total KL divergence
    obj(k) = sum(kl_matrix(:));

    % Compute KL divergence element-wise
    kl_matrix1 = Theta .* log(Theta ./ A) - Theta + A;

    % Sum over all elements to get total KL divergence
    residual(k) = sum(kl_matrix1(:));


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


end