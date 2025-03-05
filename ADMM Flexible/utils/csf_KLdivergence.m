function [Theta, W, H, residual, obj,savetime]= csf_KLdivergence()

[X, rho, maxiter, maxtime, W, H, lambda, Theta , A] = get_initial_matrices();

epsilon = 0.01;
cpu0=tic;
compute_obj=0;

X_approx = (W * H).^2;

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

    t1 = (rho*A - lambda + sqrt((lambda - rho*A).^2 + 8*X.*(rho+2))) ./ 2*(rho+2);
    t2 = (rho*A - lambda - sqrt((lambda - rho*A).^2 + 8*X.*(rho+2))) ./ 2*(rho+2);

    g1 = t1.^2 - X.* log(t1.^2) + t1.*lambda + rho/2 .* (t1-A).^2;
    g2 = t2.^2 - X.* log(t2.^2) + t2.*lambda + rho/2 .* (t2-A).^2;

    if g1(p)<g2(p)

   Theta(p) = t1(p);
   else
    Theta(p) = t2(p);
   end


Theta(q) = (rho.*A(q) - lambda(q)) ./ (rho+2); 


k=k+1; 
start_obj=tic;
X_approx = A.^2;


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