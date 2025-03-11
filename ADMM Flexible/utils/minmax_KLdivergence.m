function [Theta, W, H, residual, obj,savetime]= minmax_KLdivergence()

[X, rho, maxiter, maxtime, W, H, lambda, Theta , A] = get_initial_matrices(3);

epsilon = 0.01;
cpu0=tic;
compute_obj=0;

X_approx = min(1, max(-1, W * H));

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

    %% Updating Theta entry-wise

    p = find(X>0);
    q = find(X==0);
    %% X > 0
t11 = (1 / 2*rho) * (((rho*A - lambda - 1) + sqrt((lambda - rho*A + 1).^2 + 4*rho*X)));
t22 = (1/rho) * (rho*A - lambda);

D = sqrt((lambda - rho*A + 1).^2 + 4*rho*X);
Y = rho*A - lambda;

c11 = find(Y>rho);
c12 = find(Y > (2*rho + 1 - D));
c13 = intersect(c11,c12);
c1 = intersect(c13,p); 

c21 = find (Y < (2*rho + 1 - D));
c22 = intersect(c21,c11);
c23 = intersect(c21,c22);
c2  = intersect(c23,p);

c31 = find(Y < (2*rho + 1 - D));
c32 = intersect(c31, c11);
c3  = intersect(c32,p);

c41 = find(Y < rho);
c42 = intersect(c41, c12);
c4 = intersect(c42,p);

c51 = intersect(c41,c21);
c5  = intersect(c51,p);

g11 = t11 - X.* log(t11) + t11.*lambda + rho/2 .* (t11-A).^2;
g22 = 1 + t22.*lambda + rho/2 .* (t22-A).^2;

Theta(c1) = t22(c1);

if g11(c3) < g22(c3)
    Theta(c3) = t11(c3);
else
    Theta(c3) = t22(c3);
end

Theta(c4) = 1;

Theta(c5) = t11(c5); 

%% X = 0
t44 = (rho*A - lambda + 1) ./ rho;
t33 = (rho*A - lambda) ./ rho;

Y = rho*A - lambda;

y1 = find(Y< (-rho));
q1 = intersect(q,y1);

y22 = find(Y> (-rho));
y23 = find(Y < (-rho +1));
y2  = intersect(y22,y23);
q2 = intersect(q,y2);

y3 = find(Y> (-rho +1));
y4 = find(Y<rho);
y5 = intersect(y3,y4);
q3 = intersect(q,y5); 

y6 = find(Y > rho);
y7 = find(Y < (rho+1));
y8 = intersect(y6,y7);
q4 = intersect(q,y8);

y9 = find(Y > (rho +1));
q5 = intersect(q,y9);

Theta(q1) = t33(q1);
Theta(q2) = -1; 
Theta(q3) = t44(q3);


g44 = t44  + t44.*lambda + rho/2 .* (t44-A).^2;
g33 = 1 + t33.*lambda + rho/2 .* (t33-A).^2;

if g33(q4) < g44(q4)
    Theta(q4) = t33(q4);
else
    Theta(q4) = t44(q4);
end

Theta(q5) = t33(q5);

%%
k=k+1; 
start_obj=tic;
X_approx = min(1, max(-1,A));


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