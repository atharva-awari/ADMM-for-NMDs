function [Theta, W, H, residual, obj,savetime]= relu_L1()

[X, rho, maxiter, maxtime, W, H, lambda, Theta , A] = get_initial_matrices();


cpu0=tic;
compute_obj=0;

X_approx = max(0, W * H);

k=1;
start_obj=tic;
residual(k) = sum(abs(Theta(:) - A(:))) / sum(abs(Theta(:)));
obj(k) = sum(abs(X(:) - X_approx(:))) / sum(abs(X(:)));
compute_obj=compute_obj+toc(start_obj);
runtime=toc(cpu0)-compute_obj;
savetime(k)=runtime;

while k < maxiter && runtime<maxtime

    [Theta, W, H, lambda, A] = update_factors(Theta, W, H, lambda,A, rho);


     %% Updating Theta entry-wise
    Y =rho*A - lambda;


    p = find(Y>1);
    q = find(Y<(-1));
    r1 = find(Y> 0);
    r2 = find(Y<1);
    r = intersect(r1,r2);
    s1 = find(Y> (-1));
    s2 = find(Y<0);
    s = intersect(s1,s2);


    t1 = (rho*A - lambda) ./ (rho);
    t21 = (rho*A - lambda -1 ) ./ (rho);
    t22 = (rho*A - lambda +1 ) ./ (rho);
    t23 = X; 
    t2 = max(min(t22,X), t21); 

    g1 = X +  t1.*lambda + rho/2*(t1 - A).^2;
    

    g2 = abs(X - t2) + t2.*lambda + rho/2*(t2 - A).^2;
    
    

    Theta(p)=t2(p);
    Theta(q)=t1(q);
    Theta(r)=0;


    if g1(s)<g2(s)

        Theta(s) = t1(s);
    else
        Theta(s) = t2(s);
    end


k=k+1; 
start_obj=tic;
X_approx = max(0,A);
residual(k) = sum(abs(Theta(:) - A(:))) / sum(abs(Theta(:)));
obj(k) = sum(abs(X(:) - X_approx(:))) /  sum(abs(X(:))); 
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