function [Theta, W, H, residual, obj,savetime]= csf_L1()

[X, rho, maxiter, maxtime, W, H, lambda, Theta , A] = get_initial_matrices();


cpu0=tic;
compute_obj=0;


X_approx = (W * H).^2;

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
    Z = ((rho*A - lambda)./ (rho+2)).^2;

    if rho>2
    p = find(Y>0);
    q = find(Z >X);
   
    r = intersect(p,q);
    
    s  = find(Z<X);
    u = find(Y<0);
    v = intersect(u,q);


    t1 = (rho*A - lambda) ./ (rho-2);
    t2 = (rho*A - lambda) ./ (rho+2);
    
    
    Theta(r)=t2(r);
    Theta(v)=t2(v);
    Theta(s)=t1(s);
    

    else 
        t1 = sqrt(X);
        t2 = -sqrt(X); 
        g1 = abs(X - (t1).^2) + lambda.*t1 + rho/2 .* (t1 - A).^2;
        g2 = abs(X - (t2).^2) + lambda.*t2 + rho/2 .* (t2 - A).^2;
        l = find(g1<g2);
        m = find(g1>g2);
        Theta(l) = t1(l);
        Theta(m) = t2(m); 

        
    end




k=k+1; 
start_obj=tic;
X_approx = A.^2;
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