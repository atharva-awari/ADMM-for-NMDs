function [Theta, W, H, residual, obj,savetime]= modulus_L1()

[X, rho, maxiter, maxtime, W, H, lambda, Theta , A] = get_initial_matrices(4);


cpu0=tic;
compute_obj=0;

X_approx = abs(W * H);

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
    r1 = find(Y> (-1));
    r2 = find(Y<1);
    r = intersect(r1,r2);
   


    t11 = (rho*A - lambda +1) ./ (rho);
    t22 = (rho*A - lambda -1 ) ./ (rho);
   
    t1 = max(min(t11,X), t22);
    t2 = max(min(t11,-X), t22); 
     

    g1 = abs(X -t1) +  t1.*lambda + rho/2*(t1 - A).^2;
    g2 = abs(X + t2) + t2.*lambda + rho/2*(t2 - A).^2;
    
    

    Theta(p)=t1(p);
    Theta(q)=t2(q);
    


    if g1(r)<g2(r)

        Theta(r) = t1(r);
    else
        Theta(r) = t2(r);
    end


k=k+1; 
start_obj=tic;
X_approx = abs(A);
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



% Save them individually in the base workspace
    assignin('base', 'X', X);
    assignin('base', 'W', W);
    assignin('base', 'H', H);
    assignin('base', 'Theta', Theta);
    assignin('base', 'obj', obj);
    assignin('base', 'residual', residual);





end