function [Theta, W, H, residual, obj,savetime]= relu_frobenius()

[X, rho, maxiter, maxtime, W, H, lambda, Theta , A] = get_initial_matrices(1);


cpu0=tic;
compute_obj=0;

normX = norm(X , 'fro');
k=1;
start_obj=tic;
residual(k) = norm(Theta - W*H, 'fro') / norm(Theta,'fro');
obj(k) = norm(X - max(0, W*H), 'fro') / normX;
compute_obj=compute_obj+toc(start_obj);
runtime=toc(cpu0)-compute_obj;
savetime(k)=runtime;

while k < maxiter && runtime<maxtime

    [Theta, W, H, lambda, A] = update_factors(Theta, W, H, lambda,A, rho);

    %% Updating Theta entry-wise
    Y =lambda - rho*A;


    p = find(Y>X);
    q = find(Y<0);
    r1 = find(Y> 0);
    r2 = find(Y<X);
    r = intersect(r1,r2);


    t1 = (rho*A - lambda + X ) ./ (rho + 1);
    t2 = (rho*A - lambda ) ./ (rho + 1);

    Theta(p)=t2(p);
    Theta(q)=t1(q);

    g1 = 1/2 * (X - t1).^2 + t1.*lambda + rho/2*(t1 - A).^2;
    g2 = 1/2 * (X).^2 + t2.*lambda + rho/2*(t2 - A).^2;

  


    if g1(r)<g2(r)

        Theta(r) = t1(r);
    else
        Theta(r) = t2(r);
    end


k=k+1; 
start_obj=tic;
residual(k) = norm(Theta-A, 'fro') / norm(Theta,'fro');
obj(k) = norm(X - max(0, A), 'fro') / normX;
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