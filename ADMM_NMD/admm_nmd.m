function [Theta, W, H, residual, objective] = admm_nmd(X, rho, maxiter, maxtime, W, H, Theta, lambda, model, error_measure, options)

a = options.a;
b = options.b;

cpu0=tic;
compute_obj=0;



k=1;
start_obj=tic;
compute_obj=compute_obj+toc(start_obj);
runtime=toc(cpu0)-compute_obj;
savetime(k)=runtime;

[residual(k), objective(k)] = compute_residual_objective(X, W, H, Theta, model, error_measure, options);


while k < maxiter && runtime<maxtime
     
    
    Ht=H';
    HHt=H*Ht;
    LW=norm(HHt);
    W=W-1/(rho*LW)*(-lambda*Ht+rho*(W*HHt-Theta*Ht));


    Wt=W';
    WtW=Wt*W;
    LH=norm(WtW);
    H=H-1/(rho*LH)*(-Wt*lambda+rho*(WtW*H-Wt*Theta));

    A = W*H;

    [Theta] = update_Theta(X,rho,Theta,lambda,A, model, error_measure, options);

    lambda = lambda + rho*(Theta - A);
    k=k+1;

    start_obj=tic;
    compute_obj=compute_obj+toc(start_obj);
    runtime=toc(cpu0)-compute_obj;
    savetime(k)=runtime;

    [residual(k), objective(k)] = compute_residual_objective(X, W, H, Theta, model, error_measure, options);


end

%Plotting objective and residual

subplot(1,2,1)
semilogy(objective,'b','LineWidth',1.5);hold on; 
hold on

xlabel('Iterations');
ylabel('Objective');

subplot(1,2,2)
semilogy(residual,'r','LineWidth',1.5);hold on; 

ylabel('Residual');
xlabel('Iteration'); 