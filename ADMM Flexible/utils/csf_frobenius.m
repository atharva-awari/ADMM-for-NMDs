function [Theta, W, H, residual, obj,savetime]= csf_frobenius()

[X, rho, maxiter, maxtime, W, H, lambda, Theta , A] = get_initial_matrices(2);


cpu0=tic;
compute_obj=0;

[m,n]=size(X);
normX = norm(X , 'fro');
k=1;
start_obj=tic;
residual(k) = norm(Theta - W*H, 'fro') / norm(Theta,'fro');
obj(k) = norm(X - (W*H).^2, 'fro') / normX;
compute_obj=compute_obj+toc(start_obj);
runtime=toc(cpu0)-compute_obj;
savetime(k)=runtime;

while k < maxiter && runtime<maxtime

    [Theta, W, H, lambda, A] = update_factors(Theta, W, H, lambda,A, rho);


     %% Updating Theta entry-wise
    for i=1:m
        for j=1:n

            a =2;
            c =rho - 2 * X(i,j);
            d = lambda(i,j) - rho * A(i,j);
            Theta(i,j) = cardanoatharva(a,c, d);
        end
    end



k=k+1; 
start_obj=tic;
residual(k) = norm(Theta-A, 'fro') / norm(Theta,'fro');
obj(k) = norm(X - A.^2, 'fro') / normX;
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