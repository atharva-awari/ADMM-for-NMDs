function [Theta, W, H, residual, obj,savetime]= minmax_L1()

[X, rho, maxiter, maxtime, W, H, lambda, Theta , A] = get_initial_matrices(3);


cpu0=tic;
compute_obj=0;

X_approx = min(1, max(-1, W * H));

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


   c1 = find(Y > (rho+1));

   c2_1 = find(Y > (-rho +1));
   c2_2 = find (Y < (rho-1)); 
   c2   = intersect(c2_1,c2_2); 

   c3 = find(Y < (-rho-1));

   c4_1 = find(Y > (-rho-1));
   c4_2 = find(Y < (-rho));
   c4   = intersect(c4_1,c4_2);

   c5_1 = find (Y > rho);
   c5_2 = find(Y < (rho+1));
   c5   = intersect(c5_1,c5_2);

   c6_1 = find(Y > (rho-1));
   c6_2 = find(Y< rho);
   c6 = intersect(c6_1,c6_2);

   c7_1 = find(Y> (-rho));
   c7_2 = find(Y< (-rho +1));
   c7   = intersect(c7_1,c7_2);


    t1 = (rho*A - lambda) ./ (rho);
    
    t21 = (rho*A - lambda -1 ) ./ (rho);
    t22 = (rho*A - lambda +1 ) ./ (rho);
    t23 = X; 
    t2 = max(min(t22,X), t21);

    t3 = t1; 


    g1 = abs(X+1) +  t1.*lambda + rho/2*(t1 - A).^2;
    

    g2 = abs(X - t2) + t2.*lambda + rho/2*(t2 - A).^2;

    g3 = abs(X-1) +  t1.*lambda + rho/2*(t1 - A).^2;
    
    

    Theta(c1)=t3(c1);
    Theta(c2)=t2(c2);
    Theta(c3)=t1(c3);

    if g1(c4)<g2(c4)

       Theta(c4) = t1(c4);
    else
        Theta(c4) = t2(c4);
    end


    if g2(c5)<g3(c5)

       Theta(c5) = t2(c5);
    else
        Theta(c5) = t3(c5);
    end

    Theta(c6)=1;

    Theta(c7)=-1;


k=k+1; 
start_obj=tic;
X_approx = min(1,max(-1,A));
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