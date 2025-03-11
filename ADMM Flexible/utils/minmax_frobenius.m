function [Theta, W, H, residual, obj,savetime]= minmax_frobenius()

[X, rho, maxiter, maxtime, W, H, lambda, Theta , A] = get_initial_matrices(3);


cpu0=tic;
compute_obj=0;

normX = norm(X , 'fro');
k=1;
start_obj=tic;
residual(k) = norm(Theta - W*H, 'fro') / norm(Theta,'fro');
obj(k) = norm(X - min(1, max(-1, W*H)), 'fro') / normX;
compute_obj=compute_obj+toc(start_obj);
runtime=toc(cpu0)-compute_obj;
savetime(k)=runtime;

while k < maxiter && runtime<maxtime

    [Theta, W, H, lambda, A] = update_factors(Theta, W, H, lambda,A, rho);

    %% Updating Theta entry-wise
     Y =rho*A - lambda;


   c1_1 = find(Y >rho);
   c1_2 = find(Y > (1 + rho -X));
   c1   = intersect(c1_1,c1_2);

   c2_1 = find(Y > (-rho));
   c2_2 = find (Y < rho); 
   c2_3 = intersect(c2_1,c2_2);
   c2_4 = find(Y > (-rho -X -1));
   c2_5 = find (Y < (1+rho-X)); 
   c2_6 = intersect(c2_4,c2_5);
   c2   = intersect(c2_3,c2_6);


   c3_1 = find(Y < (-rho));
   c3_2 = find(Y < (-1 -rho -X));
   c3   = intersect(c3_1,c3_2);

   c4_1 = find(Y < (-rho));
   c4_2 = find(Y > (-1 -rho -X));
   c4_3 = find(Y < (1 + rho -X));
   c4_4 = intersect(c4_2,c4_3);
   c4   = intersect(c4_1,c4_4); 

   c5_1 = find (Y > rho);
   c5_2 = find(Y > (-1 - rho-X));
   c5_3 = find(Y < (1 + rho-X));
   c5_4   = intersect(c5_2,c5_3);
   c5   = intersect(c5_1,c5_4);

   c6_1 = find(Y > (-rho));
   c6_2 = find(Y < (rho));
   c6_3 = intersect(c6_1,c6_2);
   c6_4 = find(Y> (1+rho -X));
   c6 = intersect(c6_3,c6_4);

   
   c7   = intersect(c3_2,c6_3);


    t1 = (rho*A - lambda) ./ (rho);
    
   t2  = (X + rho*A - lambda) ./ (rho+1);

    t3 = t1; 


    g1 = 1/2 * (X-1).^2 +  t1.*lambda + rho/2*(t1 - A).^2;
    

    g2 = 1/2 * (X - t2).^2 + t2.*lambda + rho/2*(t2 - A).^2;

    g3 = 1/2 * (X+1).^2 +  t3.*lambda + rho/2*(t3 - A).^2;
    
    

    Theta(c1)=t1(c1);
    Theta(c2)=t2(c2);
    Theta(c3)=t3(c3);

    if g2(c4)<g3(c4)

       Theta(c4) = t2(c4);
    else
        Theta(c4) = t3(c4);
    end


    if g1(c5)<g2(c5)

       Theta(c5) = t1(c5);
    else
        Theta(c5) = t2(c5);
    end

    Theta(c6)=1;

    Theta(c7)=-1;



k=k+1; 
start_obj=tic;
residual(k) = norm(Theta-A, 'fro') / norm(Theta,'fro');
obj(k) = norm(X - min(1,max(-1, A)), 'fro') / normX;
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