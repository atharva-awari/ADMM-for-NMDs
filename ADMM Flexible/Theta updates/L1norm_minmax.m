clear all; clc;

m=8;
n=6;
r = 3;
W0 = randn(m,r);
H0 = randn(r,n);
X = min(1, max(-1, W0*H0));


lambda = randn(m,n);
rho=5;


[U,S,V] = svds(X,r);
Theta = U*S*V';

W = U;
H = S*V';

A = W*H;

%for k=1:100
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






    obj = abs(X - min(1, max(-1,Theta))) + lambda.*Theta + rho/2 .* (Theta - A).^2;

    %W = (1/rho) * ((H*H') \ (H* lambda' + rho * H * Theta') );
    %W = W';
    %H = (1/rho) * ((W'*W) \ (W' * lambda + rho * W' * Theta) );
    %A = W*H;
    %lambda = lambda + rho*(Theta - A);

    for i = 1 : m
        for j = 1 : n
            thetaijint = [Theta(i,j)-10:0.01:Theta(i,j)+10]; 
            for k = 1 : length(thetaijint)
                fthetaijint(k) = abs(X(i,j) - min(1,max(-1,thetaijint(k)))) + lambda(i,j).*thetaijint(k) + rho/2 .* (thetaijint(k) - A(i,j)).^2; 
            end

            %if min(fthetaijint) < obj(i,j)-1e-12
                %disp('PROBLEM in your formula Atharva')
                  figure; 
            plot(thetaijint,fthetaijint);
            hold on; 
            plot(Theta(i,j),obj(i,j),'ro'); 

                %keyboard
            %end

        end
    end
%    error(k)= obj(i,j);

%end


