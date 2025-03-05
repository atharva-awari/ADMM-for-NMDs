clear all; clc;

m=5;
n=3;
r = 2;
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






    obj = 1/2 * (X - min(1, max(-1,Theta))).^2 + lambda.*Theta + rho/2 .* (Theta - A).^2;

    %W = (1/rho) * ((H*H') \ (H* lambda' + rho * H * Theta') );
    %W = W';
    %H = (1/rho) * ((W'*W) \ (W' * lambda + rho * W' * Theta) );
    %A = W*H;
    %lambda = lambda + rho*(Theta - A);

    for i = 1 : m
        for j = 1 : n
            thetaijint = [Theta(i,j)-10:0.01:Theta(i,j)+10]; 
            for k = 1 : length(thetaijint)
                fthetaijint(k) = 1/2*(X(i,j) - min(1,max(-1,thetaijint(k))))^2 + lambda(i,j).*thetaijint(k) + rho/2 .* (thetaijint(k) - A(i,j)).^2; 
            end

            %if min(fthetaijint) < obj(i,j)-1e-12
             %   disp('PROBLEM in your formula Atharva')
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


