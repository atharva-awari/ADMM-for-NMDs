clear all; clc;

m=8;
n=6;
r = 3;
W0 = randn(m,r);
H0 = randn(r,n);
X = (W0*H0).^2;


lambda = randn(m,n);
rho=6;


[U,S,V] = svds(X,r);
Theta = U*S*V';

W = U;
H = S*V';

A = W*H;

%for k=1:100
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
        k = find(g1>g2);
        Theta(l) = t1(l);
        Theta(k) = t2(k); 

        
    end


   



    obj = abs(X - (Theta).^2) + lambda.*Theta + rho/2 .* (Theta - A).^2;

    %W = (1/rho) * ((H*H') \ (H* lambda' + rho * H * Theta') );
    %W = W';
    %H = (1/rho) * ((W'*W) \ (W' * lambda + rho * W' * Theta) );
    %A = W*H;
    %lambda = lambda + rho*(Theta - A);

    for i = 1 : m
        for j = 1 : n
            thetaijint = [Theta(i,j)-10:0.01:Theta(i,j)+10]; 
            for k = 1 : length(thetaijint)
                fthetaijint(k) = abs(X(i,j) - (thetaijint(k))^2) + lambda(i,j).*thetaijint(k) + rho/2 .* (thetaijint(k) - A(i,j)).^2; 
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


