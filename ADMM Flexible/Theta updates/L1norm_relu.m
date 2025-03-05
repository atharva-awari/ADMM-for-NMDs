clear all; clc;

m=8;
n=6;
r = 3;
W0 = randn(m,r);
H0 = randn(r,n);
X = max(0, W0*H0);


lambda = randn(m,n);
rho=5;


[U,S,V] = svds(X,r);
Theta = U*S*V';

W = U;
H = S*V';

A = W*H;

%for k=1:100
    Y =rho*A - lambda;


    p = find(Y>1);
    q = find(Y<(-1));
    r1 = find(Y> 0);
    r2 = find(Y<1);
    r = intersect(r1,r2);
    s1 = find(Y> (-1));
    s2 = find(Y<0);
    s = intersect(s1,s2);


    t1 = (rho*A - lambda) ./ (rho);
    t21 = (rho*A - lambda -1 ) ./ (rho);
    t22 = (rho*A - lambda +1 ) ./ (rho);
    t23 = X; 
    t2 = max(min(t22,X), t21); 

    g1 = X +  t1.*lambda + rho/2*(t1 - A).^2;
    % g21 = abs(X - t21) + t21.*lambda + rho/2*(t21 - A).^2;
    % g22 = abs(X - t22) + t22.*lambda + rho/2*(t22 - A).^2;
    % g23 = abs(X - t23) + t23.*lambda + rho/2*(t23 - A).^2; 

    g2 = abs(X - t2) + t2.*lambda + rho/2*(t2 - A).^2;
    
    

    Theta(p)=t2(p);
    Theta(q)=t1(q);
    Theta(r)=0;


    if g1(s)<g2(s)

        Theta(s) = t1(s);
    else
        Theta(s) = t2(s);
    end




    obj = abs(X - max(0,Theta)) + lambda.*Theta + rho/2 .* (Theta - A).^2;

    %W = (1/rho) * ((H*H') \ (H* lambda' + rho * H * Theta') );
    %W = W';
    %H = (1/rho) * ((W'*W) \ (W' * lambda + rho * W' * Theta) );
    %A = W*H;
    %lambda = lambda + rho*(Theta - A);

    for i = 1 : m
        for j = 1 : n
            thetaijint = [Theta(i,j)-10:0.01:Theta(i,j)+10]; 
            for k = 1 : length(thetaijint)
                fthetaijint(k) = abs(X(i,j) - max(0,thetaijint(k))) + lambda(i,j).*thetaijint(k) + rho/2 .* (thetaijint(k) - A(i,j)).^2; 
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


