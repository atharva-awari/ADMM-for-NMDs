clear all; clc;


m=8;
n=6;
r = 3;
W0 = randn(m,r);
H0 = randn(r,n);
X = max(0, W0*H0);


lambda = randn(m,n);
%rho=5;


[U,S,V] = svds(X,r);
Theta = U*S*V';

W = U;
H = S*V';

A = W*H;

%for k=1:100
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

    %g1 = 1/2 * (X(r) - t1(r)).^2 + t1(r).*lambda + rho/2*(t1 - A).^2;
    %g2 = 1/2 * (X + t2).^2 + t2*lambda + rho/2*(t2 - A).^2;



    if g1(r)<g2(r)

        Theta(r) = t1(r);
    else
        Theta(r) = t2(r);
    end




    obj = 1/2*(X - max(0,Theta)).^2 + lambda.*Theta + rho/2 .* (Theta - A).^2;

    %W = (1/rho) * ((H*H') \ (H* lambda' + rho * H * Theta') );
    %W = W';
    %H = (1/rho) * ((W'*W) \ (W' * lambda + rho * W' * Theta) );
    %A = W*H;
    %lambda = lambda + rho*(Theta - A);

    for i = 1 : m
        for j = 1 : n
            thetaijint = [Theta(i,j)-10:0.01:Theta(i,j)+10]; 
            for k = 1 : length(thetaijint)
                fthetaijint(k) = 1/2*(X(i,j) - max(0,thetaijint(k))).^2 + lambda(i,j).*thetaijint(k) + rho/2 .* (thetaijint(k) - A(i,j)).^2; 
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


