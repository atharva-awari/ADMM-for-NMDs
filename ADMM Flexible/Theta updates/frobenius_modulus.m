clear all; clc;

m=5;
n=2;
X = rand(m,n);

lambda = randn(m,n);
rho=5;
r = 3;

[U,S,V] = svds(X,r);
Theta = U*S*V';

W = U;
H = S*V';

A = W*H;

%for k=1:100
    Y = rho*A - lambda;


    p = find(Y>X);
    q = find(Y<(-X));
    r1 = find(Y> (-X));
    r2 = find(Y<X);
    r = intersect(r1,r2);


    t1 = (rho*A - lambda + X ) ./ (rho + 1);
    t2 = (rho*A - lambda - X ) ./ (rho + 1);

    Theta(p)=t1(p);
    Theta(q)=t2(q);

    g1 = 1/2 * (X - t1).^2 + t1.*lambda + rho/2*(t1 - A).^2;
    g2 = 1/2 * (X + t2).^2 + t2.*lambda + rho/2*(t2 - A).^2;

    %g1 = 1/2 * (X(r) - t1(r)).^2 + t1(r).*lambda + rho/2*(t1 - A).^2;
    %g2 = 1/2 * (X + t2).^2 + t2*lambda + rho/2*(t2 - A).^2;



    if g1(r)<g2(r)

        Theta(r) = t1(r);
    else
        Theta(r) = t2(r);
    end




    obj = 1/2*(X - abs(Theta)).^2 + lambda.*Theta + rho/2 .* (Theta - A).^2;

    %W = (1/rho) * ((H*H') \ (H* lambda' + rho * H * Theta') );
    %W = W';
    %H = (1/rho) * ((W'*W) \ (W' * lambda + rho * W' * Theta) );
    %A = W*H;
    %lambda = lambda + rho*(Theta - A);

    for i = 1 : m
        for j = 1 : n
            thetaijint = [Theta(i,j)-10:0.01:Theta(i,j)+10]; 
            for k = 1 : length(thetaijint)
                fthetaijint(k) = 1/2*(X(i,j) - abs(thetaijint(k))).^2 + lambda(i,j).*thetaijint(k) + rho/2 .* (thetaijint(k) - A(i,j)).^2; 
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


