clear all; clc;

m=8;
n=6;r = 3;
%X = rand(m,n);
W0 = randn(m,r);
 H0 = randn(r,n);
 X =  (W0*H0).^2;

lambda = randn(m,n);
rho=5;


p = find(X>0);
q = find(X==0);

epsilon = 0.001;

[U,S,V] = svds(X,r);
Theta = U*S*V';
Theta = max(epsilon,Theta);

W = U;
H = S*V';

A = W*H;

%for k=1:100

D = sqrt((lambda - rho.*A).^2 + 8*X.*(rho+2));
t1 = (rho.*A - lambda + D ) ./ 2*(rho+2);
t2 = (rho.*A - lambda - D) ./ 2*(rho+2);

g1 = t1.^2 - X.* log(t1.^2) + t1.*lambda + rho/2 .* (t1-A).^2;
g2 = t2.^2 - X.* log(t2.^2) + t2.*lambda + rho/2 .* (t2-A).^2;

if g1(p)<g2(p)

   Theta(p) = t1(p);
else
    Theta(p) = t2(p);
end


Theta(q) = (rho.*A(q) - lambda(q)) ./ (rho+2); 


        obj = Theta.^2 - X.* log(Theta.^2) + Theta.*lambda + rho/2 .* (Theta - A).^2;

        % W = (1/rho) * ((H*H') \ (H* lambda' + rho * H * Theta') );
        % W = W';
        % H = (1/rho) * ((W'*W) \ (W' * lambda + rho * W' * Theta) );
        % A = W*H;
        % lambda = lambda + rho*(Theta - A);

    for i = 1 : m
        for j = 1 : n
            thetaijint = [Theta(i,j)-50:0.01:Theta(i,j)+50]; 
            for k = 1 : length(thetaijint)
                fthetaijint(k) = thetaijint(k).^2 - X(i,j).* log(thetaijint(k).^2) + thetaijint(k).*lambda(i,j) + rho/2 .* (thetaijint(k) - A(i,j)).^2;
            end

            % if min(fthetaijint) < obj(i,j)-1e-12
            %     disp('PROBLEM in your formula Atharva')
                  figure; 
                  set(0, 'DefaultLineLineWidth', 2);
            plot(thetaijint,fthetaijint);
            hold on; 
            plot(Theta(i,j),obj(i,j),'ro'); 

            %     keyboard
            % end

        end
    end


        % i=3;j=3;
        % error(k)= Theta(i,j);

%end
   
% figure;
% set(0, 'DefaultAxesFontSize', 20);
% set(0, 'DefaultLineLineWidth', 2);
% semilogy(error);     


