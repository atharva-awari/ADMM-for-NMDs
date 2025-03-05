clear all; clc;

m=8;n=6;r = 3;
%X = eye(m,n);
 W0 = randn(m,r);
 H0 = randn(r,n);
 X = abs(W0*H0);

lambda = randn(m,n);
rho=1;


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

Theta(p) = (1 / 2*rho) * ((rho*A(p) - lambda(p) - 1) + sqrt((lambda(p) - rho*A(p) + 1).^2 + 4*rho*X(p)));

t1 = (rho*A - lambda - 1) ./ rho;
t2 = (rho*A - lambda +1) ./ rho;

Y = rho*A - lambda;

y1 = find(Y>1);
q1 = intersect(q,y1);

y2 = find(Y<-1);
q2 = intersect(q,y2);

y3 = find(Y>-1);
y4 = find(Y<1);
y5 = intersect(y3,y4);
q3 = intersect(q,y5); 

Theta(q1) = t1(q1);
Theta(q2) = t2(q2); 
Theta(q3) = 0;


        obj = abs(Theta) - X.* log(abs(Theta)) + Theta.*lambda + rho/2 .* (Theta - A).^2;

        % W = (1/rho) * ((H*H') \ (H* lambda' + rho * H * Theta') );
        % W = W';
        % H = (1/rho) * ((W'*W) \ (W' * lambda + rho * W' * Theta) );
        % A = W*H;
        % lambda = lambda + rho*(Theta - A);


        % i=3;j=3;
        % error(k)= Theta(i,j);

     for i = 1 : m
        for j = 1 : n
            thetaijint = [Theta(i,j)-10:0.01:Theta(i,j)+10]; 
            for k = 1 : length(thetaijint)
                fthetaijint(k) = abs(thetaijint(k)) - X(i,j).* log(abs(thetaijint(k))) + thetaijint(k).*lambda(i,j) + rho/2 .* (thetaijint(k) - A(i,j)).^2 ;
            end

            % if min(fthetaijint) < obj(i,j)-1e-12
            %     disp('PROBLEM in your formula Atharva')
                  figure; 
            plot(thetaijint,fthetaijint);
            hold on; 
            plot(Theta(i,j),obj(i,j),'ro'); 

                % keyboard
            %end

        end
    end

%end
   
% figure;
% set(0, 'DefaultAxesFontSize', 20);
% set(0, 'DefaultLineLineWidth', 2);
% semilogy(error);     


