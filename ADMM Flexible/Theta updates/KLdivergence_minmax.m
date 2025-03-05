clear all; clc;

m=8;n=6;r = 3;
X = rand(m,n);
%W0 = randn(m,r);
%H0 = randn(r,n);
%X = min(1, max(-1,W0*H0));

lambda = randn(m,n);
rho=1;


p = find(X>0);
q = find(X==0);

epsilon = 0.01;

[U,S,V] = svds(X,r);
Theta = U*S*V';
Theta = max(epsilon,Theta);

W = U;
H = S*V';

A = W*H;

%for k=1:100
%% X > 0
t11 = (1 / 2*rho) * (((rho*A - lambda - 1) + sqrt((lambda - rho*A + 1).^2 + 4*rho*X)));
t22 = (1/rho) * (rho*A - lambda);

D = sqrt((lambda - rho*A + 1).^2 + 4*rho*X);
Y = rho*A - lambda;

c11 = find(Y>rho);
c12 = find(Y > (2*rho + 1 - D));
c13 = intersect(c11,c12);
c1 = intersect(c13,p); 

c21 = find (Y < (2*rho + 1 - D));
c22 = intersect(c21,c11);
c23 = intersect(c21,c22);
c2  = intersect(c23,p);

c31 = find(Y < (2*rho + 1 - D));
c32 = intersect(c31, c11);
c3  = intersect(c32,p);

c41 = find(Y < rho);
c42 = intersect(c41, c12);
c4 = intersect(c42,p);

c51 = intersect(c41,c21);
c5  = intersect(c51,p);

g11 = t11 - X.* log(t11) + t11.*lambda + rho/2 .* (t11-A).^2;
g22 = 1 + t22.*lambda + rho/2 .* (t22-A).^2;

Theta(c1) = t22(c1);

if g11(c3) < g22(c3)
    Theta(c3) = t11(c3);
else
    Theta(c3) = t22(c3);
end

Theta(c4) = 1;

Theta(c5) = t11(c5); 

%% X = 0
t44 = (rho*A - lambda + 1) ./ rho;
t33 = (rho*A - lambda) ./ rho;

Y = rho*A - lambda;

y1 = find(Y< (-rho));
q1 = intersect(q,y1);

y22 = find(Y> (-rho));
y23 = find(Y < (-rho +1));
y2  = intersect(y22,y23);
q2 = intersect(q,y2);

y3 = find(Y> (-rho +1));
y4 = find(Y<rho);
y5 = intersect(y3,y4);
q3 = intersect(q,y5); 

y6 = find(Y > rho);
y7 = find(Y < (rho+1));
y8 = intersect(y6,y7);
q4 = intersect(q,y8);

y9 = find(Y > (rho +1));
q5 = intersect(q,y9);

Theta(q1) = t33(q1);
Theta(q2) = -1; 
Theta(q3) = t44(q3);


g44 = t44  + t44.*lambda + rho/2 .* (t44-A).^2;
g33 = 1 + t33.*lambda + rho/2 .* (t33-A).^2;

if g33(q4) < g44(q4)
    Theta(q4) = t33(q4);
else
    Theta(q4) = t44(q4);
end

Theta(q5) = t33(q5);



        obj = min(1, max(-1,Theta)) - X.* log(min(1,max(-1,Theta))) + Theta.*lambda + rho/2 .* (Theta - A).^2;
        % 
        % W = (1/rho) * ((H*H') \ (H* lambda' + rho * H * Theta') );
        % W = W';
        % H = (1/rho) * ((W'*W) \ (W' * lambda + rho * W' * Theta) );
        % A = W*H;
        % lambda = lambda + rho*(Theta - A);

     for i = 1 : m
        for j = 1 : n
            thetaijint = [Theta(i,j)-10:0.01:Theta(i,j)+10]; 
            for k = 1 : length(thetaijint)
                fthetaijint(k) = min(1,max(-1,thetaijint(k))) - X(i,j).* log(min(1, max(-1,thetaijint(k)))) + thetaijint(k).*lambda(i,j) + rho/2 .* (thetaijint(k) - A(i,j)).^2;
            end

            % if min(fthetaijint) < obj(i,j)-1e-12
            %     disp('PROBLEM in your formula Atharva')
                  figure; 
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


