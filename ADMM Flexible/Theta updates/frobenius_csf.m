clear all; clc;

m=8;
n=6;
r = 3;
W0 = randn(m,r);
H0 = randn(r,n);
X = ( W0*H0).^2;


lambda = randn(m,n);
rho=5;


[U,S,V] = svds(X,r);
Theta = U*S*V';
W = U;
H = S*V';

A = W*H;

for i=1:m
        for j=1:n

            a =2;
            c =rho - 2 * X(i,j);
            d = lambda(i,j) - rho * A(i,j);
            Theta(i,j) = cardanoatharva(a,c, d);
        end
end

 obj = 1/2*(X - (Theta).^2).^2 + lambda.*Theta + rho/2 .* (Theta - A).^2;

    for i = 1 : m
        for j = 1 : n
            thetaijint = [Theta(i,j)-10:0.01:Theta(i,j)+10]; 
            for k = 1 : length(thetaijint)
                fthetaijint(k) = 1/2*(X(i,j) - (thetaijint(k)).^2).^2 + lambda(i,j).*thetaijint(k) + rho/2 .* (thetaijint(k) - A(i,j)).^2; 
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


