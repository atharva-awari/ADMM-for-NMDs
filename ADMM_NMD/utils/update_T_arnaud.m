function [T] = update_T_arnaud(X,rho,T,lambda,A, model, error_measure, param)
%% This file contains all the updates for the matrix T, for all different
%% cases of nonlinearities and loss-functions. The user can add more nonlinear
%% models and loss functions easily after solving one-dimensional subproblems
%% for each entry of T
switch model
    %% Case 1: ReLU with Frobenius norm
    case 'ReLU'
        switch error_measure
            case 'Frobenius'
                Y = lambda - rho*A;
                p = (Y >= X);
                q = (Y <= 0);
                r = (Y > 0 & Y < X);

                t1 = (X - Y) ./ (rho + 1);
                t2 = (  - Y) ./ rho;

                T(p) = t2(p);
                T(q) = t1(q);

                g1 = 0.5*(X - t1).^2 + t1.*lambda + 0.5*rho*(t1 - A).^2;
                g2 = 0.5*(X).^2      + t2.*lambda + 0.5*rho*(t2 - A).^2;
                
                ind1better    = false(size(T));
                ind1better(r) = g1(r) < g2(r);
                T( r &  ind1better)  = t1( r &  ind1better);
                T( r & ~ind1better)  = t2( r & ~ind1better);
    %% ReLU with L1 norm
            case 'L1'
                Y  = rho*A - lambda;
                p  = (Y >= 1);
                q  = (Y <= -1);
                r  = Y > 0 & Y < 1;
                s  = (Y >= -1 & Y <= 0);
                
                t1 = Y./rho;
                t2 = max(min((Y + 1)./rho, X), (Y - 1)./rho);
                
                g1 = abs(X)      + t1.*lambda + 0.5*rho*(t1 - A).^2;
                g2 = abs(X - t2) + t2.*lambda + 0.5*rho*(t2 - A).^2;
                
                T(p) = t2(p);
                T(q) = t1(q);
                T(r) = 0;
                
                ind1better    = false(size(T));
                ind1better(s) = g1(s) < g2(s);
                T( s &  ind1better)  = t1( s &  ind1better);
                T( s & ~ind1better)  = t2( s & ~ind1better);
    %% ReLU with KL-divergence
            case 'KL divergence'
                p = X>0;
                q = X==0;
                
                D    = sqrt((lambda - rho*A + 1).^2 + 4*rho.*X);
                T(p) = (rho.*A(p) - lambda(p) - 1 + D(p)) ./ (2*rho);
                
                t1 = (rho.*A - lambda - 1) ./ rho;
                t2 = (rho.*A - lambda) ./ rho;
                Y  = rho.*A - lambda;

                q1 = q & (Y >= 1);
                q2 = q & (Y <= 0);
                q3 = q & (Y > 0 & Y < 1);
                
                T(q1) = t1(q1);
                T(q2) = t2(q2);
                T(q3) = 0;
        end
    %% CSF with frobenius norm
    case 'CSF'
        [m,n] = size(X);
        switch error_measure
            case 'Frobenius'
                for i=1:m
                    for j=1:n
                        a = 2;
                        c = rho - 2 * X(i,j);
                        d = lambda(i,j) - rho * A(i,j);
                        T(i,j) = cardanoatharva(a,c,d);
                    end
                end
    %% CSF with L1 norm
            case 'L1'
                if rho > 2
                    Y  = rho.*A - lambda;
                    t1 = Y./(rho - 2);
                    t2 = Y./(rho + 2);
                else
                    xclamp = max(X,0);
                    t1 =  sqrt(xclamp);
                    t2 = -sqrt(xclamp);
                end
                g1 = abs(X - t1.^2) + t1.*lambda + 0.5*rho.*(t1 - A).^2;
                g2 = abs(X - t2.^2) + t2.*lambda + 0.5*rho.*(t2 - A).^2;
            
                T = t1;
                ind2better    = (g2 < g1);
                T(ind2better) = t2(ind2better);
   %% CSF with KL-divergence
            case 'KL divergence'
                Y   = rho.*A - lambda;
                p   = (X > 0);
                q   = ~p;
                
                D   = sqrt( max( Y.^2 + 8*(rho+2).*X, 0) );
                t1  = ( Y + D ) ./ ( 2*(rho+2) );
                t2  = ( Y - D ) ./ ( 2*(rho+2) );
                
                g1  = t1.^2 - X.*log( max(t1.^2, eps) ) + t1.*lambda + 0.5*rho.*(t1 - A).^2;
                g2  = t2.^2 - X.*log( max(t2.^2, eps) ) + t2.*lambda + 0.5*rho.*(t2 - A).^2;
                
                ind2better = (g2 <= g1);
                T = t1;
                T(ind2better) = t2(ind2better);
                
                T0   = Y ./ (rho+2);
                T(q) = T0(q);
        end
   %% MinMax with Frobenius norm
    case 'MinMax'
        a = param.a;
        b = param.b;
        switch error_measure
            case 'Frobenius'
                tmin   = A - lambda./rho;
                t1star = min(tmin,a);
                t2star = (X + rho.*tmin) ./ (1 + rho);
                t2star = min(max(t2star,a),b);
                t3star = max(tmin,b);
                
                g1 = 0.5*(X - a).^2      + 0.5*rho.*(t1star - tmin).^2;
                g2 = 0.5*(X - t2star).^2 + 0.5*rho.*(t2star - tmin).^2;
                g3 = 0.5*(X - b).^2      + 0.5*rho.*(t3star - tmin).^2;
                
                T = t1star;
                ind2better    = (g2 < g1) & (g2 <= g3);
                ind3better    = (g3 < g1) & (g3 <  g2);
                T(ind2better) = t2star(ind2better);
                T(ind3better) = t3star(ind3better);
   %% MinMax with L1-norm
            case 'L1'
                tmin   = A - lambda./rho;
                t1star = min(tmin, a);
                t3star = max(tmin, b);
                t_med  = max(min(tmin + 1./rho, X), tmin - 1./rho);
                t2star = min(max(t_med, a), b);               
                
                g1 = abs(X - a)      + t1star.*lambda + (rho/2)*(t1star - A).^2 ;
                g2 = abs(X - t2star) + t2star.*lambda + (rho/2)*(t2star - A).^2 ;
                g3 = abs(X - b)      + t3star.*lambda + (rho/2)*(t3star - A).^2;
                
                T = t1star;
                ind2better = g2 < g1 & g2 <= g3;  T(ind2better) = t2star(ind2better);
                ind3better = g3 < g1 & g3 <  g2;  T(ind3better) = t3star(ind3better);
   %% MinMax with KL-divergence
            case 'KL divergence'
                tmin = A - lambda./rho;
                D    = sqrt(max((lambda - rho.*A + 1).^2 + 4*rho.*X, 0));

                t1star   = min(tmin,a);
                t3star   = max(tmin,b);
                t_int    = (rho.*A - lambda - 1 + D) ./ (2*rho);
                z        = (X==0);
                t_int(z) = (rho.*A(z) - lambda(z) - 1) ./ rho;
                t2star   = min(max(t_int,a),b);
                
                f1 = a;
                f2 = t2star;
                f3 = b;
                
                g1 = f1 - X.*log(max(f1,eps)) + t1star.*lambda + 0.5*rho.*(t1star - A).^2;
                g2 = f2 - X.*log(max(f2,eps)) + t2star.*lambda + 0.5*rho.*(t2star - A).^2;
                g3 = f3 - X.*log(max(f3,eps)) + t3star.*lambda + 0.5*rho.*(t3star - A).^2;
                
                g1(z) = f1(z) + t1star(z).*lambda(z) + 0.5*rho.*(t1star(z) - A(z)).^2;
                g2(z) = f2(z) + t2star(z).*lambda(z) + 0.5*rho.*(t2star(z) - A(z)).^2;
                g3(z) = f3(z) + t3star(z).*lambda(z) + 0.5*rho.*(t3star(z) - A(z)).^2;
                
                T = t1star;
                ind2better = (g2 < g1) & (g2 <= g3);
                ind3better = (g3 < g1) & (g3 <  g2);
                T(ind2better) = t2star(ind2better);
                T(ind3better) = t3star(ind3better);
        end
    %% Modulus with Frobenius norm
    case 'Modulus'
        switch error_measure
            case 'Frobenius'
                Y  = rho.*A - lambda;
                t1 = (Y + X) ./ (rho + 1); t1 = max(t1, 0);
                t2 = (Y - X) ./ (rho + 1); t2 = min(t2, 0);
                
                g1 = 0.5*(X - t1).^2 + t1.*lambda + 0.5*rho.*(t1 - A).^2;
                g2 = 0.5*(X + t2).^2 + t2.*lambda + 0.5*rho.*(t2 - A).^2;
                
                T    = t1;
                ind2better    = (g2 < g1);
                T(ind2better) = t2(ind2better);
    %% Modulus with L1-norm
            case 'L1'
                tmin  = A - lambda./rho;

                t1 = max( max( min(tmin + 1./rho, X),  tmin - 1./rho ), 0);
                t2 = min( max( min(tmin + 1./rho, -X), tmin - 1./rho ), 0);
                
                Y = rho.*A - lambda;
                
                T    = t1;
                q    = (Y <= -1);
                T(q) = t2(q);
                
                mid = (Y > -1) & (Y < 1);
                g1  = abs(X - t1) + t1.*lambda + 0.5*rho.*(t1 - A).^2;
                g2  = abs(X + t2) + t2.*lambda + 0.5*rho.*(t2 - A).^2;
                
                ind1better = (g1 < g2);
                T(mid &  ind1better) = t1(mid &  ind1better);
                T(mid & ~ind1better) = t2(mid & ~ind1better);
   %% Modulus with KL-divergence
            case 'KL divergence'
                Y  = rho.*A - lambda;
                p  = (X > 0);
                
                D1 = sqrt( max((1 - Y).^2 + 4*rho.*X, 0) );
                t1 = (Y - 1 + D1) ./ (2*rho);
                
                D2 = sqrt( max((1 + Y).^2 + 4*rho.*X, 0) );
                t2 = (1 + Y - D2) ./ (2*rho);
                
                f1 = t1;
                f2 = -t2;
                
                g1 = f1 - X.*log( max(f1, eps) ) + t1.*lambda + 0.5*rho.*(t1 - A).^2;
                g2 = f2 - X.*log( max(f2, eps) ) + t2.*lambda + 0.5*rho.*(t2 - A).^2;
                
                ind1better = (g1 <= g2);
                T(p &  ind1better) = t1(p &  ind1better);
                T(p & ~ind1better) = t2(p & ~ind1better);
                
                Y1  = (Y >= 1);
                Ym1 = (Y <= -1);
                mid = ~Y1 & ~Ym1;
                
                t1 = (Y - 1) ./ rho;
                t2 = (Y + 1) ./ rho;
                
                q  = ~p; % X==0
                T(q & Y1)  = t1(q & Y1);
                T(q & Ym1) = t2(q & Ym1);
                T(q & mid) = 0;
        end
end