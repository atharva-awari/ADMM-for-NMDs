function [T] = update_T(X,rho,T,lambda,A, model, error_measure, param)
%% This file contains all the updates for the matrix T, for all different 
%% cases of nonlinearities and loss-functions. The user can add more nonlinear
%% models and loss functions easily after solving one-dimensional subproblems
%% for each entry of T

switch model
 
     %% Case 1: ReLU with Frobenius norm
    case 'ReLU'
   
        switch error_measure
            case 'Frobenius'
         
                
    Y =lambda - rho*A;


    p = find(Y>X);
    q = find(Y<0);
    r1 = find(Y> 0);
    r2 = find(Y<X);
    r = intersect(r1,r2);


    t1 = (rho*A - lambda + X ) ./ (rho + 1);
    t2 = (rho*A - lambda ) ./ (rho);

    T(p)=t2(p);
    T(q)=t1(q);

    g1 = 1/2 * (X - t1).^2 + t1.*lambda + ((rho/2)*(t1 - A).^2);
    g2 = 1/2 * (X).^2 + t2.*lambda + ((rho/2)*(t2 - A).^2);

  
   
   ind1 = find( g1(r)<g2(r) ) ;
   ind2 = find (g1(r) >= g2(r));
   T(r(ind1)) = t1(r(ind1));
   T(r(ind2)) = t2(r(ind2));
   

   %% ReLU with L1 norm
        case 'L1'
        
            
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

    g1 = X +  t1.*lambda + ((rho/2)*(t1 - A).^2);
    

    g2 = abs(X - t2) + t2.*lambda + ((rho/2)*(t2 - A).^2);
    
    

    T(p)=t2(p);
    T(q)=t1(q);
    T(r)=0;
    
    ind1 = find( g1(s)<g2(s) ) ;
   ind2 = find (g1(s) >= g2(s));
   T(s(ind1)) = t1(s(ind1));
   T(s(ind2)) = t2(s(ind2));

   
%% ReLU with KL-divergence
    
            
    case 'KL divergence' 

    
    p = find(X>0);
    q = find(X==0);

    D = sqrt((lambda - rho*A + 1).^2 + 4*rho*X);
    T(p) =  (rho*A(p) - lambda(p) - 1 + D(p))./ (2*rho);
    

    
    t1 = (rho*A - lambda - 1) ./ rho;
    t2 = (rho*A - lambda) ./ rho;
    
    Y = rho*A - lambda;
    
    y1 = find(Y>1);
    q1 = intersect(q,y1);
    
    y2 = find(Y<0);
    q2 = intersect(q,y2);
    
    y3 = find(Y>0);
    y4 = find(Y<1);
    y5 = intersect(y3,y4);
    q3 = intersect(q,y5); 

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

            a =2;
            c =rho - 2 * X(i,j);
            d = lambda(i,j) - rho * A(i,j);
            T(i,j) = cardanoatharva(a,c, d);
        end
        end

        %% CSF with L1 norm
    
            
    case 'L1' 

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
    
    
    T(r)=t2(r);
    T(v)=t2(v);
    T(s)=t1(s);
    

    else 
        t1 = sqrt(X);
        t2 = -sqrt(X); 
        g1 = abs(X - (t1).^2) + lambda.*t1 + ((rho/2) .* (t1 - A).^2);
        g2 = abs(X - (t2).^2) + lambda.*t2 + ((rho/2) .* (t2 - A).^2);
        l = find(g1<g2);
        m = find(g1>g2);
        T(l) = t1(l);
        T(m) = t2(m); 

        
    end

        

    %% CSF with KL-divergence
      case 'KL divergence' 

    p = find(X>0);
    q = find(X==0);

    D = sqrt(((lambda - rho.*A).^2) + (8*X.*(rho+2)));
    t1 = (rho.*A - lambda + D ) ./ (2*(rho+2));
    t2 = (rho.*A - lambda - D) ./ (2*(rho+2));

    g1 = t1.^2 - X.* log(t1.^2) + t1.*lambda + ((rho/2) .* (t1-A).^2);
    g2 = t2.^2 - X.* log(t2.^2) + t2.*lambda + ((rho/2) .* (t2-A).^2);


   ind1 = find( g1(p)<g2(p) ) ;
   ind2 = find (g1(p) >= g2(p));

   T(p(ind1)) = t1(p(ind1));

   T(p(ind2)) = t2(p(ind2));


    T(q) = (rho.*A(q) - lambda(q)) ./ (rho+2); 

        end

    %% MinMax with Frobenius norm
    case 'MinMax'
       

        a = param.a;
        b = param.b;
        
        switch error_measure

      case 'Frobenius'

   Y =rho*A - lambda;


   c1_1 = find(Y > (rho*b));
   c1_2 = find(Y > (b + rho*b -X));
   c1   = intersect(c1_1,c1_2);

   c2_1 = find(Y > (rho*a));
   c2_2 = find (Y < (rho*b)); 
   c2_3 = intersect(c2_1,c2_2);
   c2_4 = find(Y > (a*rho -X +a));
   c2_5 = find (Y < (b+rho*b-X)); 
   c2_6 = intersect(c2_4,c2_5);
   c2   = intersect(c2_3,c2_6);


   c3_1 = find(Y < (rho*a));
   c3_2 = find(Y < (a + rho*a -X));
   c3   = intersect(c3_1,c3_2);

   c4_1 = find(Y < (rho*a));
   c4_2 = find(Y > (a + rho*a -X));
   c4_3 = find(Y < (b + rho*b -X));
   c4_4 = intersect(c4_2,c4_3);
   c4   = intersect(c4_1,c4_4); 

   c5_1 = find (Y > rho*b);
   c5_2 = find(Y > (a + rho*a -X));
   c5_3 = find(Y < (b + rho*b -X));
   c5_4   = intersect(c5_2,c5_3);
   c5   = intersect(c5_1,c5_4);

   c6_1 = find(Y > (rho*a));
   c6_2 = find(Y < (rho*b));
   c6_3 = intersect(c6_1,c6_2);
   c6_4 = find(Y> (b +rho*b -X));
   c6 = intersect(c6_3,c6_4);

   
   c7   = intersect(c3_2,c6_3);


    t1 = (rho*A - lambda) ./ (rho);
    
   t2  = (X + rho*A - lambda) ./ (rho+1);

    t3 = t1; 


    g1 = 1/2 * (X-1).^2 +  t1.*lambda + ((rho/2)*(t1 - A).^2);
    

    g2 = 1/2 * (X - t2).^2 + t2.*lambda + ((rho/2)*(t2 - A).^2);

    g3 = 1/2 * (X+1).^2 +  t3.*lambda + ((rho/2)*(t3 - A).^2);
    
    

    T(c1)=t1(c1);
    T(c2)=t2(c2);
    T(c3)=t3(c3);
    
    ind1 = find( g2(c4)<g3(c4) ) ;
    ind2 = find (g2(c4) >= g3(c4));
    T(c4(ind1)) = t2(c4(ind1));
    T(c4(ind2)) = t3(c4(ind2));
    
    
    ind3 = find( g1(c5)<g2(c5) ) ;
    ind4 = find (g1(c5) >= g2(c5));
    T(c5(ind3)) = t1(c5(ind3));
    T(c5(ind4)) = t2(c5(ind4));

    

    T(c6)=b;

    T(c7)=a;

%% MinMax with L1-norm
    case 'L1'
   
        
   Y =rho*A - lambda;


   
   c1 = find(Y > ((rho*b)+1));

   c2_1 = find(Y < rho*b);
   c2_2 = find(Y > rho*a);
   c2_3 = intersect(c2_1,c2_2);
   c2_4 = find(Y < rho*b +1);
   c2_5 = find(Y > rho*a+1);
   c2_6 = intersect(c2_4,c2_5);
   c2_7 = intersect(c2_3,c2_6);
   c2_8 = find(Y < rho*b -1);
   c2_9 = find(Y > rho*a-1);
   c2_10 = intersect(c2_8,c2_9);
   c2 = intersect(c2_10, c2_7);
   


   c3 = find(Y < (rho*a-1));

   c4_1 = find(Y < (rho*a));
   c4_2 = intersect(c2_6,c2_10);
   c4   = intersect(c4_1,c4_2);
   

   c5_1 = c2_3;
   c5_2 = find(Y > (rho*b-1));
   c5_3 = find(Y > (rho*b+1));
   c5_4 = intersect(c5_2,c5_3);
   c5   = intersect(c5_4,c5_1);

   c6_1 = find(Y > (rho*b));
   c6_2 = c4_2;
   c6 = intersect(c6_1,c6_2);

   c7_1 = find(Y< (rho*a-1));
   c7_2 = find(Y< (rho*a +1));
   c7_3   = intersect(c7_1,c7_2);
   c7 = intersect(c7_3,c2_3);


    t1 = (rho*A - lambda) ./ (rho);
    
    t21 = (rho*A - lambda -1 ) ./ (rho);
    t22 = (rho*A - lambda +1 ) ./ (rho);
    t23 = X; 
    t2 = max(min(t22,X), t21);

    t3 = t1; 


    g1 = abs(X-a) +  t1.*lambda + ((rho/2)*(t1 - A).^2);
    

    g2 = abs(X - t2) + t2.*lambda + ((rho/2)*(t2 - A).^2);

    g3 = abs(X-b) +  t1.*lambda + ((rho/2)*(t1 - A).^2);
    
    

    T(c1)=t3(c1);
    T(c2)=t2(c2);
    T(c3)=t1(c3);


    ind1 = find( g1(c4)<g2(c4) ) ;
    ind2 = find (g1(c4) >= g2(c4));
    T(c4(ind1)) = t1(c4(ind1));
    T(c4(ind2)) = t2(c4(ind2));
    
    

    T(c5) = b; 
    

    ind3 = find( g2(c6)<g3(c6) ) ;
    ind4 = find (g2(c6) >= g3(c6));
    T(c6(ind3)) = t2(c6(ind3));
    T(c6(ind4)) = t3(c6(ind4));
    

    T(c7)=a;

%% MinMax with KL-divergence

   case 'KL divergence' 

    p = find(X>0);
    q = find(X==0);
%% X > 0
D = sqrt((lambda - rho*A + 1).^2 + 4*rho*X);
t11 = (rho*A - lambda) ./ (rho);
t22 =(rho*A - lambda - 1 + D) ./ (2*rho);
t33 = t11;


Y = rho*A - lambda;

c11 = find(Y>(rho*b));
c12 = find(Y > (2*rho*b + 1 - D));
c13 = intersect(c11,c12);
c1 = intersect(c13,p); 

c21 = find (Y < (2*rho*b + 1 - D));
c22 = find (Y > (2*rho*a + 1 - D));
c23 = intersect(c21,c22);
c24 = intersect(c23,c11);
c2  = intersect(c24,p);

c31 = find(Y < (rho*b));
c32 = find(Y > (rho*a));
c33 = intersect(c31,c32);
c34 = intersect(c33, c23);
c3  = intersect(c34,p);

c41 = find (Y < (2*rho*a + 1 - D));
c42= intersect(c33,c41);
c4 = intersect(c42,p);

c51 = find(Y < (rho*a));
c52 = intersect(c41,c51);
c5  = intersect(c52,p);

c61 = find(Y < (rho*a));
c62 = intersect(c61, c23);
c6 = intersect(c62,p);

c71 = find (Y > (2*rho*b + 1 - D));
c72 = intersect(c33,c71);
c7 = intersect(c72,p);

g11 = b - X.*log(b) + t11.*lambda + ((rho/2) .* (t11-A).^2);
g22 = t22 - X.* log(t22) + t22.*lambda + ((rho/2) .* (t22-A).^2);
g33 = a - X.*log(a) + t33.*lambda + ((rho/2) .* (t33-A).^2);


T(c1) = t11(c1);

ind1 = find( g11(c2)<g22(c2) ) ;
ind2 = find (g11(c2) >= g22(c2));
Theta(c2(ind1)) = t11(c2(ind1));
Theta(c2(ind2)) = t22(c2(ind2));

T(c3) = t22(c3);

T(c4) = a;

T(c5) = t33(c5); 

ind3 = find( g22(c6)<g33(c6) ) ;
ind4 = find (g22(c6) >= g33(c6));
T(c6(ind3)) = t22(c6(ind3));
T(c6(ind4)) = t33(c6(ind4));

T(c7) = b;


%% X = 0
t44 = (rho*A - lambda) ./ rho;
t55 = (rho*A - lambda - 1) ./ rho;
t66 = t44;

Y = rho*A - lambda;

y1 = find(Y< (rho*a));
q1 = intersect(q,y1);

y21 = find(Y< (rho*b));
y22 = find(Y> (rho*a));
y23 = intersect(y21,y22);
y24 = find(Y < (rho*a +1));
y2  = intersect(y23,y24);
q2 = intersect(q,y2);

y31 = find(Y> (rho*a+1));
y32 = find(Y< (rho*b+1));
y33 = intersect(y31,y32);
y34 = intersect(y33,y23);
q3 = intersect(q,y34); 

y41 = find(Y > (rho*b));
y42 = intersect(y41,y33);
q4 = intersect(q,y42);

y51 = find(Y > (rho*b +1));
q5 = intersect(q,y51);

T(q1) = t44(q1);
T(q2) = a; 
T(q3) = t55(q3);



g44 = a + t44.*lambda + ((rho/2) .* (t44-A).^2);
g55 = t55  + t55.*lambda + ((rho/2) .* (t55-A).^2);
g66 = b + t66.*lambda + ((rho/2) .* (t66-A).^2);

ind5 = find( g55(q4)<g66(q4) ) ;
ind6 = find (g55(q4) >= g66(q4));
T(q4(ind5)) = t55(q4(ind5));
T(q4(ind6)) = t66(q4(ind6));


T(q5) = t66(q5);

        end

%% Modulus with Frobenius norm
    case 'Modulus' 

        switch error_measure

        case 'Frobenius'

        Y = rho*A - lambda;


    p = find(Y>X);
    q = find(Y<(-X));
    r1 = find(Y> (-X));
    r2 = find(Y<X);
    r = intersect(r1,r2);


    t1 = (rho*A - lambda + X ) ./ (rho + 1);
    t2 = (rho*A - lambda - X ) ./ (rho + 1);

    T(p)=t1(p);
    T(q)=t2(q);

    g1 = 1/2 * (X - t1).^2 + t1.*lambda + ((rho/2)*(t1 - A).^2);
    g2 = 1/2 * (X + t2).^2 + t2.*lambda + ((rho/2)*(t2 - A).^2);

    

ind1 = find( g1(r)<g2(r) ) ;
ind2 = find (g1(r) >= g2(r));
T(r(ind1)) = t1(r(ind1));
T(r(ind2)) = t2(r(ind2));

   

    %% Modulus woth L1-norm

            case 'L1' 

    Y =rho*A - lambda;


    p = find(Y>1);
    q = find(Y<(-1));
    r1 = find(Y> (-1));
    r2 = find(Y<1);
    r = intersect(r1,r2);
   


    t11 = (rho*A - lambda +1) ./ (rho);
    t22 = (rho*A - lambda -1 ) ./ (rho);
   
    t1 = max(min(t11,X), t22);
    t2 = max(min(t11,-X), t22); 
     

    g1 = abs(X -t1) +  t1.*lambda + ((rho/2)*(t1 - A).^2);
    g2 = abs(X + t2) + t2.*lambda + ((rho/2)*(t2 - A).^2);
    
    

    T(p)=t1(p);
    T(q)=t2(q);
    
ind1 = find( g1(r)<g2(r) ) ;
ind2 = find (g1(r) >= g2(r));
T(r(ind1)) = t1(r(ind1));
T(r(ind2)) = t2(r(ind2));

    
   

    %% Modulus with KL-divergence 

case 'KL divergence'

  p = find(X>0);
  q = find(X==0);
D = sqrt((lambda - rho*A + 1).^2 + 4*rho*X);
T(p) =  (rho*A(p) - lambda(p) - 1 + D(p)) ./ (2*rho) ;

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

T(q1) = t1(q1);
T(q2) = t2(q2); 
T(q3) = 0;


        end






end
