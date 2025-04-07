function [Theta] = update_Theta(X,rho,Theta,lambda,A, model, error_measure, options)

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
    t2 = (rho*A - lambda ) ./ (rho + 1);

    Theta(p)=t2(p);
    Theta(q)=t1(q);

    g1 = 1/2 * (X - t1).^2 + t1.*lambda + rho/2*(t1 - A).^2;
    g2 = 1/2 * (X).^2 + t2.*lambda + rho/2*(t2 - A).^2;

  


    if g1(r)<g2(r)

        Theta(r) = t1(r);
    else
        Theta(r) = t2(r);
    end

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

    g1 = X +  t1.*lambda + rho/2*(t1 - A).^2;
    

    g2 = abs(X - t2) + t2.*lambda + rho/2*(t2 - A).^2;
    
    

    Theta(p)=t2(p);
    Theta(q)=t1(q);
    Theta(r)=0;


    if g1(s)<g2(s)

        Theta(s) = t1(s);
    else
        Theta(s) = t2(s);
    end
%% ReLU with KL-divergence
    
            
    case 'KL divergence' 

    
    p = find(X>0);
    q = find(X==0);

    
    Theta = max(eps,Theta);

    Theta(p) = (1 / 2*rho) * ((rho*A(p) - lambda(p) - 1) + sqrt((lambda(p) - rho*A(p) + 1).^2 + 4*rho*X(p)));

    t1 = (rho*A - lambda + 1) ./ rho;
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

    Theta(q1) = t1(q1);
    Theta(q2) = t2(q2); 
    Theta(q3) = 0;

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
            Theta(i,j) = cardanoatharva(a,c, d);
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
    
    
    Theta(r)=t2(r);
    Theta(v)=t2(v);
    Theta(s)=t1(s);
    

    else 
        t1 = sqrt(X);
        t2 = -sqrt(X); 
        g1 = abs(X - (t1).^2) + lambda.*t1 + rho/2 .* (t1 - A).^2;
        g2 = abs(X - (t2).^2) + lambda.*t2 + rho/2 .* (t2 - A).^2;
        l = find(g1<g2);
        m = find(g1>g2);
        Theta(l) = t1(l);
        Theta(m) = t2(m); 

        
    end

        

    %% CSF with KL-divergence
      case 'KL divergence' 

    p = find(X>0);
    q = find(X==0);

    t1 = (rho*A - lambda + sqrt((lambda - rho*A).^2 + 8*X.*(rho+2))) ./ 2*(rho+2);
    t2 = (rho*A - lambda - sqrt((lambda - rho*A).^2 + 8*X.*(rho+2))) ./ 2*(rho+2);

    g1 = t1.^2 - X.* log(t1.^2) + t1.*lambda + rho/2 .* (t1-A).^2;
    g2 = t2.^2 - X.* log(t2.^2) + t2.*lambda + rho/2 .* (t2-A).^2;

    if g1(p)<g2(p)

   Theta(p) = t1(p);
   else
    Theta(p) = t2(p);
   end


    Theta(q) = (rho.*A(q) - lambda(q)) ./ (rho+2); 

        end

    %% MinMax with Frobenius norm
    case 'MinMax'
       

        a = options.a;
        b = options.b;
        
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


    g1 = 1/2 * (X-1).^2 +  t1.*lambda + rho/2*(t1 - A).^2;
    

    g2 = 1/2 * (X - t2).^2 + t2.*lambda + rho/2*(t2 - A).^2;

    g3 = 1/2 * (X+1).^2 +  t3.*lambda + rho/2*(t3 - A).^2;
    
    

    Theta(c1)=t1(c1);
    Theta(c2)=t2(c2);
    Theta(c3)=t3(c3);

    if g2(c4)<g3(c4)

       Theta(c4) = t2(c4);
    else
        Theta(c4) = t3(c4);
    end


    if g1(c5)<g2(c5)

       Theta(c5) = t1(c5);
    else
        Theta(c5) = t2(c5);
    end

    Theta(c6)=b;

    Theta(c7)=a;

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


    g1 = abs(X-a) +  t1.*lambda + rho/2*(t1 - A).^2;
    

    g2 = abs(X - t2) + t2.*lambda + rho/2*(t2 - A).^2;

    g3 = abs(X-b) +  t1.*lambda + rho/2*(t1 - A).^2;
    
    

    Theta(c1)=t3(c1);
    Theta(c2)=t2(c2);
    Theta(c3)=t1(c3);

    if g1(c4)<g2(c4)

       Theta(c4) = t1(c4);
    else
        Theta(c4) = t2(c4);
    end

    Theta(c5) = b; 

    if g2(c6)<g3(c6)

       Theta(c6) = t2(c6);
    else
        Theta(c6) = t3(c6);
    end

    

    Theta(c7)=a;

%% MinMax with KL-divergence

   case 'KL divergence' 

    p = find(X>0);
    q = find(X==0);
%% X > 0
t11 = (1 / 2*rho) * (((rho*A - lambda - 1) + sqrt((lambda - rho*A + 1).^2 + 4*rho*X)));
t22 = (1/rho) * (rho*A - lambda);

D = sqrt((lambda - rho*A + 1).^2 + 4*rho*X);
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

g11 = t11 - X.* log(t11) + t11.*lambda + rho/2 .* (t11-A).^2;
g22 = 1 + t22.*lambda + rho/2 .* (t22-A).^2;

Theta(c1) = t22(c1);

if g11(c2) < g22(c2)
    Theta(c2) = t11(c2);
else
    Theta(c2) = t22(c2);
end

Theta(c3) = t11(c3);

Theta(c4) = a;

Theta(c5) = t22(c5); 

if g11(c6) < g22(c6)
    Theta(c6) = t11(c6);
else
    Theta(c6) = t22(c6);
end

Theta(c7) = b;


%% X = 0
t44 = (rho*A - lambda - 1) ./ rho;
t33 = (rho*A - lambda) ./ rho;

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

Theta(q1) = t33(q1);
Theta(q2) = a; 
Theta(q3) = t44(q3);


g44 = t44  + t44.*lambda + rho/2 .* (t44-A).^2;
g33 = 1 + t33.*lambda + rho/2 .* (t33-A).^2;

if g33(q4) < g44(q4)
    Theta(q4) = t33(q4);
else
    Theta(q4) = t44(q4);
end

Theta(q5) = t33(q5);

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

    Theta(p)=t1(p);
    Theta(q)=t2(q);

    g1 = 1/2 * (X - t1).^2 + t1.*lambda + rho/2*(t1 - A).^2;
    g2 = 1/2 * (X + t2).^2 + t2.*lambda + rho/2*(t2 - A).^2;

    



    if g1(r)<g2(r)

        Theta(r) = t1(r);
    else
        Theta(r) = t2(r);
    end

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
     

    g1 = abs(X -t1) +  t1.*lambda + rho/2*(t1 - A).^2;
    g2 = abs(X + t2) + t2.*lambda + rho/2*(t2 - A).^2;
    
    

    Theta(p)=t1(p);
    Theta(q)=t2(q);
    


    if g1(r)<g2(r)

        Theta(r) = t1(r);
    else
        Theta(r) = t2(r);
    end

    %% Modulus with KL-divergence 

case 'KL divergence'

  p = find(X>0);
  q = find(X==0);

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


        end






end
