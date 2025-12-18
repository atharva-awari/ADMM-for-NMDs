function xopt=findmin(a,b,c)
    % Given vectors a, b and c, this function computes
    % xopt = argmin_x ||c-max(0,b+ax)||_2^2

    pos  = abs(a) > 1e-16;

    a = a(pos);
    b = b(pos);
    c = c(pos);
    
    breakpts  = -b./a;
    [breakpts,ind] = sort(breakpts);
    m = length(breakpts);
    
    a = a(ind); sga = sign(a);
    b = b(ind);
    c = c(ind);

    cc = c.*c;
    bb = b.*b;
    aa = a.*a;
    bc = b.*c;
    ac = a.*c;
    ab = a.*b;
    
    %Quadratic function far left
    neg = a<-1e-16;
    ti  = sum(cc)-2*sum(bc(neg))+sum(bb(neg)); %we should add cste (see above) to be exact but we dont care
    tl  = 2*(sum(ab(neg)-ac(neg)));
    tq  = sum(aa(neg));
    
    %The minimum of tq*x^2+tl*x+ti is at xmin=-tl/(2*tq)
    xmin = -tl/(2*tq);  
    
    %Which is the best: the min or the first breakpoint ?
    if xmin<breakpts(1)
        xopt=xmin;
    else
        xopt=breakpts(1);
    end
    %The value of f at xopt
    yopt = ti+tl*xopt+tq*xopt^2;
    
    %For each interval, we check if the min (or the previous breakpoint) is better
    for i=1:m
        ti = ti+sga(i)*(bb(i)-2*bc(i));
        tl = tl+sga(i)*(2*ab(i)-2*ac(i));
        tq = tq+sga(i)*(aa(i));
        xmin = -tl/(2*tq); %the minimum of tq*x^2+tl*x+ti is at xmin=-tl/(2*tq)
        if i+1<=m && (xmin<breakpts(i+1) && xmin>breakpts(i)) || (i+1>m && xmin>breakpts(m))
            xt = xmin;
        else
            xt = breakpts(i);
        end
        if ti+tl*xt+tq*xt^2<yopt
            xopt = xt;
            yopt = ti+tl*xopt+tq*xopt^2;
        end
    end
end