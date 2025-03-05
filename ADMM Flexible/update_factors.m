function [Theta, W, H, lambda, A] = update_factors(Theta, W, H, lambda, A, rho)

    Ht=H';
    HHt=H*Ht;
    LW=norm(HHt);
    W=W-1/(rho*LW)*(-lambda*Ht+rho*(W*HHt-Theta*Ht));


    Wt=W';
    WtW=Wt*W;
    LH=norm(WtW);
    H=H-1/(rho*LH)*(-Wt*lambda+rho*(WtW*H-Wt*Theta));

    A = W*H;

    lambda = lambda + rho*(Theta - A);
    
end

