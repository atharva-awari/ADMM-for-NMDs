function [rmse_test,rmse_train]= rmse_comp(X, W, H, model, param)

    M = param.mask;
    M_test = 1-M;
    p_obs = param.p_obs;
    B = param.B;
%% Calculation of rmse on training and testing data
switch model
    case 'ReLU'
     Xhat = max(0, W*H); 
     
    case 'CSF'
        Xhat =  (W*H).^2;

    case 'MinMax'
        b = param.b;
        a = param.a;
        Xhat = min(b, max(a,W*H));

    case 'Modulus'
        Xhat = abs(W*H);
end

     
    
sq_errors_test = (M_test .* (X - Xhat)).^2;     % squared error only on test set
rmse_test = sqrt(sum(sq_errors_test(:)) / sum(M_test(:)));


sq_errors_train = (M .* (X - Xhat)).^2;     % squared error only on train set
rmse_train = sqrt(sum(sq_errors_train(:)) / sum(M(:)));