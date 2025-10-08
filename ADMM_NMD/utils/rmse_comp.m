%% RMSE computation on test and train data
function [rmse_test,rmse_train]= rmse_comp(X, A, model, param)
%% Validation mask
M_valid = param.validmask;
M_test = 1-M_valid;
%% Calculation of rmse on training and testing data
switch model
    case 'ReLU'
        Xhat = max(0, A);

    case 'CSF'
        Xhat =  (A).^2;

    case 'MinMax'
        b = param.b;
        a = param.a;
        Xhat = min(b, max(a,A));

    case 'Modulus'
        Xhat = abs(A);
end
%% Calculation of Squared errors
sq_errors_test = (M_test .* (X - Xhat)).^2;     % squared error only on test set
rmse_test = sqrt(sum(sq_errors_test(:)) / sum(M_test(:)));
sq_errors_train = (M_valid .* (X - Xhat)).^2;     % squared error only on train set
rmse_train = sqrt(sum(sq_errors_train(:)) / sum(M_valid(:)));