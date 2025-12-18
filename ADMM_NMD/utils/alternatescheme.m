function [W,H,W_stop,H_stop,err,iter,savetime] = alternatescheme(X,W,H,param)

% Computes an approximate solution of the following non linear matrix
% decomposition problem (NMD)
%
%       min_{W,H} ||X-max(0,WH)||_F^2
%
% using coordinate descent algorithm
%
%****** Input ******
%   X       : m-by-n matrix, sparse and non negative
%   W  : m-by-r matrix
%   H  : r-by-n matrix; r is the desired rank of the factorization
%    
%   
%   param   : structure, containing the parameters of the algorithm
%  
%       .maxit  = maximum number of iterations (default: 1000)
%       .tol    = tolerance on the relative error (default: 1.e-4)
%       .time   = maximum runtime of the algorithm
%       .stop   = records the solution after "stop" number of iterations
% ****** Output ******
%   W       : m-by-r low-rank factor W
%   H       : r-by-n low-rank factor H
%   W_stop  : m-by-r low-rank factor W after "stop" number of iterations
%   H_stop  : r-by-n low-rank factor H after "stop" number of iterations
%   err     : vector containing evolution of relative error along
%             iterations ||X-max(0,WH)||_F / || X ||_F
%   iter    : number of iterations to reach the stopping condition

% Note: In order to use multi-threaded version of the code, you need the 
%       Parallel Computing Toolbox of MATLAB

% See the paper ''Coordinate Descent Algorithm Nonlinear Matrix Decomposition 
% with the ReLU function'', Atharva Awari, Harrison Nguyen, Samuel Wertz, 
% Arnaud Vandaeleand Nicolas Gillis, 2024. 

[m,n]=size(X);
if nargin < 4
    param = [];
end

if ~isfield(param,'maxit')
    param.maxit = 1000;
end
if ~isfield(param,'tol')
    param.tol = 1e-4;
end

if ~isfield(param,'time')
    param.time = 1000;
end

if ~isfield(param,'stop')
    param.stop = param.maxit/2;
end

cpu0=tic;
compute_obj=0;
W_stop = W;
H_stop = H;

% Scale W,H 
Z = max(0,W*H); 
optscal = sum(sum(Z.*X))/sum(Z(:).^2); 
W = optscal*W;
normX=norm(X,'fro');
iter=1;
start_obj=tic;
compute_obj=compute_obj+toc(start_obj);
runtime=toc(cpu0)-compute_obj;
savetime(iter)=runtime;

err = [norm(X-optscal*Z,'fro')/normX];
    while iter < param.maxit && runtime<param.time
        H   = updateH(X,W,H);
        W   = updateH(X',H',W')';
        err = [err norm(X-max(0,W*H),'fro')/normX];
        if err(end)<=param.tol
           break;
        end

        if iter==param.stop
            W_stop = W;
            H_stop = H;
        end
    
    iter=iter+1;
    start_obj=tic;
    compute_obj=compute_obj+toc(start_obj);
    runtime=toc(cpu0)-compute_obj;
  
    savetime(iter)=runtime;
    end
end

function H = updateH(X,W,H)
    [r,n] = size(H);
    WH    = W*H;
    
    parfor j = 1:n
        c = X(:,j);
        for i = 1:r
            a       = W(:,i);
            b       = WH(:,j)-W(:,i)*H(i,j);
            Hijold  = H(i,j);
            H(i,j)  = findmin(a,b,c);           
            WH(:,j)    = WH(:,j)+W(:,i)*(H(i,j)-Hijold);
        end
    end
end