function [W,H] = scaling_factors(X,r,W,H)
%% Scaling W and H
for i = 1:r
    norm_w = norm(W(:, i), 2);
    norm_h = norm(H(i, :), 2);
    % Avoid division by zero
    if norm_w == 0 || norm_h == 0
        continue;
    end
    W(:, i) = (W(:, i) / norm_w) * norm_h;
    H(i, :) = (H(i, :) / norm_h) * norm_w;
end
%% Scaling W and H according to the data X
A = W*H;
optscal = sum(sum(A.*X))/sum(A(:).^2);
W = optscal*W;
end