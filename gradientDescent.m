function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    differential=1/m*(X'*X*theta-X'*y);
	  theta=theta-alpha.*differential;
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
end

end
