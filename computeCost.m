function J = computeCost(X, y, theta)

m = length(y); % number of training examples
J = 0;

J=1/(2*m)*sum((X*theta-y).^2);    % Cost function for linear regression

end
