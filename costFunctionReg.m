function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%%-y(log(h(x)))-(1-y)log1-h

sig=zeros(m);
sig=theta'*X';
%fprintf('h theta of x\n');
%sig
h_theta=zeros(m);
%fprintf('sigmoid function\n');
h_theta=sigmoid(sig);

%h_theta
%fprintf('log function\n');
logh_theta=zeros(m);
logh_theta=log(h_theta);
%logh_theta
%fprintf('1st term\n')
Aterm=zeros(m);
Aterm=-1*(y'.*logh_theta);
%Aterm

onevector=ones(1, m);
%onevector
%fprintf('1-h theta \n');
h_theta2=zeros(m);
h_theta2=onevector-h_theta;
%h_theta2
%fprintf('log function 2\n');
logh_theta2=zeros(m);
logh_theta2=log(h_theta2);
%logh_theta2
%fprintf('2nd term\n')
Bterm=zeros(m);
Y=onevector-y';
Bterm=-1*(Y.*logh_theta2);
%Bterm

total_term=zeros(m);
total_term=Aterm+Bterm;
%fprintf('total term\n')
%total_term
AplusBby_m=sum(total_term)/m;

%lambda/2m sum(square of theta j=1:n)
%overfitting term
theta_square=zeros(size(theta),1);
theta_square=power(theta,2);
sumtheta_square=sum(theta_square)-theta_square(1);
Cterm=(lambda/(2*m))*sumtheta_square;
%fprintf('cost function\n')
%J
J=AplusBby_m+ Cterm;

x_i=zeros(m);
x_i=X(:,1);
grad(1)=sum((h_theta-y').*x_i')/m;

for i=2:size(theta)
x_i=zeros(m);
x_i=X(:,i);
grad(i)=(sum((h_theta-y').*x_i')/m)+((lambda/m)*theta(i));
endfor

% =============================================================

end
