function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
X = [ones(1,m); X'];
a_1 = X; %Every column is a trial. 
a_2 = [ones(1,m); sigmoid(Theta1 * X)]; 
a_3 = sigmoid(Theta2*a_2);
%Create a matrix K*N where N is the number of trial. Each column is an ...
%experiment. 
yPrep = zeros(size(a_3));
for i=1:m 
    yPrep(y(i),i)=1; 
end 
%Back propagation 
delta_3 = a_3 - yPrep;
delta_2 = (Theta2' * delta_3) .* a_2 .* (1-a_2);
delta_2 = delta_2(2:end, :);
%Start computing the gradient
for i=1:m
    Theta2_grad = Theta2_grad + delta_3(:,i) * (a_2(:,i)');
end 
Theta2_reg = Theta2;
Theta2_reg(:,1) = 0;
Theta2_grad = (1/m) * Theta2_grad + (lambda/m)*Theta2_reg; 
for i=1:m 
    Theta1_grad = Theta1_grad + delta_2(:,i) * (a_1(:,i)');
end 
Theta1_reg = Theta1;
Theta1_reg(:,1) = 0;
Theta1_grad = (1/m) * Theta1_grad + (lambda/m)*Theta1_reg; 
%Start computing the cost 
for i=1:m 
    J = J + yPrep(:,i)' * log(a_3(:,i)) + ...
        (1-yPrep(:,i))' * log(1-a_3(:,i));
end 
%Cost function 
%!!Attention!! Both summation of regularized grad and cost are all 
%over indexes exculde the first column of Thetas. Namely use Theta_reg 
J = (-1/m) * J + (lambda/(2*m)) * (sum(sum(Theta1_reg.^2)) + ...
    sum(sum(Theta2_reg.^2))); 

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
