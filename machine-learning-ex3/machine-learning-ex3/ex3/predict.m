function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);
% Add ones to the X data matrix
X = [ones(m, 1) X];

% You need to return the following variables correctly
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

a1 = zeros(size(Theta1, 1), 1);
for i = 1:m
      tmp1 = repmat(X(i,:),size(Theta1, 1),1);   
      tmp1 = tmp1.*Theta1;
      tmp1 = sigmoid(sum(tmp1,2));
      a1 = tmp1;
      a1 = [1;a1];
      tmp2 = sigmoid((a1'*Theta2')');
     [num,ind] = max(tmp2);
      p(i) = ind;   
end
    
        
        
        
        
        
        
        % =========================================================================
        
end
