function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_vec = [0.5, 0.8, 1, 2];
sigma_vec = [0.05, 0.09, 0.15, 0.3];

pred_error = zeros(length(C_vec), length(sigma_vec));

for i = 1:length(C_vec)
  
  C = C_vec(i);
  
  for j = 1:length(sigma_vec)
    
    sigma = sigma_vec(j);
    
    model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    
    predictions = svmPredict(model, Xval);
    pred_error(i,j) = mean(double(predictions ~= yval));
    
  end
end

[~, loc] = min(pred_error(:));
[loc_C, loc_sigma] = ind2sub(size(pred_error), loc);
C = C_vec(loc_C);
sigma = sigma_vec(loc_sigma);

% =========================================================================

end
