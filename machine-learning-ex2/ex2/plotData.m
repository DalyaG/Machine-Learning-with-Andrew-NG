function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%


% seperate positive from negative
idx_pos = find(y);
idx_neg = setdiff(1:length(y), idx_pos);
X_pos = X(idx_pos, :);
X_neg = X(idx_neg, :);


plot(X_pos(:,1), X_pos(:,2), 'k+', 'MarkerSize', 4, X_neg(:,1), X_neg(:,2), 'ko', 'MarkerSize', 4);


% =========================================================================



hold off;

end
