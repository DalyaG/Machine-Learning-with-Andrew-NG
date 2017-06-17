function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

dataMat = zeros(K, n, m);

% 1. for each example in X, put the data point in dataMat in the row
%    corresponding to the centroid given by idx

% 1.1 get subsripts
  [subExamples, subDim] = meshgrid(1:m, 1:n);
  subExamples = subExamples(:);
  subDim = subDim(:);

  subCentroids = repmat(idx, 1, n)';
  subCentroids = subCentroids(:);

% 1.2 get linear indices
  indX = sub2ind(size(X), subExamples, subDim);

  indDataMat = sub2ind(size(dataMat), subCentroids, subDim, subExamples);

% 1.3 put data in dataMat
  dataMat(indDataMat) = X(indX);
  

% 2. compute mean to get new centroids

centroids = sum(dataMat, 3) ./ sum(dataMat~=0, 3);



% =============================================================


end

