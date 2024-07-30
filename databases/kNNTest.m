% Load data
load('Cryotherapy.mat')
% Calculate all distances
len = sum(X .^ 2, 2);
dist = len + len' - 2 * (X * X');
[~, ind] = sort(dist, 2);
% chack results
pred = Y(ind);