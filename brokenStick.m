function [d, g] = brokenStick(eigvals)
% 
% This function implements the broken stick method of for selecting the
% number of statistically significant principle components.  The method takes 
% any principal components that have variances larger than they would be by chance.  
% Taken from page 38 of 
%
% Exploratory Data Analysis with MATLAB 
% by Martinez and Martinez.
%
% Written by:
% -- 
% John L. Weatherwax                2005-08-14
% 
% email: wax@alum.mit.edu
% 
% Please send comments and especially bug reports to the
% above email address.
% 
%-----

% make sure we have a row vector (with postive values): 
eigvals = abs(eigvals(:).'); 

% Sort eigenvalues in decreasing order:
[ eigvals ] = sort( eigvals,'descend');

% Calculate the proportional variance:
propvar = eigvals / sum(eigvals); 

% calculate the the expected length of the k-th longest segment:
p = length(eigvals); 
g = zeros(1,p);
for k = 1:p 
    for i = k:p 
      g(k) = g(k) + 1/i;
    end 
end
g = g/p;

% Find the cumulative variances that are larger than chance:
inds = find( propvar < g ); 

% Send out the first:
d = inds(1) - 1; 