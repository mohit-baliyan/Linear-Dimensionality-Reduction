function [bestT, bestErr, dir] = fisher(x, y, names)
%Calculate Fisher's discriminant direction,
%project all data points onto this direction, select the optimal threshold
%Inputs
%   x contains values for Class 1
%   y contains values for Class 2
%   names is array of strings:
%       name(1) is name of attribute
%       name(2) is name of the first class
%       name(2) is name of the second class
%       If name is omitted then graphs is not formed.
%
%Outputs
%   bestT is optimal threshold
%   bestErr is minimal error which corresponds to threshold bestT. Error is
%       one minus half of sencsitivity + specificity or 
%       1 - 0.5*(TP/Pos+TN/Neg), where 
%       TP is true positive or the number of correctly recognised casses of
%           the first class, 
%       Pos is the number of casses of the first class,
%       TN is true negative or the number of correctly recognised casses of
%           the secong class, 
%       Neg is the number of casses of the second class.
%   dir is vector with fisher direction
%
    
    % Calculate means
    uMean = mean(y);
    nMean = mean(x);
    % Calculate covariance matrices
    uCov = cov(y);
    nCov = cov(x);
    % Matrix correction
    mat = uCov + nCov;
    mat = mat + 0.001 * max(abs(diag(mat))) * eye(length(uMean));
    
    % Calculate Fisher directions
    dir = mat \ (uMean - nMean)';
    % check the empty direction
    d = sqrt(sum(dir .^ 2));
    if abs(d) < mean(abs(uMean)) * 1.e-5
        bestT = 0;
        bestErr = inf;
        return;
    end
    
    % Normalise Fisher direction
    dir = dir ./ sqrt(sum(dir .^ 2));
    
    %Calculate projection
    projX = x * dir;
    projY = y * dir;
    %Calculate threshold and return result
    if nargin < 3
        [bestT, bestErr] = oneDClass(projX, projY);
    else
        [bestT, bestErr] = oneDClass(projX, projY, names);
    end
end
