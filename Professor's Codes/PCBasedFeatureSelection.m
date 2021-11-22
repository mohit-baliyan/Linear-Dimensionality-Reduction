function res = PCBasedFeatureSelection(X, selectPC, selectAttr)
% PCBasedFeatureSelection provide feature selection on the base of
% principal components (PC). 
%
% Inputs:
%   X is n-by-m data matrix or covariance matrix or correlation matrix. If
%       X is data matrix then PCs are calculated for covariance matrix.
%   selectPC specifies method to select number of PCs. There are three
%       methods:
%       'Kaiser' for Kaiser rule: informative PCs have eigenvalue which is
%           not less than sum(eigenvalues)/m.
%       'BrokenStick' for brocken stick rule.
%       Real number for conditional number: informative PCs have
%           eigenvalues which are not less than maximal PC divided by
%           specified number. Default value is 10.
%   selectAttr specifies method to select informative attributes in PC.
%       There are three methods:
%       'Kaiser' for Kaiser rule: informative attribute have absolute value
%           of coefficient in PC off not less than 1/sqrt(m).
%       'BrokenStick' for brocken stick rule applied to absolute values of
%           PC coordinates.
%       Real number for conditional number: informative attribute have
%           absolute value of coefficient in PC off not less than 
%           maximal coefficient divided by specified number. Default value
%           is 10. 
%
% Output:
%   res is array of structures with following fields:
%       nAttr is number of attributes used in analysis.
%       sumEigenvalues is sum of eigenvalues.
%       infEigenvalues is array of eigenvalues for informative PCs.
%       infPCs is matrix with informative PCs as columns.
%       maskPCs is matrix with boolean mask of informative attributes in
%           PCs as columns.
%       toRemove is list of attributes to remove (noninfromative)

    % Check what we have
    if ~ismatrix(X) || ~isnumeric(X)
        error(['X must be n-by-m data matrix or covariance matrix or\n',...
            'correlation matrix. If X is data matrix then PCs\n',...
            ' are calculated for covariance matrix.']);
    end
    % remove nans
    X(any(isnan(X),2),:) = [];
    % Get size
    [n, m] = size(X);
    if n ~= m
        X = cov(X);
    end
    
    % Who select PCs
    if isnumeric(selectPC)
        % Conditional number
        selectPC = @(vals, total) CondNumber(vals, selectPC);
    elseif ischar(selectPC)
        if strcmpi(selectPC,'Kaiser')
            selectPC = @(vals, total) Kaiser(vals, total);
        elseif strcmpi(selectPC,'BrokenStick')
            selectPC = @(vals, total) BrokenStick(vals);
        else
            error(['Invalid value of selectPC attribute:"', selectPC, '"']);
        end
    else
        error(['Invalid value of selectPC attribute:"', selectPC, '"']);
    end
    
    % Who select Attributes
    if isnumeric(selectAttr)
        % Conditional number
        selectAttr = @(vals, total) CondNumber(vals, selectAttr);
    elseif ischar(selectAttr)
        if strcmpi(selectAttr,'Kaiser')
            selectAttr = @(vals, total) Kaiser(vals, total);
        elseif strcmpi(selectAttr,'BrokenStick')
            selectAttr = @(vals, total) BrokenStick(vals);
        else
            error(['Invalid value of selectPC attribute:"', selectAttr, '"']);
        end
    else
        error(['Invalid value of selectPC attribute:"', selectAttr, '"']);
    end

    % Create arrays for results
    nAttr = size(X, 1);
    attrs = 1:nAttr;
    res = [];
    % The main loop of selecton
    while true
        % Calculate eigenvalues and eigen vectors
        r = struct;
        r.nAttr = nAttr;
        [V, D] = eig(X);
        D = abs(diag(D));
        r.sumEigenvalues = sum(D);
        ind = selectPC(D, r.sumEigenvalues / nAttr);
        r.infEigenvalues = D(ind);
        r.infPCs = V(:, ind); 
        r.maskPCs = r.infPCs;
        tmp = 1 / sqrt(nAttr);
        for k = 1:size(r.maskPCs, 2)
            r.maskPCs(:, k) = selectAttr(abs(r.maskPCs(:, k)), tmp);
        end
        % Select attributes to remove: noninformative for all PCs
        ind = sum(r.maskPCs, 2);
        ind = ind == 0;
        r.toRemove = attrs(ind);
        res = [res, r];
        if sum(ind) > 0 && sum(ind) < nAttr
            % if there are noninformative attributes and not all attributes
            % are not infromative then continue
            % 1. Remove corresponding attributes from matrix X
            X(ind, :) = [];
            X(:, ind) = [];
            % 2. Remove corresponding attribute
            attrs(ind) = [];
            % 3. Change number of attributes under consideration
            nAttr = nAttr - sum(ind);
        else
            break
        end
    end
    
    
end

function ind = Kaiser(vals, thres)
% Kaiser implements Kaise rule to select informative PCs or attributes
% Inputs:
%   vals is vector of values to define informative elements.
%   thres is threshold of informativeness
%
% Outputs:
%   ind is boolean vector with true for informative elements of vals and
%       false for all other elements.

    % Form index to return
    ind = vals >= thres;
end

function ind = BrokenStick(vals)
% BrokenStick implements Broken Stick rule to select informative PCs or
% attributes.
% Inputs:
%   vals is vector of values to define informative elements.
%   the second attribute is not used.
%
% Outputs:
%   ind is boolean vector with true for informative elements of vals and
%       false for all other elements.

    % Make sure we have a row vector with postive values:
    vals = abs(vals(:).');

    % Sort values in decreasing order:
    [vals, ind] = sort( vals, 'descend');

    % Calculate the percentage:
    perc = vals / sum(vals);

    % calculate the the expected length of the k-th longest segment:
    p = length(vals);
    g = zeros(1,p);
    for k = 1:p
        for i = k:p
            g(k) = g(k) + 1/i;
        end
    end
    g = g / p;

    % Find the percentage that are larger than chance:
    inds = find( perc < g );

    % Send out the first:
    d = inds(1) - 1; 

    % Form index to return
    ind = ind <= d;
end

function ind = CondNumber(vals, CN)
% CondNumber implements Conditional number rule to select informative PCs
% ot attributes.
% Inputs:
%   vals is vector of values to define informative elements.
%   CN is maximal acceptable conditional number: attribute or PC k is
%       informative if vals(k) >= max(vals) / CN
%
% Outputs:
%   ind is boolean vector with true for informative elements of vals and
%       false for all other elements.

    % We work with absolute values of vals
    vals = abs(vals);
    % Calculate threshold
    thr = max(vals) / CN;
    % Form index to return
    ind = vals >= thr;
end
