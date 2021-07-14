function [bestT, bestErr] = oneDClass(x, y, name)
% oneDClass applied classification with one input attribute by searching
% the best threshold.
%Inputs
%   x contains values for Class 1
%   y contains values for Class 2
%   name is array of strings:
%       name(1) is name of attribute
%       name(2) is name of the first class
%       name(2) is name of the second class
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
%
    %Define numbers of cases
    Pos = length(x);
    Neg = length(y);
    tot = Pos + Neg;
    
    %Define set of unique values
    thr = unique([x; y])';
    %Add two boders
    thr = [thr(1) * 0.9999, (thr(2:end) + thr(1:end - 1)) / 2,...
        thr(end) * 1.0001];
    errs = zeros(1, length(thr));
    
    %Define meaning of "class 1"
    xLt =  mean(x) > mean(y);
    
    %Define variabled to search
    bestErr = tot;
    bestT = -Inf;
    %Check each threshold
    for k = 1:length(thr)
        t = thr(k);
        nX = sum(x < t);
        nY = sum(y >= t);
        if xLt
            nX = Pos - nX;
            nY = Neg - nY;
        end
        err = 1 - (nX / Pos + nY / Neg) / 2;
        if err < bestErr
            bestErr = err;
            bestT = t;
        end
        errs(k) = err;
    end

    if nargin == 3
        if ~iscell(name)
            name = cellstr(name);
        end
        
        %Define min and max to form bines
        mi = min([x; y]);
        ma = max([x; y]);
        edges = mi:(ma-mi)/20:ma;
        
        %Draw histograms
        figure;
        histogram(x, edges, 'Normalization','probability');
        hold on;
        histogram(y, edges, 'Normalization','probability');
        title(name{1});
        xlabel(['Value of ', name{1}]);
        ylabel('Fraction of cases');
        
        %Draw graph of errors
        sizes = axis();
        plot(thr, errs * sizes(4), 'g');
        %Draw the best threshold
        plot([bestT, bestT], sizes(3:4), 'k', 'LineWidth', 2);
        legend(name{2}, name{3}, 'Error', 'Threshold');
    end
end
