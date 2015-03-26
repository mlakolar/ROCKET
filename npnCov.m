function covM = npnCov(Y)
% Computes the correlation matrix from the nonparanormal data Y
% Method is described in Section 4 of Liu et al. 
% http://jmlr.csail.mit.edu/papers/volume10/liu09a/liu09a.pdf


[n, p] = size(Y);

deltan = 1 / 4 / n^(1/4) / sqrt(pi*log(n));
IX=zeros(size(Y));
for j=1:p
    [~, ~, temp] = unique(Y(:,j));
    IX(:,j)=temp;
end
IX = IX / n;
IX(IX < deltan) = deltan;
IX(IX > 1-deltan) = 1 - deltan;
covM = corr(norminv(IX, 0, 1));

end
