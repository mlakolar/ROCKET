function covM = rankCovIID(Y)
% Computes Kendall's tau correlation matrix

[n, p] = size(Y);
covM = zeros(p, p);
denom = 0;

for i=1:n
    t = sign(repmat(Y(i, :), (n-i), 1) - Y(i+1:n, :));
    covM = covM + (t' * t);
    denom = denom + (n-i);
end

covM = covM / denom;
covM = sin(pi/2*covM);

end


