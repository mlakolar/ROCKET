function [eP, eVar, ISa, ISb] = teInference(Y, a, b, methodType, covarianceType, varargin)
% teInference - Inference for an edge in an elliptical coopula model.
%
% Arguments:
%               Y               (n x p) data matrix
%               a, b            index of an edge for which inference is made
%               methodType      number from 1-3 indicating the procedure
%
% Method Type:
%               1 (default)     uses lasso to estimate the neighborhood
%                               of node a and node b
%               2               uses dantzig to estimate the neighborhood
%                               of node a and node b
%               3               user inputs gamma_a and gamma_b
%               4               OLS without model selection
%
% Covariance Type:
%               0 (default)     rank correlation
%               1               pearson correlation
%               2               pearson correlation after marginal
%                               transformatoin


if nargin == 3
    methodType = 1; 
    covarianceType=0;
end
if nargin == 4
    covarianceType=0;
end


assert( a ~= b );        % not sure if our method works for diagonal elements

[n,p] = size(Y);

if methodType == 2
    lambdaDefault = 0.5/sqrt(n)*norminv(1-(.1/log(n))/(2*p));
else
    lambdaDefault = 1.1/sqrt(n)*norminv(1-(.1/log(n))/(2*p));
end
    
[lambda, lambdaWeigth, zeroThreshold, refit, mu, gamma_a, gamma_b] = ...
    process_options(varargin,...
    'lambda', lambdaDefault,...
    'lambdaWeight', ones(p-2, 2), ...
    'zeroThreshold', 1e-4, ...    
    'refit', 1, ...
    'mu', 2, ...
    'gamma_a', zeros(p-2,1), ...
    'gamma_b', zeros(p-2,1) ...
    );

if covarianceType == 1
    covM = corr(Y);
elseif covarianceType == 2
    covM = npnCov(Y);
elseif covarianceType == 0
    covM = rankCovIID(Y);
else 
    error('Wrong gaussianModel');
end

I = setdiff(1:p, [a,b]);

if methodType == 1 || methodType == 2
    XX = covM(I, I);
    Xya = covM(I, a);
    Xyb = covM(I, b);

    if isscalar(lambda)
        lambda_a = lambda;
        lambda_b = lambda;
    else
        lambda_a = lambda(1);
        lambda_b = lambda(2);
    end
    
    if methodType == 1         
        gamma_a = copulaLasso(XX, Xya, lambda_a, lambdaWeigth(:,1), 'Verbose', 0);                
        gamma_b = copulaLasso(XX, Xyb, lambda_b, lambdaWeigth(:,2), 'Verbose', 0);                
    else
        gamma_a = copulaDantzig(XX, Xya, lambda_a, mu);
        gamma_b = copulaDantzig(XX, Xyb, lambda_b, mu);
    end

    S_a = abs(gamma_a) > zeroThreshold ;
    S_b = abs(gamma_b) > zeroThreshold ;    
    ISa = I(S_a); ISb = I(S_b);
    S = union(ISa, ISb);        
    
    if refit
        gamma_a = covM(S, S) \ covM(S, a);
        gamma_b = covM(S, S) \ covM(S, b);    
    else
        S = I;
    end
    
elseif methodType == 3
    % do nothing gamma_a and gamma_b should be given by user
    S = I;
    ISa = I;
    ISb = I;    
elseif methodType == 4
    S = I;   
    ISa = I;
    ISb = I;
    gamma_a = covM(S, S) \ covM(S, a);
    gamma_b = covM(S, S) \ covM(S, b);    
else
    error('Method type not implemented');
end

% compute theta
thetaM = covM([a,b], [a,b]) ...
        - [gamma_a, gamma_b]' * covM(S, [a,b]) ...
        - covM(S, [a,b])' * [gamma_a, gamma_b] ...
        + [gamma_a, gamma_b]' * covM(S, S) * [gamma_a, gamma_b];
iThetaM = inv(thetaM);
eP = iThetaM(1,2);

% compute variance
if covarianceType
    eVar = (iThetaM(1,1)*iThetaM(2,2)+iThetaM(1,2)^2) / n;     
else
    S = [a,b,S];
    hs = length(S);
    sCovM = covM(S, S);
    
    T = 2/pi*asin(sCovM(:));
    eVar = 0;
    
    a = 1; b = 2;
    ua = zeros(hs, 1); ub = zeros(hs, 1);
    ua(a) = 1; ua(b) = 0; ua(3:hs) = -gamma_a;
    ub(a) = 0; ub(b) = 1; ub(3:hs) = -gamma_b;    
    
    x = sqrt(1 - sCovM(:).^2) .* kron(ua, ub);
    
    for i=1:n
        J = (1:n) ~= i;
        tmp = sign(repmat(Y(i, S), n-1, 1) - Y(J, S));
        tmp = repmat(tmp, [1 1 hs]);
        tmp = tmp .* permute(tmp, [1 3 2]);
        tmp = reshape(tmp, n-1, hs*hs);
        eVar = eVar + ((sum(tmp) / (n-1) - T')*x)^2;
    end
    eVar = pi^2/n * eVar * (n-1) / (n-2)^2;    
    eVar = eVar / (thetaM(1,1)*thetaM(2,2) - thetaM(1,2)^2)^2;
end


end
