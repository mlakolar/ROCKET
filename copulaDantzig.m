function varargout = copulaDantzig( XX, Xy, delta, mu )
% Uses TFOCS to solve the following Dantzig like optimization probelm
%
%        minimize norm(x,1) + (1/2)*mu*norm(x-x0).^2
%        s.t.     norm(XX*x-Xy,Inf) <= delta


x0   = []; 
z0   = []; 
opts = struct();
opts.printEvery = 0;

% Call TFOCS
objectiveF = prox_l1;
affineF    = { XX, -Xy };
dualproxF  = prox_l1( delta );
[varargout{1:max(nargout,1)}] = ...
    tfocs_SCD( objectiveF, affineF, dualproxF, mu, x0, z0, opts );

