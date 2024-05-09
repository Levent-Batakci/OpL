function [G, f, chi] = learn(U,V,m,S,K,Y)
%LEARN learn with a kernel encoder/decoder idea

    % U = phi(u); % Column Vector in R^n
    % V = psy(v); % Column Vector in R^m

    chi = @(y,V_) K(y,Y)*((K(Y,Y)+1e-8*eye(m)) \ V_); 
    
    N = size(U,2);
    G = cell(m,1);
    sinner = S(U,U);

    inner = ((sinner+1e-8*eye(N,N)) \ V.');
    f = @(u) (S(u,U)*inner).';
    G = @(U_) (@(y) chi(y, f(U_)));
end

