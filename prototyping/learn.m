function [G, f, chi] = learn(U,V,m,S,K,Y)
%LEARN learn with a kernel encoder/decoder idea

    % U = phi(u); % Column Vector in R^n
    % V = psy(v); % Column Vector in R^m
    
    chi = @(y,V_) K(y,Y)*((K(Y,Y)+1e-12*eye(m)) \ V_); 
    
    N = size(U,2);
    G = cell(m,1);
    sinner = S(U,U);
    % for j = 1:m
    %     inner = sinner \ V(:,j).';
    %     fj = @(u) S(phi(u),U)*inner;
    %     Gj = @(u) chi(fj(u)); % DOES THIS WORK?
    %     G{j} = Gj;
    % end

    inner = (sinner \ V.').';
    f = @(u) S(u,U)*inner;
    G = @(U_) (@(y) chi(y, f(U_)));
end

