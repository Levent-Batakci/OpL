% Levent Batakci            %
% AMATH563 - Final project  %
% Operator learning (OpL)   %

% This script is just for me to do some fast prototyping
%   of the kernel method. I work with the basic example G = d/dx


% Functions for data creation
f1 = @(x) x; df1 = @(x) ones(size(x));
f2 = @(x) x.^2; df2 = @(x) 2*x;
f3 = @(x) sin(x); df3 = @(x) cos(x);

% Define some kernels
poly = @(x,y,a,b) (x'*y + a).^b;

%parameters
n = 100;
m = 100;

% Kernel (lame choices for now)
Q = @(x,y) (x*y.'+1).^2;
% Gamma = @(x,y) kron(S(x,y),eye(m));
S = @(x,y) (x.'*y+1).^2; % Linear kernel, R^dxR^d -> R
K = @(x,y) (x*y.'+1).^2; % Linear kernel, outer product

% Let's try some stuff
xphi = linspace(0,1,n).'; % COLUMN vector
phi = @(f) f(xphi);

ypsy = linspace(0,1,m).';  % COLUMN vector
psy = @(f) f(ypsy);

% U = [phi(f1) phi(f2) phi(f3)];
% V = [psy(df1) psy(df2) psy(df3)];
U = [phi(f2)];
V = [psy(df2)];
[G, f, chi] = learn(U,V,m,S,K,ypsy);

