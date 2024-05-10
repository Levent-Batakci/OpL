import numpy as np

# Kernel approach for operator learning
def KLearn(U,V,m,S,K,Y):
    """
    Kernel-method for learning an operator G

    input
    -----
    U : arraylike
            argument data, U = [phi(u_1) ... phi(u_N)], an n-by-N matrix
    V : arraylike
            output data, V = [psi(v_1) ... psi(v_N)], and m-by-N matrix.
    ~~note: U,V are the 'training' data for the operator, i.e., G(u_j) = v_j
    
    m : positive integer
            dimension of the codomain of psi
    S : function (R^n,R^n) -> R
            the Kernel for the mapping of R^n to R^m
    K : function (R,R) -> R
            the Kernel used for recovery of v from psi(v)
    Y : arraylike
            vector of collocation points used in psi

    output
    ------
    G_approx : function R^n -> (R -> R)
            Approximation of G
    """
    eta = 1e-8; # Nugget term, for numerical stability
    
    # Compute the recovery function
    chi_inner = K(Y,Y) + eta*np.identity(m);
    chi = (lambda V_,y : K(d2(y),Y) @ np.linalg.solve(chi_inner, V_.T));
    
    # Solve the generalized interpolation from R^n -> R^m
    if(len(np.shape(U))==1):
        U = U.reshape(1,-1)
    N = np.shape(U)[1];
    f_inner = np.linalg.solve(S(U,U)+1e-8*np.identity(N), V.T);
    f = lambda u : (S(u,U) @ f_inner).T;
    
    # Get the approximate operator
    G_approx = lambda U_ : (lambda y : chi(f(U_), y));
    return G_approx

def d2(y):
    return np.atleast_2d(y);
    