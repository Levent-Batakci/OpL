import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold

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
    chi = (lambda V_,y : K(d2(y),Y) @ np.linalg.solve(chi_inner, V_));
    
    # Solve the generalized interpolation from R^n -> R^m
    if(len(np.shape(U))==1):
        U = U.reshape(1,-1)
    N = np.shape(U)[1];
    f_inner = np.linalg.solve(S(U,U)+1e-8*np.identity(N), V.T);
    f = lambda u : (S(u,U) @ f_inner).T;
    
    # Get the approximate operator
    G_approx = lambda U_ : (lambda y : chi(f(U_), y));
    return G_approx,f

"""  NOT WORKING!!!!
def KLearn2(U,V,m,K,Y):
    ""
    Kernel-method for learning an operator G,
        but we use sklearn w/ cross-validation
    ""
    alpha = 1; # Regularization strength
    
    # Recovery map
    recover = (lambda V_ : KernelRidge(alpha=alpha,kernel="rbf",gamma=50).fit(Y,V_)); #sklearn model
    chi = (lambda V_ : (lambda y : MyPredict(recover(V_), y) )); #function of the form we want
    
    eta = 1e-8; # Nugget term, for numerical stability
    
    # Compute the recovery function
    chi_inner = K(Y,Y) + eta*np.identity(m);
    chi = (lambda V_ : (lambda y : K(d2(y),Y) @ np.linalg.solve(chi_inner, V_)));
    
    # Solve the generalized interpolation from R^n -> R^m
    if(len(np.shape(U))==1):
        U = U.reshape(1,-1)
    N = np.shape(U)[1]; # Number of data points
    f_model = KernelRidge(alpha=alpha,kernel="rbf", gamma=0.00001).fit(U.T,V.T);
    f = lambda u : MyPredict(f_model, u.T).T;
    
    G_approx = lambda U_ : chi(f(U_));
    return G_approx,f;
"""

def OpLearn(U,V, S,SGrid, K,KGrid ,Y, report_scores = False):
    
    # Compute the recovery function
    chi = ( lambda V_ : KLearnCV(Y,V_,K,KGrid,10) )
    
    # Generalized interpolation thingy
    if(report_scores):
        f,scores = KLearnCV(U.T,V.T,S,SGrid,5, report_scores=True)
    else:
        f = KLearnCV(U.T,V.T,S,SGrid,5, report_scores=False)
    
    # Get the approximate operator
    G_approx = lambda U_ : chi(f(U_.T).T);
    if(report_scores):
        return G_approx,scores
    else:
        return G_approx

def KLearnCV(X,Y,Kernel,grid,n_splits, report_scores = False):
    """
    Uses the kernel trick with cross validation
    """   
    cv = 10000; # Largest positive number I'm aware of
    g = "none";
    model = "none";
    scores = [];
    
    for g_ in grid:
        K = Kernel(g_)
        cv_ = KCV(X,Y,K,n_splits)
        scores.append(cv_)
        if(cv_ < cv):
            cv = cv_
            g = g_
            model = KernelTrick(X,Y,K)        
    print("Optimal gamma = " + str(g))
    
    if(report_scores):
        return model,scores
    else:
        return model
            
def KCV(X,Y,K,n_splits):
    """
    Uses the kernel trick and cross validates
    """
    fldr = KFold(n_splits = n_splits);
    folds = fldr.split(X);

    cv = 0;
    for i, (train_index, test_index) in enumerate(folds):
        Xtrain = X[train_index, :]
        Ytrain = Y[train_index, :]
        Xtest = X[test_index, :]
        Ytest = Y[test_index, :]
        
        G = KernelTrick(Xtrain,Ytrain,K)
        y_mod = G(Xtest)
        cv += mse(y_mod,Ytest) / n_splits;
    
    return cv;

def mse(a,b):
    return np.linalg.norm(a-b, ord='fro') / np.size(a); # Relative Mean squared error, MAYBE

def KernelTrick(X,Y,K):
    eta = 1e-8;
    inner = K(X,X);
    m = np.shape(inner)[1];
    return (lambda y: K(y,X) @ np.linalg.solve(inner + eta*np.identity(m),Y));            

# Calls to this should probably be replaced with the correct reshape call...
# But what is more human than to embrace imperfection?
def d2(y):
    return np.atleast_2d(y);
    