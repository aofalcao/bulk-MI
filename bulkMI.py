import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.metrics import mutual_info_score

#check for numba
try:
    from numba import jit
    numba_available = True
except ImportError:
    #this is very ugly! but just in case Numba is not installed
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    numba_available = False

#check for torch
try:
    import torch
    torch_available = True
except ImportError:
    torch_available = False
    

def pairwise_MI(D):
    n, m = D.shape
    s=1/np.log(2)
    mi_mat=np.ones((m,m))
    for i in range(m): 
        for j in range(0, i): 
            mi_mat[i,j] =  mutual_info_score(D[:,i], D[:,j])*s
            mi_mat[j,i] =  mi_mat[i,j]
    return mi_mat


def MyDecorator(decorator, condition, *args, **kwargs):
    def wrapper(func):
        return decorator(*args, **kwargs)(func) if condition else func
    return wrapper


def conditional_jit(func):
    if numba_available:
        return jit(func)  
    return func  

#@jit(nopython=True)
@MyDecorator(jit, numba_available, nopython=True)
def gram_matrix(X1, X2):
    return np.dot(X1, X2)


def bulk_MI_base(D):
    nD = 1 - D
    N = D.shape[0]

    # these are the building blocks (counting matrices)
    #gram_11 = ( D.T @ D ) / N
    #gram_00 = (nD.T @ nD) / N
    #gram_01 = (nD.T @ D ) / N
    #gram_10 = gram_01.T # ( D.T @ nD) / N
    
    gram_11 = gram_matrix( D.T,  D ) / N
    gram_00 = gram_matrix(nD.T , nD) / N
    gram_01 = gram_matrix(nD.T , D ) / N
    gram_10 = gram_01.T # ( D.T @ nD) / N
    
    diag_11 = np.diag(gram_11)
    diag_00 = np.diag(gram_00)

    Pi_11 = np.outer(diag_11, diag_11)
    Pi_00 = np.outer(diag_00, diag_00)
    Pi_10 = np.outer(diag_11, diag_00)
    Pi_01 = Pi_10.T

    # each possible value of the sum for the MI computation
    # binary values only have 4 possible combinations
    eps = 1e-30
    v11 = gram_11 * np.log2(gram_11 / Pi_11 + eps)
    v00 = gram_00 * np.log2(gram_00 / Pi_00 + eps)
    v01 = gram_01 * np.log2(gram_01 / Pi_01 + eps)
    v10 = gram_10 * np.log2(gram_10 / Pi_10 + eps)

    res = v11 + v00 + v01 + v10 
    np.fill_diagonal(res, 1)
    return res


def bulk_MI_torch(D):
    # Calculates the mutual information between all pairs of binary variables in a dataset.
    # This is a FAST implementation, computing only one Gram matrix directly
    # D needs to be a torch array

    n, m = D.shape         # rows, columns
    device=D.device
    # Calculate the Gram matrices <-only one potentially giant multiplication!
    
    N =  n*torch.ones((m,m)).to(device)   # helping matrices for speeding up computations
    v  = D.sum(axis=0)       # vector with number of 1s in each column
    gram_11 =  D.T @  D
        
    C = torch.ones((m, 1)).to(device) * v     

    gram_00 =  (N - C - C.T + gram_11) #this may seem silly, but should be MUCH faster, especially for sparse matrices
    gram_01 =  (C - gram_11)
    gram_10 =  gram_01.T

    #only now we can get the probs
    gram_11 = gram_11 / n
    gram_00 = gram_00 / n
    gram_01 = gram_01 / n
    gram_10 = gram_10 / n
    
    # Get the diagonal elements
    diag_11 = torch.diag(gram_11)
    diag_00 = torch.diag(gram_00)
    
    Pi_11 = torch.outer(diag_11, diag_11)
    Pi_00 = torch.outer(diag_00, diag_00)
    Pi_10 = torch.outer(diag_11, diag_00)
    Pi_01 = Pi_10.T
    
    # Calculate the mutual information terms
    eps = 1e-30  # Small value to avoid division by zero
    v11 = gram_11 * torch.log2(gram_11 / Pi_11 + eps)
    v00 = gram_00 * torch.log2(gram_00 / Pi_00 + eps)
    v01 = gram_01 * torch.log2(gram_01 / Pi_01 + eps)
    v10 = gram_10 * torch.log2(gram_10 / Pi_10 + eps)
    
    # Return the sum of the mutual information terms
    res = v11 + v00 + v01 + v10
    res.diagonal().fill_(1)
    return res


def bulk_MI(D):
    # Calculates the mutual information between all pairs of binary variables in a dataset.
    # This is a FAST implementation, computing only one Gram matrix directly
    # D can be a numpy array or a scipy sparse matrix or a Torch Tensor

    #If this is a Torch Tensor, just offload it to the torch processor
    if torch_available:
        if type(D) is torch.Tensor: 
            #print("This is Torch")
            return bulk_MI_torch(D)

    n, m = D.shape         # rows, columns
    # Calculate the Gram matrices <-only one potentially giant multiplication!
    
    N = np.ones((m,m))*n   # helping matrices for speeding up computations
    v  = D.sum(axis=0)       # vector with number of 1s in each column
    if issparse(D):
        #print("This is Sparse")
        gram_11 = (D.T @ D).toarray()
        v  = np.array(v)
    else:
        gram_11 = gram_matrix( D.T,  D )
        
    C = np.ones((m, 1)) * v     

    gram_00 =  (N - C - C.T + gram_11) #this may seem silly, but should be MUCH faster, especially for sparse matrices
    gram_01 =  (C - gram_11)
    gram_10 =  gram_01.T

    #only now we can get the probs
    gram_11 = gram_11 / n
    gram_00 = gram_00 / n
    gram_01 = gram_01 / n
    gram_10 = gram_10 / n
    
    # Get the diagonal elements
    diag_11 = np.diag(gram_11)
    diag_00 = np.diag(gram_00)
    
    Pi_11 = np.outer(diag_11, diag_11)
    Pi_00 = np.outer(diag_00, diag_00)
    Pi_10 = np.outer(diag_11, diag_00)
    Pi_01 = Pi_10.T
    
    # Calculate the mutual information terms
    eps = 1e-10  # Small value to avoid division by zero
    v11 = gram_11 * np.log2(gram_11 / Pi_11 + eps)
    v00 = gram_00 * np.log2(gram_00 / Pi_00 + eps)
    v01 = gram_01 * np.log2(gram_01 / Pi_01 + eps)
    v10 = gram_10 * np.log2(gram_10 / Pi_10 + eps)
    
    # Return the sum of the mutual information terms
    res = v11 + v00 + v01 + v10
    np.fill_diagonal(res, 1)
    return res
