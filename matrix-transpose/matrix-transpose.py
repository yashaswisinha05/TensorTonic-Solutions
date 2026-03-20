import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    A=np.asarray(A)
    A_new=np.zeros((A.shape[1],A.shape[0]))
    for i in range(A.shape[0]):
        A_new[:,i]=A[i,:]
    return A_new
    pass
