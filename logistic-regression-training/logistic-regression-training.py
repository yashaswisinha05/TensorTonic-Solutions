import numpy as np

def binary_cross_entropy(y,y1,m):
    epsilon = 1e-15
    y1=np.clip(y,epsilon,1-epsilon)
    L=-1/m*(np.sum(y*np.log(y1)+(1-y)*np.log(1-y1)))
    return L
    
def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    
    m, d = X.shape
    
    # Initialize W as a 1D array to perfectly match a 1D y
    W = np.zeros(d)
    b = 0.0
    num_epochs =1000
    i=0
    loss_prev=np.inf
    loss_diff=np.inf
    thres=1e-9
    while i<num_epochs and loss_diff>thres:
        z=np.dot(X,W)+b
        y_pred= _sigmoid(z)
        # loss=binary_cross_entropy(np.asarray(y), np.asarray(y_pred),m)
        loss=y-y_pred
        dw=1/m*(np.dot(X.T,loss))
        db=1/m*(np.sum(loss))
        W+=lr*dw
        b+=lr*db
        i+=1
        
        loss_diff=np.linalg.norm(loss-loss_prev)
        print(loss_diff)
        loss_prev=loss
    return W,b# Write code here
    pass