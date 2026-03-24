import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code 
    pe=np.zeros((seq_len,d_model))
    for seq in range(seq_len):
        for i in range(d_model):
            if i%2==0:
                pe[seq,i]=np.sin(seq/(base**(i/d_model)))
            else:
                pe[seq,i]=np.cos(seq/(base**((i-1)/d_model)))
    return pe
        
        
    pass