import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    
    m=len(seqs)
    cur=0
    if max_len is None:
        for i in range(len(seqs)):
            max=len(seqs[i])
            if max>=cur:
                cur=max
    else:
        cur=max_len
    
    padded_array=np.full((len(seqs),cur),pad_value)
    for i,seq in enumerate(seqs):
        trunc=min(len(seq),cur)
        padded_array[i,:trunc]=seq[:trunc]
        
    return padded_array
    pass