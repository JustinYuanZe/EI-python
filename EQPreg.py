import numpy as np

def EQPreg(t, x):
    """
    Python implementation of MATLAB EQPreg.m function
    
    Parameters:
    t : array-like
        Input data (independent variable, e.g., time)
    x : array-like
        Parameter vector containing alpha, w, beta for 2 groups
        
    Returns:
    Tuple containing (y, dy, d2y, d3y, y1, dy1, d2y1)
    """
    
    t = np.array(t)
    x = np.array(x).flatten()
    
    n = x.size
    k = n // 6
    
    alpha = x[0:k].reshape(-1, 1)
    w = x[k:2*k].reshape(-1, 1)
    beta = x[2*k:3*k].reshape(-1, 1)
    
    alpha1 = x[3*k:4*k].reshape(-1, 1)
    w1 = x[4*k:5*k].reshape(-1, 1)
    beta1 = x[5*k:6*k].reshape(-1, 1)
    
    t = t.reshape(1, -1)
    
    # --- GROUP 1 ---
    u = -beta - w * t
    r = np.exp(u)
    S = 1.0 / (1.0 + r)
    
    y = np.sum(alpha * S, axis=0)
    dy = np.sum((alpha * w) * (S * (1 - S)), axis=0)
    d2y = np.sum((alpha * w**2) * ((1 - 2*S) * S * (1 - S)), axis=0)
    d3y = np.sum((alpha * w**3) * ((1 - 6*S + 6*S**2) * S * (1 - S)), axis=0)
    
    # --- GROUP 2 ---
    u1 = -beta1 - w1 * t
    r1 = np.exp(u1)
    S1 = 1.0 / (1.0 + r1)
    
    y1 = np.sum(alpha1 * S1, axis=0)
    dy1 = np.sum((alpha1 * w1) * (S1 * (1 - S1)), axis=0)
    d2y1 = np.sum((alpha1 * w1**2) * ((1 - 2*S1) * S1 * (1 - S1)), axis=0)
    
    return y, dy, d2y, d3y, y1, dy1, d2y1