import numpy as np
from EQPreg import EQPreg

# Global parameters (adjust as needed)
A = 1.0
B = 1.0
Z = 1.0
d = 1.0
s = 0.0
pr = 0.7

def fitreg(x):
    """
    Fitness function to calculate total error.
    
    Parameters:
    x : array-like
        Model parameter vector.
        
    Returns:
    e : float
        Total error.
    """
    global A, B, Z, d, pr, s
    
    # Part 1: Calculate error over t domain (0 to 10)
    t_range = np.linspace(0, 10, 41)
    
    y, dy, d2y, d3y, _, dy1, d2y1 = EQPreg(t_range, x)
    
    term_exp = Z * np.exp(-d * t_range)
    ee1 = (d3y + A * y * d2y + B * (dy**2) + term_exp)**2
    ee2 = (d2y1 + A * pr * y * dy1)**2
    
    e1 = np.mean(ee1)
    e2 = np.mean(ee2)
    
    # Part 2: Boundary conditions at t = 0
    t0 = np.array([0.0])
    y_0, dy_0, _, _, y1_0, _, _ = EQPreg(t0, x)
    
    e3 = (y_0[0] - s)**2
    e4 = (dy_0[0] - 1.0)**2
    e5 = (y1_0[0] - 1.0)**2
    
    # Part 3: Boundary conditions at t = 10
    t10 = np.array([10.0])
    _, dy_10, _, _, y1_10, _, _ = EQPreg(t10, x)
    
    e6 = (dy_10[0] - 0.0)**2
    e7 = (y1_10[0] - 0.0)**2
    
    # Part 4: Combine errors
    e = e1 + e2 + (e3 + e4 + e5 + e6 + e7) / 5.0
    
    return e