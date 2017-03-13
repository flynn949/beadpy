from scipy.optimize import brent
import numpy as np

def confidenceThreshold( N, OneMa = 0.99):
    N = float(N)
    h = ((np.log(N))**(3/2))/N
    T = np.log((1-h * h)/(h * h))
    pfunc = lambda p: ((p * p)/2) * np.exp(-(p * p)/2) * (T - (2 * T)/(p * p) + 4/(p * p)) + OneMa - 1
    root = brentq(pfunc, 3.5, 5)
    return root;