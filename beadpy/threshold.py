import numpy as np
import scipy as sp
from scipy import optimize
from scipy.optimize import brentq
import math

def confidenceThreshold( N, OneMa = 0.99):
    N = float(N)
    num1 = np.log(N)
    num2 = math.pow(num1, (3./2.))
    h = num2/N
    T = np.log((1.-h * h)/(h * h))
    pfunc = lambda p: ((p * p)/2.) * np.exp(-(p * p)/2.) * (T - (2. * T)/(p * p) + 4./(p * p)) + OneMa - 1.
    root = brentq(pfunc, 3.5, 5)
    return root