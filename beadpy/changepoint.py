import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import scipy as sp
from scipy import linalg, optimize
from scipy.optimize import minimize, minimize_scalar, rosen, rosen_der, brentq, fminbound, curve_fit
import numba
from numba import jit
import math
import beadpy


@jit
def lsq(x,y):
    return np.linalg.lstsq(np.vstack([x, np.ones(len(x))]).T, y)[1][0];
	
""" Performs a least squares linear fit.
Parameters
----------
x: An array of time values.
y: An array of nucleotides/position values.
Returns
-------
The sum of squares for the least squares linear fit.
"""
	
@jit
def ss2lines(j, a):
    segA = lsq(a[:,0][a[:,0]<j], a[:,1][a[:,0]<j])
    segB = lsq(a[:,0][a[:,0]>j], a[:,1][a[:,0]>j])
    out = segA + segB
    return out;
	
""" Finds the combined sum of squares for a pair of least squares linear fits, one to the left of timepoint j and the other to the right of timepoint j.
Parameters
----------
j: The candidate changepoint.
a: An array containing time in column 1 and nucleotides/position in column 0.
Returns
-------
The combined sum of squares for the pair of least squares linear fits.
"""

def confidenceThreshold( N, OneMa = 0.99):
    if (N > 2) & (N < 2000) & (OneMa == 0.99):
        root = beadpy.confidencevals[int(N)]
    else:
        N = float(N)
        num1 = np.log(N)
        num2 = math.pow(num1, (3./2.))
        h = num2/N
        T = np.log((1.-h * h)/(h * h))
        pfunc = lambda p: ((p * p)/2.) * np.exp(-(p * p)/2.) * (T - (2. * T)/(p * p) + 4./(p * p)) + OneMa - 1.
        root = brentq(pfunc, 2, 7)
    return root

""" Calculates the critical value at a given confidence level and number of data points for the stastical significance of a log likelihood ratio of a two line fit versus a single line fit.
Parameters
----------
N: The number of timepoints
OneMa: The confidence level.
Returns
-------
The critical value.
"""
	
	
@jit
def loglik(a, leng, ssval, sigma):
    segnull = lsq(a[:,0], a[:,1])
    llnull = leng * np.log(1.0/sigma * 2.506628275) - segnull/(2 * sigma * sigma)
    ll2lines = leng * np.log(1.0/sigma * 2.506628275) - ssval/(2 * sigma * sigma)
    loglik = -1 * (ll2lines - llnull)
    return loglik;
	
""" Calculates the log likelihood ratio for a two line fit around the candidate changepoint versus a single line fit ignoring the changepoint.
Parameters
----------
a: An array containing time in column 1 and nucleotides/position in column 0.
leng: The number of timepoints in a.
ssval: The combined sum of squares for the two line fit.
sigma: The user-defined noise level.
Returns
-------
The log likelihood ratio for a two line fit versus a single line fit.
"""
	
@jit
def changePoint(array, startX, endX, offset, sigma, OneMa):
    a = array[startX - offset:endX - offset]
    leng = len(a)
    if (leng > 15):
        mini = minimize_scalar(ss2lines, bounds =((a[:,0][3]), (a[:,0][-3])), 
                                   method='bounded', args=(a))
            
        minll = loglik(a, leng, mini.fun, sigma)

        if (np.sqrt(-2 * float(minll))) > confidenceThreshold(leng, OneMa):
            chpttime = mini.x
            chpt = int(np.abs(array[:,0]-mini.x).argmin() + offset)
                    
        else:
            chpt = -1
            chpttime = -1
    else:
        chpt = -1
        chpttime = -1
    return chpt, chpttime;
""" Uses a minimising function to search for the best candidate changepoint j at which the log likelihood ratio for a two line fit versus a single line fit is maximised. Then tests this log likelihood ratio against the appropriate critical value and returns the changepoint if it passes.
Parameters
----------
array: An array containing time in column 1 and nucleotides/position in column 0.
startX: The left-hand boundary of the interval to be tested.
endX: The right-hand boundary of the interval to be tested.
offset: The index value of the first row of the current trajectory in the array.
sigma: The user-defined noise level.
OneMa: The confidence level.
Returns
-------
If the changepoint passes the significance test, the changepoint is returned as a row index. If it fails, -1 is returned.
"""
	
@jit
def linefit(array, cp1, cp2, cpt1, cpt2):
    x = array[cp1:cp2][:,0]
    y = array[cp1:cp2][:,1]
    a = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(a, y)[0]
    x1 = array[:,0][cp1]
    x2 = array[:,0][cp2]
    y1 = (array[:,0][cp1] * m) + c
    y2 =(array[:,0][cp2] * m) + c
    displacement = y2 - y1
    duration = x2 - x1
    trajectorynumber = array[:,2][cp1]
    
    return m, c, x1, x2, y1, y2, displacement, duration, trajectorynumber, cpt1, cpt2; #duration is still integer
	
""" Another least squares linear fit, this time returning many parameters to help build up the segments table.
Parameters
----------
array: An array containing time in column 1 and nucleotides/position in column 0.
cp1: The changepoint which defines the start of the current segment.
cp2: The changepoint which defines the end of the current segment.
Returns
-------
Several parameters:
m: The gradient - i.e. the rate of the segment in nt/s.
c: the y-intercept.
x1: the start time.
x2: the end time.
y1: the starting nucleotides/position value.
y2: the end nucleotides/position value.
displacement: the distance travelled: y2 - y1.
duration: in time: x2 - x1.
trajectorynumber: the number of the current trajectory.
"""
@jit	
def binary_search(array, offset, length, sigma, OneMa):
         
    cp_positions = [offset, offset + length - 1]
    cpt_positions = [array[offset,0], array[offset + length - 1,0]]
    q = 0
    
    while (q < len(cp_positions) - 1):
        cps = changePoint(array[offset:(offset + length - 1)],cp_positions[q],
                         cp_positions[q + 1], offset, sigma, OneMa)
        cp = cps[0]
        cpt = cps[1]
        
        if (cp != -1):
            cp_positions.insert(q + 1, cp)
            cpt_positions.insert(q + 1, cpt)
        else:
            q += 1

    if (len(cp_positions) > 3):
        q = 0
        while (q < len(cp_positions) - 2):
            cps = changePoint(array[offset:(offset + length - 1)],cp_positions[q],
                             cp_positions[q + 2],offset,  sigma, OneMa)
            cp = cps[0]
            cpt = cps[1]
        
            cp_positions.pop(q + 1)
            if (cp != -1):
                cp_positions.insert(q + 1, cp)
                q += 1
                
    if not cp_positions:
        return 0;  
    
    else:
        line_fits = [0]
        for q in range(0,(len(cp_positions) - 1)):
            line_fits.append(linefit(array[offset:(offset + length)], cp_positions[q] - offset,
                                                cp_positions[q + 1] - offset,
                                               cpt_positions[q],
                                               cpt_positions[q+1]))
        line_fits.pop(0)
        
    return line_fits;
    
def segment_finder(datatable, xcolumn = 'time', ycolumn = 'nucleotides', indexcolumn = 'trajectory', sigma_start = 0, sigma_end = 100, sigma = 500, method = 'global', traj = 'none', returnsigma = 'no'):
    
    datatable = datatable.reset_index(drop=True)
    
    if isinstance(traj, int):
        datatable = datatable[datatable[indexcolumn]==traj]
       
    if indexcolumn == 'none':
        a = datatable.as_matrix(columns = [xcolumn, ycolumn])
        resultsarray = np.hstack((a, np.atleast_2d(np.zeros(len(a))).T)) #Adds an index column filled with zeroes when there is only one trajectory.
        datatable['trajectory'] = 0
        indexcolumn = 'trajectory'
    else:
        resultsarray = datatable.as_matrix(columns = [xcolumn, ycolumn, indexcolumn])
   
    indextable = np.unique(datatable[indexcolumn], return_index=True, return_counts=True)
    
    if method == 'auto':
        sigmaregion = datatable[(datatable[xcolumn] > sigma_start) & (datatable[xcolumn] < sigma_end)]
        datatable = datatable[datatable[indexcolumn].isin(sigmaregion[indexcolumn])]
        resultsarray = datatable.as_matrix(columns = [xcolumn, ycolumn, indexcolumn])
        sigmavals = sigmaregion.groupby(indexcolumn)[ycolumn].apply(lambda x:x.rolling(center=False,window=20).std().mean())
        trajectories = sigmavals.index.tolist()
        sigmalist = sigmavals.tolist()
        
    if isinstance(sigma, list): #allows an externally provided list of sigma values, method should be specified as global.
        method = 'auto'
        sigmalist = sigma
        
    first = 0
    last = len(np.unique(resultsarray[:,2]))
    
    cptable = []
    for i in range(first, last):
        if method == 'auto':
            sigmaval = sigmalist[i]
        if method == 'global':
            sigmaval = sigma
        temp = binary_search(resultsarray, indextable[1][i], indextable[2][i], float(sigmaval), 0.99)
        if temp != 0:
            cptable.append(temp)
        del temp
        
        
    collist = ['rate', 'intercept', 'x1', 'x2', 'y1', 'y2', 'displacement', 'duration', 'trajectory', 'floatx1', 'floatx2']
    decimals = pd.Series([1,0,2,2,1,1,1,2,0, 2, 2], index = collist)
    
    appended_data = []
    for i in range(0,len(cptable)):
        data = DataFrame(cptable[i], columns = collist)
        appended_data.append(data)
    segmentstable = pd.concat(appended_data, axis=0)
    segmentstable = segmentstable.round(decimals)
    
    if (traj != 'none') & (method=='global'):
        segmentstable.to_csv("segments_traj_"+str(traj)+"_sigma_"+str(sigma)+".csv", index = False)
    elif (traj != 'none') & (method=='auto'):
        segmentstable.to_csv("segments_traj_"+str(traj)+"_sigma_"+str(int(sigmalist[0]))+".csv", index = False)
    elif isinstance(sigma, list):
        segmentstable.to_csv("segments_events.csv", index = False)   
    elif (isinstance(sigma, int)) & (not isinstance(traj, int)):
        segmentstable.to_csv('segments_sigma'+str(sigma)+'.csv', index = False)
    else:
        segmentstable.to_csv('segments_autosigma.csv', index = False)
    
    if (returnsigma == 'yes') & (traj != 'none') & (method=='auto'):
        return segmentstable, int(sigmalist[0])
    else:
        return segmentstable;
