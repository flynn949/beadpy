import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import scipy as sp
from scipy import linalg, optimize
from scipy.optimize import minimize, minimize_scalar, rosen, rosen_der, brentq, fminbound, curve_fit
import numba
from numba import jit

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
    segA = lsq(a[:,1][a[:,1]<j], a[:,0][a[:,1]<j])
    segB = lsq(a[:,1][a[:,1]>j], a[:,0][a[:,1]>j])
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
#No jit here, doesn't work with the root finder.
def confidenceThreshold( N, OneMa = 0.99):
    N = float(N)
    h = ((np.log(N))**(3/2))/N
    T = np.log((1-h * h)/(h * h))
    pfunc = lambda p: ((p * p)/2) * np.exp(-(p * p)/2) * (T - (2 * T)/(p * p) + 4/(p * p)) + OneMa - 1
    root = brentq(pfunc, 3.5, 5)
    return root;
	
	
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
    segnull = lsq(a[:,1], a[:,0])
    llnull = leng * np.log(1/sigma * 2.506628275) - segnull/(2 * sigma * sigma)
    ll2lines = leng * np.log(1/sigma * 2.506628275) - ssval/(2 * sigma * sigma)
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
        mini = minimize_scalar(ss2lines, bounds =((a[:,1][3]), (a[:,1][-3])), 
                                   method='bounded', args=(a))
            
        minll = loglik(a, leng, mini.fun, sigma)

        if ((-2 * float(minll))**0.5) > confidenceThreshold(leng, OneMa):
            chpt = int(np.abs(array[:,1]-mini.x).argmin() + offset - 1)
                    
        else:
            chpt = -1
    else:
        chpt = -1
    return chpt;
	
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
def linefit(array, cp1, cp2):
    x = array[cp1:cp2][:,1]
    y = array[cp1:cp2][:,0]
    a = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(a, y)[0]
    x1 = array[:,1][cp1]
    x2 = array[:,1][cp2]
    y1 = (array[:,1][cp1] * m) + c
    y2 =(array[:,1][cp2] * m) + c
    displacement = y2 - y1
    duration = x2 - x1
    trajectorynumber = array[:,2][cp1]
    
    return m, c, x1, x2, y1, y2, displacement, duration, trajectorynumber;
	
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
    q = 0
    
    while (q < len(cp_positions) - 1):
        cp = changePoint(array[offset:(offset + length - 1)],cp_positions[q],
                         cp_positions[q + 1], offset, sigma, OneMa)
        if (cp != -1):
            cp_positions.insert(q + 1, cp)
        else:
            q += 1

    if (len(cp_positions) > 3):
        q = 0
        while (q < len(cp_positions) - 2):
            cp = changePoint(array[offset:(offset + length - 1)],cp_positions[q],
                             cp_positions[q + 2],offset,  sigma, OneMa)
            cp_positions.pop(q + 1)
            if (cp != -1):
                cp_positions.insert(q + 1, cp)
                q += 1
    
    if(len(cp_positions) < 2):
        cp_positions = []
    
    if not cp_positions:
        return 0;
    else:
        line_fits = [0]
        for q in range(0,(len(cp_positions) - 1)):
            line_fits.append(linefit(array[offset:(offset + length)], cp_positions[q] - offset, cp_positions[q + 1] - offset))
        line_fits.pop(0)
        
    return line_fits;

""" Searches for changepoints in a given trajectory, following the binary segmentation approach, with a refinement step.

Parameters
----------
array: An array containing time in column 1 and nucleotides/position in column 0.
offset: the index value of the first row of the trajectory.
length: the number of rows in the trajectory.
sigma: The user-defined noise level.
OneMa: The confidence level.

Returns
-------
Line segments with all the information given by bead.linefit, each defined by the consecutive changepoint boundaries.
"""	
	

def segments_finder(restable, sigma):
    
    restable = restable.reset_index(drop=True)
    resultslite = DataFrame({
        'trajectory' : restable.trajectory,
        'time' : restable.time,
        'nucleotides' : restable.nucleotides})
    
    resultsarray = resultslite.as_matrix(columns = [resultslite.columns[0:3]])
    indextable = np.unique(resultslite.trajectory, return_index=True, return_counts=True)
    
    first = 0
    last = resultslite.trajectory.nunique()
    
    cptable = []
    for i in range(first, last):
        temp = binary_search(resultsarray, indextable[1][i], indextable[2][i], float(sigma), 0.99)
        if temp != 0:
            cptable.append(temp)
        del temp
    
    collist = ['rate', 'intercept', 'x1', 'x2', 'y1', 'y2', 'displacement', 'duration', 'trajectory']
    decimals = pd.Series([1,0,2,2,1,1,1,2,0], index = collist)
    
    appended_data = []
    for i in range(0,len(cptable)):
        data = DataFrame(cptable[i], columns = collist)
        appended_data.append(data)
    segmentstable = pd.concat(appended_data, axis=0)
    segmentstable = segmentstable.round(decimals)
    
    segmentstable.to_csv('pythonsegments_sigma'+str(sigma)+'.csv', index = False)

    return segmentstable;
	
""" Generates a changepoint segment table for all the trajectories in a Pandas results table.

Parameters
----------
restable: A pandas results array containing trajectory, time and nucleotides columns.
sigma: The user-defined noise level.

Returns
-------
A segments table containing information about all the changepoint-defined line segments for all the trajectories in the results table.

"""

def segments_finder_singletraj(traj, restable, sigma):    
    restable = restable[restable['trajectory']==traj]
    restable = restable.reset_index(drop=True)
    resultslite = DataFrame({
        'trajectory' : restable['trajectory'],
        'time' : restable['time'],
        'nucleotides' : restable['nucleotides']})
    resultsarray = resultslite.as_matrix(columns = [resultslite.columns[0:3]])
    cptable = binary_search(resultsarray, 0, len(resultsarray), float(sigma), 0.99)
    collist = ['rate', 'intercept', 'x1', 'x2', 'y1', 'y2', 'displacement', 'duration', 'trajectory']
    decimals = pd.Series([1,0,2,2,1,1,1,2,0], index = collist)
    segmentstable = DataFrame(cptable, columns = collist)
    segmentstable = segmentstable.round(decimals)
    return segmentstable;