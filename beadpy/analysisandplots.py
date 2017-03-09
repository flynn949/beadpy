import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import scipy as sp
from scipy import linalg, optimize
from scipy.optimize import minimize, minimize_scalar, rosen, rosen_der, brentq, fminbound, curve_fit
import math
from scipy.stats import norm
import pylab as P
import matplotlib.mlab as mlab
import beadpy


def segmentplotter(table,maxrate, ymin, ymax, legloc = 1, scale = 10):
    table = table[abs(table['rate']) < maxrate]
    x = table['x1']
    y = table['displacement']
    size = abs(table['rate'])/scale
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.scatter(x, y, s = size, alpha=0.5, color='magenta', edgecolors='black')
    bins = np.linspace(0, maxrate, 4) 
    l1 = ax.scatter([],[], s=(1)/scale, c = 'magenta')
    l2 = ax.scatter([],[], s=bins[1] / scale, c = 'magenta')
    l3 = ax.scatter([],[], s=bins[2] / scale, c = 'magenta')
    l4 = ax.scatter([],[], s=bins[3] / scale,c = 'magenta')
    labels = [1, int(bins[1]), int(bins[2]), int(bins[3])]
    ax.legend([l1, l2, l3, l4], 
                labels, 
                frameon = True, 
                fontsize = 16,
                handlelength = 2, 
                loc = legloc, 
                borderpad = 0.5,
                handletextpad = 1, 
                title ='Rate (nt/s)', 
                scatterpoints = 1)
    ax.set_xlabel('Segment start time (s)', fontsize=16)
    ax.set_ylabel('Segment length (nt)', fontsize=16)
    ax.set_xlim((0, max(x)))
    ax.set_ylim((ymin, ymax))
    fig.tight_layout(pad=2);
    ax.grid(True)
    fig.savefig('segments.png', dpi = 300)
    #plt.clf()
    return ax;
	
def filterer(resultstable, segmentstable, minrate, maxrate, mindisplacement, starttime, endtime):
    filtsegments = segmentstable[(abs(segmentstable['rate']) > minrate) 
                        & (segmentstable['rate'] < maxrate)
                        & (segmentstable['displacement'] >= mindisplacement) 
                        & (segmentstable['x1'] > starttime)
                        & (segmentstable['x1'] < endtime)]
    filtresults = resultstable[resultstable.trajectory.isin(filtsegments.trajectory)]
    filtresults.to_csv('filtresults.csv', index = False, float_format='%.4f')
    filtsegments.to_csv('filtsegments.csv', index = False)
    return filtresults, filtsegments;
	

def trajectory_plotter(resultstable, exampletraj, sigmaval):
        fig, ax = plt.subplots(figsize = (10, 7.5))
        ax.plot(resultstable['time'][resultstable['trajectory'] == exampletraj],
                                        resultstable['nucleotides'][resultstable['trajectory'] == exampletraj]/1000,
                                        lw = 3)
        ax.set_xlabel("Time (s)", fontsize=16)
        ax.set_ylabel("Nucleotides synthesised (kb)", fontsize=16)              
        ax.set_xlim((-50,resultstable['time'][resultstable['trajectory'] == exampletraj].max()+50))
        ax.set_ylim((-0.5 + resultstable['nucleotides'][resultstable['trajectory'] == exampletraj].min()/1000,0.5 + resultstable['nucleotides'][resultstable['trajectory'] == exampletraj].max()/1000))                                              
        if sigmaval > 10:
            fig.suptitle('Trajectory '+str(exampletraj)+', sigma '+str(sigmaval), fontsize = 16)
            exampletrajseg = beadpy.segments_finder_singletraj(exampletraj, resultstable, sigmaval)
            for row_index, row in exampletrajseg[exampletrajseg.trajectory==exampletraj].iterrows():
                ax.plot([row['x1'], row['x2']], [row['y1']/1000, row['y2']/1000],'k-', lw=2, color='Magenta', linestyle='-')
        else:
            fig.suptitle('Trajectory '+str(exampletraj), fontsize = 16)
        ax.tick_params(axis='both',  labelsize=14)                                           
        fig.tight_layout(pad=4);               
        fig.savefig(str(exampletraj)+'_'+str(sigmaval)+'.png', dpi = 300)
        return exampletrajseg

def weighted_avg_and_std(values, weights):
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)  # Fast and numerically precise
    return (average, math.sqrt(variance))

def ratefinder(restable, segtable, sigmaval = 300):
	
    #Filter the results to only be between the first data point of the filtered segments, and the last for each trajectory.
	
    groupedsegs = segtable.groupby(['trajectory'], as_index=False)
    starttimes = groupedsegs['x1'].min()
    endtimes = groupedsegs['x2'].max()
    startendtimes = pd.merge(left=starttimes, right = endtimes, how='left', left_on='trajectory', right_on='trajectory')
    mergedfiltresults = pd.merge(left=restable,right=startendtimes, how='left', left_on='trajectory', right_on='trajectory')
    finefiltresults = mergedfiltresults[(mergedfiltresults['time'] >= mergedfiltresults['x1'])
                        & (mergedfiltresults['time'] <= mergedfiltresults['x2'])]
							 
    #Do change point analysis on these events:						 
    segmentsfine = segments_finder(finefiltresults,sigmaval)
    return segmentsfine;
	

	
def ratehist(segtable, minimumrate = 5,  maximumrate = 1000, numbins = 50, weighting = 'displacement'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = segtable[(segtable['rate'] > minimumrate) & (segtable['rate'] < maximumrate)]
    if weighting == 'none':
        n, bins, patches = P.hist(x.rate, bins = numbins)
        P.text(0.95, 0.95, str(len(x)) + ' segments \n' + 'from ' + str(x.trajectory.nunique()) + ' events',
        verticalalignment='top', horizontalalignment='right',
        color='magenta', fontsize=15, transform=ax.transAxes)
    elif weighting == 'fastest':
        xgrouped = x.groupby(['trajectory'])
        maxrates = xgrouped['rate'].max()
        n, bins, patches = P.hist(maxrates, bins = numbins)
        P.text(0.95, 0.95, str(x.trajectory.nunique()) + ' events',
        verticalalignment='top', horizontalalignment='right',
        color='magenta', fontsize=15, transform=ax.transAxes)
    elif weighting == 'longest':
        xgrouped = x.groupby(['trajectory'])
        x = xgrouped.apply(lambda g: g[g['displacement'] == g['displacement'].max()])
        n, bins, patches = P.hist(x.rate, bins = numbins)
        P.text(0.95, 0.95, str(x.trajectory.nunique()) + ' events',
        verticalalignment='top', horizontalalignment='right',
        color='magenta', fontsize=15, transform=ax.transAxes)
    else:
        n, bins, patches = P.hist(x.rate, bins = numbins, weights = x[weighting]/sum(x[weighting]), normed = 1)
        P.text(0.95, 0.95, str(len(x)) + ' segments \n' + 'from ' + str(x.trajectory.nunique()) + ' events',
        verticalalignment='top', horizontalalignment='right',
        color='magenta', fontsize=15, transform=ax.transAxes)
    P.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
    P.xlabel('Rate (nt/s)')
    P.savefig('rates.png', dpi = 300)
	
	
def processivity(segtable, minimumprocessivity = 0, maximumprocessivity=15000, binno = 15, skipno = 0, initialguesses = 'default'):
    fig = plt.figure()
    ax = plt.subplot(111)
	
    groupedsegs = segtable.groupby(['trajectory'], as_index=False)
    starty = groupedsegs['y1'].min()
    endy = groupedsegs['y2'].max()
    displacements = pd.merge(left=starty, right = endy, how='left', left_on='trajectory', right_on='trajectory')
    displacements['displacement'] = displacements['y2'] - displacements['y1']
    meandisplacement = displacements['displacement'].mean()
    eventno = len(displacements['displacement'])
	#binno = math.sqrt(eventno)
    x = displacements['displacement'][(displacements.displacement > 0) & (displacements.displacement < 21000)]
	
    n, bins, patches = P.hist(x, bins = int(binno))
	
    ydata = n[skipno - binno:]
    binwidth = bins[1]-bins[0]
    xdata = bins[skipno - binno:] - binwidth/2
    def fitfunc(x, a1,t1):
        return a1*np.exp(-x/t1)
    if initialguesses == 'default':
        initialguesses = [10,5000]
    popt, pcov = curve_fit(fitfunc, xdata, ydata, p0 = initialguesses)
	
    plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
    plt.xlabel('Total processivity (nt)')
	
    xdata = np.linspace(xdata[0], xdata[-1],100)
    yfit = fitfunc(xdata, popt[0], popt[1])
    l = ax.plot(xdata, yfit, linewidth=5,color="magenta")	

    plt.text(0.95, 0.95, r'$y = $' + str(popt[0].round(2))+ r'$\ast\mathrm{exp}(-x/$'+str((popt[1].round(2))) + r'$)$' + '\n' + r'$N = $' + str(len(x)) + ' trajectories',
        verticalalignment='top', horizontalalignment='right',
        transform=ax.transAxes,
        color='magenta', fontsize=15)		
	
    plt.savefig('processivity.png', dpi = 300)
	