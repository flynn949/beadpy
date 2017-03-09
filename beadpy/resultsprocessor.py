import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt

def drift_subtractor(resultstable):
	resultstable.rename(columns={'particle':'trajectory'}, inplace=True)
	resultstable.rename(columns={'frame':'slice'}, inplace=True)
	resultstable = resultstable.sort_values(by=['trajectory', 'slice'])
	resultstraj = resultstable.groupby(['trajectory'])
	resultstable['x2'] = resultstraj['x'].transform(lambda bzz: bzz - bzz.mean())
	resultstable['y2'] = resultstraj['y'].transform(lambda bzz: bzz - bzz.mean())
	stuckslices = resultstable[resultstable['slice'] >= resultstable['slice'].max()]
	stucktraj = resultstable[resultstable['trajectory'].isin(stuckslices['trajectory'])]
	stuckgroupslice = stucktraj.groupby(['slice'])
	driftx = stuckgroupslice['x2'].aggregate(np.median)
	drifty = stuckgroupslice['y2'].aggregate(np.median)
	drift = DataFrame({
	 'slice' : Series(range(0,1 + resultstable['slice'].idxmax())),
	 'xdrift' : driftx,
	 'ydrift' : drifty
	 })
	mergedresults = pd.merge(left=resultstable,right=drift, how='left', left_on='slice', right_on='slice')
	mergedresults = mergedresults.sort_values(by=['trajectory', 'slice'])
	mergedresults['x3'] = mergedresults['x2'] - mergedresults['xdrift']
	mergedresults['y3'] = mergedresults['y2'] - mergedresults['ydrift']
	results = mergedresults.drop(['x2','y2', 'xdrift', 'ydrift'], axis=1)
	fig, ax = plt.subplots()
	ax.plot(drift.slice, drift.xdrift)
	return results;
	
""" Finds the mean position at each point in time for trajectories which endure from the start to the finish, and then subtracts this drift from each trajectory.
"""
	
def unit_converter(resultstable, exposuretime, ntconversion, micronpixel):
	resultstable['time'] = resultstable['slice'] * exposuretime #For 250 ms frames.
	resultstable['nt'] = resultstable['x3']*ntconversion
	resultstable['transverse'] = resultstable['y3'] * micronpixel
	return resultstable;
	
""" Converts from the trackpy units to units appropriate to DNA replication experiments.
"""
	
def spurious_removal(resultstable):
	resultstraj = resultstable.groupby(['trajectory'])
	startset = resultstraj['time'].aggregate(np.min) #Find the start time of each trajectory
	startset = startset[startset <= 150] #Find only the trajectories which begin prior to 150 s.
	resultstable = resultstable.loc[resultstable['trajectory'].isin(startset.index)] #Keep only the trajectories of interest.
	resultstraj = resultstable.groupby(['trajectory'])
	endset = resultstraj['time'].aggregate(np.max) #Find the end time of each trajectory
	endset = endset[endset >= 150] #Find only the trajectories which continue beyond the 150 s timepoint.
	resultstable = resultstable.loc[resultstable['trajectory'].isin(endset.index)] #Keep only the trajectories of interest.
	return resultstable;
	
""" Removes trajectories which start after a certain timepoint, as well as those which do not endure beyond a certain timepoint.
"""
	
	
def baseline(resultstable, exposuretime):
	resultstraj = resultstable.groupby(['trajectory'])
	nucleotides = resultstraj['nt'].transform(lambda bzz: bzz - bzz.head(150*int(1/exposuretime)).median())
	resultstable['nucleotides'] = nucleotides
	del resultstable['mass'], resultstable['nt'], resultstable['x3'], resultstable['y3']
	return resultstable;
	
""" Calculates the median position for the first 150 seconds of the trajectory, and then subtracts this from the whole trajectory.
"""
	
def trajectory_renumber(resultstable):
	test = []
	trajindex = np.unique(resultstable.trajectory, return_index=True, return_counts=True)
	for i in range(0,len(trajindex[2])):
		test.append([i]*trajindex[2][i])
	resultstable['trajectory'] = list(chain.from_iterable(test))
	del test
	resultstable.fillna(0)
	resultstable.slice = resultstable.slice.astype(int)
	resultstable.trajectory = resultstable.trajectory.astype(int)
	resultstable = resultstable.reset_index(drop=True)
	return resultstable;
	
# def flipxy(resultstable):
    # col_list = list(resultstable)
    # col_list[0], col_list[1] = col_list[1],col_list[0]
    # resultstable.columns = col_list
    # return resultstable
	
# def flipx(resultstable, xwidth):
    # resultstable['x'] = -1*resultstable.x + width
    # return resultstable
