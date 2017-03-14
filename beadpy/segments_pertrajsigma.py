  
def ratefinder_autosigma(restable, segtable, sigma = 0, sigma_start = 0, sigma_end = 0, method = 'auto'):
    if method == 'auto':
        restable = restable[restable.trajectory.isin(segtable.trajectories)]
        sigmavals, trajectories = sigmaval_finder(restable, sigma_start, sigma_end)
        restable = restable[restable.trajectory.isin(trajectories)]
        segtable = segtable[segtable.trajectory.isin(trajectories)]
    
    groupedsegs = segtable.groupby(['trajectory'], as_index=False)    
    starttimes = groupedsegs['x1'].min()
    endtimes = groupedsegs['x2'].max()    
    startendtimes = pd.merge(left=starttimes, right = endtimes, how='left', left_on='trajectory', right_on='trajectory')
    mergedfiltresults = pd.merge(left=restable,right=startendtimes, how='left', left_on='trajectory', right_on='trajectory')
    finefiltresults = mergedfiltresults[(mergedfiltresults['time'] >= mergedfiltresults['x1'])
                        & (mergedfiltresults['time'] <= mergedfiltresults['x2'])]
    segmentsfine = beadpy.segments_finder(finefiltresults, sigmavals)
    return segmentsfine;