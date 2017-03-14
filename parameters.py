sequencelocation = str('G:/Flynn/20170307LCoreChewPreassembleNoWash/13-24-53.804.seq')
framecut = 10
method = 'parallel' #Set method to 'streaming' or 'parallel'. If parallel, remember to start the ipcluster.
exposuretime = 1 #seconds per frame.
ntmicron = 2805 #Nucleotides per micron (use 2805 for 2.8 micron beads at 15 ul/min in the standard flow cell).
#micronpixel = 0.8 # Microns per pixel.
mintrajlength = 500 #Minimum trajectory duration, in frames.
minrate = 1 #Minimum rate segments to display in segments plot, and also for event filtering.
maxrate = 25 #Maximum rate to display in segments plot, and also for event filtering and the rate plot.
mindisplacement = 1000 #minimum displacement for event filtering.
starttime = 200 #Minimum start time (seconds) for event filtering.
endtime = 15000 #Maximum end time (seconds) for event filtering (normally just leave as a high number).
coarsesigma = 500 #Coarse grained sigma value for event identification and scatter plot.
finesigma = 500 #Fine grained sigma value for rate distribution.
ymaxscatter = 15000 #y max value for the segment scatter plot.
yminscatter = -6000 #y min value for the segment scatter plot.
ratebins = 10 #Number of bins in the rate histogram.
binning = 1
if binning == 2:
	ymincrop = 1#150
	ymaxcrop = 1950
	xmincrop = 1
	xmaxcrop = 3276
	beadsize = 9
	linklength = 1
	micronpixel = 1.6
elif binning == 1:
	ymincrop = 1
	ymaxcrop = 3790
	xmincrop = 1
	xmaxcrop = 6568
	beadsize = 15
	linklength = 2
	micronpixel = 0.8
	
ntconversion = ntmicron * micronpixel