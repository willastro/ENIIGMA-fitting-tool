import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, ScalarFormatter, LogLocator, AutoMinorLocator
import pandas as pd

def to_sub(s):
	
    
    subs = {u'0': u'\u2080',
            u'1': u'\u2081',
            u'2': u'\u2082',
            u'3': u'\u2083',
            u'4': u'\u2084',
            u'5': u'\u2085',
            u'6': u'\u2086',
            u'7': u'\u2087',
            u'8': u'\u2088',
            u'9': u'\u2089'}

    return ''.join(subs.get(char, char) for char in s)


def barp(file, fmin, fmax):
	"""
	Create barplots of column densities.
	
	Parameters
	-------------
	
	file : 'str'
		Path to the column density file.
	
	fmin : 'string'
		Path to the lower confidence interval column density.
	
	fmax : 'string'
		Path to the high confidence interval column density.
	
	Returns
	-------------
	Barplot
	
	    
	"""
	t = pd.read_csv(file, sep=',')
	sp = t['Label']
	CDtot = np.array(t['CDtot'].T.values.tolist())
	CDinmix = np.array(t['CDinmix'].T.values.tolist())
	CDpure = np.array(t['CDpure'].T.values.tolist())
	
	barWidth = 0.20
	
	tmin = pd.read_csv(fmin, sep=',')
	spmin = tmin['Label']
	CDtotmin = np.array(tmin['CDtot'].T.values.tolist())
	CDinmixmin = np.array(tmin['CDinmix'].T.values.tolist())
	CDpuremin = np.array(tmin['CDpure'].T.values.tolist())
	
	tmax = pd.read_csv(fmax, sep=',')
	spmax = tmax['Label']
	CDtotmax = np.array(tmax['CDtot'].T.values.tolist())
	CDinmixmax = np.array(tmax['CDinmix'].T.values.tolist())
	CDpuremax = np.array(tmax['CDpure'].T.values.tolist())
	
	fig = plt.figure()
	ax2=fig.add_subplot(211)
	#Liter = Lit
	Total = CDtot
	Mix = CDinmix
	Pure = CDpure
	
	Totalmin = CDtotmin
	Mixmin = CDinmixmin
	Puremin = CDpuremin
	
	Totalmax = CDtotmax
	Mixmax = CDinmixmax
	Puremax = CDpuremax
	
	
		
	#errL = [4.1e17, 0.1e17, 0, 0, 0]
	errT = 1.9e-1*Total#[1e17, 7.6e17, 6.6e16, 0e18, 0]
	errP = 1.8e-1*Pure#[0e17, 1.4e17, 0e16, 0e18, 0]
	errM = 1.85e-1*Mix#[1e17, 2.9e17, 6.6e16, 0e18, 0]
	
	
	# Set position of bar on X axis
	r1 = np.arange(len(CDtot))
	r2 = [x + barWidth for x in r1]
	r3 = [x + barWidth for x in r2]
	#r4 = [x + barWidth for x in r3]
	
	r1min = np.arange(len(CDtotmin))
	r2min = [x + barWidth for x in r1min]
	r3min = [x + barWidth for x in r2min]
	
	r1max = np.arange(len(CDtotmax))
	r2max = [x + barWidth for x in r1max]
	r3max = [x + barWidth for x in r2max]
		 
	# Make the plot
	#ax2.bar(r1, Liter, color='black', width=barWidth, yerr=errL, edgecolor='white', label='Literature')
	
	ax2.bar(r1, Total, color='red', width=barWidth, yerr=[-(Totalmin-Total), Totalmax-Total], edgecolor='white', label='Total')
	ax2.bar(r2, Mix, color='green', width=barWidth, yerr=[-(Mixmin-Mix), Mixmax-Mix], edgecolor='white', label='in mix')
	ax2.bar(r3, Pure, color='blue', width=barWidth, yerr=[-(Puremin-Pure), Puremax-Pure], edgecolor='white', label='Pure')
	
	ax2.set_ylabel(r'$\ \mathrm{Column \; Density: N} \;\;\; \mathrm{[cm^{-2}]}$')
		
	ax2.tick_params(direction='in', which='both')
	ml = MultipleLocator(4)
	ax2.yaxis.set_minor_locator(AutoMinorLocator(5))
		 
	# Add xticks on the middle of the group bars
	#ax2.xlabel('group', fontweight='bold')
	plt.xticks([r + barWidth for r in range(len(CDtot))], sp)
		
	# Create legend & Show graphic
	plt.legend(loc='best',frameon=False, fontsize=7)
	plt.tight_layout()
	plt.grid(b=True, which='major', linestyle=':')
	plt.grid(b=True, which='minor', linestyle=':')
	
	#######  LOG PLOT AFTER HERE ##########
	
	ax3=fig.add_subplot(212)
	# set height of bar
	#Liter = Lit
	"""
	Total = CDtot
	Mix = CDinmix
	Pure = CDpure
	"""
		
	#errL = [4.1e17, 0.1e17, 0, 0, 0]
	errT = ([-(Totalmin-Total), Totalmax-Total])/Total
	errM = ([-(Mixmin-Mix), Mixmax-Mix])/Mix
	errP = ([-(Puremin-Pure), Puremax-Pure])/Pure
	
	
	# Set position of bar on X axis
	r1 = np.arange(len(CDtot))
	r2 = [x + barWidth for x in r1]
	r3 = [x + barWidth for x in r2]
	#r4 = [x + barWidth for x in r3]
		 
	# Make the plot
	#ax2.bar(r1, Liter, color='black', width=barWidth, yerr=errL, edgecolor='white', label='Literature')
	ax3.bar(r1, np.log10(Total), color='red', width=barWidth, yerr=0.434*errT, edgecolor='white', label='Total')
	ax3.bar(r2, np.log10(Mix), color='green', width=barWidth, yerr=0.434*errM, edgecolor='white', label='in mix')
	ax3.bar(r3, np.log10(Pure), color='blue', width=barWidth, yerr=0.434*errP, edgecolor='white', label='Pure')
	
	ax3.set_ylabel(r'$\ \mathrm{log_{10} (N)}$')
		
	ax3.tick_params(direction='in', which='both')
	ml = MultipleLocator(4)
	ax3.yaxis.set_minor_locator(AutoMinorLocator(5))
		 
	# Add xticks on the middle of the group bars
	#ax2.xlabel('group', fontweight='bold')
	plt.xticks([r + barWidth for r in range(len(CDtot))], sp)
	plt.ylim(15,20)
		
	# Create legend & Show graphic
	plt.legend(loc='best',frameon=False, fontsize=7)
	plt.tight_layout()
	plt.grid(b=True, which='major', linestyle=':')
	plt.grid(b=True, which='minor', linestyle=':')
