import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.interpolate import interp1d
from numpy import exp, loadtxt, pi, sqrt
from lmfit import Model
from lmfit.model import save_modelresult
import site


#======================================================================
print('Performing deconvolution by Lorentzian or Gaussian of the silicate profile...')
#======================================================================

def get_line_number2(value, matrix):
	"""
	Function used to get the line number of a value in a array.
	
	Parameters
	-------------
	
	value : 'float'
		Value of interest.
	
	matrix : 'numpy.ndarray'
		Vector with the value of interest.
	
	Returns
	------------
	
	Index
	    
	"""
	for i, line in enumerate(matrix, 1):
		if line == value:
			return i


def find_nearest(array, value):
    """
	Find nearest value of a given number.
	
	Parameters
	-------------
	
	array : 'numpy.ndarray'
		Vector with the value of interest.
	
	value : 'float'
		Value of interest.
	
	Returns
	-------------
	Nearest value
	    
	"""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

###############  DECONVOLUTION  ############################################
def gaussian(x, amp1, amp2, amp3, amp4, amp5, amp6, wid1, wid2, wid3, wid4, wid5, wid6):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    p1 = (amp1 / (sqrt(2*pi) * wid1)) * exp(-(x-8.2)**2 / (2*wid1**2))
    p2 = (amp2 / (sqrt(2*pi) * wid2)) * exp(-(x-9.5)**2 / (2*wid2**2))
    p3 = (amp3 / (sqrt(2*pi) * wid3)) * exp(-(x-11.0)**2 / (2*wid3**2))
    p4 = (amp4 / (sqrt(2*pi) * wid4)) * exp(-(x-16.2)**2 / (2*wid4**2))
    p5 = (amp5 / (sqrt(2*pi) * wid5)) * exp(-(x-18.0)**2 / (2*wid5**2))
    p6 = (amp6 / (sqrt(2*pi) * wid6)) * exp(-(x-20.0)**2 / (2*wid6**2))
    return p1 + p2 + p3 + p4 + p5 + p6


def fit(filename, xmin, xmax, npoints = 1000., pathlib = site.getsitepackages()[0]+'/ENIIGMA', silicate_guess_factor = 0.25, values_wid = [0.1, 0.5, 0.8, 0.9, 0.9, 0.9], perc_wid_silicate=2.5, perc_wid_data = 0.4, perc_amp_min_data=0.3, perc_amp_max_data=0.41):
	"""
	Used to combine files in the genetic algorithm spectral decomposition.
	
	Parameters
	-------------
	
	filename : 'str'
		Optical depth file.
	
	xmin, xmax : 'float'
		Range used in the spectral decomposition.
	
	n_points : 'float'
		Number of data points used for interpolating the data.
		Default: 1000
	
	pathlib : 'str'
		Directory of ice data library. Reads GCS3 silicate profile.
		Default: site-packages in the python pathway.
	
	silicate_guess_factor : 'float'
		Initial guess for the CGS3 silicate feature to match the observed profile.
		Default: 0.25
	
	values_wid : 'array'
		six width guesses for the initial decomposition.
		Default: [0.1, 0.5, 0.8, 0.9, 0.9, 0.9]
	
	perc_wid_silicate : 'float'
		Factor for the silicate profile variation.
		Default: 2.5
	
	perc_wid_data : 'float'
		Factor width for the silicate profile variation.
		Default: 0.4
	
	perc_amp_min_data : 'float'
		Minimum factor amplitude for the silicate profile variation.
		Default: 0.31
	
	perc_amp_max_data : 'float'
		Minimum factor amplitude for the silicate profile variation.
		Default: 0.3
		
	
	Returns
	-------------
	
	File 'New_tau_no_silic_'+name_source+'.txt' With the optical depth without the silicate profile.
	
	"""
	fig = plt.figure()
	###################SILICATE PROFILE############
	DIR = os.getcwd() + '/'
	#filet = 'Optical_depth_GY92274.od'
	tau_data = filename
	t = np.loadtxt(tau_data).T
	xtau = t[0]
	ytau = t[1]
	etau = t[2]
	
	name_source = filename[len(DIR):].split('_')[2].split('.')[0]
	
	file_s = pathlib+'/ICE_LIB/Silicate/'+'Silicate_GCS3_new.out'
	file_silic0 = file_s
	
	t0 = np.loadtxt(file_silic0).T
	x_silic0 = t0[0]
	y_silic0 = t0[1]*silicate_guess_factor
	
	#exit()
	
	npoints = npoints
	pp1 = find_nearest(xtau, xmin)
	pp2 = find_nearest(xtau, xmax)
	
	tt = np.linspace(pp1, pp2, int(npoints))
	
	
	Fd_silic0 = interp1d(x_silic0,y_silic0, kind='cubic')	#interpolate data
	Fsilic0 = Fd_silic0(tt)
	
	plt.plot(tt, Fsilic0, label='CGS3 silicate (guess)', color='limegreen')
	
	gmodel_gcs3 = Model(gaussian)
	
	pars_gcs3 = gmodel_gcs3.make_params(amp1=0.01, amp2=1.0, amp3=1.0, amp4=0.8, amp5=0.8, amp6=0.8)
	
	values_wid = values_wid#[0.1, 0.5, 0.8, 0.9, 0.9, 0.9]
	percentage_wid = 2.5
	
	pars_gcs3.add('wid1', value=values_wid[0], min=values_wid[0]-percentage_wid*values_wid[0], max=values_wid[0]+percentage_wid*values_wid[0])
	pars_gcs3.add('wid2', value=values_wid[1], min=values_wid[1]-percentage_wid*values_wid[1], max=values_wid[1]+percentage_wid*values_wid[1])
	pars_gcs3.add('wid3', value=values_wid[2], min=values_wid[2]-percentage_wid*values_wid[2], max=values_wid[2]+percentage_wid*values_wid[2])
	pars_gcs3.add('wid4', value=values_wid[3], min=values_wid[3]-percentage_wid*values_wid[3], max=values_wid[3]+percentage_wid*values_wid[3])
	pars_gcs3.add('wid5', value=values_wid[4], min=values_wid[4]-percentage_wid*values_wid[4], max=values_wid[4]+percentage_wid*values_wid[4])
	pars_gcs3.add('wid6', value=values_wid[5], min=values_wid[5]-percentage_wid*values_wid[5], max=values_wid[5]+percentage_wid*values_wid[5])
	
	result_gcs3 = gmodel_gcs3.fit(Fsilic0, x=tt, params=pars_gcs3)
	
	all_pars = []
	for name, param in list(result_gcs3.params.items()):
	    #print('{:7s} {:11.5f}'.format(name, param.value))
	    all_pars.__iadd__([param.value])
	
	amplitude_gcs3, width_gcs3 = all_pars[0:6], all_pars[6:12]
	#print amplitude_gcs3, width_gcs3
	
	"""
	plt.plot(tt, ytau, 'bo')
	plt.plot(tt, result_gcs3.init_fit, 'k--', label='initial fit')
	plt.plot(tt, result_gcs3.best_fit, 'r-', label='best fit')
	plt.legend(loc='best')
	plt.plot(tt, ytau, label='CRBR2422')
	plt.plot(tt, ytau*0., ':')
	plt.xlim(5.25,20.0)
	plt.ylim(0.8,-0.1)
	plt.legend(loc='best', frameon=False, fontsize=9)
	plt.ylabel(r'Optical Depth$\mathrm{(\tau_{\lambda})}$',fontsize=10)
	plt.xlabel(r'$\lambda\ \mathrm{[\mu m]}$',fontsize=10)
	
	plt.savefig('Silicate_deconv_1st.pdf',format='pdf', bbox_inches='tight', dpi=300)	
	"""
	
	#exit()
	
	###################################
	
	
	gmodel = Model(gaussian)
	
	amp = amplitude_gcs3
	wid = width_gcs3
	
	pars = gmodel.make_params()
	#pars['amp1'].vary = False
	
	values_wid = [wid[0], wid[1], wid[2], wid[3], wid[4], wid[5]]
	print(values_wid)
	percentage_wid = 0.4
	
	pars.add('wid1', value=values_wid[0], min=values_wid[0]-percentage_wid*values_wid[0], max=values_wid[0]+percentage_wid*values_wid[0])
	pars.add('wid2', value=values_wid[1], min=values_wid[1]-percentage_wid*values_wid[1], max=values_wid[1]+percentage_wid*values_wid[1])
	pars.add('wid3', value=values_wid[2], min=values_wid[2]-percentage_wid*values_wid[2], max=values_wid[2]+percentage_wid*values_wid[2])
	pars.add('wid4', value=values_wid[3], min=values_wid[3]-percentage_wid*values_wid[3], max=values_wid[3]+percentage_wid*values_wid[3])
	pars.add('wid5', value=values_wid[4], min=values_wid[4]-percentage_wid*values_wid[4], max=values_wid[4]+percentage_wid*values_wid[4])
	pars.add('wid6', value=values_wid[5], min=values_wid[5]-percentage_wid*values_wid[5], max=values_wid[5]+percentage_wid*values_wid[5])
	
	
	values_amp = [amp[0], amp[1], amp[2], amp[3], amp[4], amp[5]]
	print(values_amp)
	percentage_amp_min = 0.3
	percentage_amp_max = 0.41
	
	pars.add('amp1', value=values_amp[0], min=values_amp[0]-percentage_amp_min*values_amp[0], max=values_amp[0]+percentage_amp_max*values_amp[0])
	pars.add('amp2', value=values_amp[1], min=values_amp[1]-percentage_amp_min*values_amp[1], max=values_amp[1]+percentage_amp_max*values_amp[1])
	pars.add('amp3', value=values_amp[2], min=values_amp[2]-percentage_amp_min*values_amp[2], max=values_amp[2]+percentage_amp_max*values_amp[2])
	pars.add('amp4', value=values_amp[3], min=values_amp[3]-percentage_amp_min*values_amp[3], max=values_amp[3]+percentage_amp_max*values_amp[3])
	pars.add('amp5', value=values_amp[4], min=values_amp[4]-percentage_amp_min*values_amp[4], max=values_amp[4]+percentage_amp_max*values_amp[4])
	pars.add('amp6', value=values_amp[5], min=values_amp[5]-percentage_amp_min*values_amp[5], max=values_amp[5]+percentage_amp_max*values_amp[5])
	
	result = gmodel.fit(ytau, x=tt, params=pars)#, method='least_squares')
	
	#exit()
	print((result.fit_report()))
	
	with open('fit_result_'+name_source+'_silicate_removal.txt', 'w') as fh:
	    fh.write(result.fit_report())
	
	print('-------------------------------')
	print('Parameter    Value       Stderr')
	value = []
	for name, param in list(result.params.items()):
		check = isinstance(param.stderr, float)
		if check == False:
			print(('{:7s} {:11.5f} {:11.5f}'.format(name, param.value, np.nanstd(etau))))
			value.__iadd__([param.value])
		else:
			print(('{:7s} {:11.5f} {:11.5s}'.format(name, param.value, param.stderr)))
			value.__iadd__([param.value])
	
	tau_nosilic = ytau - result.best_fit
	np.savetxt('New_tau_no_silic_'+name_source+'.txt', np.transpose([tt,tau_nosilic,etau]))
	np.savetxt('Best_fit_'+name_source+'.txt', np.transpose([tt,result.best_fit]))
	
	
	#plt.plot(tt, result.init_fit, 'k--', label='initial fit')
	plt.plot(tt, result.best_fit, 'r-', label='Best fit')
	plt.legend(loc='best')
	
	check = isinstance(param.stderr, float)
	if check == False:
		dely = 3*np.nanstd(etau)
		plt.fill_between(tt, result.best_fit-dely, result.best_fit+dely, color="#ABABAB", label='3-$\sigma$ uncertainty')
	else:
		dely = result.eval_uncertainty(sigma=3)
		plt.fill_between(tt, result.best_fit-dely, result.best_fit+dely, color="#ABABAB", label='3-$\sigma$ uncertainty')
	
	
	taumax = max(ytau) + 0.1*max(ytau)
	
	plt.plot(tt, ytau, color= 'black', label=name_source)
	plt.plot(tt, tau_nosilic, color= 'blue', label=name_source+' '+'(Removed Silicate)')
	plt.plot(tt, ytau*0., ':', color='lightgrey')
	plt.minorticks_on()
	plt.tick_params(which='major', direction='in', labelsize=10)
	plt.tick_params(which='minor', direction='in', labelsize=10)
	plt.xlim(min(tt),max(tt))
	plt.ylim(taumax,-0.1)
	plt.legend(loc='best', frameon=False, fontsize=9)
	plt.ylabel(r'Optical Depth$\mathrm{(\tau_{\lambda})}$',fontsize=10)
	plt.xlabel(r'$\lambda\ \mathrm{[\mu m]}$',fontsize=10)
	
	plt.savefig('Silicate_deconv.pdf',format='pdf', bbox_inches='tight', dpi=300)	
	
#filename = '/Users/will_rocha_starplan/Downloads/Data_StarPlan_Project/Fitting/T_silicate/'+'Optical_depth_GY92274.od'
#Silicate_decomposition(filename, 5.15, 30., npoints = 1000., pathlib = '/Users/will_rocha_starplan/Downloads/Data_StarPlan_Project', silicate_guess_factor = 0.25, values_wid = [0.1, 0.5, 0.8, 0.9, 0.9, 0.9], perc_wid_silicate=2.5, perc_wid_data = 0.4, perc_amp_min_data=0.3, perc_amp_max_data=0.41)