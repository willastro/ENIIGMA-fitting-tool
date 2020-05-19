


Importing modules
===============================


Continuum 
-------------------------

Both polynomial and blackbody continuum fit are imported via:
::
	>>> from ENIIGMA.Continuum import Fit
	>>> Fit.Continuum_poly(filename, 5.15, 30., order = 1., range_limits=[[5.25, 5.30], [26.5, 27.5], [27.5,30.]])
	>>> Fit.Continuum_BB(filename, 5.15, 30., range_limits=[[5.25, 8.], [16., 20.], [20., 30.]], guess = (500, 1e-18, 200, 3e-16), guess_view = False)


Silicate feature decomposition 
-------------------------

The silicate features at 9.8 and 18 microns are decomposed by 6 Gaussian function each one:
::
	>>> from ENIIGMA.Silicate import Silicate_deconv
	>>> Silicate_deconv.Silicate_decomposition(filename, xmin, xmax, npoints = 1000., pathlib = Default, silicate_guess_factor = 0.25, values_wid = [0.1, 0.5, 0.8, 0.9, 0.9, 0.9], perc_wid_silicate=2.5, perc_wid_data = 0.4, perc_amp_min_data=0.3, perc_amp_max_data=0.41)


Genetic algorithm optimisation
-------------------------

Called as:
::
	>>> from ENIIGMA.GA import optimize
	>>> optimize.ENIIGMA(filename, 2.5,15., list_sp, group_comb=6, skip=True, pathlib = Default)

Once finished the optimisation, the fitness function evolution can be checked as:
::
	
	>>> from ENIIGMA.GA import check_ga
	>>> check_ga.check(combination=185, option=-9)

The best five combinations can be checked by:
::
	
	>>> from ENIIGMA.GA import check_ga
	>>> check_ga.top_five_scaled()
	

Statistical analysis
-------------------------

The statistical module is imported from ENIIGMA.Stats. The following options are available:
::

Pie chart
	
	>>> from ENIIGMA.Stats import Pie_chart_plots
	>>> Pie_chart_plots.pie(sig_level=16.81)

Confidence interval
	
	>>> from ENIIGMA.Stats import Stats_Module
	>>> Stats_Module.stat(f_sig=3)

Degeneracy analysis
	
	>>> from ENIIGMA.Stats import Degen_plots
	>>> Degen_plots.merge_components_cd()
	>>> Degen_plots.hist_plot()