from ENIIGMA.Continuum import Fit
from ENIIGMA.GA import optimize
from ENIIGMA.GA import check_ga
from ENIIGMA.Stats import Pie_chart_plots
from ENIIGMA.Stats import Stats_Module
from ENIIGMA.Stats import Degen_plots
import ENIIGMA
import os


DIR = os.getcwd() + '/'

#Continuum
filename = DIR+'svs_49.txt'
#Fit.Continuum_poly(filename, 1.24, 4.0, order = 2., range_limits=[[1.24, 2.85], [3.8, 4.]])
#Fit.Continuum_BB(filename, 1.2, 4., range_limits=[[1.25, 2.5], [3.8, 4.]], guess = (1000, 1e-20), guess_view = False)

#exit()

filename = 'Optical_depth_svs_49.od'
list_sp = ['H2O_40K', 'CH3OH', 'HNCO_NH3'] #guess list

#optimize.ENIIGMA(filename, 2.9, 4., list_sp, group_comb=3, skip=False, ga_max = 1.0, gen=200, mutp = 0.05, popsize = 150, fitness='rmse')

#exit()
#Statistic of generations for each combinations
#check_ga.top_five_scaled(savepdf=False)
#check_ga.check(combination=14, option=-4, savepdf=False)

#Run the statistical module
#Stats_Module.stat(f_sig=3)

#exit()
#Pie plot
#Pie_chart_plots.pie(sig_level=9.)
#exit()


#Check components and create histogram
Degen_plots.merge_components_cd()
Degen_plots.hist_plot()