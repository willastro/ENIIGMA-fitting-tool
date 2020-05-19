from ENIIGMA.Continuum import Fit
from ENIIGMA.GA import optimize
from ENIIGMA.GA import check_ga
from ENIIGMA.Stats import Pie_chart_plots
from ENIIGMA.Stats import Stats_Module
from ENIIGMA.Stats import Degen_plots
import ENIIGMA


#Continuum
#filename = '/Users/will_rocha_starplan/Downloads/Data_StarPlan_Project/Fitting/TPY/'+'WL17.dat'
#filename = '/Users/will_rocha_starplan/eniigma_doc/doc/Tutorial/svs49.txt'
#Fit.Continuum_poly(filename, 1.24, 4.0, order = 2., range_limits=[[1.24, 2.85], [3.8, 4.]])
#Fit.Continuum_BB(filename, 1.2, 4., range_limits=[[1.25, 2.5], [3.8, 4.]], guess = (1000, 1e-20), guess_view = False)
#exit()
#exit()


#exit()
filename = 'Optical_depth_svs49.od'
list_sp = ['H2O_40K', 'H2O_NH3_CO2_CH4_10_1_1_1_72K_b', 'd_NH3_CH3OH_50_10K_I10m_Baselined', 'CO_NH3_10K', 'H2O_CH4_10_0.6_a_V3', 'CO_CH3OH_10K', 'HNCO_NH3'] #guess list


#dir_libice = ENIIGMA.__file__#'/Users/will_rocha_starplan/Downloads/Data_StarPlan_Project/'

#libice = '/Users/will_rocha_starplan/Downloads/Data_StarPlan_Project/'#dir_libice[:len(dir_libice)-12]
#exit()
#optimize.ENIIGMA(filename, 2.84, 4., list_sp, group_comb=3, skip=False, pathlib = None, ga_max = 0.5)

#exit()
#Statistic of generations for each combinations
#check_ga.top_five_scaled(savepdf=True)
check_ga.check(combination=26, option=-3, savepdf=False)

#exit()
#Pie plot
#Pie_chart_plots.pie(sig_level=9.)
#exit()
#Run the statistical module
#Stats_Module.stat(f_sig=2)

#Check components and create histogram
#Degen_plots.merge_components_cd()
#Degen_plots.hist_plot()	