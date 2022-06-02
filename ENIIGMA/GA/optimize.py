import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.interpolate import interp1d
import time
from itertools import combinations
import shutil
import sh
import pandas as pd
import sys
import site
from pyevolve import G1DList, GSimpleGA, Selectors, Crossovers
from pyevolve import Initializators, Mutators, Consts
from pyevolve import Interaction
from pyevolve import Statistics
from pyevolve import DBAdapters
from pyevolve import Scaling

tic = time.time()

from matplotlib.ticker import MultipleLocator, FormatStrFormatter, ScalarFormatter, LogLocator
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams['axes.linewidth'] = 1.5

# ======= nature constants (seem global) =====================
cl = 2.99792458E+08  # speed of light [m/s]
hplanck = 6.62607554E-34  # Planck's constant [J s]
bk = 1.38065812E-23  # Boltzmann's constant [J/K]
pi = np.pi  # just pi


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


def ENIIGMA(od_file, xmin, xmax, list_sp, n_points=500, group_comb=5, skip=False, pathlib=None, factor1=2, factor2=2,
            dtype='Obs', St='False', freqs=10, gen=100, ga_min=0., ga_max=0.5, mutp=0.01, popsize=50, fitness='rmse',
            initializator=Initializators.G1DListInitializatorReal, mutator=Mutators.G1DListMutatorRealGaussian,
            crossover=Crossovers.G1DBinaryStringXTwoPoint, scaling=Scaling.LinearScaling,
            selector=Selectors.GTournamentSelector, termination=GSimpleGA.RawScoreCriteria):
    """
	Used to combine files in the genetic algorithm spectral decomposition.

	Parameters
	-------------

	od_file : 'str'
		Optical depth file.

	xmin, xmax : 'float'
		Range used in the spectral decomposition.

	list_sp : 'str'
		List of ice data used as initial guess.

	n_points : 'float'
		Number of data points used for interpolating the data.

	group_comb : 'float'
		Value used to group species in the final part of the optimization.
		Default = 5. E.g. 6 files will be combined in group of 5 without repetition. In total, 6 combination will be performed.

	skip : 'bool'
		Default: False
		If True, the chi-square selection of files will not be performed.

	pathlib : 'str'
		Directory of ice data library.
		Default: site-packages in the python pathway.

	factor1 : 'int'
		Select files factor1 times higher than the minimum chi-square in the first performance, namely, only pure ice data.
		Default: 2

	factor2 : 'int'
		Select files factor1 times higher than the minimum chi-square in the second performance, namely, pure and mixture ice data.
		Default: 2

	dtype : 'str'
		If 'Obs' use the observational optical depth.
		If 'Lab' use the experimental absobance data.
		Default: 'Obs'

	St : 'bool'
		If 'True' shows the genetic algorithm statistic.
		Default: 'False'

	freqs : 'int'
		Show the genetic algorithm statistics at a given number of generations.
		Default: 10

	gen : 'int'
		Number of generations in the genetic algorithm optimization.
		Default: 100

	mutp : 'float'
		Mutation rate.
		Default: 0.01

	popsize : 'int'
		Population size.
		Default = 50


	Returns
	-------------
	Create a directory tree where the output files will be stored.

	Folder 'Standard' stores the initial guess files.

	Folder 'New_standard' stores the files selected after the initial guess.

	Folder 'Mutation' stores files based on the 'New_standard' folder.

	Folder 'Workspace' stores output files regarding the genetic algorithm optimization.

	File 'best_comb.csv' contains the chi-square values of all combination.

	File 'Best_score.txt' contains the best chi-square among all combination.

	File 'Best_values.txt' contains the best genes (weights) among all combination.

	File 'comb_score.txt' contains the chi-squares over all combination.

	File 'Pyevolve.db' contains the genetic algorithm statistics over all combination.



	"""
    print(' ')
    print(' ')
    print('=================================================================')
    print('|			ENIIGMA CODE				|')
    print('| A Python module for the decomposition of IR ice spectra using |')
    print('|        laboratory data and genetic modeling algorithm         |')
    print('|                   Current Version: V.0                        |')
    print('|                         April 2020                            |')
    print('=================================================================')

    DIR = os.getcwd() + '/'
    print('DIR:',DIR)

    if os.path.isdir(DIR + 'Workspace') == False:
        os.makedirs('Workspace')
    else:
        shutil.rmtree(DIR + 'Workspace')

    if os.path.isdir(DIR + 'Standard') == False:
        os.makedirs('Standard')
    else:
        try:
            sh.rm(sh.glob(DIR + 'Standard/*.dat'))
        except:
            pass

    if os.path.isdir(DIR + 'Mutation') == False:
        os.makedirs('Mutation')
    else:
        try:
            sh.rm(sh.glob(DIR + 'Mutation/*.dat'))
        except:
            pass

    if os.path.isdir(DIR + 'New_standard') == False:
        os.makedirs('New_standard')
    else:
        try:
            sh.rm(sh.glob(DIR + 'New_standard/*.dat'))
        except:
            pass

    if not os.path.exists(DIR + 'Workspace/Interp'):
        os.makedirs('Workspace/Interp')

    if not os.path.exists(DIR + 'Workspace/Interp2'):
        os.makedirs('Workspace/Interp2')

    if not os.path.exists(DIR + 'Workspace/Processing'):
        os.makedirs('Workspace/Processing')

    if not os.path.exists(DIR + 'Workspace/Processing/Interp_proc'):
        os.makedirs('Workspace/Processing/Interp_proc')

    if not os.path.exists(DIR + 'Workspace/Processing/Interp_proc/Degeneracy'):
        os.makedirs('Workspace/Processing/Interp_proc/Degeneracy')

    if not os.path.exists(DIR + 'Workspace/Processing/Store_interp'):
        os.makedirs('Workspace/Processing/Store_interp')

    if not os.path.exists(DIR + 'Workspace/R'):
        os.makedirs('Workspace/R')

    if not os.path.exists(DIR + 'Workspace/Store'):
        os.makedirs('Workspace/Store')

    if not os.path.exists(DIR + 'Workspace/Store_interp_0'):
        os.makedirs('Workspace/Store_interp_0')

    if not os.path.exists(DIR + 'Workspace/Store_interp_1'):
        os.makedirs('Workspace/Store_interp_1')

    t = pd.read_csv(od_file, sep='\s+', header=None)
    if dtype == 'Obs':
        x_lam = t[0]
        Abs = t[1]
        etau = t[2]
        tau = Abs
        ssize = len(x_lam) - 1
    else:
        wavenumber = t[0]
        Abs = t[1]
        etau = t[2]
        x_lam = 1e4 / wavenumber
        tau = 2.3 * Abs
        ssize = len(x_lam) - 1

    a = find_nearest(x_lam, xmin)
    b = find_nearest(x_lam, xmax)
    npoints = n_points

    ind1 = get_line_number2(a, x_lam) - 1
    ind2 = get_line_number2(b, x_lam) + 1

    ntt = np.linspace(a, b, npoints)
    try:
        new_tau = interp1d(x_lam[ind1:ind2], tau[ind1:ind2], kind='linear', bounds_error=False,
                           fill_value=0.05)  # interpolate data
        new_tau = new_tau(ntt)

        new_etau = interp1d(x_lam[ind1:ind2], etau[ind1:ind2], kind='linear', bounds_error=False,
                            fill_value=0.05)  # interpolate data
        new_etau = new_etau(ntt)

    except ValueError:
        new_tau = interp1d(x_lam[ind1:ind2], tau[ind1:ind2], kind='cubic', bounds_error=False,
                           fill_value=0.05)  # interpolate data
        new_tau = new_tau(ntt)

        new_etau = interp1d(x_lam[ind1:ind2], etau[ind1:ind2], kind='cubic', bounds_error=False,
                            fill_value=0.05)  # interpolate data
        new_etau = new_etau(ntt)

    np.savetxt('New_tau_GA.txt', np.transpose([ntt, new_tau, new_etau]))

    # plt.clf()
    # pp.close()
    # exit()

    # ======================================================================
    print('Performing a linear combination of pure ices...')
    # ======================================================================

    # ==================DIRECTORIES======================

    R = DIR + 'Workspace/R/'

    if pathlib == None:
        import ENIIGMA
        dir_libice = ENIIGMA.__file__
        libice = dir_libice[:len(dir_libice) - 12]

        p_ices = libice + '/ICE_LIB/P_ICES/*.dat'  # PURE ICES
        files_p_ices = glob.glob(p_ices)

        pt_ices = libice + '/ICE_LIB/PT_ICES/*.dat'  # HEATED PURE ICES
        files_pt_ices = glob.glob(pt_ices)

        mt_ices = libice + '/ICE_LIB/MT_ICES/*.dat'  # HEATED MIXTURES
        files_mt_ices = glob.glob(mt_ices)

        mi_ices = libice + '/ICE_LIB/MI_ICES/*.dat'  # IRRADIATED MIXTURES
        files_mi_ices = glob.glob(mi_ices)

    else:
        p_ices = pathlib + '/ICE_LIB/P_ICES/*.dat'  # PURE ICES
        files_p_ices = glob.glob(p_ices)

        pt_ices = pathlib + '/ICE_LIB/PT_ICES/*.dat'  # HEATED PURE ICES
        files_pt_ices = glob.glob(pt_ices)

        mt_ices = pathlib + '/ICE_LIB/MT_ICES/*.dat'  # HEATED MIXTURES
        files_mt_ices = glob.glob(mt_ices)

        mi_ices = pathlib + '/ICE_LIB/MI_ICES/*.dat'  # IRRADIATED MIXTURES
        files_mi_ices = glob.glob(mi_ices)

    dest1 = DIR + 'Workspace/'
    store = DIR + 'Workspace/Store/'
    store_in = DIR + 'Workspace/Store_interp_0/'
    store_in1 = DIR + 'Workspace/Store_interp_1/'
    proc = DIR + 'Workspace/Processing/'

    nstd = DIR + 'New_standard/'
    nstd_f = DIR + 'New_standard/*.dat'
    files_nstd_f = glob.glob(nstd_f)

    mutation = DIR + 'Mutation/'

    f1 = open(DIR + 'Workspace/Processing/none.dat', 'w')
    f2 = open(DIR + 'Workspace/Processing/Interp_proc/none.dat', 'w')
    f3 = open(DIR + 'Workspace/Processing/Interp_proc/none.txt', 'w')
    f4 = open(DIR + 'Workspace/Store/none.dat', 'w')
    f1.close()
    f2.close()
    f3.close()
    f4.close()

    sh.rm(sh.glob(DIR + 'Workspace/Processing/*.dat'))
    sh.rm(sh.glob(DIR + 'Workspace/Processing/Interp_proc/*.dat'))
    sh.rm(sh.glob(DIR + 'Workspace/Processing/Interp_proc/*.txt'))
    sh.rm(sh.glob(DIR + 'Workspace/Store/*.dat'))

    ###############  LINEAR COMBINATION  ############################################

    sp_guess = list_sp

    if skip == False:
        for i in range(len(files_p_ices)):
            str0 = files_p_ices[i]
            for bb in range(len(sp_guess)):
                if str0.find(sp_guess[bb] + '.dat') != -1:
                    shutil.copy(str0, DIR + 'Standard/')

        for i in range(len(files_pt_ices)):
            str0 = files_pt_ices[i]
            for bb in range(len(sp_guess)):
                if str0.find(sp_guess[bb] + '.dat') != -1:
                    shutil.copy(str0, DIR + 'Standard/')

        for i in range(len(files_mt_ices)):
            str0 = files_mt_ices[i]
            for bb in range(len(sp_guess)):
                if str0.find(sp_guess[bb] + '.dat') != -1:
                    shutil.copy(str0, DIR + 'Standard/')

        for i in range(len(files_mi_ices)):
            str0 = files_mi_ices[i]
            for bb in range(len(sp_guess)):
                if str0.find(sp_guess[bb] + '.dat') != -1:
                    shutil.copy(str0, DIR + 'Standard/')

        # exit()
        standard = DIR + 'Standard/*.dat'
        files_std = glob.glob(standard)

        for files_std in glob.glob(standard):  # Copying all standard data to Workspace
            shutil.copy(files_std, dest1)

        for files_std in glob.glob(standard):  # Copying all standard data to Store
            shutil.copy(files_std, store)

        i_var = len(files_p_ices)
        count00 = 0.
        for index0 in range(i_var):
            f = files_p_ices[index0]
            res = [ele for ele in sp_guess if (ele in f)]
            if bool(res) == True:
                res = res
            else:
                shutil.copy(f, dest1)

                mypath = DIR + 'Workspace/*.dat'
                files = glob.glob(mypath)

                i = len(files)
                for index in range(i):
                    t = pd.read_csv(files[index], sep='\s+', header=None)
                    x_lam = t[0]
                    x_nu = cl / (x_lam * 1e-6)
                    y = t[1] / max(t[1])
                    x_mic = 1e4 * (1 / x_lam)
                    tau = 2.3 * y

                    aa = find_nearest(x_mic, xmin)
                    bb = find_nearest(x_mic, xmax)
                    ind11 = get_line_number2(aa, x_mic) - 1
                    ind22 = get_line_number2(bb, x_mic) + 1

                    try:
                        tau_lab = interp1d(x_mic[ind11:ind22], tau[ind11:ind22], kind='cubic', bounds_error=False,
                                           fill_value=0.05)
                        tau_lab = (tau_lab(ntt))  # +0.01
                    except ValueError:
                        tau_lab = interp1d(x_mic[ind11:ind22], tau[ind11:ind22], kind='linear', bounds_error=False,
                                           fill_value=0.05)
                        tau_lab = (tau_lab(ntt))  # +0.01

                    interp_dir = DIR + 'Workspace/Interp/'
                    os.chdir(interp_dir)

                    np.savetxt('i_var.txt', np.transpose([i_var]), fmt='%1d')

                    f = open('interp__' + files[index][len(mypath) - 5:], 'w')
                    for v in range(len(ntt)):
                        f.write('{0:f} {1:f}\n'.format(ntt[v], tau_lab[v]))
                    f.close()

                    filess = glob.glob(DIR + 'Workspace/Interp/*.dat')

                    if sys.version_info[0] == 3:
                        from ENIIGMA.GA import create3
                        create3.create_interp3(filess)
                    else:
                        import create
                        create.create_interp2(filess)

                with open('interp_all.txt', 'r') as f2:
                    lines = f2.readlines()
                data = []
                data = [line.split() for line in lines]
                data2 = np.asfarray(data)

                # exit()

                from ENIIGMA.GA import GA_module_pyev as gam
                Sp0 = DIR + 'Workspace/Store_interp_0/'
                sh.cp(sh.glob(DIR + 'Workspace/Interp/*.dat'), Sp0)

                home1 = DIR + 'Workspace/Interp/'
                home2 = DIR + '/Workspace/Store_interp_0/'

                f = gam.gamod(new_tau, new_etau, home1, home2, Stats=St, freq_stat=freqs, gen=gen, ga_min=ga_min,
                              ga_max=ga_max, mutp=mutp, popsize=popsize, cc=1., fitness=fitness,
                              initializator=initializator, mutator=mutator, crossover=crossover, scaling=scaling,
                              selector=selector, termination=termination)

                tscore = np.loadtxt(
                    DIR + 'Workspace/Interp/' + 'comb_score0.txt').T  # pd.read_csv(DIR+'Workspace/Interp/'+'comb_score0.txt',sep='\s+', header=None)
                score = tscore[0]

                count00 = count00 + 1
                print('round', count00, 'of', i_var - 1, 'Score =', score)

                filename = 'interp__' + files_p_ices[index0][len(p_ices) - 5:]
                if os.path.exists(filename):
                    # shutil.move(filename, store)
                    os.remove(filename)

                shutil.copy(dest1 + files_p_ices[index0][len(p_ices) - 5:], store)
                os.remove(dest1 + files_p_ices[index0][len(p_ices) - 5:])
                try:
                    sh.rm(sh.glob(DIR + 'Workspace/Interp/*.dat'))
                    sh.rm(sh.glob(DIR + 'Workspace/Interp/OutFile.txt'))
                except:
                    raise Exception('Please, give your initial guess!')

        # exit()

        # Selection

        df = pd.read_csv(DIR + 'Workspace/Interp/Best_comb.csv', sep=',')
        matrix = df.shape[1] - 3

        cuttof = max(df['best_chi'][:factor1 * matrix])

        df = df[df.best_chi <= cuttof]
        df.to_csv('Best_comb_selected.csv', index=False)

        sp = np.loadtxt(DIR + 'Workspace/Interp/Best_comb_selected.csv', dtype=str, delimiter=',', usecols=[matrix],
                        skiprows=1).T
        for i in range(matrix):
            try:
                shutil.copy(store + sp[i].split('__')[1] + '.dat', nstd)
            except IndexError:
                print(' ')
                print('ENIIGMA info: No file was selected for the next step. Please, check the cuttof value.')
                print(' ')

        # exit()

        # SELECT BRANCH CONTAINING THE BEST SPECIES (EXCEPT WATER, ONCE MOST OF THE BRANCHS HAS ALREADY WATER INCLUDED - first approximation)

        spn = np.loadtxt(DIR + 'Workspace/Interp/Best_comb_selected.csv', dtype=str, delimiter=',', usecols=[matrix],
                         skiprows=1).T

        for files_pt_ices in glob.glob(pt_ices):
            str1 = files_pt_ices
            for bb in range(matrix):
                if str1.find(spn[bb].split('__')[1]) != -1:
                    shutil.copy(files_pt_ices, mutation)

        for files_mt_ices in glob.glob(mt_ices):
            str1 = files_mt_ices
            for bb in range(matrix):
                if str1.find(spn[bb].split('__')[1]) != -1:
                    shutil.copy(files_mt_ices, mutation)

        for files_mi_ices in glob.glob(mi_ices):
            str1 = files_mi_ices
            for bb in range(matrix):
                if str1.find(spn[bb].split('__')[1]) != -1:
                    shutil.copy(files_mi_ices, mutation)

        sh.rm(sh.glob(DIR + 'Workspace/*.dat'))
        # exit()

        # PERFORMANCE AFTER SELECTION OF MUTATION

        nstd = DIR + 'New_standard/'
        nstd_f = DIR + 'New_standard/*.dat'
        files_nstd_f = glob.glob(nstd_f)

        for instd in range(len(files_nstd_f)):
            shutil.copy(files_nstd_f[instd], dest1)

        # for files_nstd_f in glob.glob(nstd_f): #Copying all standard data to Workspace
        #	shutil.copy(files_nstd_f, dest1)

        list2 = []
        for ins in range(len(files_nstd_f)):
            fn = files_nstd_f[ins][len(DIR + 'New_standard/'):]
            list2.append(fn)

        # exit()
        mut = DIR + 'Mutation/*.dat'
        files_mut = glob.glob(mut)

        listmut = []
        countl = 0
        for imut in range(len(files_mut)):
            flmut = files_mut[imut][len(DIR + 'Mutation/'):]
            listmut.append(flmut)
            if flmut not in list2:
                countl = countl + 1
            else:
                list2 = list2

        count0 = 0.
        i_var = len(files_mut)
        # exit()
        for index0 in range(i_var):
            f = files_mut[index0]
            # print f[len(DIR+'Mutation/'):]
            # print listmut[index0]

            if listmut[index0] in list2:
                list2 = list2
            else:
                shutil.copy(f, dest1)
                # print 'here-1'
                mypath2 = DIR + 'Workspace/*.dat'
                files = glob.glob(mypath2)

                i = len(files)
                for index in range(i):
                    # print(files[index])
                    df = pd.read_csv(files[index], sep='\s+', header=None)
                    x_lam = df[0]
                    y = df[1] / max(df[1])
                    x_nu = cl / (x_lam * 1e-6)
                    x_mic = 1e4 * (1 / x_lam)
                    tau = 2.3 * y

                    aa = find_nearest(x_mic, xmin)
                    bb = find_nearest(x_mic, xmax)
                    ind11 = get_line_number2(aa, x_mic) - 1
                    ind22 = get_line_number2(bb, x_mic) + 1

                    try:
                        tau_lab = interp1d(x_mic[ind11:ind22], tau[ind11:ind22], kind='cubic', bounds_error=False,
                                           fill_value=0.05)
                        tau_lab = (tau_lab(ntt))  # +0.01
                    except ValueError:
                        tau_lab = interp1d(x_mic[ind11:ind22], tau[ind11:ind22], kind='linear', bounds_error=False,
                                           fill_value=0.05)
                        tau_lab = (tau_lab(ntt))  # +0.01

                    interp_dir = DIR + 'Workspace/Interp2/'
                    os.chdir(interp_dir)

                    f = open('interp__' + files[index][len(mypath2) - 5:], 'w')
                    for v in range(len(ntt)):
                        f.write('{0:f} {1:f}\n'.format(ntt[v], tau_lab[v]))
                    f.close()

                    filess = glob.glob(DIR + 'Workspace/Interp2/*.dat')

                    if sys.version_info[0] == 3:
                        from ENIIGMA.GA import create3
                        create3.create_interp3(filess)
                    else:
                        import create
                        create.create_interp2(filess)

                np.savetxt('i_var.txt', np.transpose([countl]), fmt='%1d')

                with open('interp_all.txt', 'r') as f2:
                    lines = f2.readlines()
                data = []
                data = [line.split() for line in lines]
                data2 = np.asfarray(data)

                from ENIIGMA.GA import GA_module_pyev as gam
                Sp1 = DIR + 'Workspace/Store_interp_1/'
                sh.cp(sh.glob(DIR + 'Workspace/Interp2/*.dat'), Sp1)

                home1 = DIR + 'Workspace/Interp2/'
                home2 = DIR + '/Workspace/Store_interp_1/'

                f = gam.gamod(new_tau, new_etau, home1, home2, Stats=St, freq_stat=freqs, gen=gen, ga_min=ga_min,
                              ga_max=ga_max, mutp=mutp, popsize=popsize, cc=count0, fitness=fitness,
                              initializator=initializator, mutator=mutator, crossover=crossover, scaling=scaling,
                              selector=selector, termination=termination)

                tscore = np.loadtxt(
                    DIR + 'Workspace/Interp2/' + 'comb_score0.txt').T  # pd.read_csv(DIR+'Workspace/Interp/'+'comb_score0.txt',sep='\s+', header=None)
                score = tscore[0]
                count0 = count0 + 1
                print('round', count0, 'of', countl, 'Score =', score)

                filename = 'interp__' + files_mut[index0][len(mut) - 5:]
                if os.path.exists(filename):
                    # shutil.move(filename, store)
                    os.remove(filename)

                shutil.copy(dest1 + files_mut[index0][len(mut) - 5:], store)
                os.remove(dest1 + files_mut[index0][len(mut) - 5:])
                try:
                    sh.rm(sh.glob(DIR + 'Workspace/Interp2/*.dat'))
                    sh.rm(sh.glob(DIR + 'Workspace/Interp2/OutFile.txt'))
                except:
                    raise Exception('Add error message....!')

        # sh.cp(sh.glob(DIR+'Workspace/Interp2/*.db'),DIR+'Workspace/Interp2/Pyevolve_DB/')
        # sh.rm(sh.glob(DIR+'Workspace/Interp2/*.db'))
        # exit()
        # Selection

        df = pd.read_csv(DIR + 'Workspace/Interp2/Best_comb.csv', sep=',')
        matrix = df.shape[1] - 3

        cuttof = max(df['best_chi'][:factor2 * matrix])

        df = df[df.best_chi <= 1. * cuttof]
        df.to_csv('Best_comb_selected.csv', index=False)

        sp = np.loadtxt(DIR + 'Workspace/Interp2/Best_comb_selected.csv', dtype=str, delimiter=',', usecols=[matrix],
                        skiprows=1).T
        for i in range(df.shape[0]):
            # print store_in1+sp[i]+'.dat'
            shutil.copy(store_in1 + sp[i] + '.dat', R)

        # exit()

        interp_dir = DIR + 'Workspace/Processing/Interp_proc/'
        os.chdir(interp_dir)

        filess = glob.glob(DIR + 'Workspace/R/*.dat')

        if sys.version_info[0] == 3:
            from ENIIGMA.GA import create3
            create3.create_R3(filess)
        else:
            import create
            create.create_R2(filess)

        df = pd.read_csv(DIR + 'Workspace/Processing/Interp_proc/All_R.txt', sep='\s+', header=None)
        # print df.shape
        # exit()
        q = df.shape[1] / 2

        if group_comb <= q:
            qq = range(int(q))

            def func(i):
                filess = glob.glob(DIR + 'Workspace/R/*.dat')
                spp = filess[i][len(R):]
                spp1 = spp.split('__')[1]
                spp2 = spp1.split('.dat')[0]

                in0, in1 = i * 2, (i * 2) + 1

                f = open(spp2 + '.dat', 'w')
                for v in range(df.shape[0]):
                    f.write('{0:f} {1:f}\n'.format(df[in0][v], df[in1][v]))
                f.close()

            combin = combinations(qq, group_comb)  # number is the group size for each combination
            count = 0
            all = list(combin)
            np.savetxt('i_var.txt', np.transpose([len(all)]), fmt='%1d')

            def func2(all):
                count = 0
                for k in range(len(all)):
                    at = all[k]
                    # print 'at', at
                    for indexf in at:
                        func(indexf)

                    from ENIIGMA.GA import GA_module_pyev as gam
                    Sp = DIR + 'Workspace/Processing/Store_interp/'
                    sh.cp(sh.glob(DIR + 'Workspace/Processing/Interp_proc/*.dat'), Sp)
                    home1 = DIR + 'Workspace/Processing/Interp_proc/'
                    home2 = DIR + '/Workspace/Processing/Store_interp/'

                    f = gam.gamod(new_tau, new_etau, home1, home2, Stats=St, freq_stat=freqs, gen=gen, ga_min=ga_min,
                                  ga_max=ga_max, mutp=mutp, popsize=popsize, cc=count, fitness=fitness,
                                  initializator=initializator, mutator=mutator, crossover=crossover, scaling=scaling,
                                  selector=selector, termination=termination)

                    sh.rm(sh.glob(DIR + 'Workspace/Processing/Interp_proc/*.dat'))
                    sh.rm(sh.glob(DIR + 'Workspace/Processing/Interp_proc/OutFile.txt'))

                    tscore = np.loadtxt(
                        DIR + 'Workspace/Processing/Interp_proc/' + 'comb_score0.txt').T  # pd.read_csv(DIR+'Workspace/Interp/'+'comb_score0.txt',sep='\s+', header=None)
                    score = tscore[0]
                    count = count + 1
                    print('round', count, 'of', len(all), 'Score =', score)

            func2(all)
        else:
            print(' ')
            print(' ')
            print('ENIIGMA info: ERROR: Cannot combine' + ' ' + str(q) + ' ' + 'files in groups of' + ' ' + str(
                group_comb) + '!')
            print('ENIIGMA info: Please, use group_comb =' + ' ' + str(q) + ' ' + 'or less.')
            print(' ')
            print(' ')
        # exit()

        ############
        from ENIIGMA.GA import Plot_fitting as plot
        path = DIR + 'Workspace/Processing/'
        plot.plot(path)

        plt.plot(ntt, new_tau, color='black', label='Data')
        plt.plot(ntt, 0. * new_tau, ':', color='gray')
        vmin = min(new_tau) - 0.1 * (min(new_tau))
        vmax = max(new_tau) + 0.1 * (max(new_tau))
        plt.xlim(min(ntt), max(ntt))
        plt.ylim(vmax, vmin)
        # plt.tight_layout()
        plt.legend(loc='best', ncol=2, frameon=False, fontsize=9)
        plt.ylabel(r'$\mathrm{\tau_{\lambda}}$', fontsize=10)
        plt.xlabel(r'$\lambda\ \mathrm{[\mu m]}$', fontsize=10)

        plt.savefig(DIR + 'Final_plot.pdf')
        os.chdir('../../../')

        toc = time.time()

        dt = toc - tic

        print("\n The elapsed time was:", int(dt), "sec")

    elif skip == True:
        for i in range(len(files_p_ices)):
            str0 = files_p_ices[i]
            for bb in range(len(sp_guess)):
                if str0.find(sp_guess[bb] + '.dat') != -1:
                    shutil.copy(str0, DIR + 'Standard/')

        for i in range(len(files_pt_ices)):
            str0 = files_pt_ices[i]
            for bb in range(len(sp_guess)):
                if str0.find(sp_guess[bb] + '.dat') != -1:
                    shutil.copy(str0, DIR + 'Standard/')

        for i in range(len(files_mt_ices)):
            str0 = files_mt_ices[i]
            for bb in range(len(sp_guess)):
                if str0.find(sp_guess[bb] + '.dat') != -1:
                    shutil.copy(str0, DIR + 'Standard/')

        for i in range(len(files_mi_ices)):
            str0 = files_mi_ices[i]
            for bb in range(len(sp_guess)):
                if str0.find(sp_guess[bb] + '.dat') != -1:
                    shutil.copy(str0, DIR + 'Standard/')

        # exit()
        standard = DIR + 'Standard/*.dat'
        files_std = glob.glob(standard)

        for files_std in glob.glob(standard):
            shutil.copy(files_std, R)

        R = DIR + 'Workspace/R/'
        filename = DIR + 'Workspace/R/*.dat'
        file_R = glob.glob(filename)

        # a = list(product(c, repeat= size))
        q = len(file_R)
        qq = range(int(q))

        combin = combinations(qq, group_comb)  # number is the group size for each combination
        all = list(combin)

        count = 0
        for k in range(len(all)):
            at = all[k]
            print(at)
            for indexf in at:
                print(indexf)
                spp = file_R[indexf][len(R):]
                shutil.copy(R + spp, proc)
                print(spp)  #

            mypath2 = DIR + 'Workspace/Processing/*.dat'
            files = glob.glob(mypath2)

            fig = plt.figure()

            i = len(files)
            for index in range(i):
                df = pd.read_csv(files[index], sep='\s+', header=None)
                x_lam = df[0]
                x_nu = cl / (x_lam * 1e-6)
                y = df[1] / max(df[1])
                print('max is', max(y))
                x_mic = 1e4 * (1 / x_lam)
                tau = 2.3 * y

                aa = find_nearest(x_mic, xmin)
                bb = find_nearest(x_mic, xmax)
                ind11 = get_line_number2(aa, x_mic) - 1
                ind22 = get_line_number2(bb, x_mic) + 1

                try:
                    tau_lab = interp1d(x_mic[ind11:ind22], tau[ind11:ind22], kind='cubic', bounds_error=False,
                                       fill_value=0.05)
                    tau_lab = (tau_lab(ntt))  # +0.01
                except ValueError:
                    tau_lab = interp1d(x_mic[ind11:ind22], tau[ind11:ind22], kind='linear', bounds_error=False,
                                       fill_value=0.05)
                    tau_lab = (tau_lab(ntt))  # +0.01

                interp_dir = DIR + 'Workspace/Processing/Interp_proc/'
                os.chdir(interp_dir)
                np.savetxt('i_var.txt', np.transpose([len(all)]), fmt='%1d')

                f = open(files[index][len(DIR) + 21:], 'w')
                for v in range(len(ntt)):
                    # tau_lab = tau_lab[v]
                    f.write('{0:f} {1:f}\n'.format(ntt[v], tau_lab[v]))
                f.close()

            from ENIIGMA.GA import GA_module_pyev as gam
            Sp = DIR + 'Workspace/Processing/Store_interp/'
            sh.cp(sh.glob(DIR + 'Workspace/Processing/Interp_proc/*.dat'), Sp)
            home1 = DIR + 'Workspace/Processing/Interp_proc/'
            home2 = DIR + '/Workspace/Processing/Store_interp/'

            f = gam.gamod(new_tau, new_etau, home1, home2, Stats=St, freq_stat=freqs, gen=gen, ga_min=ga_min,
                          ga_max=ga_max, mutp=mutp, popsize=popsize, cc=count, fitness=fitness,
                          initializator=initializator, mutator=mutator, crossover=crossover, scaling=scaling,
                          selector=selector, termination=termination)

            sh.rm(sh.glob(DIR + 'Workspace/Processing/*.dat'))
            sh.rm(sh.glob(DIR + 'Workspace/Processing/Interp_proc/*.dat'))
            sh.rm(sh.glob(DIR + 'Workspace/Processing/Interp_proc/OutFile.txt'))

            tscore = np.loadtxt(
                DIR + 'Workspace/Processing/Interp_proc/' + 'comb_score0.txt').T  # pd.read_csv(DIR+'Workspace/Interp/'+'comb_score0.txt',sep='\s+', header=None)
            score = tscore[0]
            count = count + 1
            print('round', count, 'of', len(all), 'Score =', score)

        ############
        from ENIIGMA.GA import Plot_fitting as plot
        path = DIR + 'Workspace/Processing/'
        plot.plot(path)

        plt.plot(ntt, new_tau, color='black', label='Data')
        plt.plot(ntt, 0. * new_tau, ':', color='gray')
        vmin = min(new_tau) - 0.1 * (min(new_tau))
        vmax = max(new_tau) + 0.1 * (max(new_tau))
        plt.xlim(min(ntt), max(ntt))
        plt.ylim(vmax, vmin)
        # plt.tight_layout()
        plt.legend(loc='best', ncol=2, frameon=False, fontsize=9)
        plt.ylabel(r'$\mathrm{\tau_{\lambda}}$', fontsize=10)
        plt.xlabel(r'$\lambda\ \mathrm{[\mu m]}$', fontsize=10)

        plt.savefig(DIR + 'Final_plot.pdf')
        os.chdir('../../../')

        toc = time.time()

        dt = toc - tic

        print("\n The elapsed time was:", int(dt), "sec")

