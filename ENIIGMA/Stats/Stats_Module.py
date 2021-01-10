import pandas as pd
import numpy as np
import os
import glob
import sys


def stat(DIR=os.getcwd() + '/', f_sig=5):
    """
	Statistic module.

	Parameters
	-------------

	f_sig : 'float'
		Factor for the standard deviation in the statistic module.

	"""
    print('====================================================')
    print('		RUNNING STATISTICAL MODULE')
    print('====================================================')

    # DIR = os.getcwd()+'/'
    # DIR = DIR[:len(DIR)-33]
    # /Users/will_rocha_starplan/Downloads/Data_StarPlan_Project/Fitting/TC/Workspace/Processing/Interp_proc/
    home1 = DIR + 'Workspace/Processing/Interp_proc/'
    home2 = DIR + 'Workspace/Processing/Store_interp/'
    pathb = home1 + 'Best_comb.csv'

    os.chdir(home1)

    df = pd.read_csv(home1 + 'Best_comb.csv', sep=',', header=1)
    n_genes = df.shape[1] - 3  # number of genes

    data = pd.read_csv(pathb, delimiter=",", low_memory=True, usecols=['name'], nrows=n_genes)
    sp = home2 + data + '.dat'
    list = sp.T.values.tolist()[0]

    if sys.version_info[0] == 3:
        from ENIIGMA.Stats import create3
        create3.create_file3(list)
    else:
        import create
        create.create_file2(list)

    ####Perform statistic here!

    from ENIIGMA.Stats import stats_ga as sga
    pathdata0 = DIR + 'New_tau_GA.txt'
    tdata0 = pd.read_csv(pathdata0, sep='\s+', header=None)
    xdata0 = tdata0[0]
    new_tau = tdata0[1]
    err_tau = tdata0[2]
    fileout = home1 + 'output_file.txt'
    file_et = DIR + 'New_tau_GA.txt'
    filecl = home1 + 'Best_comb.csv'
    sga.stats_blockt(xdata0, new_tau, err_tau, fileout, filecl, home1, f_sig=f_sig)

    for f2 in glob.glob(home1 + 'Column_*.csv'):
        os.remove(f2)
    for f3 in glob.glob(home1 + 'trans_*.csv'):
        os.remove(f3)
    for f4 in glob.glob(home1 + 'Column_density_*.csv'):
        os.remove(f4)

    os.remove(home1 + 'MergeCD.csv')
    os.remove(home1 + 'MergeCD2.csv')
    os.remove(home1 + 'MergeCD3.csv')
    os.remove(home1 + 'MergeCD4.csv')

    os.remove(home1 + 'MergeCD_min.csv')
    os.remove(home1 + 'MergeCD2_min.csv')
    os.remove(home1 + 'MergeCD3_min.csv')
    os.remove(home1 + 'MergeCD4_min.csv')

    os.remove(home1 + 'MergeCD_max.csv')
    os.remove(home1 + 'MergeCD2_max.csv')
    os.remove(home1 + 'MergeCD3_max.csv')
    os.remove(home1 + 'MergeCD4_max.csv')

    print('====================================================')
    print('		Created by the ENIIGMA Team')
    print('		ENIIGMA code V.0 - April 2020')
    print('====================================================')
