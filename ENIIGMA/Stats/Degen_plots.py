import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import os
import glob
from pandas import DataFrame
import sh
import pandas as pd
import sys


def chi_values(dir=os.getcwd() + '/'):
    """
	Chi-square values.

	"""
    dir = dir + 'Workspace/Processing/Interp_proc/Degeneracy/'
    dir_b1 = dir[:len(dir) - 11]
    dir_b2 = dir[:len(dir_b1) - 33]

    file1 = dir_b1 + 'Confidence_limits_2nd.dat'
    file2 = dir_b1 + 'output_file.txt'
    fileqmin = dir_b1 + 'q_min.txt'
    fileqmax = dir_b1 + 'q_max.txt'
    fobs = dir_b2 + 'New_tau_GA.txt'

    tdata0 = pd.read_csv(fobs, sep='\s+', header=None)
    xd = tdata0[0]
    yd = tdata0[1]
    ey = tdata0[2]

    t0 = pd.read_csv(file1, sep='\s+', header=None)
    Cmin0 = pd.read_csv(file1, sep='\s+', header=None, usecols=list(range(t0.shape[1] - 1)), nrows=1)
    Cmin = Cmin0.T.values.tolist()

    t1 = pd.read_csv(file2, sep='\s+', header=None)
    Ysp = pd.read_csv(file2, sep='\s+', header=None, usecols=list(range(1, t1.shape[1], 2)))

    fileqmin = fileqmin
    fileqmax = fileqmax

    tmin = pd.read_csv(fileqmin, sep='\s+', header=None)
    tmax = pd.read_csv(fileqmax, sep='\s+', header=None)

    crange = list(range(t0.shape[1] - 1))
    ysprange = list(range(1, t1.shape[1], 2))

    f0 = 0.
    fmin = 0.
    fmax = 0.
    for i, j in zip(crange, ysprange):
        f0 += Cmin[i] * Ysp[j]
        fmin += tmin[0][i] * Ysp[j]
        fmax += tmax[0][i] * Ysp[j]

    p0 = (1. / (len(yd) - 1 - (t0.shape[1] - 1))) * np.sum(((yd - f0) / ey) ** 2)
    pmin = (1. / (len(yd) - 1 - (t0.shape[1] - 1))) * np.sum(((yd - fmin) / ey) ** 2)
    pmax = (1. / (len(yd) - 1 - (t0.shape[1] - 1))) * np.sum(((yd - fmax) / ey) ** 2)

    return p0, pmin, pmax


def merge_components_cd(dir=os.getcwd() + '/'):
    """
	Merge components.

	"""

    print('-------------------------------------------------------------------------')
    print('Searching for combinations inside 3 sigma confidence interval...')
    print('-------------------------------------------------------------------------')

    dir = dir + 'Workspace/Processing/Interp_proc/Degeneracy/'
    os.chdir(dir)
    dir_b1 = dir[:len(dir) - 11]
    dir_b2 = dir[:len(dir_b1) - 33]

    fobs = dir_b2 + 'New_tau_GA.txt'
    fileb = dir_b1 + 'Best_comb.csv'

    tdata0 = pd.read_csv(fobs, sep='\s+', header=None)
    xd = tdata0[0]
    yd = tdata0[1]
    ey = tdata0[2]

    pathb4 = fileb
    tb4 = pd.read_csv(pathb4, sep=',', header=1)
    size4 = tb4.shape[0]
    n_genes = tb4.shape[1] - 3

    path0 = pathb4[:len(pathb4) - 25] + 'Store_interp/'

    n = 0
    for i in range(0, size4, n_genes):
        c_vals = np.loadtxt(pathb4, dtype=float, delimiter=',', usecols=list(range(n_genes)), skiprows=1).T
        cc = []
        for j in range(n_genes):
            cc.append(c_vals[j][i])

        sp_names = np.loadtxt(pathb4, dtype=str, delimiter=',', usecols=(n_genes), skiprows=1).T
        # print sp_names
        step_i = i
        step_f = step_i + n_genes
        sp = []
        sp_id = []

        for k in range(step_i, step_f):
            sp.append(path0 + sp_names[k] + '.dat')
            sp_id.append(sp_names[k])

        if sys.version_info[0] == 3:
            from ENIIGMA.Stats import create3
            create3.create_file3four(sp)
        else:
            import create
            create.create_file2four(sp)

        fileout = dir + 'output_file4.txt'

        t1 = pd.read_csv(dir + 'output_file4.txt', sep='\s+', header=None)
        Ysp = pd.read_csv(dir + 'output_file4.txt', sep='\s+', header=None, usecols=list(range(1, t1.shape[1], 2)))

        crange = list(range(n_genes))
        ysprange = list(range(1, t1.shape[1], 2))

        f0 = 0.
        count = 1
        for i, j in zip(crange, ysprange):
            name = sp_id[i]
            f0 += cc[i] * Ysp[j]

            f0c = cc[i] * Ysp[j]

            Data1 = {str(name): f0c}
            df1 = DataFrame(Data1, columns=[str(name)])
            df1.to_csv('Comp_' + str(count) + '_' + str(name) + '.csv', index=False)
            count = count + 1

        # print 'f0 is:'
        # print f0

        chi_calc = (1. / (len(yd) - 1 - n_genes)) * np.sum(((yd - f0) / ey) ** 2)

        red_chi2_vals = chi_values()

        if chi_calc <= red_chi2_vals[2]:
            # print red_chi2_vals[2]
            Wav = {'Wavelength': t1[0]}
            df_wav = DataFrame(Wav, columns=['Wavelength'])
            df_wav.to_csv('Comp_0_wav.csv', index=False)

            D0 = {'all': f0}
            df_wav = DataFrame(D0, columns=['all'])
            df_wav.to_csv('Comp_' + str(n_genes + 1) + '_all.csv', index=False)

            all_filenames = [i for i in sorted(glob.glob(dir + 'Comp_*'))]
            combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames], axis=1)
            combined_csv.to_csv("Components_" + str(n) + ".csv", index=False, encoding='utf-8-sig')
            sh.rm(sh.glob(dir + 'Comp_*'))
        else:
            sh.rm(sh.glob(dir + 'Comp_*'))

        n = n + 1

    from ENIIGMA.Stats import deconvolution as dcv
    from ENIIGMA.Stats.Merge_colden import mergecd_no_bp

    pathdir = dir + '/'
    files_comp = 'Components_*.csv'
    file_csv = sorted(glob.glob(pathdir + files_comp))

    print('-------------------------------------------------------------------------')
    print(len(file_csv), ' combinations found inside 3 sigma confidence interval')
    print('')
    print('Deconvolving components and merging column densities...')
    print('')
    print('Creating file: All_merge_final.csv')
    print('-------------------------------------------------------------------------')

    count = 0
    for index in range(len(file_csv)):
        file = file_csv[index]
        dcv.deconv_all(file)
        filename = pathdir + 'Column_density_*.csv'
        mergecd_no_bp(filename, pathdir)

        try:
            orig_name = pathdir + 'MergeCD5.csv'
            new_name = pathdir + 'MergeCD5_' + str(count) + '.csv'
            os.rename(orig_name, new_name)

            for f1 in glob.glob(pathdir + 'Analytic*.dat'):
                os.remove(f1)
            for f2 in glob.glob(pathdir + 'Column_*.csv'):
                os.remove(f2)
            for f3 in glob.glob(pathdir + 'trans_*.csv'):
                os.remove(f3)
            for f4 in glob.glob(pathdir + 'Column_density_*.csv'):
                os.remove(f4)
            os.remove(pathdir + 'MergeCD.csv')
            os.remove(pathdir + 'MergeCD2.csv')
            os.remove(pathdir + 'MergeCD3.csv')
            os.remove(pathdir + 'MergeCD4.csv')
        except:
            pass

        count = count + 1

    sp_files = 'MergeCD5*.csv'
    f_sp = sorted(glob.glob(pathdir + sp_files))

    cnt = 0
    for index2 in range(len(f_sp)):
        pd.read_csv(f_sp[index2], header=None).T.to_csv('output_' + str(cnt) + '.csv', header=False, index=False)
        cnt = cnt + 1

    df = pd.concat(list(map(pd.read_csv, glob.glob(os.path.join('', "output_*.csv")))),
                   sort='False')  # sort = False because pandas warning
    df.fillna('nan', inplace=True)
    df.replace(0.0, np.nan, inplace=True)
    df.fillna('nan', inplace=True)
    df = df[~df['Label'].isin(['CDinmix', 'CDpure'])]
    df.insert(0, 'index', list(range(1, len(f_sp) + 1)))
    df = df.drop(['Label'], axis=1)
    df.to_csv('All_merge_final.csv', index=False)

    for f5 in glob.glob(pathdir + 'MergeCD5_*.csv'):
        os.remove(f5)
    for f6 in glob.glob(pathdir + 'output_*.csv'):
        os.remove(f6)


def replace(tup, x, y):  # Replace elements in a tuple
    tup_list = list(tup)
    for element in tup_list:
        if element == x:
            tup_list[tup_list.index(element)] = y
    new_tuple = tuple(tup_list)
    return new_tuple


def hist_plot(dir=os.getcwd() + '/'):
    os.chdir(dir + 'Workspace/Processing/Interp_proc/Degeneracy/')
    file = dir + 'Workspace/Processing/Interp_proc/Degeneracy/All_merge_final.csv'
    # file = os.getcwd()+'/All_merge_final.csv'

    t = np.loadtxt(file, dtype=float, delimiter=',', skiprows=1).T

    X = np.genfromtxt(file, dtype=float, delimiter=',', names=True)
    header0 = X.dtype.names
    header = replace(header0, 'NH4', 'NH4+')

    print('---------------------------------------')
    print('Making Fig_comb_hist.pdf...')
    print('---------------------------------------')

    count1 = 0

    r1 = list(range(1, len(header)))
    r2 = list(range(len(header) - 1))
    fig = plt.figure(figsize=(10, 20))
    for index, index1 in zip(r1, r2):
        ax = fig.add_subplot(6, 2, count1 + 1)
        sp = t[index]
        sp_v = np.array([sp])
        sp_v2 = sp_v[~np.isnan(sp_v)]
        m = np.median(sp_v2)
        n, bins, patches = ax.hist(np.log(sp_v2 / m), bins='fd', range=[-3, 3], ls='solid', facecolor='darkred',
                                   alpha=1., histtype='bar', ec='black', label='$N_\mathrm{H_2O,\ tot}$')
        maxfreq = n.max()
        plt.ylim(0, maxfreq + 0.4 * maxfreq)
        try:
            q_min, q_avg, q_max = np.percentile(sp_v2, [1., 50., 99.])
            q_min, q_avg, q_max = q_min / 1e17, q_avg / 1e17, q_max / 1e17
        except:
            print('ENIIGMA Warning: Histogram for' + ' ' + header[index] + ' ' + 'not created.')
            continue

        new_header = header[index].replace('2', '$_2$').replace('3', '$_3$').replace('4', '$_4$').replace('+',
                                                                                                          '$^+$')  # .replace('COO','$\mathrm{COO}^-$')
        plt.title('Molecule:' + ' ' + new_header + '\n' + '3$\sigma$ C.I.' + ':' + ' ' + str(
            '{:.2f}'.format(q_avg)) + '$_{' + str('{:.2f}'.format(q_min)) + '}^{' + str(
            '{:.2f}'.format(q_max)) + '}$' + ' ' + 'x' + '$10^{17}$' + '  ' + '$\mathrm{cm^{-2}}$',
                  position=(0.5, 0.75), fontsize=13, color='blue')
        if count1 % 2 != 0:
            plt.setp(ax.get_yticklabels(), visible=True)
        elif count1 % 2 == 0:
            plt.ylabel(r'$\ \mathrm{Number \; of \; combinations}$', fontsize=14)
            plt.setp(ax.get_yticklabels(), visible=True)
        if count1 == len(r1) - 2 or count1 == len(r1) - 1:
            plt.xlabel(r'$\ \mathrm{log (X/X_{median})}$', fontsize=14)

        ax.tick_params(which='both', direction="out", labelsize=14)
        plt.savefig('Fig_comb_hist.pdf', bbox_inches='tight', dpi=300)

        count1 = count1 + 1

# red_chi2_vals = chi_values()
# merge_components_cd()
# hist_plot()