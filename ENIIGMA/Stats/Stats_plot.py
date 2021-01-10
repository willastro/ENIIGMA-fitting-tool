import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.lines import Line2D
import scipy.interpolate
from itertools import combinations
import os
import pandas as pd
from pandas import DataFrame
import glob
import sh


def min_max(xd, yd, emin, emax, pathdir):
    """
	Plot the minimum and maximum confidence intervals.

	Parameters
	-------------

	xd : 'array'

	yd : 'array'

	emin : 'array'

	emax : 'str'

	"""
    print('---------------------------------------------------')
    print('Making upper and lower confidence intervals...')
    print('---------------------------------------------------')

    t0 = pd.read_csv(pathdir + 'Confidence_limits_2nd.dat', sep='\s+', header=None)
    Cmin0 = pd.read_csv(pathdir + 'Confidence_limits_2nd.dat', sep='\s+', header=None,
                        usecols=list(range(t0.shape[1] - 1)), nrows=1)
    Cmin = Cmin0.T.values.tolist()

    t1 = pd.read_csv(pathdir + 'output_file.txt', sep='\s+', header=None)
    Ysp = pd.read_csv(pathdir + 'output_file.txt', sep='\s+', header=None, usecols=list(range(1, t1.shape[1], 2)))

    fileqmin = pathdir + 'q_min.txt'
    fileqmax = pathdir + 'q_max.txt'

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

    plt.plot(xd, 0 * f0, color='black', ls=':')
    plt.plot(xd, f0, color='limegreen', linewidth=2, zorder=5)
    plt.plot(xd, yd, color='black', label='CRBR 2422.8-3423', zorder=4)
    plt.plot(xd, fmin, color='red', linestyle='--')
    plt.plot(xd, fmax, color='blue', linestyle='--')
    plt.fill_between(xd, emin, emax, color='gray', label='_nolegend_')
    plt.ylabel(r'Optical Depth$\mathrm{(\tau_{\lambda})}$', fontsize=10)
    plt.xlabel(r'$\lambda\ \mathrm{[\mu m]}$', fontsize=10)
    plt.xlim(min(xd), max(xd))
    custom_lines = [Line2D([0], [0], color='black', lw=1.5), Line2D([0], [0], color='limegreen', lw=1.5),
                    Line2D([0], [0], color='red', lw=1.5, ls='--'),
                    Line2D([0], [0], color='blue', lw=1.5, ls='--')]  # Line2D([0], [0], color='grey', lw=1.5),
    plt.legend(custom_lines, ['Observation', 'Best fit', 'Lower confidence limit', 'Upper confidence limit'],
               loc='lower left', frameon=False, fontsize='small')
    plt.minorticks_on()
    plt.tick_params(which='major', length=5, width=1, direction='in', labelsize=14)
    plt.tick_params(which='minor', length=3, width=1, direction='in', labelsize=14)
    maxtau = max(yd) + 0.4 * max(yd)
    plt.ylim(maxtau, -0.05)


def deconv_best(xd, yd, emin, emax, pathdir):
    """
	Plot the best combination from GA.

	Parameters
	-------------

	xd : 'array'

	yd : 'array'

	emin : 'array'

	emax : 'str'

	"""

    print('---------------------------')
    print('Making deconvolution plot...')
    print('---------------------------')

    t0 = pd.read_csv(pathdir + 'Confidence_limits_2nd.dat', sep='\s+', header=None)
    Cmin0 = pd.read_csv(pathdir + 'Confidence_limits_2nd.dat', sep='\s+', header=None,
                        usecols=list(range(t0.shape[1] - 1)), nrows=1)
    Cmin = Cmin0.T.values.tolist()

    t1 = pd.read_csv(pathdir + 'output_file.txt', sep='\s+', header=None)
    Ysp = pd.read_csv(pathdir + 'output_file.txt', sep='\s+', header=None, usecols=list(range(1, t1.shape[1], 2)))

    fileqmin, fileqmax = pathdir + 'q_min.txt', pathdir + 'q_max.txt'

    tmin, tmax = pd.read_csv(fileqmin, sep='\s+', header=None), pd.read_csv(fileqmax, sep='\s+', header=None)

    pathb = pathdir + 'Best_comb.csv'
    df = pd.read_csv(pathb, sep=',', header=1)
    n_genes = df.shape[1] - 3  # number of genes

    spnames = pd.read_csv(pathb, delimiter=",", low_memory=True, usecols=['name'], nrows=n_genes)
    list_names = spnames.T.values.tolist()[0]

    crange = list(range(t0.shape[1] - 1))
    ysprange = list(range(1, t1.shape[1], 2))

    f0 = 0.
    fmin = 0.
    fmax = 0.
    count = 1
    for i, j in zip(crange, ysprange):
        # print i,j
        name = list_names[i]
        f0 += Cmin[i] * Ysp[j]
        fmin += tmin[0][i] * Ysp[j]
        # print tmin[0][i]
        fmax += tmax[0][i] * Ysp[j]

        f0c = Cmin[i] * Ysp[j]
        fminc = tmin[0][i] * Ysp[j]
        fmaxc = tmax[0][i] * Ysp[j]

        Data1 = {str(name): fminc}
        df1 = DataFrame(Data1, columns=[str(name)])
        df1.to_csv('Cmin_' + str(count) + '_' + str(name) + '.csv', index=False)

        Data2 = {str(name): fmaxc}
        df2 = DataFrame(Data2, columns=[str(name)])
        df2.to_csv('Cmax_' + str(count) + '_' + str(name) + '.csv', index=False)

        Data3 = {str(name): f0c}
        df3 = DataFrame(Data3, columns=[str(name)])
        df3.to_csv('C0_' + str(count) + '_' + str(name) + '.csv', index=False)

        count = count + 1

    # Min#######################################################################
    Wav = {'Wavelength': t1[0]}
    df_wav = DataFrame(Wav, columns=['Wavelength'])
    df_wav.to_csv('Cmin_0_wav.csv', index=False)

    Dmin = {'all': fmin}
    df_wav = DataFrame(Dmin, columns=['all'])
    df_wav.to_csv('Cmin_' + str(n_genes + 1) + '_all.csv', index=False)

    all_filenames = [i for i in sorted(glob.glob(pathdir + 'Cmin_*'))]
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames], axis=1)
    combined_csv.to_csv("Components_min.csv", index=False, encoding='utf-8-sig')
    sh.rm(sh.glob(pathdir + 'Cmin_*'))
    # Min#######################################################################

    # Min#######################################################################
    Wav = {'Wavelength': t1[0]}
    df_wav = DataFrame(Wav, columns=['Wavelength'])
    df_wav.to_csv('Cmax_0_wav.csv', index=False)

    Dmax = {'all': fmax}
    df_wav = DataFrame(Dmax, columns=['all'])
    df_wav.to_csv('Cmax_' + str(n_genes + 1) + '_all.csv', index=False)

    all_filenames = [i for i in sorted(glob.glob(pathdir + 'Cmax_*'))]
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames], axis=1)
    combined_csv.to_csv("Components_max.csv", index=False, encoding='utf-8-sig')
    sh.rm(sh.glob(pathdir + 'Cmax_*'))
    # Min#######################################################################

    # Best#######################################################################
    Wav = {'Wavelength': t1[0]}
    df_wav = DataFrame(Wav, columns=['Wavelength'])
    df_wav.to_csv('C0_0_wav.csv', index=False)

    D0 = {'all': f0}
    df_wav = DataFrame(D0, columns=['all'])
    df_wav.to_csv('C0_' + str(n_genes + 1) + '_all.csv', index=False)

    all_filenames = [i for i in sorted(glob.glob(pathdir + 'C0_*'))]
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames], axis=1)
    combined_csv.to_csv("Components.csv", index=False, encoding='utf-8-sig')
    sh.rm(sh.glob(pathdir + 'C0_*'))
    # Min#######################################################################

    print('-----------------------------')
    print('Performing deconvolutions...')
    print('I am here!!')
    print('-----------------------------')

    from ENIIGMA.Stats import deconvolution as dcv
    filemin = pathdir + 'Components_min.csv'
    dcv.deconv_all(filemin)
    from ENIIGMA.Stats.Merge_colden import mergecd_min
    filename = pathdir + 'Column_density_*.csv'
    try:
        mergecd_min(filename, pathdir)
    except:
        print(' ')
        print('Merge file empty - Check if best files is present in the deconvolution routine')
        print(' ')

    from ENIIGMA.Stats import deconvolution as dcv
    filemax = pathdir + 'Components_max.csv'
    dcv.deconv_all(filemax)
    from ENIIGMA.Stats.Merge_colden import mergecd_max
    filename = pathdir + 'Column_density_*.csv'
    try:
        mergecd_max(filename, pathdir)
    except:
        print(' ')
        print('Merge file empty - Check if best files is present in the deconvolution routine')
        print(' ')

    from ENIIGMA.Stats import deconvolution as dcv
    file = pathdir + 'Components.csv'
    try:
        dcv.deconv_all(file)
    except:
        print(' ')
        print('Merge file empty - Check if best files is present in the deconvolution routine')
        print(' ')

    fig1 = plt.figure()
    frame1 = fig1.add_axes((.1, .3, .8, .6))
    # plt.plot(xsp1,0*xd)
    plt.plot(xd, yd, color='black', label='source')
    plt.plot(t1[0], f0, color='limegreen', linestyle='-', label='Model')
    cbest0 = pd.read_csv(file, sep=',', header=1)
    cbest = cbest0.T.values.tolist()

    rname = list(range(len(list_names)))
    rcomp = list(range(1, len(list_names) + 1))

    for k1, k2 in zip(rname, rcomp):
        plt.plot(cbest[0], cbest[k2], linestyle=':', label=list_names[k1])

    plt.fill_between(xd, emin, emax, color='gray')
    plt.legend(ncol=1, fontsize='small', frameon=False)
    plt.ylabel(r'Optical Depth$\mathrm{(\tau_{\lambda})}$', fontsize=10)
    plt.xlabel(r'$\lambda\ \mathrm{[\mu m]}$', fontsize=10)
    plt.xlim(min(xd), max(xd))
    maxtau = max(yd) + 0.3 * max(yd)
    plt.ylim(maxtau, -0.05)

    frame2 = fig1.add_axes((.1, .1, .8, .2))
    residual = yd - f0
    frame2.plot(xd, residual, color='black', label='Residual')
    plt.xlim(min(xd), max(xd))
    frame2.yaxis.tick_right()
    frame2.yaxis.set_label_position("right")
    min_y = min(residual) + 0.1 * (min(residual))
    max_y = max(residual) + 0.1 * (max(residual))
    plt.ylim(0.2, -0.2)
    plt.grid(b=True, which='major', linestyle=':')
    plt.grid(b=True, which='minor', linestyle=':')
    plt.ylabel(r'$\mathrm{Residual}$', fontsize=10)
    plt.xlabel(r'$\lambda\ \mathrm{[\mu m]}$', fontsize=10)
    plt.tight_layout()