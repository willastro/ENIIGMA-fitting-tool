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
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

plt.rcParams['axes.linewidth'] = 1.5


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def get_line_number2(value, matrix):
    for i, line in enumerate(matrix, 1):
        if line == value:
            return i


def to_sub(s):
    subs = {'0': '\u2080',
            '1': '\u2081',
            '2': '\u2082',
            '3': '\u2083',
            '4': '\u2084',
            '5': '\u2085',
            '6': '\u2086',
            '7': '\u2087',
            '8': '\u2088',
            '9': '\u2089'}

    return ''.join(subs.get(char, char) for char in s)


def st_plot_contour2(file1, pathdir):
    """
	Chi-square maps for 2 variables.

	Parameters
	-------------

	file1 : 'str'
		Path chi-squares from the GA optimization.

	"""
    sig1 = 2.41
    sig2 = 4.61
    sig3 = 9.21

    with open(file1, 'r') as f2:
        lines = f2.readlines()

    data = []
    data = [line.split() for line in lines]
    data2 = np.asfarray(data)
    home1 = pathdir

    pathb = pathdir + 'Best_comb.csv'

    tfil = pd.read_csv(pathb, delimiter=",", low_memory=True, usecols=['name'])  # output_file.txt
    nsp1, nsp2 = tfil['name'][0], tfil['name'][1]
    nsp1, nsp2 = os.path.splitext(os.path.basename(nsp1))[0], os.path.splitext(os.path.basename(nsp2))[0]
    nsp1, nsp2 = nsp1.replace('_', ':'), nsp2.replace('_', ':')

    labels = [nsp1, nsp2]

    ucv = data2.shape[1]  # number of columns
    w_values = data2.shape[1] - 2  # weight values

    chi = data2[:, ucv - 1]
    deltachi = chi - min(chi)
    b = find_nearest(deltachi, 50.)
    a = get_line_number2(b, deltachi)

    min0 = 0
    max0 = a

    chi = data2[:, ucv - 1]  # [min0:max0]
    deltachi = chi - min(chi)

    c = np.linspace(0, w_values, ucv - 1)
    a = list(combinations(c, 2))

    idx0 = 0
    idx1 = 1
    count1 = 0
    count2 = 0

    w1_min, w1_max, w2_min, w2_max = [], [], [], []
    fig = plt.figure()

    pos = np.array([1])
    ar1 = list(range(len(a)))
    ar2 = list(range(len(pos)))
    for i, j in zip(ar1, ar2):
        # print a[i][idx0], a[i][idx1]
        s0 = int(a[i][idx0])
        s1 = int(a[i][idx1])
        p1 = data2[:, s0][min0:max0]
        p2 = data2[:, s1][min0:max0]
        p3 = deltachi[min0:max0]
        cc = pos[j]
        ax = fig.add_subplot(1, 1, cc)
        # print count1+1
        x, y, z = p1, p2, p3

        # xll = x.min();  xul = x.max();  yll = y.min();  yul = y.max()
        xmin, xmax, ymin, ymax = 0.0, 1., 0., 1.
        xll = xmin;
        xul = xmax;
        yll = ymin;
        yul = ymax
        xmin0 = np.where(z == np.min(z))
        xmin0 = x[xmin0[0][0]]
        ymin0 = np.where(z == np.min(z))
        ymin0 = y[ymin0[0][0]]
        interp = scipy.interpolate.Rbf(x, y, z, smooth=3, function='linear')
        yi, xi = np.mgrid[yll:yul:100j, xll:xul:100j]
        zi = interp(xi, yi)
        contours1 = plt.contour(xi, yi, zi, [sig1], colors='olive', linestyle=':.', linewidths=2)
        contours2 = plt.contour(xi, yi, zi, [sig2], colors='gold', linestyle=':', linewidths=2)
        contours3 = plt.contour(xi, yi, zi, [sig3], colors='red', linestyle=':.', linewidths=2)
        pv = contours2.collections[0].get_paths()[0]
        vv = pv.vertices
        xv = vv[:, 0]
        yv = vv[:, 1]
        # print('-------------------------------')
        # print('Index	Specie    C_I_Min       C_I_Max')
        # print('{:1f} {:7s} {:11.5f} {:11.5f}'.format(s0, labels[s0], min(xv), max(xv)))
        # print('{:1f} {:7s} {:11.5f} {:11.5f}'.format(s1, labels[s1], min(yv), max(yv)))

        if labels[s0] == nsp1:
            w1_min.append(min(xv))
            w1_max.append(max(xv))

        if labels[s1] == nsp2:
            w2_min.append(min(yv))
            w2_max.append(max(yv))

        from matplotlib import ticker, colors
        # r1 = np.floor((min(z)))
        # r2 = np.ceil(np.log10(z.max())+1)
        # print r1, r2
        # lev_exp = np.arange(r1, r2)
        # levs = np.power(100, lev_exp)
        # cs = ax.contourf(xi, yi, zi, levs, norm=colors.LogNorm())
        # print min(z), max(z)
        levs = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        cs = ax.contourf(xi, yi, zi, levs, cmap=plt.cm.gray, vmin=min(z), vmax=max(z), zorder=0)

        # Version with w1 and specie names separated

        plt.setp(ax.get_xticklabels(), visible=True)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.tick_params(bottom='on')
        ax.set_xlabel(r'$\mathrm{' + labels[s0].replace('H2O:15K', 'H_2O \; (15K)') + '}$', fontsize=16)
        ax.set_ylabel(
            r'$\mathrm{' + labels[s1].replace('CO:CH3OH:CH3CH2OH:15K', 'CO:CH_3OH:CH_3CH_2OH \; (15K)').replace(
                'H2O:75K', 'H_2O \; (75K)').replace('H2O:40K', 'H_2O \; (40K)') + '}$', fontsize=16)
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        ax.minorticks_on()
        ax.set_yticks((0.2, 0.4, 0.6, 0.8, 1.))
        ax.tick_params(which='major', length=10, width=1, direction='in', labelsize=16, zorder=10)
        ax.tick_params(which='minor', length=5, width=1, direction='in', labelsize=16, zorder=10)

        count1 = count1 + 1

    w1n_min, w1n_max, w2n_min, w2n_max = np.mean(w1_min), np.mean(w1_max), np.mean(w2_min), np.mean(w2_max)
    np.savetxt('q_min.txt', np.transpose([w1n_min, w2n_min]))
    np.savetxt('q_max.txt', np.transpose([w1n_max, w2n_max]))
    textstr = '\n'.join((r'w$_1$:' + nsp1.replace('CO:CH3OH:CH3OCH3:15.0K', 'CO:CH$_3$OH:CH$_3$OCH$_3$ 15K').replace(
        'H2O:CH3OH:1:1:c', 'H2O:CH3OH:1:1-CR').replace('CO:NH3:10K', 'CO:NH$_3$ 10K').replace('CO:CH3OH:10K',
                                                                                              'CO:CH$_3$OH 10K').replace(
        'CO:CH3CH2OH:30.0K', 'CO:CH$_3$CH$_2$OH 30K').replace('HCOOH:30.0K', 'HCOOH 30K').replace('HCOOH:15.0K',
                                                                                                  'HCOOH 15K').replace(
        'H2O:NH3:CO:1.0:0.6:0.4:c', 'H$_2$O:NH$_3$:CO (1.0:0.6:0.4)-CR').replace('H2CO', 'H$_2$CO').replace(
        'NH4NO3:200K', 'NH$_4^+$ (UV+heating)').replace('CO:CH3CHO:30.0K', 'CO:CH$_3$CHO 30K').replace('H2O:40K',
                                                                                                       'H$_2$O 40K').replace(
        'NH3:CH3OH:50:10K', 'NH$_3$:CH$_3$OH 10K').replace('CH3CN', 'CH$_3$CN').replace('HNCO:NH3',
                                                                                        'NH$_4^+$ (heating)'),
                         r'w$_2$:' + nsp2.replace('CO:CH3OH:CH3OCH3:15.0K', 'CO:CH$_3$OH:CH$_3$OCH$_3$ 15K').replace(
                             'H2O:CH3OH:1:1:c', 'H2O:CH3OH:1:1-CR').replace('HCOOH:15.0K', 'HCOOH 15K').replace(
                             'CO:NH3:10K', 'CO:NH$_3$ 10K').replace('CO:CH3OH:10K', 'CO:CH$_3$OH 10K').replace(
                             'CO:CH3CH2OH:30.0K', 'CO:CH$_3$CH$_2$OH 30K').replace('HCOOH:30.0K', 'HCOOH 30K').replace(
                             'H2O:NH3:CO:1.0:0.6:0.4:c', 'H$_2$O:NH$_3$:CO (1.0:0.6:0.4)-CR').replace('H2CO',
                                                                                                      'H$_2$CO').replace(
                             'NH4NO3:200K', 'NH$_4^+$ (UV+heating)').replace('CO:CH3CHO:30.0K',
                                                                             'CO:CH$_3$CHO 30K').replace('H2O:40K',
                                                                                                         'H$_2$O 40K').replace(
                             'NH3:CH3OH:50:10K', 'NH$_3$:CH$_3$OH 10K').replace('CH3CN', 'CH$_3$CN').replace('HNCO:NH3',
                                                                                                             'NH$_4^+$ (heating)')))
    props = dict(boxstyle='round', facecolor='white')

    cb_ax = fig.add_axes([0.02, -0.04, 0.98, 0.02])
    cbar = fig.colorbar(cs, cax=cb_ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.set_xlabel(r'$\mathrm{\Delta(\nu,\alpha)=\chi^2 - \chi^2_{min}}$', fontsize=16)
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.subplots_adjust(wspace=0, hspace=0)


def st_plot_contour3(file1, pathdir):
    """
	Chi-square maps for 3 variables.

	Parameters
	-------------

	file1 : 'str'
		Path chi-squares from the GA optimization.

	"""
    sig1 = 3.66
    sig2 = 6.25
    sig3 = 11.34

    with open(file1, 'r') as f2:
        lines = f2.readlines()

    data = []
    data = [line.split() for line in lines]
    data2 = np.asfarray(data)
    home1 = pathdir

    pathb = pathdir + 'Best_comb.csv'

    tfil = pd.read_csv(pathb, delimiter=",", low_memory=True, usecols=['name'])  # output_file.txt
    nsp1, nsp2, nsp3 = tfil['name'][0], tfil['name'][1], tfil['name'][2]
    nsp1, nsp2, nsp3 = os.path.splitext(os.path.basename(nsp1))[0], os.path.splitext(os.path.basename(nsp2))[0], \
                       os.path.splitext(os.path.basename(nsp3))[0]
    nsp1, nsp2, nsp3 = nsp1.replace('_', ':'), nsp2.replace('_', ':'), nsp3.replace('_', ':')
    labels = [nsp1, nsp2, nsp3]

    ucv = data2.shape[1]  # number of columns
    w_values = data2.shape[1] - 2  # weight values
    chi = data2[:, ucv - 1]
    deltachi = chi - min(chi)
    b = find_nearest(deltachi, 20.)
    a = get_line_number2(b, deltachi)

    min0 = 0
    max0 = a

    chi = data2[:, ucv - 1]  # [min0:max0]
    deltachi = chi - min(chi)

    c = np.linspace(0, w_values, ucv - 1)
    a = list(combinations(c, 2))

    idx0 = 0
    idx1 = 1
    count1 = 0
    count2 = 0

    w1_min, w1_max, w2_min, w2_max, w3_min, w3_max = [], [], [], [], [], []
    fig = plt.figure()

    pos = np.array([1, 3, 4])
    ar1 = list(range(len(a)))
    ar2 = list(range(len(pos)))
    for i, j in zip(ar1, ar2):
        # print a[i][idx0], a[i][idx1]
        s0 = int(a[i][idx0])
        s1 = int(a[i][idx1])
        p1 = data2[:, s0][min0:max0]
        p2 = data2[:, s1][min0:max0]
        p3 = deltachi[min0:max0]
        cc = pos[j]

        ax = fig.add_subplot(2, 2, cc)  # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9.75, 3))
        # print count1+1
        x, y, z = p1, p2, p3

        xll = x.min();
        xul = x.max();
        yll = y.min();
        yul = y.max()
        xmin0 = np.where(z == np.min(z))
        xmin0 = x[xmin0[0][0]]
        ymin0 = np.where(z == np.min(z))
        ymin0 = y[ymin0[0][0]]
        interp = scipy.interpolate.Rbf(x, y, z, smooth=3, function='linear')
        yi, xi = np.mgrid[yll:yul:100j, xll:xul:100j]
        # min_fitness_idx = numpy.where(fitness == numpy.min(fitness))
        # min_xi = np.where(np.min(xi) == sig3)
        zi = interp(xi, yi)
        contours1 = plt.contour(xi, yi, zi, [sig1], colors='olive', linewidths=2)
        contours2 = plt.contour(xi, yi, zi, [sig2], colors='gold', linewidths=2)
        contours3 = plt.contour(xi, yi, zi, [sig3], colors='red', linewidths=2)
        pv = contours2.collections[0].get_paths()[0]
        vv = pv.vertices
        xv = vv[:, 0]
        yv = vv[:, 1]
        # print labels[s0], min(xv), max(xv), labels[s1], min(yv), max(yv)
        # print('-------------------------------')
        # print('Index	Specie    C_I_Min       C_I_Max')
        # print('{:1f} {:7s} {:11.5f} {:11.5f}'.format(s0, labels[s0], min(xv), max(xv)))
        # print('{:1f} {:7s} {:11.5f} {:11.5f}'.format(s1, labels[s1], min(yv), max(yv)))

        # print 'labels[s0] == nsp8', labels[s1], nsp8

        if labels[s0] == nsp1:
            # print labels[s0] == nsp1
            w1_min.append(min(xv))
            w1_max.append(max(xv))
        elif labels[s0] == nsp2:
            # print labels[s0] == nsp2
            w2_min.append(min(xv))
            w2_max.append(max(xv))
        if labels[s1] == nsp3:
            # print labels[s0] == nsp3
            w3_min.append(min(yv))
            w3_max.append(max(yv))

        cs = ax.contourf(xi, yi, zi, cmap=plt.cm.gray, vmin=min(z), vmax=max(z), zorder=0)
        # plt.xticks(fontsize = 18)
        # plt.yticks(fontsize = 18)

        # Version with w1 and specie names separated
        if cc == 1:
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.tick_params(bottom='on')
            # ax.set_xlabel(r'$\mathrm{'+labels[s0]+'}$', fontsize=18)
            ax.set_ylabel('w' + '$_' + str(s1 + 1) + '$', fontsize=45)
            ax.yaxis.set_ticks_position('both')
            ax.xaxis.set_ticks_position('both')
            ax.tick_params(which='major', length=10, width=1, direction='in', labelsize=16, zorder=10)
            ax.tick_params(which='minor', length=5, width=1, direction='in', labelsize=16, zorder=10)
        else:
            # ax.set_xlabel(r'$\mathrm{'+'w'+'_'+str(s0+1)+'('+labels[s0]+'}$'+')', fontsize=45)
            ax.set_xlabel('w' + '$_' + str(s0 + 1) + '$', fontsize=16)
            ax.yaxis.set_ticks_position('both')
            ax.xaxis.set_ticks_position('both')
            ax.tick_params(which='major', length=10, width=1, direction='in', labelsize=16, zorder=10, rotation=45)
            ax.tick_params(which='minor', length=5, width=1, direction='in', labelsize=16, zorder=10)

        if cc == 4:
            plt.gca().axes.get_yaxis().set_visible(True)
            plt.tick_params(left='on')
            ax.yaxis.set_ticks_position('both')
            ax.xaxis.set_ticks_position('both')
            ax.tick_params(which='major', length=10, width=1, direction='in', labelsize=16, zorder=10)
            ax.tick_params(which='minor', length=5, width=1, direction='in', labelsize=16, zorder=10)
            ax.set_yticklabels([])
            # plt.axis('off')
            # plt.setp(ax.get_yticklabels(), visible=True)
            # plt.tick_params(left='off')
            # ax.set_xlabel(r'$\mathrm{'+labels[s0]+'}$', fontsize=18)
            ax.set_ylabel(r'$\mathrm{' + 'w' + '_' + str(s1 + 1) + '(' + labels[s1] + '}$' + ')', fontsize=1)
        else:
            ax.yaxis.set_ticks_position('both')
            ax.xaxis.set_ticks_position('both')
            ax.set_ylabel('w' + '$_' + str(s1 + 1) + '$', fontsize=16)
            ax.tick_params(which='major', length=10, width=1, direction='in', labelsize=16, zorder=10)
            ax.tick_params(which='minor', length=5, width=1, direction='in', labelsize=16, zorder=10)

        count1 = count1 + 1

    w1n_min, w1n_max, w2n_min, w2n_max, w3n_min, w3n_max = np.mean(w1_min), np.mean(w1_max), np.mean(w2_min), np.mean(
        w2_max), np.mean(w3_min), np.mean(w3_max)
    np.savetxt('q_min.txt', np.transpose([w1n_min, w2n_min, w3n_min]))
    np.savetxt('q_max.txt', np.transpose([w1n_max, w2n_max, w3n_max]))
    textstr = '\n'.join((r'w$_1$:' + nsp1.replace('CO:CH3OH:CH3OCH3:15.0K', 'CO:CH$_3$OH:CH$_3$OCH$_3$ 15K').replace(
        'H2O:CH3OH:1:1:c', 'H2O:CH3OH:1:1-CR').replace('CO:NH3:10K', 'CO:NH$_3$ 10K').replace('CO:CH3OH:10K',
                                                                                              'CO:CH$_3$OH 10K').replace(
        'CO:CH3CH2OH:30.0K', 'CO:CH$_3$CH$_2$OH 30K').replace('HCOOH:30.0K', 'HCOOH 30K').replace('HCOOH:15.0K',
                                                                                                  'HCOOH 15K').replace(
        'H2O:NH3:CO:1.0:0.6:0.4:c', 'H$_2$O:NH$_3$:CO (1.0:0.6:0.4)-CR').replace('H2CO', 'H$_2$CO').replace(
        'NH4NO3:200K', 'NH$_4^+$ (UV+heating)').replace('CO:CH3CHO:30.0K', 'CO:CH$_3$CHO 30K').replace('H2O:40K',
                                                                                                       'H$_2$O 40K').replace(
        'NH3:CH3OH:50:10K', 'NH$_3$:CH$_3$OH 10K').replace('CH3CN', 'CH$_3$CN').replace('HNCO:NH3',
                                                                                        'NH$_4^+$ (heating)'),
                         r'w$_2$:' + nsp2.replace('CO:CH3OH:CH3OCH3:15.0K', 'CO:CH$_3$OH:CH$_3$OCH$_3$ 15K').replace(
                             'H2O:CH3OH:1:1:c', 'H2O:CH3OH:1:1-CR').replace('HCOOH:15.0K', 'HCOOH 15K').replace(
                             'CO:NH3:10K', 'CO:NH$_3$ 10K').replace('CO:CH3OH:10K', 'CO:CH$_3$OH 10K').replace(
                             'CO:CH3CH2OH:30.0K', 'CO:CH$_3$CH$_2$OH 30K').replace('HCOOH:30.0K', 'HCOOH 30K').replace(
                             'H2O:NH3:CO:1.0:0.6:0.4:c', 'H$_2$O:NH$_3$:CO (1.0:0.6:0.4)-CR').replace('H2CO',
                                                                                                      'H$_2$CO').replace(
                             'NH4NO3:200K', 'NH$_4^+$ (UV+heating)').replace('CO:CH3CHO:30.0K',
                                                                             'CO:CH$_3$CHO 30K').replace('H2O:40K',
                                                                                                         'H$_2$O 40K').replace(
                             'NH3:CH3OH:50:10K', 'NH$_3$:CH$_3$OH 10K').replace('CH3CN', 'CH$_3$CN').replace('HNCO:NH3',
                                                                                                             'NH$_4^+$ (heating)'),
                         r'w$_3$:' + nsp3.replace('CO:CH3OH:CH3OCH3:15.0K', 'CO:CH$_3$OH:CH$_3$OCH$_3$ 15K').replace(
                             'H2O:CH3OH:1:1:c', 'H2O:CH3OH:1:1-CR').replace('HCOOH:15.0K', 'HCOOH 15K').replace(
                             'CO:NH3:10K', 'CO:NH$_3$ 10K').replace('CO:CH3OH:10K', 'CO:CH$_3$OH 10K').replace(
                             'CO:CH3CH2OH:30.0K', 'CO:CH$_3$CH$_2$OH 30K').replace('HCOOH:30.0K', 'HCOOH 30K').replace(
                             'H2O:NH3:CO:1.0:0.6:0.4:c', 'H$_2$O:NH$_3$:CO (1.0:0.6:0.4)-CR').replace('H2CO',
                                                                                                      'H$_2$CO').replace(
                             'NH4NO3:200K', 'NH$_4^+$ (UV+heating)').replace('CO:CH3CHO:30.0K',
                                                                             'CO:CH$_3$CHO 30K').replace('H2O:40K',
                                                                                                         'H$_2$O 40K').replace(
                             'NH3:CH3OH:50:10K', 'NH$_3$:CH$_3$OH 10K').replace('CH3CN', 'CH$_3$CN').replace(
                             'H2O:CH4:10:0.6:a', 'H$_2$O:CH$_4$:10:0.6').replace('HNCO:NH3', 'NH$_4^+$ (heating)')))
    props = dict(boxstyle='round', facecolor='white')
    ax.text(0.04, 1.97, textstr, transform=ax.transAxes, fontsize=13, verticalalignment='top', bbox=props)

    # ax2 = fig.add_subplot(1,1,1)
    # fig.colorbar(cs, orientation='vertical', fraction=0.046, pad=0.04)
    # plt.tight_layout()
    cb_ax = fig.add_axes([0.02, -0.02, 0.98, 0.01])
    # cb_ax = fig.add_axes([.85, .3, .02, .8])
    cbar = fig.colorbar(cs, cax=cb_ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.set_xlabel(r'$\mathrm{\Delta(\nu,\alpha)=\chi^2 - \chi^2_{min}}$', fontsize=16)
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.subplots_adjust(wspace=0, hspace=0)


def st_plot_contour4(file1, pathdir):
    """
	Chi-square maps for 4 variables.

	Parameters
	-------------

	file1 : 'str'
		Path chi-squares from the GA optimization.

	"""
    sig1 = 4.88
    sig2 = 7.78
    sig3 = 13.28

    with open(file1, 'r') as f2:
        lines = f2.readlines()

    data = []
    data = [line.split() for line in lines]
    data2 = np.asfarray(data)
    home1 = pathdir

    pathb = pathdir + 'Best_comb.csv'

    tfil = pd.read_csv(pathb, delimiter=",", low_memory=True, usecols=['name'])  # output_file.txt
    nsp1, nsp2, nsp3, nsp4 = tfil['name'][0], tfil['name'][1], tfil['name'][2], tfil['name'][3]
    nsp1, nsp2, nsp3, nsp4 = os.path.splitext(os.path.basename(nsp1))[0], os.path.splitext(os.path.basename(nsp2))[0], \
                             os.path.splitext(os.path.basename(nsp3))[0], os.path.splitext(os.path.basename(nsp4))[0]
    nsp1, nsp2, nsp3, nsp4 = nsp1.replace('_', ':'), nsp2.replace('_', ':'), nsp3.replace('_', ':'), nsp4.replace('_',
                                                                                                                  ':')

    labels = [nsp1, nsp2, nsp3, nsp4]

    ucv = data2.shape[1]  # number of columns
    w_values = data2.shape[1] - 2  # weight values

    chi = data2[:, ucv - 1]
    deltachi = chi - min(chi)
    b = find_nearest(deltachi, 50.)
    a = get_line_number2(b, deltachi)

    min0 = 0
    max0 = a

    chi = data2[:, ucv - 1]  # [min0:max0]
    deltachi = chi - min(chi)

    c = np.linspace(0, w_values, ucv - 1)
    a = list(combinations(c, 2))

    idx0 = 0
    idx1 = 1
    count1 = 0
    count2 = 0

    w1_min, w1_max, w2_min, w2_max, w3_min, w3_max, w4_min, w4_max = [], [], [], [], [], [], [], []
    fig = plt.figure(figsize=(30, 33.7))

    pos = np.array([1, 4, 5, 7, 8, 9])
    ar1 = list(range(len(a)))
    ar2 = list(range(len(pos)))
    for i, j in zip(ar1, ar2):
        # print a[i][idx0], a[i][idx1]
        s0 = int(a[i][idx0])
        s1 = int(a[i][idx1])
        p1 = data2[:, s0][min0:max0]
        p2 = data2[:, s1][min0:max0]
        p3 = deltachi[min0:max0]
        cc = pos[j]
        ax = fig.add_subplot(3, 3, cc)  # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9.75, 3))
        # print count1+1
        x, y, z = p1, p2, p3

        xll = x.min();
        xul = x.max();
        yll = y.min();
        yul = y.max()
        xmin0 = np.where(z == np.min(z))
        xmin0 = x[xmin0[0][0]]
        ymin0 = np.where(z == np.min(z))
        ymin0 = y[ymin0[0][0]]
        interp = scipy.interpolate.Rbf(x, y, z, smooth=3., function='linear')
        yi, xi = np.mgrid[yll:yul:100j, xll:xul:100j]
        # min_fitness_idx = numpy.where(fitness == numpy.min(fitness))
        # min_xi = np.where(np.min(xi) == sig3)
        zi = interp(xi, yi)
        contours1 = plt.contour(xi, yi, zi, [sig1], colors='olive', linestyle=':.', linewidths=4)
        contours2 = plt.contour(xi, yi, zi, [sig2], colors='gold', linestyle=':', linewidths=4)
        contours3 = plt.contour(xi, yi, zi, [sig3], colors='red', linestyle=':.', linewidths=4)
        pv = contours2.collections[0].get_paths()[0]
        vv = pv.vertices
        xv = vv[:, 0]
        yv = vv[:, 1]
        # print labels[s0], min(xv), max(xv), labels[s1], min(yv), max(yv)
        # print('-------------------------------')
        # print('Index	Specie    C_I_Min       C_I_Max')
        # print('{:1f} {:7s} {:11.5f} {:11.5f}'.format(s0, labels[s0], min(xv), max(xv)))
        # print('{:1f} {:7s} {:11.5f} {:11.5f}'.format(s1, labels[s1], min(yv), max(yv)))

        # print 'labels[s0] == nsp8', labels[s1], nsp8

        if labels[s0] == nsp1:
            # print labels[s0] == nsp1
            w1_min.append(min(xv))
            w1_max.append(max(xv))
        elif labels[s0] == nsp2:
            # print labels[s0] == nsp2
            w2_min.append(min(xv))
            w2_max.append(max(xv))
        elif labels[s0] == nsp3:
            # print labels[s0] == nsp3
            w3_min.append(min(xv))
            w3_max.append(max(xv))

        if labels[s1] == nsp4:
            # print labels[s0] == nsp4
            w4_min.append(min(yv))
            w4_max.append(max(yv))

        cs = ax.contourf(xi, yi, zi, cmap=plt.cm.gray, vmin=min(z), vmax=max(z), zorder=0)
        # plt.xticks(fontsize = 18)
        # plt.yticks(fontsize = 18)

        # Version with w1 and specie names separated
        if cc == 1 or cc == 4:
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.tick_params(bottom='on')
            # ax.set_xlabel(r'$\mathrm{'+labels[s0]+'}$', fontsize=18)
            ax.set_ylabel('w' + '$_' + str(s1 + 1) + '$', fontsize=45)
            ax.yaxis.set_ticks_position('both')
            ax.xaxis.set_ticks_position('both')
            ax.tick_params(which='major', length=10, width=1, direction='in', labelsize=45, zorder=10)
            ax.tick_params(which='minor', length=5, width=1, direction='in', labelsize=45, zorder=10)
        else:
            # ax.set_xlabel(r'$\mathrm{'+'w'+'_'+str(s0+1)+'('+labels[s0]+'}$'+')', fontsize=45)
            ax.set_xlabel('w' + '$_' + str(s0 + 1) + '$', fontsize=45)
            ax.yaxis.set_ticks_position('both')
            ax.xaxis.set_ticks_position('both')
            ax.tick_params(which='major', length=10, width=1, direction='in', labelsize=45, zorder=10, rotation=45)
            ax.tick_params(which='minor', length=5, width=1, direction='in', labelsize=45, zorder=10)

        if cc == 5 or cc == 8 or cc == 9:
            plt.gca().axes.get_yaxis().set_visible(True)
            plt.tick_params(left='on')
            ax.yaxis.set_ticks_position('both')
            ax.xaxis.set_ticks_position('both')
            ax.tick_params(which='major', length=10, width=1, direction='in', labelsize=45, zorder=10)
            ax.tick_params(which='minor', length=5, width=1, direction='in', labelsize=45, zorder=10)
            ax.set_yticklabels([])
            # plt.axis('off')
            # plt.setp(ax.get_yticklabels(), visible=True)
            # plt.tick_params(left='off')
            # ax.set_xlabel(r'$\mathrm{'+labels[s0]+'}$', fontsize=18)
            ax.set_ylabel(r'$\mathrm{' + 'w' + '_' + str(s1 + 1) + '(' + labels[s1] + '}$' + ')', fontsize=1)
        else:
            ax.yaxis.set_ticks_position('both')
            ax.xaxis.set_ticks_position('both')
            ax.set_ylabel('w' + '$_' + str(s1 + 1) + '$', fontsize=45)
            ax.tick_params(which='major', length=10, width=1, direction='in', labelsize=45, zorder=10)
            ax.tick_params(which='minor', length=5, width=1, direction='in', labelsize=45, zorder=10)

        count1 = count1 + 1

    w1n_min, w1n_max, w2n_min, w2n_max, w3n_min, w3n_max, w4n_min, w4n_max = np.mean(w1_min), np.mean(w1_max), np.mean(
        w2_min), np.mean(w2_max), np.mean(w3_min), np.mean(w3_max), np.mean(w4_min), np.mean(w4_max)
    np.savetxt('q_min.txt', np.transpose([w1n_min, w2n_min, w3n_min, w4n_min]))
    np.savetxt('q_max.txt', np.transpose([w1n_max, w2n_max, w3n_max, w4n_max]))
    textstr = '\n'.join((r'w$_1$:' + nsp1.replace('CO:CH3OH:CH3OCH3:15.0K', 'CO:CH$_3$OH:CH$_3$OCH$_3$ 15K').replace(
        'H2O:CH3OH:1:1:c', 'H2O:CH3OH:1:1-CR').replace('CO:NH3:10K', 'CO:NH$_3$ 10K').replace('CO:CH3OH:10K',
                                                                                              'CO:CH$_3$OH 10K').replace(
        'CO:CH3CH2OH:30.0K', 'CO:CH$_3$CH$_2$OH 30K').replace('HCOOH:30.0K', 'HCOOH 30K').replace('HCOOH:15.0K',
                                                                                                  'HCOOH 15K').replace(
        'H2O:NH3:CO:1.0:0.6:0.4:c', 'H$_2$O:NH$_3$:CO (1.0:0.6:0.4)-CR').replace('H2CO', 'H$_2$CO').replace(
        'NH4NO3:200K', 'NH$_4^+$ (UV+heating)').replace('CO:CH3CHO:30.0K', 'CO:CH$_3$CHO 30K').replace('H2O:40K',
                                                                                                       'H$_2$O 40K').replace(
        'NH3:CH3OH:50:10K', 'NH$_3$:CH$_3$OH 10K').replace('CH3CN', 'CH$_3$CN').replace('HNCO:NH3',
                                                                                        'NH$_4^+$ (heating)'),
                         r'w$_2$:' + nsp2.replace('CO:CH3OH:CH3OCH3:15.0K', 'CO:CH$_3$OH:CH$_3$OCH$_3$ 15K').replace(
                             'H2O:CH3OH:1:1:c', 'H2O:CH3OH:1:1-CR').replace('HCOOH:15.0K', 'HCOOH 15K').replace(
                             'CO:NH3:10K', 'CO:NH$_3$ 10K').replace('CO:CH3OH:10K', 'CO:CH$_3$OH 10K').replace(
                             'CO:CH3CH2OH:30.0K', 'CO:CH$_3$CH$_2$OH 30K').replace('HCOOH:30.0K', 'HCOOH 30K').replace(
                             'H2O:NH3:CO:1.0:0.6:0.4:c', 'H$_2$O:NH$_3$:CO (1.0:0.6:0.4)-CR').replace('H2CO',
                                                                                                      'H$_2$CO').replace(
                             'NH4NO3:200K', 'NH$_4^+$ (UV+heating)').replace('CO:CH3CHO:30.0K',
                                                                             'CO:CH$_3$CHO 30K').replace('H2O:40K',
                                                                                                         'H$_2$O 40K').replace(
                             'NH3:CH3OH:50:10K', 'NH$_3$:CH$_3$OH 10K').replace('CH3CN', 'CH$_3$CN').replace('HNCO:NH3',
                                                                                                             'NH$_4^+$ (heating)'),
                         r'w$_3$:' + nsp3.replace('CO:CH3OH:CH3OCH3:15.0K', 'CO:CH$_3$OH:CH$_3$OCH$_3$ 15K').replace(
                             'H2O:CH3OH:1:1:c', 'H2O:CH3OH:1:1-CR').replace('HCOOH:15.0K', 'HCOOH 15K').replace(
                             'CO:NH3:10K', 'CO:NH$_3$ 10K').replace('CO:CH3OH:10K', 'CO:CH$_3$OH 10K').replace(
                             'CO:CH3CH2OH:30.0K', 'CO:CH$_3$CH$_2$OH 30K').replace('HCOOH:30.0K', 'HCOOH 30K').replace(
                             'H2O:NH3:CO:1.0:0.6:0.4:c', 'H$_2$O:NH$_3$:CO (1.0:0.6:0.4)-CR').replace('H2CO',
                                                                                                      'H$_2$CO').replace(
                             'NH4NO3:200K', 'NH$_4^+$ (UV+heating)').replace('CO:CH3CHO:30.0K',
                                                                             'CO:CH$_3$CHO 30K').replace('H2O:40K',
                                                                                                         'H$_2$O 40K').replace(
                             'NH3:CH3OH:50:10K', 'NH$_3$:CH$_3$OH 10K').replace('CH3CN', 'CH$_3$CN').replace(
                             'H2O:CH4:10:0.6:a', 'H$_2$O:CH$_4$:10:0.6').replace('HNCO:NH3', 'NH$_4^+$ (heating)'),
                         r'w$_4$:' + nsp4.replace('CO:CH3OH:CH3OCH3:15.0K', 'CO:CH$_3$OH:CH$_3$OCH$_3$ 15K').replace(
                             'H2O:CH3OH:1:1:c', 'H2O:CH3OH:1:1-CR').replace('HCOOH:15.0K', 'HCOOH 15K').replace(
                             'CO:NH3:10K', 'CO:NH$_3$ 10K').replace('CO:CH3OH:10K', 'CO:CH$_3$OH 10K').replace(
                             'CO:CH3CH2OH:30.0K', 'CO:CH$_3$CH$_2$OH 30K').replace('HCOOH:30.0K', 'HCOOH 30K').replace(
                             'H2O:NH3:CO:1.0:0.6:0.4:c', 'H$_2$O:NH$_3$:CO (1.0:0.6:0.4)-CR').replace('H2CO',
                                                                                                      'H$_2$CO').replace(
                             'NH4NO3:200K', 'NH$_4^+$ (UV+heating)').replace('CO:CH3CHO:30.0K',
                                                                             'CO:CH$_3$CHO 30K').replace(
                             'H2O:CH4:10:0.6:a', 'H$_2$O:CH$_4$:10:0.6').replace('H2O:CH4:10:0.6:a:V3',
                                                                                 'H$_2$O:CH$_4$:10:0.6').replace(
                             'H2O:40K', 'H$_2$O 40K').replace('NH3:CH3OH:50:10K', 'NH$_3$:CH$_3$OH 10K').replace(
                             'CH3CN', 'CH$_3$CN').replace('HNCO:NH3', 'NH$_4^+$ (heating)')))
    props = dict(boxstyle='round', facecolor='white')
    ax.text(-0.6, 2.80, textstr, transform=ax.transAxes, fontsize=45, verticalalignment='top', bbox=props)

    # ax2 = fig.add_subplot(1,1,1)
    # fig.colorbar(cs, orientation='vertical', fraction=0.046, pad=0.04)
    # plt.tight_layout()
    cb_ax = fig.add_axes([0.02, -0.02, 0.98, 0.01])
    # cb_ax = fig.add_axes([.85, .3, .02, .8])
    cbar = fig.colorbar(cs, cax=cb_ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=45)
    cbar.ax.set_xlabel(r'$\mathrm{\Delta(\nu,\alpha)=\chi^2 - \chi^2_{min}}$', fontsize=45)
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.subplots_adjust(wspace=0, hspace=0)


def st_plot_contour5(file1, pathdir):
    """
	Chi-square maps for 5 variables.

	Parameters
	-------------

	file1 : 'str'
		Path chi-squares from the GA optimization.

	"""
    sig1 = 6.06
    sig2 = 9.24
    sig3 = 15.09

    with open(file1, 'r') as f2:
        lines = f2.readlines()

    data = []
    data = [line.split() for line in lines]
    data2 = np.asfarray(data)
    home1 = pathdir

    pathb = pathdir + 'Best_comb.csv'

    tfil = pd.read_csv(pathb, delimiter=",", low_memory=True, usecols=['name'])  # output_file.txt
    nsp1, nsp2, nsp3, nsp4, nsp5 = tfil['name'][0], tfil['name'][1], tfil['name'][2], tfil['name'][3], tfil['name'][4]
    nsp1, nsp2, nsp3, nsp4, nsp5 = os.path.splitext(os.path.basename(nsp1))[0], \
                                   os.path.splitext(os.path.basename(nsp2))[0], \
                                   os.path.splitext(os.path.basename(nsp3))[0], \
                                   os.path.splitext(os.path.basename(nsp4))[0], \
                                   os.path.splitext(os.path.basename(nsp5))[0]
    nsp1, nsp2, nsp3, nsp4, nsp5 = nsp1.replace('_', ':'), nsp2.replace('_', ':'), nsp3.replace('_', ':'), nsp4.replace(
        '_', ':'), nsp5.replace('_', ':')

    labels = [nsp1, nsp2, nsp3, nsp4, nsp5]

    ucv = data2.shape[1]  # number of columns
    w_values = data2.shape[1] - 2  # weight values

    chi = data2[:, ucv - 1]
    deltachi = chi - min(chi)
    b = find_nearest(deltachi, 60.)
    a = get_line_number2(b, deltachi)

    min0 = 0
    max0 = a

    chi = data2[:, ucv - 1]  # [min0:max0]
    deltachi = chi - min(chi)

    c = np.linspace(0, w_values, ucv - 1)
    a = list(combinations(c, 2))

    idx0 = 0
    idx1 = 1
    count1 = 0
    count2 = 0

    w1_min, w1_max, w2_min, w2_max, w3_min, w3_max, w4_min, w4_max, w5_min, w5_max = [], [], [], [], [], [], [], [], [], []
    fig = plt.figure(figsize=(30, 33.7))

    pos = np.array([1, 5, 6, 9, 10, 11, 13, 14, 15, 16])
    ar1 = list(range(len(a)))
    ar2 = list(range(len(pos)))
    for i, j in zip(ar1, ar2):
        # print a[i][idx0], a[i][idx1]
        s0 = int(a[i][idx0])
        s1 = int(a[i][idx1])
        p1 = data2[:, s0][min0:max0]
        p2 = data2[:, s1][min0:max0]
        p3 = deltachi[min0:max0]
        cc = pos[j]
        ax = fig.add_subplot(4, 4, cc)  # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9.75, 3))
        # print count1+1
        x, y, z = p1, p2, p3

        xll = x.min();
        xul = x.max();
        yll = y.min();
        yul = y.max()
        xmin0 = np.where(z == np.min(z))
        xmin0 = x[xmin0[0][0]]
        ymin0 = np.where(z == np.min(z))
        ymin0 = y[ymin0[0][0]]
        interp = scipy.interpolate.Rbf(x, y, z, smooth=3, function='linear')
        yi, xi = np.mgrid[yll:yul:100j, xll:xul:100j]
        # min_fitness_idx = numpy.where(fitness == numpy.min(fitness))
        # min_xi = np.where(np.min(xi) == sig3)
        zi = interp(xi, yi)
        contours1 = plt.contour(xi, yi, zi, [sig1], colors='olive', linestyle=':.', linewidths=4)
        contours2 = plt.contour(xi, yi, zi, [sig2], colors='gold', linestyle=':', linewidths=4)
        contours3 = plt.contour(xi, yi, zi, [sig3], colors='red', linestyle=':.', linewidths=4)
        pv = contours2.collections[0].get_paths()[0]
        vv = pv.vertices
        xv = vv[:, 0]
        yv = vv[:, 1]
        # print labels[s0], min(xv), max(xv), labels[s1], min(yv), max(yv)
        # print('-------------------------------')
        # print('Index	Specie    C_I_Min       C_I_Max')
        # print('{:1f} {:7s} {:11.5f} {:11.5f}'.format(s0, labels[s0], min(xv), max(xv)))
        # print('{:1f} {:7s} {:11.5f} {:11.5f}'.format(s1, labels[s1], min(yv), max(yv)))

        # print 'labels[s0] == nsp8', labels[s1], nsp8

        if labels[s0] == nsp1:
            # print labels[s0] == nsp1
            w1_min.append(min(xv))
            w1_max.append(max(xv))
        elif labels[s0] == nsp2:
            w2_min.append(min(xv))
            w2_max.append(max(xv))
        elif labels[s0] == nsp3:
            print(max(xv))
            # print labels[s0] == nsp3
            w3_min.append(min(xv))
            w3_max.append(max(xv))
        elif labels[s0] == nsp4:
            # print labels[s0] == nsp4
            w4_min.append(min(xv))
            w4_max.append(max(xv))

        if labels[s1] == nsp5:
            # print labels[s0] == nsp5
            w5_min.append(min(yv))
            w5_max.append(max(yv))

        cs = ax.contourf(xi, yi, zi, cmap=plt.cm.gray, vmin=min(z), vmax=max(z), zorder=0)
        # plt.xticks(fontsize = 18)
        # plt.yticks(fontsize = 18)

        # Version with w1 and specie names separated
        if cc == 1 or cc == 5 or cc == 9:
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.tick_params(bottom='on')
            ax.set_xlabel(r'$\mathrm{' + labels[s0] + '}$', fontsize=18)
            ax.set_ylabel('w' + '$_' + str(s1 + 1) + '$', fontsize=45)
            ax.yaxis.set_ticks_position('both')
            ax.xaxis.set_ticks_position('both')
            ax.tick_params(which='major', length=10, width=1, direction='in', labelsize=45, zorder=10)
            ax.tick_params(which='minor', length=5, width=1, direction='in', labelsize=45, zorder=10)
        else:
            # ax.set_xlabel(r'$\mathrm{'+'w'+'_'+str(s0+1)+'('+labels[s0]+'}$'+')', fontsize=45)
            ax.set_xlabel('w' + '$_' + str(s0 + 1) + '$', fontsize=45)
            ax.yaxis.set_ticks_position('both')
            ax.xaxis.set_ticks_position('both')
            ax.tick_params(which='major', length=10, width=1, direction='in', labelsize=45, zorder=10, rotation=45)
            ax.tick_params(which='minor', length=5, width=1, direction='in', labelsize=45, zorder=10)

        if cc == 6 or cc == 10 or cc == 11 or cc == 14 or cc == 15 or cc == 16:
            plt.gca().axes.get_yaxis().set_visible(True)
            plt.tick_params(left='on')
            ax.yaxis.set_ticks_position('both')
            ax.xaxis.set_ticks_position('both')
            ax.tick_params(which='major', length=10, width=1, direction='in', labelsize=45, zorder=10)
            ax.tick_params(which='minor', length=5, width=1, direction='in', labelsize=45, zorder=10)
            ax.set_yticklabels([])
            # plt.axis('off')
            # plt.setp(ax.get_yticklabels(), visible=True)
            # plt.tick_params(left='off')
            # ax.set_xlabel(r'$\mathrm{'+labels[s0]+'}$', fontsize=18)
            ax.set_ylabel(r'$\mathrm{' + 'w' + '_' + str(s1 + 1) + '(' + labels[s1] + '}$' + ')', fontsize=1)
        else:
            ax.yaxis.set_ticks_position('both')
            ax.xaxis.set_ticks_position('both')
            ax.set_ylabel('w' + '$_' + str(s1 + 1) + '$', fontsize=45)
            ax.tick_params(which='major', length=10, width=1, direction='in', labelsize=45, zorder=10)
            ax.tick_params(which='minor', length=5, width=1, direction='in', labelsize=45, zorder=10)

        count1 = count1 + 1

    w1n_min, w1n_max, w2n_min, w2n_max, w3n_min, w3n_max, w4n_min, w4n_max, w5n_min, w5n_max = np.mean(w1_min), np.mean(
        w1_max), np.mean(w2_min), np.mean(w2_max), np.mean(w3_min), np.mean(w3_max), np.mean(w4_min), np.mean(
        w4_max), np.mean(w5_min), np.mean(w5_max)
    # print 'w1n_min, w1n_max', w1n_min, w1n_max
    # print w1_min, w2_min, w3_min, w4_min, w5_min
    np.savetxt('q_min.txt', np.transpose([w1n_min, w2n_min, w3n_min, w4n_min, w5n_min]))
    np.savetxt('q_max.txt', np.transpose([w1n_max, w2n_max, w3n_max, w4n_max, w5n_max]))
    textstr = '\n'.join((r'w$_1$:' + nsp1.replace('CO:CH3OH:CH3OCH3:15.0K', 'CO:CH$_3$OH:CH$_3$OCH$_3$ 15K').replace(
        'H2O:CH3OH:1:1:c', 'H2O:CH3OH:1:1-CR').replace('CO:NH3:10K', 'CO:NH$_3$ 10K').replace('CO:CH3OH:10K',
                                                                                              'CO:CH$_3$OH 10K').replace(
        'CO:CH3CH2OH:30.0K', 'CO:CH$_3$CH$_2$OH 30K').replace('HCOOH:30.0K', 'HCOOH 30K').replace('HCOOH:15.0K',
                                                                                                  'HCOOH 15K').replace(
        'H2O:NH3:CO:1.0:0.6:0.4:c', 'H$_2$O:NH$_3$:CO (1.0:0.6:0.4)-CR').replace('H2CO', 'H$_2$CO').replace(
        'NH4NO3:200K', 'NH$_4^+$ (UV+heating)').replace('CO:CH3CHO:30.0K', 'CO:CH$_3$CHO 30K').replace('H2O:40K',
                                                                                                       'H$_2$O 40K').replace(
        'NH3:CH3OH:50:10K', 'NH$_3$:CH$_3$OH 10K').replace('CH3CN', 'CH$_3$CN').replace('HNCO:NH3',
                                                                                        'NH$_4^+$ (heating)'),
                         r'w$_2$:' + nsp2.replace('CO:CH3OH:CH3OCH3:15.0K', 'CO:CH$_3$OH:CH$_3$OCH$_3$ 15K').replace(
                             'H2O:CH3OH:1:1:c', 'H2O:CH3OH:1:1-CR').replace('HCOOH:15.0K', 'HCOOH 15K').replace(
                             'CO:NH3:10K', 'CO:NH$_3$ 10K').replace('CO:CH3OH:10K', 'CO:CH$_3$OH 10K').replace(
                             'CO:CH3CH2OH:30.0K', 'CO:CH$_3$CH$_2$OH 30K').replace('HCOOH:30.0K', 'HCOOH 30K').replace(
                             'H2O:NH3:CO:1.0:0.6:0.4:c', 'H$_2$O:NH$_3$:CO (1.0:0.6:0.4)-CR').replace('H2CO',
                                                                                                      'H$_2$CO').replace(
                             'NH4NO3:200K', 'NH$_4^+$ (UV+heating)').replace('CO:CH3CHO:30.0K',
                                                                             'CO:CH$_3$CHO 30K').replace('H2O:40K',
                                                                                                         'H$_2$O 40K').replace(
                             'NH3:CH3OH:50:10K', 'NH$_3$:CH$_3$OH 10K').replace('CH3CN', 'CH$_3$CN').replace('HNCO:NH3',
                                                                                                             'NH$_4^+$ (heating)'),
                         r'w$_3$:' + nsp3.replace('CO:CH3OH:CH3OCH3:15.0K', 'CO:CH$_3$OH:CH$_3$OCH$_3$ 15K').replace(
                             'H2O:CH3OH:1:1:c', 'H2O:CH3OH:1:1-CR').replace('HCOOH:15.0K', 'HCOOH 15K').replace(
                             'CO:NH3:10K', 'CO:NH$_3$ 10K').replace('CO:CH3OH:10K', 'CO:CH$_3$OH 10K').replace(
                             'CO:CH3CH2OH:30.0K', 'CO:CH$_3$CH$_2$OH 30K').replace('HCOOH:30.0K', 'HCOOH 30K').replace(
                             'H2O:NH3:CO:1.0:0.6:0.4:c', 'H$_2$O:NH$_3$:CO (1.0:0.6:0.4)-CR').replace('H2CO',
                                                                                                      'H$_2$CO').replace(
                             'NH4NO3:200K', 'NH$_4^+$ (UV+heating)').replace('CO:CH3CHO:30.0K',
                                                                             'CO:CH$_3$CHO 30K').replace('H2O:40K',
                                                                                                         'H$_2$O 40K').replace(
                             'NH3:CH3OH:50:10K', 'NH$_3$:CH$_3$OH 10K').replace('CH3CN', 'CH$_3$CN').replace(
                             'H2O:CH4:10:0.6:a', 'H$_2$O:CH$_4$:10:0.6').replace('HNCO:NH3', 'NH$_4^+$ (heating)'),
                         r'w$_4$:' + nsp4.replace('CO:CH3OH:CH3OCH3:15.0K', 'CO:CH$_3$OH:CH$_3$OCH$_3$ 15K').replace(
                             'H2O:CH3OH:1:1:c', 'H2O:CH3OH:1:1-CR').replace('HCOOH:15.0K', 'HCOOH 15K').replace(
                             'CO:NH3:10K', 'CO:NH$_3$ 10K').replace('CO:CH3OH:10K', 'CO:CH$_3$OH 10K').replace(
                             'CO:CH3CH2OH:30.0K', 'CO:CH$_3$CH$_2$OH 30K').replace('HCOOH:30.0K', 'HCOOH 30K').replace(
                             'H2O:NH3:CO:1.0:0.6:0.4:c', 'H$_2$O:NH$_3$:CO (1.0:0.6:0.4)-CR').replace('H2CO',
                                                                                                      'H$_2$CO').replace(
                             'NH4NO3:200K', 'NH$_4^+$ (UV+heating)').replace('CO:CH3CHO:30.0K',
                                                                             'CO:CH$_3$CHO 30K').replace(
                             'H2O:CH4:10:0.6:a', 'H$_2$O:CH$_4$:10:0.6').replace('H2O:CH4:10:0.6:a:V3',
                                                                                 'H$_2$O:CH$_4$:10:0.6').replace(
                             'H2O:40K', 'H$_2$O 40K').replace('NH3:CH3OH:50:10K', 'NH$_3$:CH$_3$OH 10K').replace(
                             'CH3CN', 'CH$_3$CN').replace('HNCO:NH3', 'NH$_4^+$ (heating)'),
                         r'w$_5$:' + nsp5.replace('CO:CH3OH:CH3OCH3:15.0K', 'CO:CH$_3$OH:CH$_3$OCH$_3$ 15K').replace(
                             'H2O:CH3OH:1:1:c', 'H2O:CH3OH:1:1-CR').replace('HCOOH:15.0K', 'HCOOH 15K').replace(
                             'CO:NH3:10K', 'CO:NH$_3$ 10K').replace('H2O:CH4:10:0.6:a:V3',
                                                                    'H$_2$O:CH$_4$:10:0.6').replace('CO:CH3OH:10K',
                                                                                                    'CO:CH$_3$OH 10K').replace(
                             'CO:CH3CH2OH:30.0K', 'CO:CH$_3$CH$_2$OH 30K').replace('HCOOH:30.0K', 'HCOOH 30K').replace(
                             'H2O:NH3:CO:1.0:0.6:0.4:c', 'H$_2$O:NH$_3$:CO (1.0:0.6:0.4)-CR').replace('H2CO',
                                                                                                      'H$_2$CO').replace(
                             'NH4NO3:200K', 'NH$_4^+$ (UV+heating)').replace('CO:CH3CHO:30.0K',
                                                                             'CO:CH$_3$CHO 30K').replace(
                             'H2O:CH4:10:0.6:a:V3', 'H$_2$O:CH$_4$:10:0.6').replace('H2O:40K', 'H$_2$O 40K').replace(
                             'NH3:CH3OH:50:10K', 'NH$_3$:CH$_3$OH 10K').replace('CH3CN', 'CH$_3$CN').replace('HNCO:NH3',
                                                                                                             'NH$_4^+$ (heating)')))
    props = dict(boxstyle='round', facecolor='white')
    ax.text(-0.6, 3.7, textstr, transform=ax.transAxes, fontsize=45, verticalalignment='top', bbox=props)

    # ax2 = fig.add_subplot(1,1,1)
    # fig.colorbar(cs, orientation='vertical', fraction=0.046, pad=0.04)
    # plt.tight_layout()
    cb_ax = fig.add_axes([0.02, -0.02, 0.98, 0.01])
    # cb_ax = fig.add_axes([.85, .3, .02, .8])
    cbar = fig.colorbar(cs, cax=cb_ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=45)
    cbar.ax.set_xlabel(r'$\mathrm{\Delta(\nu,\alpha)=\chi^2 - \chi^2_{min}}$', fontsize=45)
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.subplots_adjust(wspace=0, hspace=0)


def st_plot_contour6(file1, pathdir):
    """
	Chi-square maps for 6 variables.

	Parameters
	-------------

	file1 : 'str'
		Path chi-squares from the GA optimization.

	"""
    sig1 = 7.23
    sig2 = 10.64
    sig3 = 16.81

    with open(file1, 'r') as f2:
        lines = f2.readlines()

    data = []
    data = [line.split() for line in lines]
    data2 = np.asfarray(data)
    home1 = pathdir

    pathb = pathdir + 'Best_comb.csv'

    tfil = pd.read_csv(pathb, delimiter=",", low_memory=True, usecols=['name'])  # output_file.txt
    nsp1, nsp2, nsp3, nsp4, nsp5, nsp6 = tfil['name'][0], tfil['name'][1], tfil['name'][2], tfil['name'][3], \
                                         tfil['name'][4], tfil['name'][5]
    nsp1, nsp2, nsp3, nsp4, nsp5, nsp6 = os.path.splitext(os.path.basename(nsp1))[0], \
                                         os.path.splitext(os.path.basename(nsp2))[0], \
                                         os.path.splitext(os.path.basename(nsp3))[0], \
                                         os.path.splitext(os.path.basename(nsp4))[0], \
                                         os.path.splitext(os.path.basename(nsp5))[0], \
                                         os.path.splitext(os.path.basename(nsp6))[0]
    nsp1, nsp2, nsp3, nsp4, nsp5, nsp6 = nsp1.replace('_', ':'), nsp2.replace('_', ':'), nsp3.replace('_',
                                                                                                      ':'), nsp4.replace(
        '_', ':'), nsp5.replace('_', ':'), nsp6.replace('_', ':')

    labels = [nsp1, nsp2, nsp3, nsp4, nsp5, nsp6]

    ucv = data2.shape[1]  # number of columns
    w_values = data2.shape[1] - 2  # weight values

    chi = data2[:, ucv - 1]
    deltachi = chi - min(chi)
    b = find_nearest(deltachi, 50.)
    a = get_line_number2(b, deltachi)

    min0 = 0
    max0 = a

    chi = data2[:, ucv - 1]  # [min0:max0]
    deltachi = chi - min(chi)

    c = np.linspace(0, w_values, ucv - 1)
    a = list(combinations(c, 2))

    idx0 = 0
    idx1 = 1
    count1 = 0
    count2 = 0

    w1_min, w1_max, w2_min, w2_max, w3_min, w3_max, w4_min, w4_max, w5_min, w5_max, w6_min, w6_max = [], [], [], [], [], [], [], [], [], [], [], []
    fig = plt.figure(figsize=(30, 33.7))

    pos = np.array([1, 6, 7, 11, 12, 13, 16, 17, 18, 19, 21, 22, 23, 24, 25])
    ar1 = list(range(len(a)))
    ar2 = list(range(len(pos)))
    for i, j in zip(ar1, ar2):
        # print a[i][idx0], a[i][idx1]
        s0 = int(a[i][idx0])
        s1 = int(a[i][idx1])
        p1 = data2[:, s0][min0:max0]
        p2 = data2[:, s1][min0:max0]
        p3 = deltachi[min0:max0]
        cc = pos[j]
        ax = fig.add_subplot(5, 5, cc)  # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9.75, 3))
        # print count1+1
        x, y, z = p1, p2, p3

        xll = x.min();
        xul = x.max();
        yll = y.min();
        yul = y.max()
        interp = scipy.interpolate.Rbf(x, y, z, smooth=3, function='linear')
        yi, xi = np.mgrid[yll:yul:100j, xll:xul:100j]
        # min_fitness_idx = numpy.where(fitness == numpy.min(fitness))
        # min_xi = np.where(np.min(xi) == sig3)
        zi = interp(xi, yi)
        contours1 = plt.contour(xi, yi, zi, [sig1], colors='olive', linestyle=':.', linewidths=4)
        contours2 = plt.contour(xi, yi, zi, [sig2], colors='gold', linestyle=':', linewidths=4)
        contours3 = plt.contour(xi, yi, zi, [sig3], colors='red', linestyle=':.', linewidths=4)
        pv = contours2.collections[0].get_paths()[0]
        vv = pv.vertices
        xv = vv[:, 0]
        yv = vv[:, 1]
        # print labels[s0], min(xv), max(xv), labels[s1], min(yv), max(yv)
        # print('-------------------------------')
        # print('Index	Specie    C_I_Min       C_I_Max')
        # print('{:1f} {:7s} {:11.5f} {:11.5f}'.format(s0, labels[s0], min(xv), max(xv)))
        # print('{:1f} {:7s} {:11.5f} {:11.5f}'.format(s1, labels[s1], min(yv), max(yv)))

        # print 'labels[s0] == nsp8', labels[s1], nsp8

        if labels[s0] == nsp1:
            # print labels[s0] == nsp1
            w1_min.append(min(xv))
            w1_max.append(max(xv))
        elif labels[s0] == nsp2:
            # print labels[s0] == nsp2
            w2_min.append(min(xv))
            w2_max.append(max(xv))
        elif labels[s0] == nsp3:
            # print labels[s0] == nsp3
            w3_min.append(min(xv))
            w3_max.append(max(xv))
        elif labels[s0] == nsp4:
            # print labels[s0] == nsp4
            w4_min.append(min(xv))
            w4_max.append(max(xv))
        elif labels[s0] == nsp5:
            # print labels[s0] == nsp5
            w5_min.append(min(xv))
            w5_max.append(max(xv))

        if labels[s1] == nsp6:
            # print labels[s0] == nsp6
            w6_min.append(min(yv))
            w6_max.append(max(yv))

        cs = ax.contourf(xi, yi, zi, cmap=plt.cm.gray, vmin=min(z), vmax=max(z), zorder=0)
        # plt.xticks(fontsize = 18)
        # plt.yticks(fontsize = 18)

        # Version with w1 and specie names separated
        if cc == 1 or cc == 6 or cc == 11 or cc == 16:
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.tick_params(bottom='on')
            # ax.set_xlabel(r'$\mathrm{'+labels[s0]+'}$', fontsize=18)
            ax.set_ylabel('w' + '$_' + str(s1 + 1) + '$', fontsize=45)
            ax.yaxis.set_ticks_position('both')
            ax.xaxis.set_ticks_position('both')
            ax.tick_params(which='major', length=10, width=1, direction='in', labelsize=45, zorder=10)
            ax.tick_params(which='minor', length=5, width=1, direction='in', labelsize=45, zorder=10)
        else:
            # ax.set_xlabel(r'$\mathrm{'+'w'+'_'+str(s0+1)+'('+labels[s0]+'}$'+')', fontsize=45)
            ax.set_xlabel('w' + '$_' + str(s0 + 1) + '$', fontsize=45)
            ax.yaxis.set_ticks_position('both')
            ax.xaxis.set_ticks_position('both')
            ax.tick_params(which='major', length=10, width=1, direction='in', labelsize=45, zorder=10, rotation=45)
            ax.tick_params(which='minor', length=5, width=1, direction='in', labelsize=45, zorder=10)

        if cc == 7 or cc == 12 or cc == 13 or cc == 17 or cc == 18 or cc == 19 or cc == 21 or cc == 22 or cc == 23 or cc == 24 or cc == 25:
            plt.gca().axes.get_yaxis().set_visible(True)
            plt.tick_params(left='on')
            ax.yaxis.set_ticks_position('both')
            ax.xaxis.set_ticks_position('both')
            ax.tick_params(which='major', length=10, width=1, direction='in', labelsize=45, zorder=10)
            ax.tick_params(which='minor', length=5, width=1, direction='in', labelsize=45, zorder=10)
            ax.set_yticklabels([])
            # plt.axis('off')
            # plt.setp(ax.get_yticklabels(), visible=True)
            # plt.tick_params(left='off')
            # ax.set_xlabel(r'$\mathrm{'+labels[s0]+'}$', fontsize=18)
            ax.set_ylabel(r'$\mathrm{' + 'w' + '_' + str(s1 + 1) + '(' + labels[s1] + '}$' + ')', fontsize=1)
        else:
            ax.yaxis.set_ticks_position('both')
            ax.xaxis.set_ticks_position('both')
            ax.set_ylabel('w' + '$_' + str(s1 + 1) + '$', fontsize=45)
            ax.tick_params(which='major', length=10, width=1, direction='in', labelsize=45, zorder=10)
            ax.tick_params(which='minor', length=5, width=1, direction='in', labelsize=45, zorder=10)

        count1 = count1 + 1

    w1n_min, w1n_max, w2n_min, w2n_max, w3n_min, w3n_max, w4n_min, w4n_max, w5n_min, w5n_max, w6n_min, w6n_max = np.mean(
        w1_min), np.mean(w1_max), np.mean(w2_min), np.mean(w2_max), np.mean(w3_min), np.mean(w3_max), np.mean(
        w4_min), np.mean(w4_max), np.mean(w5_min), np.mean(w5_max), np.mean(w6_min), np.mean(w6_max)
    np.savetxt('q_min.txt', np.transpose([w1n_min, w2n_min, w3n_min, w4n_min, w5n_min, w6n_min]))
    np.savetxt('q_max.txt', np.transpose([w1n_max, w2n_max, w3n_max, w4n_max, w5n_max, w6n_max]))
    textstr = '\n'.join((r'w$_1$:' + nsp1.replace('CO:CH3OH:CH3OCH3:15.0K', 'CO:CH$_3$OH:CH$_3$OCH$_3$ 15K').replace(
        'H2O:CH3OH:1:1:c', 'H2O:CH3OH:1:1-CR').replace('CO:NH3:10K', 'CO:NH$_3$ 10K').replace('CO:CH3OH:10K',
                                                                                              'CO:CH$_3$OH 10K').replace(
        'CO:CH3CH2OH:30.0K', 'CO:CH$_3$CH$_2$OH 30K').replace('HCOOH:30.0K', 'HCOOH 30K').replace('HCOOH:15.0K',
                                                                                                  'HCOOH 15K').replace(
        'H2O:NH3:CO:1.0:0.6:0.4:c', 'H$_2$O:NH$_3$:CO (1.0:0.6:0.4)-CR').replace('H2CO', 'H$_2$CO').replace(
        'NH4NO3:200K', 'NH$_4^+$ (UV+heating)').replace('CO:CH3CHO:30.0K', 'CO:CH$_3$CHO 30K').replace('H2O:40K',
                                                                                                       'H$_2$O 40K').replace(
        'NH3:CH3OH:50:10K', 'NH$_3$:CH$_3$OH 10K').replace('CH3CN', 'CH$_3$CN').replace('HNCO:NH3',
                                                                                        'NH$_4^+$ (heating)'),
                         r'w$_2$:' + nsp2.replace('CO:CH3OH:CH3OCH3:15.0K', 'CO:CH$_3$OH:CH$_3$OCH$_3$ 15K').replace(
                             'H2O:CH3OH:1:1:c', 'H2O:CH3OH:1:1-CR').replace('HCOOH:15.0K', 'HCOOH 15K').replace(
                             'CO:NH3:10K', 'CO:NH$_3$ 10K').replace('CO:CH3OH:10K', 'CO:CH$_3$OH 10K').replace(
                             'CO:CH3CH2OH:30.0K', 'CO:CH$_3$CH$_2$OH 30K').replace('HCOOH:30.0K', 'HCOOH 30K').replace(
                             'H2O:NH3:CO:1.0:0.6:0.4:c', 'H$_2$O:NH$_3$:CO (1.0:0.6:0.4)-CR').replace('H2CO',
                                                                                                      'H$_2$CO').replace(
                             'NH4NO3:200K', 'NH$_4^+$ (UV+heating)').replace('CO:CH3CHO:30.0K',
                                                                             'CO:CH$_3$CHO 30K').replace('H2O:40K',
                                                                                                         'H$_2$O 40K').replace(
                             'NH3:CH3OH:50:10K', 'NH$_3$:CH$_3$OH 10K').replace('CH3CN', 'CH$_3$CN').replace('HNCO:NH3',
                                                                                                             'NH$_4^+$ (heating)'),
                         r'w$_3$:' + nsp3.replace('CO:CH3OH:CH3OCH3:15.0K', 'CO:CH$_3$OH:CH$_3$OCH$_3$ 15K').replace(
                             'H2O:CH3OH:1:1:c', 'H2O:CH3OH:1:1-CR').replace('HCOOH:15.0K', 'HCOOH 15K').replace(
                             'CO:NH3:10K', 'CO:NH$_3$ 10K').replace('CO:CH3OH:10K', 'CO:CH$_3$OH 10K').replace(
                             'CO:CH3CH2OH:30.0K', 'CO:CH$_3$CH$_2$OH 30K').replace('HCOOH:30.0K', 'HCOOH 30K').replace(
                             'H2O:NH3:CO:1.0:0.6:0.4:c', 'H$_2$O:NH$_3$:CO (1.0:0.6:0.4)-CR').replace('H2CO',
                                                                                                      'H$_2$CO').replace(
                             'NH4NO3:200K', 'NH$_4^+$ (UV+heating)').replace('CO:CH3CHO:30.0K',
                                                                             'CO:CH$_3$CHO 30K').replace('H2O:40K',
                                                                                                         'H$_2$O 40K').replace(
                             'NH3:CH3OH:50:10K', 'NH$_3$:CH$_3$OH 10K').replace('CH3CN', 'CH$_3$CN').replace(
                             'H2O:CH4:10:0.6:a', 'H$_2$O:CH$_4$:10:0.6').replace('HNCO:NH3', 'NH$_4^+$ (heating)'),
                         r'w$_4$:' + nsp4.replace('CO:CH3OH:CH3OCH3:15.0K', 'CO:CH$_3$OH:CH$_3$OCH$_3$ 15K').replace(
                             'H2O:CH3OH:1:1:c', 'H2O:CH3OH:1:1-CR').replace('HCOOH:15.0K', 'HCOOH 15K').replace(
                             'CO:NH3:10K', 'CO:NH$_3$ 10K').replace('CO:CH3OH:10K', 'CO:CH$_3$OH 10K').replace(
                             'CO:CH3CH2OH:30.0K', 'CO:CH$_3$CH$_2$OH 30K').replace('HCOOH:30.0K', 'HCOOH 30K').replace(
                             'H2O:NH3:CO:1.0:0.6:0.4:c', 'H$_2$O:NH$_3$:CO (1.0:0.6:0.4)-CR').replace('H2CO',
                                                                                                      'H$_2$CO').replace(
                             'NH4NO3:200K', 'NH$_4^+$ (UV+heating)').replace('CO:CH3CHO:30.0K',
                                                                             'CO:CH$_3$CHO 30K').replace(
                             'H2O:CH4:10:0.6:a', 'H$_2$O:CH$_4$:10:0.6').replace('H2O:CH4:10:0.6:a:V3',
                                                                                 'H$_2$O:CH$_4$:10:0.6').replace(
                             'H2O:40K', 'H$_2$O 40K').replace('NH3:CH3OH:50:10K', 'NH$_3$:CH$_3$OH 10K').replace(
                             'CH3CN', 'CH$_3$CN').replace('HNCO:NH3', 'NH$_4^+$ (heating)'),
                         r'w$_5$:' + nsp5.replace('CO:CH3OH:CH3OCH3:15.0K', 'CO:CH$_3$OH:CH$_3$OCH$_3$ 15K').replace(
                             'H2O:CH3OH:1:1:c', 'H2O:CH3OH:1:1-CR').replace('HCOOH:15.0K', 'HCOOH 15K').replace(
                             'CO:NH3:10K', 'CO:NH$_3$ 10K').replace('H2O:CH4:10:0.6:a:V3',
                                                                    'H$_2$O:CH$_4$:10:0.6').replace('CO:CH3OH:10K',
                                                                                                    'CO:CH$_3$OH 10K').replace(
                             'CO:CH3CH2OH:30.0K', 'CO:CH$_3$CH$_2$OH 30K').replace('HCOOH:30.0K', 'HCOOH 30K').replace(
                             'H2O:NH3:CO:1.0:0.6:0.4:c', 'H$_2$O:NH$_3$:CO (1.0:0.6:0.4)-CR').replace('H2CO',
                                                                                                      'H$_2$CO').replace(
                             'NH4NO3:200K', 'NH$_4^+$ (UV+heating)').replace('CO:CH3CHO:30.0K',
                                                                             'CO:CH$_3$CHO 30K').replace(
                             'H2O:CH4:10:0.6:a:V3', 'H$_2$O:CH$_4$:10:0.6').replace('H2O:40K', 'H$_2$O 40K').replace(
                             'NH3:CH3OH:50:10K', 'NH$_3$:CH$_3$OH 10K').replace('CH3CN', 'CH$_3$CN').replace('HNCO:NH3',
                                                                                                             'NH$_4^+$ (heating)'),
                         r'w$_6$:' + nsp6.replace('CO:CH3OH:CH3OCH3:15.0K', 'CO:CH$_3$OH:CH$_3$OCH$_3$ 15K').replace(
                             'H2O:CH3OH:1:1:c', 'H2O:CH3OH:1:1-CR').replace('HCOOH:15.0K', 'HCOOH 15K').replace(
                             'CO:NH3:10K', 'CO:NH$_3$ 10K').replace('H2O:CH4:10:0.6:a:V3',
                                                                    'H$_2$O:CH$_4$:10:0.6').replace('CO:CH3OH:10K',
                                                                                                    'CO:CH$_3$OH 10K').replace(
                             'CO:CH3CH2OH:30.0K', 'CO:CH$_3$CH$_2$OH 30K').replace('HCOOH:30.0K', 'HCOOH 30K').replace(
                             'H2O:NH3:CO:1.0:0.6:0.4:c', 'H$_2$O:NH$_3$:CO (1.0:0.6:0.4)-CR').replace('H2CO',
                                                                                                      'H$_2$CO').replace(
                             'NH4NO3:200K', 'NH$_4^+$ (UV+heating)').replace('CO:CH3CHO:30.0K',
                                                                             'CO:CH$_3$CHO 30K').replace(
                             'H2O:CH4:10:0.6:a:V3', 'H$_2$O:CH$_4$:10:0.6').replace('H2O:40K', 'H$_2$O 40K').replace(
                             'NH3:CH3OH:50:10K', 'NH$_3$:CH$_3$OH 10K').replace('CH3CN', 'CH$_3$CN').replace('HNCO:NH3',
                                                                                                             'NH$_4^+$ (heating)')))
    props = dict(boxstyle='round', facecolor='white')
    ax.text(-1.2, 4.5, textstr, transform=ax.transAxes, fontsize=45, verticalalignment='top', bbox=props)

    # ax2 = fig.add_subplot(1,1,1)
    # fig.colorbar(cs, orientation='vertical', fraction=0.046, pad=0.04)
    # plt.tight_layout()
    cb_ax = fig.add_axes([0.02, -0.02, 0.98, 0.01])
    # cb_ax = fig.add_axes([.85, .3, .02, .8])
    cbar = fig.colorbar(cs, cax=cb_ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=45)
    cbar.ax.set_xlabel(r'$\mathrm{\Delta(\nu,\alpha)=\chi^2 - \chi^2_{min}}$', fontsize=45)
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.subplots_adjust(wspace=0, hspace=0)


def st_plot_contour7(file1, pathdir):
    """
	Chi-square maps for 7 variables.

	Parameters
	-------------

	file1 : 'str'
		Path chi-squares from the GA optimization.

	"""
    sig1 = 8.38
    sig2 = 12.02
    sig3 = 18.48

    with open(file1, 'r') as f2:
        lines = f2.readlines()

    data = []
    data = [line.split() for line in lines]
    data2 = np.asfarray(data)
    home1 = pathdir

    pathb = pathdir + 'Best_comb.csv'

    tfil = pd.read_csv(pathb, delimiter=",", low_memory=True, usecols=['name'])  # output_file.txt
    nsp1, nsp2, nsp3, nsp4, nsp5, nsp6, nsp7 = tfil['name'][0], tfil['name'][1], tfil['name'][2], tfil['name'][3], \
                                               tfil['name'][4], tfil['name'][5], tfil['name'][6]
    nsp1, nsp2, nsp3, nsp4, nsp5, nsp6, nsp7 = os.path.splitext(os.path.basename(nsp1))[0], \
                                               os.path.splitext(os.path.basename(nsp2))[0], \
                                               os.path.splitext(os.path.basename(nsp3))[0], \
                                               os.path.splitext(os.path.basename(nsp4))[0], \
                                               os.path.splitext(os.path.basename(nsp5))[0], \
                                               os.path.splitext(os.path.basename(nsp6))[0], \
                                               os.path.splitext(os.path.basename(nsp7))[0]
    nsp1, nsp2, nsp3, nsp4, nsp5, nsp6, nsp7 = nsp1.replace('_', ':'), nsp2.replace('_', ':'), nsp3.replace('_',
                                                                                                            ':'), nsp4.replace(
        '_', ':'), nsp5.replace('_', ':'), nsp6.replace('_', ':'), nsp7.replace('_', ':')

    labels = [nsp1, nsp2, nsp3, nsp4, nsp5, nsp6, nsp7]

    ucv = data2.shape[1]  # number of columns
    w_values = data2.shape[1] - 2  # weight values

    chi = data2[:, ucv - 1]
    deltachi = chi - min(chi)
    b = find_nearest(deltachi, 50.)
    a = get_line_number2(b, deltachi)

    min0 = 0
    max0 = a

    chi = data2[:, ucv - 1]  # [min0:max0]
    deltachi = chi - min(chi)

    c = np.linspace(0, w_values, ucv - 1)
    a = list(combinations(c, 2))

    idx0 = 0
    idx1 = 1
    count1 = 0
    count2 = 0

    w1_min, w1_max, w2_min, w2_max, w3_min, w3_max, w4_min, w4_max, w5_min, w5_max, w6_min, w6_max, w7_min, w7_max = [], [], [], [], [], [], [], [], [], [], [], [], [], []
    fig = plt.figure(figsize=(30, 33.7))

    pos = np.array([1, 8, 15, 22, 29, 36, 9, 16, 23, 30, 37, 17, 24, 31, 38, 25, 32, 39, 33, 40, 41])
    ar1 = list(range(len(a)))
    ar2 = list(range(len(pos)))
    for i, j in zip(ar1, ar2):
        # print a[i][idx0], a[i][idx1]
        s0 = int(a[i][idx0])
        s1 = int(a[i][idx1])
        p1 = data2[:, s0][min0:max0]
        p2 = data2[:, s1][min0:max0]
        p3 = deltachi[min0:max0]
        cc = pos[j]
        ax = fig.add_subplot(7, 7, cc)  # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9.75, 3))
        # print count1+1
        x, y, z = p1, p2, p3

        # xll = x.min();  xul = x.max();  yll = y.min();  yul = y.max()
        xmin, xmax, ymin, ymax = 0.0, 1., 0., 1.
        xll = xmin;
        xul = xmax;
        yll = ymin;
        yul = ymax
        xmin0 = np.where(z == np.min(z))
        xmin0 = x[xmin0[0][0]]
        ymin0 = np.where(z == np.min(z))
        ymin0 = y[ymin0[0][0]]
        interp = scipy.interpolate.Rbf(x, y, z, smooth=3, function='linear')
        yi, xi = np.mgrid[yll:yul:100j, xll:xul:100j]
        # min_fitness_idx = numpy.where(fitness == numpy.min(fitness))
        # min_xi = np.where(np.min(xi) == sig3)
        zi = interp(xi, yi)
        contours1 = plt.contour(xi, yi, zi, [sig1], colors='olive', linestyle=':.', linewidths=4)
        contours2 = plt.contour(xi, yi, zi, [sig2], colors='gold', linestyle=':', linewidths=4)
        contours3 = plt.contour(xi, yi, zi, [sig3], colors='red', linestyle=':.', linewidths=4)
        pv = contours2.collections[0].get_paths()[0]
        vv = pv.vertices
        xv = vv[:, 0]
        yv = vv[:, 1]
        # print labels[s0], min(xv), max(xv), labels[s1], min(yv), max(yv)
        # print('-------------------------------')
        # print('Index	Specie    C_I_Min       C_I_Max')
        # print('{:1f} {:7s} {:11.5f} {:11.5f}'.format(s0, labels[s0], min(xv), max(xv)))
        # print('{:1f} {:7s} {:11.5f} {:11.5f}'.format(s1, labels[s1], min(yv), max(yv)))

        # print 'labels[s0] == nsp8', labels[s1], nsp8

        if labels[s0] == nsp1:
            # print labels[s0] == nsp1
            w1_min.append(min(xv))
            w1_max.append(max(xv))
        elif labels[s0] == nsp2:
            # print labels[s0] == nsp2
            w2_min.append(min(xv))
            w2_max.append(max(xv))
        elif labels[s0] == nsp3:
            # print labels[s0] == nsp3
            w3_min.append(min(xv))
            w3_max.append(max(xv))
        elif labels[s0] == nsp4:
            # print labels[s0] == nsp4
            w4_min.append(min(xv))
            w4_max.append(max(xv))
        elif labels[s0] == nsp5:
            # print labels[s0] == nsp5
            w5_min.append(min(xv))
            w5_max.append(max(xv))
        elif labels[s0] == nsp6:
            # print labels[s0] == nsp6
            w6_min.append(min(xv))
            w6_max.append(max(xv))

        if labels[s1] == nsp7:
            print(labels[s0], nsp7)
            w7_min.append(min(yv))
            w7_max.append(max(yv))

        levs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                28, 29, 30]
        cs = ax.contourf(xi, yi, zi, levs, cmap=plt.cm.gray, vmin=min(z), vmax=max(z), zorder=0)
        # cs = ax.contourf(xi, yi, zi, cmap=plt.cm.gray, vmin=min(z), vmax=max(z), zorder=0)
        # plt.xticks(fontsize = 18)
        # plt.yticks(fontsize = 18)

        # Version with w1 and specie names separated
        if cc == 1 or cc == 8 or cc == 15 or cc == 22 or cc == 29 or cc == 36 or cc == 9 or cc == 16 or cc == 23 or cc == 30 or cc == 37 or cc == 17 or cc == 24 or cc == 31 or cc == 38 or cc == 25 or cc == 32 or cc == 39 or cc == 33 or cc == 40 or cc == 41:
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.tick_params(bottom='on')
            ax.set_yticks((0., 0.5))
            ax.set_xticks((0., 0.5))
            # ax.set_xlabel(r'$\mathrm{'+labels[s0]+'}$', fontsize=18)
            ax.set_xlabel('w' + '$_' + str(s0 + 1) + '$', fontsize=45)
            ax.set_ylabel('w' + '$_' + str(s1 + 1) + '$', fontsize=45)
            ax.yaxis.set_ticks_position('both')
            ax.xaxis.set_ticks_position('both')
            ax.minorticks_on()
            ax.tick_params(which='major', length=10, width=1, direction='in', labelsize=45, zorder=10)
            ax.tick_params(which='minor', length=5, width=1, direction='in', labelsize=45, zorder=10)
        else:
            ax.minorticks_on()
            plt.setp(ax.get_xticklabels(), visible=False)
            # ax.set_xlabel(r'$\mathrm{'+'w'+'_'+str(s0+1)+'('+labels[s0]+'}$'+')', fontsize=45)
            ax.set_xlabel('w' + '$_' + str(s0 + 1) + '$', fontsize=45)
            ax.yaxis.set_ticks_position('both')
            ax.xaxis.set_ticks_position('both')
            ax.tick_params(which='major', length=10, width=1, direction='in', labelsize=45, zorder=10, rotation=45)
            ax.tick_params(which='minor', length=5, width=1, direction='in', labelsize=45, zorder=10)

        if cc == 9 or cc == 16 or cc == 23 or cc == 30 or cc == 44 or cc == 17 or cc == 24 or cc == 31 or cc == 45 or cc == 25 or cc == 32 or cc == 46 or cc == 33 or cc == 47 or cc == 48 or cc == 49:
            plt.gca().axes.get_yaxis().set_visible(False)
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.tick_params(left='on')
            ax.yaxis.set_ticks_position('both')
            ax.xaxis.set_ticks_position('both')
            ax.tick_params(which='major', length=10, width=1, direction='in', labelsize=45, zorder=10)
            ax.tick_params(which='minor', length=5, width=1, direction='in', labelsize=45, zorder=10)
            ax.set_yticklabels([])
            ax.minorticks_on()
        elif cc == 37 or cc == 38 or cc == 39 or cc == 40 or cc == 41:
            # plt.gca().axes.get_yaxis().set_visible(True)
            plt.setp(ax.get_xticklabels(), visible=True)
            plt.tick_params(left='on')
            ax.yaxis.set_ticks_position('both')
            ax.xaxis.set_ticks_position('both')
            ax.tick_params(which='major', length=10, width=1, direction='in', labelsize=45, zorder=10)
            ax.tick_params(which='minor', length=5, width=1, direction='in', labelsize=45, zorder=10)
            ax.set_yticklabels([])
            ax.minorticks_on()
            # plt.axis('off')
            # plt.setp(ax.get_yticklabels(), visible=True)
            # plt.tick_params(left='off')
            # ax.set_xlabel(r'$\mathrm{'+labels[s0]+'}$', fontsize=18)
            ax.set_ylabel(r'$\mathrm{' + 'w' + '_' + str(s1 + 1) + '(' + labels[s1] + '}$' + ')', fontsize=1)
        else:
            ax.minorticks_on()
            # plt.setp(ax.get_xticklabels(), visible=True)
            ax.yaxis.set_ticks_position('both')
            ax.xaxis.set_ticks_position('both')
            ax.set_xlabel('w' + '$_' + str(s0 + 1) + '$', fontsize=45)
            ax.set_ylabel('w' + '$_' + str(s1 + 1) + '$', fontsize=45)
            ax.tick_params(which='major', length=10, width=1, direction='in', labelsize=45, zorder=10)
            ax.tick_params(which='minor', length=5, width=1, direction='in', labelsize=45, zorder=10)

        count1 = count1 + 1

    w1n_min, w1n_max, w2n_min, w2n_max, w3n_min, w3n_max, w4n_min, w4n_max, w5n_min, w5n_max, w6n_min, w6n_max, w7n_min, w7n_max = np.mean(
        w1_min), np.mean(w1_max), np.mean(w2_min), np.mean(w2_max), np.mean(w3_min), np.mean(w3_max), np.mean(
        w4_min), np.mean(w4_max), np.mean(w5_min), np.mean(w5_max), np.mean(w6_min), np.mean(w6_max), np.mean(
        w7_min), np.mean(w7_max)
    np.savetxt('q_min.txt', np.transpose([w1n_min, w2n_min, w3n_min, w4n_min, w5n_min, w6n_min, w7n_min]))
    np.savetxt('q_max.txt', np.transpose([w1n_max, w2n_max, w3n_max, w4n_max, w5n_max, w6n_max, w7n_max]))
    textstr = '\n'.join((r'w$_1$:' + nsp1.replace('CO:CH3OH:CH3OCH3:15.0K', 'CO:CH$_3$OH:CH$_3$OCH$_3$ 15K').replace(
        'H2O:CH3OH:1:1:c', 'H2O:CH3OH:1:1-CR').replace('CO:NH3:10K', 'CO:NH$_3$ 10K').replace('CO:CH3OH:10K',
                                                                                              'CO:CH$_3$OH 10K').replace(
        'CO:CH3CH2OH:30.0K', 'CO:CH$_3$CH$_2$OH 30K').replace('HCOOH:30.0K', 'HCOOH 30K').replace('HCOOH:15.0K',
                                                                                                  'HCOOH 15K').replace(
        'H2O:NH3:CO:1.0:0.6:0.4:c', 'H$_2$O:NH$_3$:CO (1.0:0.6:0.4)-CR').replace('H2CO', 'H$_2$CO').replace(
        'NH4NO3:200K', 'NH$_4^+$ (UV+heating)').replace('CO:CH3CHO:30.0K', 'CO:CH$_3$CHO 30K').replace('H2O:40K',
                                                                                                       'H$_2$O 40K').replace(
        'NH3:CH3OH:50:10K', 'NH$_3$:CH$_3$OH 10K').replace('CH3CN', 'CH$_3$CN').replace('HNCO:NH3',
                                                                                        'NH$_4^+$ (heating)'),
                         r'w$_2$:' + nsp2.replace('CO:CH3OH:CH3OCH3:15.0K', 'CO:CH$_3$OH:CH$_3$OCH$_3$ 15K').replace(
                             'H2O:CH3OH:1:1:c', 'H2O:CH3OH:1:1-CR').replace('HCOOH:15.0K', 'HCOOH 15K').replace(
                             'CO:NH3:10K', 'CO:NH$_3$ 10K').replace('CO:CH3OH:10K', 'CO:CH$_3$OH 10K').replace(
                             'CO:CH3CH2OH:30.0K', 'CO:CH$_3$CH$_2$OH 30K').replace('HCOOH:30.0K', 'HCOOH 30K').replace(
                             'H2O:NH3:CO:1.0:0.6:0.4:c', 'H$_2$O:NH$_3$:CO (1.0:0.6:0.4)-CR').replace('H2CO',
                                                                                                      'H$_2$CO').replace(
                             'NH4NO3:200K', 'NH$_4^+$ (UV+heating)').replace('CO:CH3CHO:30.0K',
                                                                             'CO:CH$_3$CHO 30K').replace('H2O:40K',
                                                                                                         'H$_2$O 40K').replace(
                             'NH3:CH3OH:50:10K', 'NH$_3$:CH$_3$OH 10K').replace('CH3CN', 'CH$_3$CN').replace('HNCO:NH3',
                                                                                                             'NH$_4^+$ (heating)'),
                         r'w$_3$:' + nsp3.replace('CO:CH3OH:CH3OCH3:15.0K', 'CO:CH$_3$OH:CH$_3$OCH$_3$ 15K').replace(
                             'H2O:CH3OH:1:1:c', 'H2O:CH3OH:1:1-CR').replace('HCOOH:15.0K', 'HCOOH 15K').replace(
                             'CO:NH3:10K', 'CO:NH$_3$ 10K').replace('CO:CH3OH:10K', 'CO:CH$_3$OH 10K').replace(
                             'CO:CH3CH2OH:30.0K', 'CO:CH$_3$CH$_2$OH 30K').replace('HCOOH:30.0K', 'HCOOH 30K').replace(
                             'H2O:NH3:CO:1.0:0.6:0.4:c', 'H$_2$O:NH$_3$:CO (1.0:0.6:0.4)-CR').replace('H2CO',
                                                                                                      'H$_2$CO').replace(
                             'NH4NO3:200K', 'NH$_4^+$ (UV+heating)').replace('CO:CH3CHO:30.0K',
                                                                             'CO:CH$_3$CHO 30K').replace('H2O:40K',
                                                                                                         'H$_2$O 40K').replace(
                             'NH3:CH3OH:50:10K', 'NH$_3$:CH$_3$OH 10K').replace('CH3CN', 'CH$_3$CN').replace(
                             'H2O:CH4:10:0.6:a', 'H$_2$O:CH$_4$:10:0.6').replace('HNCO:NH3', 'NH$_4^+$ (heating)'),
                         r'w$_4$:' + nsp4.replace('CO:CH3OH:CH3OCH3:15.0K', 'CO:CH$_3$OH:CH$_3$OCH$_3$ 15K').replace(
                             'H2O:CH3OH:1:1:c', 'H2O:CH3OH:1:1-CR').replace('HCOOH:15.0K', 'HCOOH 15K').replace(
                             'CO:NH3:10K', 'CO:NH$_3$ 10K').replace('CO:CH3OH:10K', 'CO:CH$_3$OH 10K').replace(
                             'CO:CH3CH2OH:30.0K', 'CO:CH$_3$CH$_2$OH 30K').replace('HCOOH:30.0K', 'HCOOH 30K').replace(
                             'H2O:NH3:CO:1.0:0.6:0.4:c', 'H$_2$O:NH$_3$:CO (1.0:0.6:0.4)-CR').replace('H2CO',
                                                                                                      'H$_2$CO').replace(
                             'NH4NO3:200K', 'NH$_4^+$ (UV+heating)').replace('CO:CH3CHO:30.0K',
                                                                             'CO:CH$_3$CHO 30K').replace(
                             'H2O:CH4:10:0.6:a', 'H$_2$O:CH$_4$:10:0.6').replace('H2O:CH4:10:0.6:a:V3',
                                                                                 'H$_2$O:CH$_4$:10:0.6').replace(
                             'H2O:40K', 'H$_2$O 40K').replace('NH3:CH3OH:50:10K', 'NH$_3$:CH$_3$OH 10K').replace(
                             'CH3CN', 'CH$_3$CN').replace('HNCO:NH3', 'NH$_4^+$ (heating)'),
                         r'w$_5$:' + nsp5.replace('CO:CH3OH:CH3OCH3:15.0K', 'CO:CH$_3$OH:CH$_3$OCH$_3$ 15K').replace(
                             'H2O:CH3OH:1:1:c', 'H2O:CH3OH:1:1-CR').replace('HCOOH:15.0K', 'HCOOH 15K').replace(
                             'CO:NH3:10K', 'CO:NH$_3$ 10K').replace('H2O:CH4:10:0.6:a:V3',
                                                                    'H$_2$O:CH$_4$:10:0.6').replace('CO:CH3OH:10K',
                                                                                                    'CO:CH$_3$OH 10K').replace(
                             'CO:CH3CH2OH:30.0K', 'CO:CH$_3$CH$_2$OH 30K').replace('HCOOH:30.0K', 'HCOOH 30K').replace(
                             'H2O:NH3:CO:1.0:0.6:0.4:c', 'H$_2$O:NH$_3$:CO (1.0:0.6:0.4)-CR').replace('H2CO',
                                                                                                      'H$_2$CO').replace(
                             'NH4NO3:200K', 'NH$_4^+$ (UV+heating)').replace('CO:CH3CHO:30.0K',
                                                                             'CO:CH$_3$CHO 30K').replace(
                             'H2O:CH4:10:0.6:a:V3', 'H$_2$O:CH$_4$:10:0.6').replace('H2O:40K', 'H$_2$O 40K').replace(
                             'NH3:CH3OH:50:10K', 'NH$_3$:CH$_3$OH 10K').replace('CH3CN', 'CH$_3$CN').replace('HNCO:NH3',
                                                                                                             'NH$_4^+$ (heating)'),
                         r'w$_6$:' + nsp6.replace('CO:CH3OH:CH3OCH3:15.0K', 'CO:CH$_3$OH:CH$_3$OCH$_3$ 15K').replace(
                             'H2O:CH3OH:1:1:c', 'H2O:CH3OH:1:1-CR').replace('HCOOH:15.0K', 'HCOOH 15K').replace(
                             'CO:NH3:10K', 'CO:NH$_3$ 10K').replace('H2O:CH4:10:0.6:a:V3',
                                                                    'H$_2$O:CH$_4$:10:0.6').replace('CO:CH3OH:10K',
                                                                                                    'CO:CH$_3$OH 10K').replace(
                             'CO:CH3CH2OH:30.0K', 'CO:CH$_3$CH$_2$OH 30K').replace('HCOOH:30.0K', 'HCOOH 30K').replace(
                             'H2O:NH3:CO:1.0:0.6:0.4:c', 'H$_2$O:NH$_3$:CO (1.0:0.6:0.4)-CR').replace('H2CO',
                                                                                                      'H$_2$CO').replace(
                             'NH4NO3:200K', 'NH$_4^+$ (UV+heating)').replace('CO:CH3CHO:30.0K',
                                                                             'CO:CH$_3$CHO 30K').replace(
                             'H2O:CH4:10:0.6:a:V3', 'H$_2$O:CH$_4$:10:0.6').replace('H2O:40K', 'H$_2$O 40K').replace(
                             'NH3:CH3OH:50:10K', 'NH$_3$:CH$_3$OH 10K').replace('CH3CN', 'CH$_3$CN').replace('HNCO:NH3',
                                                                                                             'NH$_4^+$ (heating)'),
                         r'w$_7$:' + nsp7.replace('CO:CH3OH:CH3OCH3:15.0K', 'CO:CH$_3$OH:CH$_3$OCH$_3$ 15K').replace(
                             'H2O:CH3OH:1:1:c', 'H2O:CH3OH:1:1-CR').replace('HCOOH:15.0K', 'HCOOH 15K').replace(
                             'CO:NH3:10K', 'CO:NH$_3$ 10K').replace('H2O:CH4:10:0.6:a:V3',
                                                                    'H$_2$O:CH$_4$:10:0.6').replace('CO:CH3OH:10K',
                                                                                                    'CO:CH$_3$OH 10K').replace(
                             'CO:CH3CH2OH:30.0K', 'CO:CH$_3$CH$_2$OH 30K').replace('HCOOH:30.0K', 'HCOOH 30K').replace(
                             'H2O:NH3:CO:1.0:0.6:0.4:c', 'H$_2$O:NH$_3$:CO (1.0:0.6:0.4)-CR').replace('H2CO',
                                                                                                      'H$_2$CO').replace(
                             'NH4NO3:200K', 'NH$_4^+$ (UV+heating)').replace('CO:CH3CHO:30.0K',
                                                                             'CO:CH$_3$CHO 30K').replace(
                             'H2O:CH4:10:0.6:a:V3', 'H$_2$O:CH$_4$:10:0.6').replace('H2O:40K', 'H$_2$O 40K').replace(
                             'NH3:CH3OH:50:10K', 'NH$_3$:CH$_3$OH 10K').replace('CH3CN', 'CH$_3$CN').replace('HNCO:NH3',
                                                                                                             'NH$_4^+$ (heating)')))
    props = dict(boxstyle='round', facecolor='white')
    ax.text(-2.0, 5.97, textstr, transform=ax.transAxes, fontsize=45, verticalalignment='top', bbox=props)

    # ax2 = fig.add_subplot(1,1,1)
    # fig.colorbar(cs, orientation='vertical', fraction=0.046, pad=0.04)
    # plt.tight_layout()
    cb_ax = fig.add_axes([0.02, 0.07, 0.85, 0.02])
    # cb_ax = fig.add_axes([.85, .3, .02, .8])
    cbar = fig.colorbar(cs, cax=cb_ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=45)
    cbar.ax.set_xlabel(r'$\mathrm{\Delta(\nu,\alpha)=\chi^2 - \chi^2_{min}}$', fontsize=45)
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.subplots_adjust(wspace=0, hspace=0)


def st_plot_contour8(file1, pathdir):
    """
	Chi-square maps for 8 variables.

	Parameters
	-------------

	file1 : 'str'
		Path chi-squares from the GA optimization.

	"""

    print('---------------------------')
    print('Making Contour Plot...')
    print('---------------------------')

    sig1 = 9.52
    sig2 = 13.36
    sig3 = 20.09

    with open(file1, 'r') as f2:
        lines = f2.readlines()

    data = []
    data = [line.split() for line in lines]
    data2 = np.asfarray(data)
    home1 = pathdir

    pathb = pathdir + 'Best_comb.csv'
    tfil = pd.read_csv(pathb, delimiter=",", low_memory=True, usecols=['name'])
    nsp1, nsp2, nsp3, nsp4, nsp5, nsp6, nsp7, nsp8 = tfil['name'][0], tfil['name'][1], tfil['name'][2], tfil['name'][3], \
                                                     tfil['name'][4], tfil['name'][5], tfil['name'][6], tfil['name'][7]
    nsp1, nsp2, nsp3, nsp4, nsp5, nsp6, nsp7, nsp8 = os.path.splitext(os.path.basename(nsp1))[0], \
                                                     os.path.splitext(os.path.basename(nsp2))[0], \
                                                     os.path.splitext(os.path.basename(nsp3))[0], \
                                                     os.path.splitext(os.path.basename(nsp4))[0], \
                                                     os.path.splitext(os.path.basename(nsp5))[0], \
                                                     os.path.splitext(os.path.basename(nsp6))[0], \
                                                     os.path.splitext(os.path.basename(nsp7))[0], \
                                                     os.path.splitext(os.path.basename(nsp8))[0]
    nsp1, nsp2, nsp3, nsp4, nsp5, nsp6, nsp7, nsp8 = nsp1.replace('_', ':'), nsp2.replace('_', ':'), nsp3.replace('_',
                                                                                                                  ':'), nsp4.replace(
        '_', ':'), nsp5.replace('_', ':'), nsp6.replace('_', ':'), nsp7.replace('_', ':'), nsp8.replace('_', ':')

    labels = [nsp1, nsp2, nsp3, nsp4, nsp5, nsp6, nsp7, nsp8]

    ucv = data2.shape[1]  # number of columns
    w_values = data2.shape[1] - 2  # weight values

    chi = data2[:, ucv - 1]
    deltachi = chi - min(chi)
    b = find_nearest(deltachi, 50.)
    a = get_line_number2(b, deltachi)

    min0 = 0
    max0 = a

    chi = data2[:, ucv - 1]  # [min0:max0]
    deltachi = chi - min(chi)  # - 35
    # deltachi = deltachi0 - min(deltachi0)

    c = np.linspace(0, w_values, ucv - 1)
    a = list(combinations(c, 2))

    idx0 = 0
    idx1 = 1
    count1 = 0
    count2 = 0

    w1_min, w1_max, w2_min, w2_max, w3_min, w3_max, w4_min, w4_max, w5_min, w5_max, w6_min, w6_max, w7_min, w7_max, w8_min, w8_max = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    fig = plt.figure(figsize=(30, 33.7))

    pos = np.array(
        [1, 8, 15, 22, 29, 36, 43, 9, 16, 23, 30, 37, 44, 17, 24, 31, 38, 45, 25, 32, 39, 46, 33, 40, 47, 41, 48, 49])
    ar1 = list(range(len(a)))
    ar2 = list(range(len(pos)))
    for i, j in zip(ar1, ar2):
        # print a[i][idx0], a[i][idx1]
        s0 = int(a[i][idx0])
        s1 = int(a[i][idx1])
        p1 = data2[:, s0][min0:max0]
        p2 = data2[:, s1][min0:max0]
        p3 = deltachi[min0:max0]
        cc = pos[j]
        ax = fig.add_subplot(7, 7, cc)  # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9.75, 3))
        # print count1+1
        x, y, z = p1, p2, p3

        # xll = x.min();  xul = x.max();  yll = y.min();  yul = y.max()
        xmin, xmax, ymin, ymax = 0.0, 1., 0., 1.
        xll = xmin;
        xul = xmax;
        yll = ymin;
        yul = ymax
        xmin0 = np.where(z == np.min(z))
        xmin0 = x[xmin0[0][0]]
        ymin0 = np.where(z == np.min(z))
        ymin0 = y[ymin0[0][0]]
        interp = scipy.interpolate.Rbf(x, y, z, smooth=3, function='linear')
        yi, xi = np.mgrid[yll:yul:100j, xll:xul:100j]
        # min_fitness_idx = numpy.where(fitness == numpy.min(fitness))
        # min_xi = np.where(np.min(xi) == sig3)
        zi = interp(xi, yi)
        contours1 = plt.contour(xi, yi, zi, [sig1], colors='olive', linestyle=':.', linewidths=4)
        contours2 = plt.contour(xi, yi, zi, [sig2], colors='gold', linestyle=':', linewidths=4)
        contours3 = plt.contour(xi, yi, zi, [sig3], colors='red', linestyle=':.', linewidths=4)
        pv = contours2.collections[0].get_paths()[0]
        vv = pv.vertices
        xv = vv[:, 0]
        yv = vv[:, 1]

        # print labels[s0], min(xv), max(xv), labels[s1], min(yv), max(yv)
        # print('-------------------------------')
        # print('Index	Specie    C_I_Min       C_I_Max')
        # print('{:1f} {:7s} {:11.5f} {:11.5f}'.format(s0, labels[s0], min(xv), max(xv)))
        # print('{:1f} {:7s} {:11.5f} {:11.5f}'.format(s1, labels[s1], min(yv), max(yv)))

        # print 'labels[s0] == nsp8', labels[s1], nsp8

        if labels[s0] == nsp1:
            # print labels[s0] == nsp1
            w1_min.append(min(xv))
            w1_max.append(max(xv))
        elif labels[s0] == nsp2:
            # print labels[s0] == nsp2
            w2_min.append(min(xv))
            w2_max.append(max(xv))
        elif labels[s0] == nsp3:
            # print labels[s0] == nsp3
            w3_min.append(min(xv))
            w3_max.append(max(xv))
        elif labels[s0] == nsp4:
            # print labels[s0] == nsp4
            w4_min.append(min(xv))
            w4_max.append(max(xv))
        elif labels[s0] == nsp5:
            # print labels[s0] == nsp5
            w5_min.append(min(xv))
            w5_max.append(max(xv))
        elif labels[s0] == nsp6:
            # print labels[s0] == nsp6
            w6_min.append(min(xv))
            w6_max.append(max(xv))
        elif labels[s0] == nsp7:
            # print labels[s0], nsp7
            w7_min.append(min(xv))
            w7_max.append(max(xv))

        if labels[s1] == nsp8:
            # print labels[s1],nsp8
            w8_min.append(min(yv))
            w8_max.append(max(yv))

        # levs = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 70, 80, 100]
        levs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                28, 29, 30]
        # levs = [0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0,8.5,9.0,9.5,10.,10.5,11.,11.5,12.0,12.5,13.,13.5,14.,14.5,15.,15.5,16.,16.5,17.,17.5,18.,18.5,19.,19.5,20.,20.5,21.,21.5,22.,22.5,23.,23.5,24.,24.5,25.]
        cs = ax.contourf(xi, yi, zi, levs, cmap=plt.cm.gray, vmin=min(z), vmax=max(z), zorder=0)
        # cs = ax.contourf(xi, yi, zi, cmap=plt.cm.gray, vmin=min(z), vmax=max(z), zorder=0)
        # plt.xticks(fontsize = 18)
        # plt.yticks(fontsize = 18)

        # Version with w1 and specie names separated
        if cc == 1 or cc == 8 or cc == 15 or cc == 22 or cc == 29 or cc == 36 or cc == 9 or cc == 16 or cc == 23 or cc == 30 or cc == 37 or cc == 17 or cc == 24 or cc == 31 or cc == 38 or cc == 25 or cc == 32 or cc == 39 or cc == 33 or cc == 40 or cc == 41:
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.tick_params(bottom='on')
            ax.minorticks_on()
            ax.set_yticks((0.5, 1.0))
            ax.set_xticks((0.5, 1.0))
            # ax.set_xlabel(r'$\mathrm{'+labels[s0]+'}$', fontsize=18)
            ax.set_ylabel('w' + '$_' + str(s1 + 1) + '$', fontsize=45)
            ax.yaxis.set_ticks_position('both')
            ax.xaxis.set_ticks_position('both')
            ax.tick_params(which='major', length=10, width=1, direction='in', labelsize=45, zorder=10)
            ax.tick_params(which='minor', length=5, width=1, direction='in', labelsize=45, zorder=10)
        else:
            # ax.set_xlabel(r'$\mathrm{'+'w'+'_'+str(s0+1)+'('+labels[s0]+'}$'+')', fontsize=45)
            ax.minorticks_on()
            ax.set_xlabel('w' + '$_' + str(s0 + 1) + '$', fontsize=45)
            ax.yaxis.set_ticks_position('both')
            ax.xaxis.set_ticks_position('both')
            ax.tick_params(which='major', length=10, width=1, direction='in', labelsize=45, zorder=10, rotation=45)
            ax.tick_params(which='minor', length=5, width=1, direction='in', labelsize=45, zorder=10)

        if cc == 9 or cc == 16 or cc == 23 or cc == 30 or cc == 37 or cc == 44 or cc == 17 or cc == 24 or cc == 31 or cc == 38 or cc == 45 or cc == 25 or cc == 32 or cc == 39 or cc == 46 or cc == 33 or cc == 40 or cc == 47 or cc == 41 or cc == 48 or cc == 49:
            plt.gca().axes.get_yaxis().set_visible(True)
            plt.tick_params(left='on')
            ax.minorticks_on()
            ax.set_yticks((0.5, 1.0))
            ax.set_xticks((0.5, 1.0))
            ax.yaxis.set_ticks_position('both')
            ax.xaxis.set_ticks_position('both')
            ax.tick_params(which='major', length=10, width=1, direction='in', labelsize=45, zorder=10)
            ax.tick_params(which='minor', length=5, width=1, direction='in', labelsize=45, zorder=10)
            ax.set_yticklabels([])
            # plt.axis('off')
            # plt.setp(ax.get_yticklabels(), visible=True)
            # plt.tick_params(left='off')
            # ax.set_xlabel(r'$\mathrm{'+labels[s0]+'}$', fontsize=18)
            ax.set_ylabel(r'$\mathrm{' + 'w' + '_' + str(s1 + 1) + '(' + labels[s1] + '}$' + ')', fontsize=1)
        else:
            ax.minorticks_on()
            ax.yaxis.set_ticks_position('both')
            ax.xaxis.set_ticks_position('both')
            ax.set_ylabel('w' + '$_' + str(s1 + 1) + '$', fontsize=45)
            ax.tick_params(which='major', length=10, width=1, direction='in', labelsize=45, zorder=10)
            ax.tick_params(which='minor', length=5, width=1, direction='in', labelsize=45, zorder=10)

        count1 = count1 + 1

    w1n_min, w1n_max, w2n_min, w2n_max, w3n_min, w3n_max, w4n_min, w4n_max, w5n_min, w5n_max, w6n_min, w6n_max, w7n_min, w7n_max = np.mean(
        w1_min), np.mean(w1_max), np.mean(w2_min), np.mean(w2_max), np.mean(w3_min), np.mean(w3_max), np.mean(
        w4_min), np.mean(w4_max), np.mean(w5_min), np.mean(w5_max), np.mean(w6_min), np.mean(w6_max), np.mean(
        w7_min), np.mean(w7_max)
    w8n_min, w8n_max = np.mean(w8_min), np.mean(w8_max)
    # print 'w1n_min, w1n_max', w1n_min, w1n_max
    np.savetxt('q_min.txt', np.transpose([w1n_min, w2n_min, w3n_min, w4n_min, w5n_min, w6n_min, w7n_min, w8n_min]))
    np.savetxt('q_max.txt', np.transpose([w1n_max, w2n_max, w3n_max, w4n_max, w5n_max, w6n_max, w7n_max, w8n_max]))
    textstr = '\n'.join((r'w$_1$:' + nsp1.replace('CO2', 'CO$_2$').replace('CH4', 'CH$_4$').replace('CH3OH',
                                                                                                    'CH$_3$OH').replace(
        'CH3CH2OH', 'CH$_3$CH$_2$OH').replace('CH3CHO', 'CH$_3$CHO').replace('NH3', 'NH$_3$').replace('H2Oa',
                                                                                                      'H$_2$O').replace(
        'CO:CH3OH:CH3OCH3:15.0K', 'CO:CH$_3$OH:CH$_3$OCH$_3$ 15K').replace('H2O:CH3OH:1:1:c',
                                                                           'H2O:CH3OH:1:1-CR').replace('CO:NH3:10K',
                                                                                                       'CO:NH$_3$ 10K').replace(
        'CO:CH3OH:10K', 'CO:CH$_3$OH 10K').replace('CO:CH3CH2OH:30.0K', 'CO:CH$_3$CH$_2$OH 30K').replace('HCOOH:30.0K',
                                                                                                         'HCOOH 30K').replace(
        'HCOOH:15.0K', 'HCOOH 15K').replace('H2O:NH3:CO:1.0:0.6:0.4:c', 'H$_2$O:NH$_3$:CO (1.0:0.6:0.4)-CR').replace(
        'H2CO', 'H$_2$CO').replace('NH4NO3:200K', 'NH$_4^+$ (UV+heating)').replace('CO:CH3CHO:30.0K',
                                                                                   'CO:CH$_3$CHO 30K').replace(
        'H2O:40K', 'H$_2$O 40K').replace('NH3:CH3OH:50:10K', 'NH$_3$:CH$_3$OH 10K').replace('CH3CN',
                                                                                            'CH$_3$CN').replace(
        'HNCO:NH3', 'NH$_4^+$ (heating)'),
                         r'w$_2$:' + nsp2.replace('CO2', 'CO$_2$').replace('CH4', 'CH$_4$').replace('CH3OH',
                                                                                                    'CH$_3$OH').replace(
                             'CH3CH2OH', 'CH$_3$CH$_2$OH').replace('CH3CHO', 'CH$_3$CHO').replace('NH3',
                                                                                                  'NH$_3$').replace(
                             'H2Oa', 'H$_2$O').replace('CO:CH3OH:CH3OCH3:15.0K',
                                                       'CO:CH$_3$OH:CH$_3$OCH$_3$ 15K').replace('H2O:CH3OH:1:1:c',
                                                                                                'H2O:CH3OH:1:1-CR').replace(
                             'HCOOH:15.0K', 'HCOOH 15K').replace('CO:NH3:10K', 'CO:NH$_3$ 10K').replace('CO:CH3OH:10K',
                                                                                                        'CO:CH$_3$OH 10K').replace(
                             'CO:CH3CH2OH:30.0K', 'CO:CH$_3$CH$_2$OH 30K').replace('HCOOH:30.0K', 'HCOOH 30K').replace(
                             'H2O:NH3:CO:1.0:0.6:0.4:c', 'H$_2$O:NH$_3$:CO (1.0:0.6:0.4)-CR').replace('H2CO',
                                                                                                      'H$_2$CO').replace(
                             'NH4NO3:200K', 'NH$_4^+$ (UV+heating)').replace('CO:CH3CHO:30.0K',
                                                                             'CO:CH$_3$CHO 30K').replace('H2O:40K',
                                                                                                         'H$_2$O 40K').replace(
                             'NH3:CH3OH:50:10K', 'NH$_3$:CH$_3$OH 10K').replace('CH3CN', 'CH$_3$CN').replace('HNCO:NH3',
                                                                                                             'NH$_4^+$ (heating)'),
                         r'w$_3$:' + nsp3.replace('CO2', 'CO$_2$').replace('CH4', 'CH$_4$').replace('CH3OH',
                                                                                                    'CH$_3$OH').replace(
                             'CH3CH2OH', 'CH$_3$CH$_2$OH').replace('CH3CHO', 'CH$_3$CHO').replace('NH3',
                                                                                                  'NH$_3$').replace(
                             'H2Oa', 'H$_2$O').replace('CO:CH3OH:CH3OCH3:15.0K',
                                                       'CO:CH$_3$OH:CH$_3$OCH$_3$ 15K').replace('H2O:CH3OH:1:1:c',
                                                                                                'H2O:CH3OH:1:1-CR').replace(
                             'HCOOH:15.0K', 'HCOOH 15K').replace('CO:NH3:10K', 'CO:NH$_3$ 10K').replace('CO:CH3OH:10K',
                                                                                                        'CO:CH$_3$OH 10K').replace(
                             'CO:CH3CH2OH:30.0K', 'CO:CH$_3$CH$_2$OH 30K').replace('HCOOH:30.0K', 'HCOOH 30K').replace(
                             'H2O:NH3:CO:1.0:0.6:0.4:c', 'H$_2$O:NH$_3$:CO (1.0:0.6:0.4)-CR').replace('H2CO',
                                                                                                      'H$_2$CO').replace(
                             'NH4NO3:200K', 'NH$_4^+$ (UV+heating)').replace('CO:CH3CHO:30.0K',
                                                                             'CO:CH$_3$CHO 30K').replace('H2O:40K',
                                                                                                         'H$_2$O 40K').replace(
                             'NH3:CH3OH:50:10K', 'NH$_3$:CH$_3$OH 10K').replace('CH3CN', 'CH$_3$CN').replace(
                             'H2O:CH4:10:0.6:a', 'H$_2$O:CH$_4$:10:0.6').replace('HNCO:NH3', 'NH$_4^+$ (heating)'),
                         r'w$_4$:' + nsp4.replace('CO2', 'CO$_2$').replace('CH4', 'CH$_4$').replace('CH3OH',
                                                                                                    'CH$_3$OH').replace(
                             'CH3CH2OH', 'CH$_3$CH$_2$OH').replace('CH3CHO', 'CH$_3$CHO').replace('NH3',
                                                                                                  'NH$_3$').replace(
                             'H2Oa', 'H$_2$O').replace('CO:CH3OH:CH3OCH3:15.0K',
                                                       'CO:CH$_3$OH:CH$_3$OCH$_3$ 15K').replace('H2O:CH3OH:1:1:c',
                                                                                                'H2O:CH3OH:1:1-CR').replace(
                             'HCOOH:15.0K', 'HCOOH 15K').replace('CO:NH3:10K', 'CO:NH$_3$ 10K').replace('CO:CH3OH:10K',
                                                                                                        'CO:CH$_3$OH 10K').replace(
                             'CO:CH3CH2OH:30.0K', 'CO:CH$_3$CH$_2$OH 30K').replace('HCOOH:30.0K', 'HCOOH 30K').replace(
                             'H2O:NH3:CO:1.0:0.6:0.4:c', 'H$_2$O:NH$_3$:CO (1.0:0.6:0.4)-CR').replace('H2CO',
                                                                                                      'H$_2$CO').replace(
                             'NH4NO3:200K', 'NH$_4^+$ (UV+heating)').replace('CO:CH3CHO:30.0K',
                                                                             'CO:CH$_3$CHO 30K').replace(
                             'H2O:CH4:10:0.6:a', 'H$_2$O:CH$_4$:10:0.6').replace('H2O:CH4:10:0.6:a:V3',
                                                                                 'H$_2$O:CH$_4$:10:0.6').replace(
                             'H2O:40K', 'H$_2$O 40K').replace('NH3:CH3OH:50:10K', 'NH$_3$:CH$_3$OH 10K').replace(
                             'CH3CN', 'CH$_3$CN').replace('HNCO:NH3', 'NH$_4^+$ (heating)'),
                         r'w$_5$:' + nsp5.replace('CO2', 'CO$_2$').replace('CH4', 'CH$_4$').replace('CH3OH',
                                                                                                    'CH$_3$OH').replace(
                             'CH3CH2OH', 'CH$_3$CH$_2$OH').replace('CH3CHO', 'CH$_3$CHO').replace('NH3',
                                                                                                  'NH$_3$').replace(
                             'H2Oa', 'H$_2$O').replace('CO:CH3OH:CH3OCH3:15.0K',
                                                       'CO:CH$_3$OH:CH$_3$OCH$_3$ 15K').replace('H2O:CH3OH:1:1:c',
                                                                                                'H2O:CH3OH:1:1-CR').replace(
                             'HCOOH:15.0K', 'HCOOH 15K').replace('CO:NH3:10K', 'CO:NH$_3$ 10K').replace(
                             'H2O:CH4:10:0.6:a:V3', 'H$_2$O:CH$_4$:10:0.6').replace('CO:CH3OH:10K',
                                                                                    'CO:CH$_3$OH 10K').replace(
                             'CO:CH3CH2OH:30.0K', 'CO:CH$_3$CH$_2$OH 30K').replace('HCOOH:30.0K', 'HCOOH 30K').replace(
                             'H2O:NH3:CO:1.0:0.6:0.4:c', 'H$_2$O:NH$_3$:CO (1.0:0.6:0.4)-CR').replace('H2CO',
                                                                                                      'H$_2$CO').replace(
                             'NH4NO3:200K', 'NH$_4^+$ (UV+heating)').replace('CO:CH3CHO:30.0K',
                                                                             'CO:CH$_3$CHO 30K').replace(
                             'H2O:CH4:10:0.6:a:V3', 'H$_2$O:CH$_4$:10:0.6').replace('H2O:40K', 'H$_2$O 40K').replace(
                             'NH3:CH3OH:50:10K', 'NH$_3$:CH$_3$OH 10K').replace('CH3CN', 'CH$_3$CN').replace('HNCO:NH3',
                                                                                                             'NH$_4^+$ (heating)'),
                         r'w$_6$:' + nsp6.replace('CO2', 'CO$_2$').replace('CH4', 'CH$_4$').replace('CH3OH',
                                                                                                    'CH$_3$OH').replace(
                             'CH3CH2OH', 'CH$_3$CH$_2$OH').replace('CH3CHO', 'CH$_3$CHO').replace('NH3',
                                                                                                  'NH$_3$').replace(
                             'H2Oa', 'H$_2$O').replace('CO:CH3OH:CH3OCH3:15.0K',
                                                       'CO:CH$_3$OH:CH$_3$OCH$_3$ 15K').replace('H2O:CH3OH:1:1:c',
                                                                                                'H2O:CH3OH:1:1-CR').replace(
                             'HCOOH:15.0K', 'HCOOH 15K').replace('CO:NH3:10K', 'CO:NH$_3$ 10K').replace(
                             'H2O:CH4:10:0.6:a:V3', 'H$_2$O:CH$_4$:10:0.6').replace('CO:CH3OH:10K',
                                                                                    'CO:CH$_3$OH 10K').replace(
                             'CO:CH3CH2OH:30.0K', 'CO:CH$_3$CH$_2$OH 30K').replace('HCOOH:30.0K', 'HCOOH 30K').replace(
                             'H2O:NH3:CO:1.0:0.6:0.4:c', 'H$_2$O:NH$_3$:CO (1.0:0.6:0.4)-CR').replace('H2CO',
                                                                                                      'H$_2$CO').replace(
                             'NH4NO3:200K', 'NH$_4^+$ (UV+heating)').replace('CO:CH3CHO:30.0K',
                                                                             'CO:CH$_3$CHO 30K').replace(
                             'H2O:CH4:10:0.6:a:V3', 'H$_2$O:CH$_4$:10:0.6').replace('H2O:40K', 'H$_2$O 40K').replace(
                             'NH3:CH3OH:50:10K', 'NH$_3$:CH$_3$OH 10K').replace('CH3CN', 'CH$_3$CN').replace('HNCO:NH3',
                                                                                                             'NH$_4^+$ (heating)'),
                         r'w$_7$:' + nsp7.replace('CO2', 'CO$_2$').replace('CH4', 'CH$_4$').replace('CH3OH',
                                                                                                    'CH$_3$OH').replace(
                             'CH3CH2OH', 'CH$_3$CH$_2$OH').replace('CH3CHO', 'CH$_3$CHO').replace('NH3',
                                                                                                  'NH$_3$').replace(
                             'H2Oa', 'H$_2$O').replace('CO:CH3OH:CH3OCH3:15.0K',
                                                       'CO:CH$_3$OH:CH$_3$OCH$_3$ 15K').replace('H2O:CH3OH:1:1:c',
                                                                                                'H2O:CH3OH:1:1-CR').replace(
                             'HCOOH:15.0K', 'HCOOH 15K').replace('CO:NH3:10K', 'CO:NH$_3$ 10K').replace(
                             'H2O:CH4:10:0.6:a:V3', 'H$_2$O:CH$_4$:10:0.6').replace('CO:CH3OH:10K',
                                                                                    'CO:CH$_3$OH 10K').replace(
                             'CO:CH3CH2OH:30.0K', 'CO:CH$_3$CH$_2$OH 30K').replace('HCOOH:30.0K', 'HCOOH 30K').replace(
                             'H2O:NH3:CO:1.0:0.6:0.4:c', 'H$_2$O:NH$_3$:CO (1.0:0.6:0.4)-CR').replace('H2CO',
                                                                                                      'H$_2$CO').replace(
                             'NH4NO3:200K', 'NH$_4^+$ (UV+heating)').replace('CO:CH3CHO:30.0K',
                                                                             'CO:CH$_3$CHO 30K').replace(
                             'H2O:CH4:10:0.6:a:V3', 'H$_2$O:CH$_4$:10:0.6').replace('H2O:40K', 'H$_2$O 40K').replace(
                             'NH3:CH3OH:50:10K', 'NH$_3$:CH$_3$OH 10K').replace('CH3CN', 'CH$_3$CN').replace('HNCO:NH3',
                                                                                                             'NH$_4^+$ (heating)'),
                         r'w$_8$:' + nsp8.replace('CO2', 'CO$_2$').replace('CH4', 'CH$_4$').replace('CH3OH',
                                                                                                    'CH$_3$OH').replace(
                             'CH3CH2OH', 'CH$_3$CH$_2$OH').replace('CH3CHO', 'CH$_3$CHO').replace('NH3',
                                                                                                  'NH$_3$').replace(
                             'H2Oa', 'H$_2$O').replace('CO:CH3OH:CH3OCH3:15.0K',
                                                       'CO:CH$_3$OH:CH$_3$OCH$_3$ 15K').replace('H2O:CH3OH:1:1:c',
                                                                                                'H2O:CH3OH:1:1-CR').replace(
                             'HCOOH:15.0K', 'HCOOH 15K').replace('CO:NH3:10K', 'CO:NH$_3$ 10K').replace(
                             'H2O:CH4:10:0.6:a:V3', 'H$_2$O:CH$_4$:10:0.6').replace('CO:CH3OH:10K',
                                                                                    'CO:CH$_3$OH 10K').replace('H2CO',
                                                                                                               'H$_2$CO').replace(
                             'NH4NO3:200K', 'NH$_4^+$ (UV+heating)').replace('CO:CH3CHO:30.0K',
                                                                             'CO:CH$_3$CHO 30K').replace(
                             'H2O:CH4:10:0.6:a:V3', 'H$_2$O:CH$_4$:10:0.6').replace('H2O:40K', 'H$_2$O 40K').replace(
                             'NH3:CH3OH:50:10K', 'NH$_3$:CH$_3$OH 10K').replace('CH3CN', 'CH$_3$CN').replace('HNCO:NH3',
                                                                                                             'NH$_4^+$ (heating)').replace(
                             'CO:CH3CH2OH:30.0K', 'CO:CH$_3$CH$_2$OH 30K').replace('HCOOH:30.0K', 'HCOOH 30K').replace(
                             'H2O:NH3:CO:1.0:0.6:0.4:c', 'H$_2$O:NH$_3$:CO (1.0:0.6:0.4)-CR')))
    props = dict(boxstyle='round', facecolor='white')
    ax.text(-2.0, 5.97, textstr, transform=ax.transAxes, fontsize=45, verticalalignment='top', bbox=props)

    # ax2 = fig.add_subplot(1,1,1)
    # fig.colorbar(cs, orientation='vertical', fraction=0.046, pad=0.04)
    # plt.tight_layout()
    cb_ax = fig.add_axes([0.02, -0.02, 0.98, 0.01])
    # cb_ax = fig.add_axes([.85, .3, .02, .8])
    cbar = fig.colorbar(cs, cax=cb_ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=45)
    cbar.ax.set_xlabel(r'$\mathrm{\Delta(\nu,\alpha)=\chi^2 - \chi^2_{min}}$', fontsize=45)
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.subplots_adjust(wspace=0, hspace=0)
