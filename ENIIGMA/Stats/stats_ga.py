import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os
import sh
from pandas import DataFrame
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, ScalarFormatter, LogLocator, AutoMinorLocator
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams['axes.linewidth'] = 1.5
pp = PdfPages('Stats.pdf')


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def get_line_number2(value, matrix):
    for i, line in enumerate(matrix, 1):
        if line == value:
            return i


def stats_blockt(ntt, new_tau, etau, fileout, filecl, pathdir, f_sig=5):
    """
	Confidence intervals function.

	Parameters
	-------------

	ntt : 'array'

	new_tau : 'array'

	etau : 'array'

	fileout : 'str'

	filecl : 'str'

	f_sig : 'float'

	"""

    xdata0 = ntt
    ydata0 = new_tau
    err_y = etau
    e_min = ydata0 - err_y
    e_max = ydata0 + err_y

    with open(fileout, 'r') as f2:
        lines = f2.readlines()

    da = []
    da = [line.split() for line in lines]
    dainp = np.asfarray(da)
    ucv = dainp.shape[1]  # number of columns
    w_values = dainp.shape[1] - (ucv / 2)  # weight values

    pathcl = filecl
    tcl = np.loadtxt(pathcl, dtype=float, delimiter=',', usecols=(list(range(int(w_values)))), skiprows=1).T

    f1 = open(pathdir + 'Confidence_limits_2nd.dat', 'w')
    f1.close()
    sh.rm(sh.glob(pathdir + 'Confidence_limits_2nd.dat'))
    ###################

    # Creatin random matrix and sorting
    if w_values == 2:
        c1, c2 = tcl[0][0], tcl[1][0]
        m = f_sig * np.std(err_y)
        int1 = np.random.normal(c1, m, 5000)
        int2 = np.random.normal(c2, m, 5000)

        list2 = list(range(len(int1)))

        for i1, j1 in zip(list2, list2):
            f1 = int1[i1] * dainp[:, 1] + int2[j1] * dainp[:, 3]

            p = (1. / (len(ydata0) - 1. - w_values)) * np.sum(((ydata0 - f1) / err_y) ** 2)

            if int1[i1] >= 0. and int2[j1] >= 0. and p >= 0. and p <= 500.:
                fp = open('Confidence_limits_2nd.dat', 'a')
                fp.write('{0:f} {1:f} {2:f}\n'.format(int1[i1], int2[j1], p))
                fp.close()

        path = pathdir + 'Confidence_limits_2nd.dat'
        t = pd.read_csv(path, sep='\s+', header=None)  # astropy.io.ascii.read(path, format='no_header')
        w1 = t[0]
        w2 = t[1]
        y = t[2]
        delchi = y - min(y)

        Data1 = {'w1': w1, 'w2': w2, 'y': y}
        df1 = DataFrame(Data1, columns=['w1', 'w2', 'y'])
        df1.sort_values(by=['y'], inplace=True)
        np.savetxt('Confidence_limits_2nd.dat', df1, fmt='%1.4f')
        Data2 = {'w1': w1, 'w2': w2, 'delchi': delchi}
        df2 = DataFrame(Data2, columns=['w1', 'w2', 'delchi'])
        df2.sort_values(by=['delchi'], inplace=True)
        np.savetxt('Confidence_limits_Delta_2nd.dat', df2, fmt='%1.4f')

        # Statistical Plots
        fig = plt.figure()
        from ENIIGMA.Stats import Stats_contour as stc
        file1 = pathdir + 'Confidence_limits_2nd.dat'
        stc.st_plot_contour2(file1, pathdir)
        plt.savefig(pp, format='pdf', bbox_inches='tight')

        from ENIIGMA.Stats import Stats_plot as stp

        fig = plt.figure()
        stp.min_max(xdata0, ydata0, e_min, e_max, pathdir)
        plt.savefig(pp, format='pdf', bbox_inches='tight')

        fig = plt.figure()
        file2 = pathdir + 'output_file.txt'
        fileqmin = pathdir + 'q_min.txt'
        fileqmax = pathdir + 'q_max.txt'
        for fn in glob.glob(pathdir + "Merge*"):  # Remove the Merge files
            os.remove(fn)
        stp.deconv_best(xdata0, ydata0, e_min, e_max, pathdir)
        plt.savefig(pp, format='pdf', bbox_inches='tight')

        try:
            fig = plt.figure()
            from ENIIGMA.Stats import Merge_colden as mc
            filename = pathdir + 'Column_density_*.csv'
            mc.mergecd(filename, pathdir)
            plt.savefig(pp, format='pdf')
        except:
            print('Analytical decomposition failed!')



    elif w_values == 3:
        c1, c2, c3 = tcl[0][0], tcl[1][0], tcl[2][0]
        m = f_sig * np.std(err_y)
        int1 = np.random.normal(c1, m, 1000)
        int2 = np.random.normal(c2, m, 1000)
        int3 = np.random.normal(c3, m, 1000)

        list2 = list(range(len(int1)))

        for i1, j1, k1 in zip(list2, list2, list2):
            f1 = int1[i1] * dainp[:, 1] + int2[j1] * dainp[:, 3] + int3[k1] * dainp[:, 5]

            p = (1. / (len(ydata0) - 1. - w_values)) * np.sum(((ydata0 - f1) / err_y) ** 2)

            if int1[i1] >= 0. and int2[j1] >= 0. and int3[k1] >= 0. and p >= 0. and p <= 1050.:
                fp = open('Confidence_limits_2nd.dat', 'a')
                fp.write('{0:f} {1:f} {2:f} {3:f}\n'.format(int1[i1], int2[j1], int3[k1], p))
                fp.close()

        path = pathdir + 'Confidence_limits_2nd.dat'
        t = pd.read_csv(path, sep='\s+', header=None)  # astropy.io.ascii.read(path, format='no_header')
        w1 = t[0]
        w2 = t[1]
        w3 = t[2]
        y = t[3]
        delchi = y - min(y)

        Data1 = {'w1': w1, 'w2': w2, 'w3': w3, 'y': y}
        df1 = DataFrame(Data1, columns=['w1', 'w2', 'w3', 'y'])
        df1.sort_values(by=['y'], inplace=True)
        np.savetxt('Confidence_limits_2nd.dat', df1, fmt='%1.4f')
        Data2 = {'w1': w1, 'w2': w2, 'w3': w3, 'delchi': delchi}
        df2 = DataFrame(Data2, columns=['w1', 'w2', 'w3', 'delchi'])
        df2.sort_values(by=['delchi'], inplace=True)
        np.savetxt('Confidence_limits_Delta_2nd.dat', df2, fmt='%1.4f')

        # Statistical Plots
        fig = plt.figure()
        from ENIIGMA.Stats import Stats_contour as stc
        file1 = pathdir + 'Confidence_limits_2nd.dat'
        stc.st_plot_contour3(file1, pathdir)
        plt.savefig(pp, format='pdf', bbox_inches='tight')

        from ENIIGMA.Stats import Stats_plot as stp

        fig = plt.figure()
        stp.min_max(xdata0, ydata0, e_min, e_max, pathdir)
        plt.savefig(pp, format='pdf', bbox_inches='tight')

        fig = plt.figure()
        file2 = pathdir + 'output_file.txt'
        fileqmin = pathdir + 'q_min.txt'
        fileqmax = pathdir + 'q_max.txt'
        for fn in glob.glob(pathdir + "Merge*"):  # Remove the Merge files
            os.remove(fn)
        stp.deconv_best(xdata0, ydata0, e_min, e_max, pathdir)
        plt.savefig(pp, format='pdf', bbox_inches='tight')

        try:
            fig = plt.figure()
            from ENIIGMA.Stats import Merge_colden as mc
            filename = pathdir + 'Column_density_*.csv'
            mc.mergecd(filename, pathdir)
            plt.savefig(pp, format='pdf')
        except:
            print('Analytical decomposition failed!')


    elif w_values == 4:
        c1, c2, c3, c4 = tcl[0][0], tcl[1][0], tcl[2][0], tcl[3][0]
        m = f_sig * np.std(err_y)
        int1 = np.random.normal(c1, m, 1000)
        int2 = np.random.normal(c2, m, 1000)
        int3 = np.random.normal(c3, m, 1000)
        int4 = np.random.normal(c4, m, 1000)

        list2 = list(range(len(int1)))

        for i1, j1, k1, l1 in zip(list2, list2, list2, list2):
            f1 = int1[i1] * dainp[:, 1] + int2[j1] * dainp[:, 3] + int3[k1] * dainp[:, 5] + int4[l1] * dainp[:, 7]

            p = (1. / (len(ydata0) - 1. - w_values)) * np.sum(((ydata0 - f1) / err_y) ** 2)

            if int1[i1] >= 0. and int2[j1] >= 0. and int3[k1] >= 0. and int4[l1] >= 0. and p >= 0. and p <= 500.:
                fp = open('Confidence_limits_2nd.dat', 'a')
                fp.write('{0:f} {1:f} {2:f} {3:f} {4:f}\n'.format(int1[i1], int2[j1], int3[k1], int4[l1], p))
                fp.close()

        path = pathdir + 'Confidence_limits_2nd.dat'
        t = pd.read_csv(path, sep='\s+', header=None)  # astropy.io.ascii.read(path, format='no_header')
        w1 = t[0]
        w2 = t[1]
        w3 = t[2]
        w4 = t[3]
        y = t[4]
        delchi2 = y - min(y)

        Data1 = {'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4, 'y': y}
        Data2 = {'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4, 'delchi2': delchi2}
        df1 = DataFrame(Data1, columns=['w1', 'w2', 'w3', 'w4', 'y'])
        df2 = DataFrame(Data2, columns=['w1', 'w2', 'w3', 'w4', 'delchi2'])
        df1.sort_values(by=['y'], inplace=True)
        df2.sort_values(by=['delchi2'], inplace=True)
        np.savetxt('Confidence_limits_2nd.dat', df1, fmt='%1.4f')
        np.savetxt('Confidence_limits_Delta_2nd.dat', df2, fmt='%1.4f')

        # Statistical Plots
        fig = plt.figure()
        from ENIIGMA.Stats import Stats_contour as stc
        file1 = pathdir + 'Confidence_limits_2nd.dat'
        stc.st_plot_contour4(file1, pathdir)
        plt.savefig(pp, format='pdf', bbox_inches='tight')

        from ENIIGMA.Stats import Stats_plot as stp
        fig = plt.figure()
        stp.min_max(xdata0, ydata0, e_min, e_max, pathdir)
        plt.savefig(pp, format='pdf', bbox_inches='tight')

        fig = plt.figure()
        file2 = pathdir + 'output_file.txt'
        fileqmin = pathdir + 'q_min.txt'
        fileqmax = pathdir + 'q_max.txt'
        for fn in glob.glob(pathdir + "Merge*"):  # Remove the Merge files
            os.remove(fn)
        stp.deconv_best(xdata0, ydata0, e_min, e_max, pathdir)
        plt.savefig(pp, format='pdf', bbox_inches='tight')

        try:
            fig = plt.figure()
            from ENIIGMA.Stats import Merge_colden as mc
            filename = pathdir + 'Column_density_*.csv'
            mc.mergecd(filename, pathdir)
            plt.savefig(pp, format='pdf')
        except:
            print('Analytical decomposition failed!')



    elif w_values == 5:
        c1, c2, c3, c4, c5 = tcl[0][0], tcl[1][0], tcl[2][0], tcl[3][0], tcl[4][0]
        m = f_sig * np.std(err_y)
        int1 = np.random.normal(c1, m, 1000)
        int2 = np.random.normal(c2, m, 1000)
        int3 = np.random.normal(c3, m, 1000)
        int4 = np.random.normal(c4, m, 1000)
        int5 = np.random.normal(c5, m, 1000)

        list2 = list(range(len(int1)))

        for i1, j1, k1, l1, m1 in zip(list2, list2, list2, list2, list2):
            f1 = int1[i1] * dainp[:, 1] + int2[j1] * dainp[:, 3] + int3[k1] * dainp[:, 5] + int4[l1] * dainp[:, 7] + \
                 int5[m1] * dainp[:, 9]

            p = (1. / (len(ydata0) - 1. - w_values)) * np.sum(((ydata0 - f1) / err_y) ** 2)

            if int1[i1] >= 0. and int2[j1] >= 0. and int3[k1] >= 0. and int4[l1] >= 0. and int5[
                m1] >= 0. and p >= 0. and p <= 1100.:
                fp = open('Confidence_limits_2nd.dat', 'a')
                fp.write(
                    '{0:f} {1:f} {2:f} {3:f} {4:f} {5:f}\n'.format(int1[i1], int2[j1], int3[k1], int4[l1], int5[m1], p))
                fp.close()

        path = pathdir + 'Confidence_limits_2nd.dat'
        t = pd.read_csv(path, sep='\s+', header=None)  # astropy.io.ascii.read(path, format='no_header')
        w1 = t[0]
        w2 = t[1]
        w3 = t[2]
        w4 = t[3]
        w5 = t[4]
        y = t[5]
        delchi2 = y - min(y)

        Data1 = {'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4, 'w5': w5, 'y': y}
        Data2 = {'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4, 'w5': w5, 'delchi2': delchi2}
        df1 = DataFrame(Data1, columns=['w1', 'w2', 'w3', 'w4', 'w5', 'y'])
        df2 = DataFrame(Data2, columns=['w1', 'w2', 'w3', 'w4', 'w5', 'delchi2'])
        df1.sort_values(by=['y'], inplace=True)
        df2.sort_values(by=['delchi2'], inplace=True)
        np.savetxt('Confidence_limits_2nd.dat', df1, fmt='%1.4f')
        np.savetxt('Confidence_limits_Delta_2nd.dat', df2, fmt='%1.4f')

        # Statistical Plots
        fig = plt.figure()
        from ENIIGMA.Stats import Stats_contour as stc
        file1 = pathdir + 'Confidence_limits_2nd.dat'
        stc.st_plot_contour5(file1, pathdir)
        plt.savefig(pp, format='pdf', bbox_inches='tight')

        from ENIIGMA.Stats import Stats_plot as stp

        fig = plt.figure()
        stp.min_max(xdata0, ydata0, e_min, e_max, pathdir)
        plt.savefig(pp, format='pdf', bbox_inches='tight')

        fig = plt.figure()
        file2 = pathdir + 'output_file.txt'
        fileqmin = pathdir + 'q_min.txt'
        fileqmax = pathdir + 'q_max.txt'
        for fn in glob.glob(pathdir + "Merge*"):  # Remove the Merge files
            os.remove(fn)
        stp.deconv_best(xdata0, ydata0, e_min, e_max, pathdir)
        plt.savefig(pp, format='pdf', bbox_inches='tight')

        try:
            fig = plt.figure()
            from ENIIGMA.Stats import Merge_colden as mc
            filename = pathdir + 'Column_density_*.csv'
            mc.mergecd(filename, pathdir)
            plt.savefig(pp, format='pdf')
        except:
            print('Analytical decomposition failed!')


    elif w_values == 6:
        c1, c2, c3, c4, c5, c6 = tcl[0][0], tcl[1][0], tcl[2][0], tcl[3][0], tcl[4][0], tcl[5][0]
        m = f_sig * np.std(err_y)
        int1 = np.random.normal(c1, m, 1000)
        int2 = np.random.normal(c2, m, 1000)
        int3 = np.random.normal(c3, m, 1000)
        int4 = np.random.normal(c4, m, 1000)
        int5 = np.random.normal(c5, m, 1000)
        int6 = np.random.normal(c6, m, 1000)

        list2 = list(range(len(int1)))

        for i1, j1, k1, l1, m1, n1 in zip(list2, list2, list2, list2, list2, list2):
            f1 = int1[i1] * dainp[:, 1] + int2[j1] * dainp[:, 3] + int3[k1] * dainp[:, 5] + int4[l1] * dainp[:, 7] + \
                 int5[m1] * dainp[:, 9] + int6[n1] * dainp[:, 11]

            p = (1. / (len(ydata0) - 1. - w_values)) * np.sum(((ydata0 - f1) / err_y) ** 2)

            if int1[i1] >= 0. and int2[j1] >= 0. and int3[k1] >= 0. and int4[l1] >= 0. and int5[m1] >= 0. and int6[
                n1] >= 0. and p >= 0. and p <= 1100.:
                fp = open('Confidence_limits_2nd.dat', 'a')
                fp.write('{0:f} {1:f} {2:f} {3:f} {4:f} {5:f} {6:f}\n'.format(int1[i1], int2[j1], int3[k1], int4[l1],
                                                                              int5[m1], int6[n1], p))
                fp.close()

        path = pathdir + 'Confidence_limits_2nd.dat'
        t = pd.read_csv(path, sep='\s+', header=None)  # astropy.io.ascii.read(path, format='no_header')
        w1 = t[0]
        w2 = t[1]
        w3 = t[2]
        w4 = t[3]
        w5 = t[4]
        w6 = t[5]
        y = t[6]
        delchi2 = y - min(y)

        Data1 = {'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4, 'w5': w5, 'w6': w6, 'y': y}
        Data2 = {'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4, 'w5': w5, 'w6': w6, 'delchi2': delchi2}
        df1 = DataFrame(Data1, columns=['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'y'])
        df2 = DataFrame(Data2, columns=['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'delchi2'])
        df1.sort_values(by=['y'], inplace=True)
        df2.sort_values(by=['delchi2'], inplace=True)
        np.savetxt('Confidence_limits_2nd.dat', df1, fmt='%1.4f')
        np.savetxt('Confidence_limits_Delta_2nd.dat', df2, fmt='%1.4f')

        # Statistical Plots
        fig = plt.figure()
        from ENIIGMA.Stats import Stats_contour as stc
        file1 = pathdir + 'Confidence_limits_2nd.dat'
        stc.st_plot_contour6(file1, pathdir)
        plt.savefig(pp, format='pdf', bbox_inches='tight')

        from ENIIGMA.Stats import Stats_plot as stp

        fig = plt.figure()
        stp.min_max(xdata0, ydata0, e_min, e_max, pathdir)
        plt.savefig(pp, format='pdf', bbox_inches='tight')

        fig = plt.figure()
        file2 = pathdir + 'output_file.txt'
        fileqmin = pathdir + 'q_min.txt'
        fileqmax = pathdir + 'q_max.txt'
        for fn in glob.glob(pathdir + "Merge*"):  # Remove the Merge files
            os.remove(fn)
        stp.deconv_best(xdata0, ydata0, e_min, e_max, pathdir)
        plt.savefig(pp, format='pdf', bbox_inches='tight')

        try:
            fig = plt.figure()
            from ENIIGMA.Stats import Merge_colden as mc
            filename = pathdir + 'Column_density_*.csv'
            mc.mergecd(filename, pathdir)
            plt.savefig(pp, format='pdf')
        except:
            print('Analytical decomposition failed!')


    elif w_values == 7:
        c1, c2, c3, c4, c5, c6, c7 = tcl[0][0], tcl[1][0], tcl[2][0], tcl[3][0], tcl[4][0], tcl[5][0], tcl[6][0]
        m = f_sig * np.std(err_y)
        int1 = np.random.normal(c1, m, 2000)
        int2 = np.random.normal(c2, m, 2000)
        int3 = np.random.normal(c3, m, 2000)
        int4 = np.random.normal(c4, m, 2000)
        int5 = np.random.normal(c5, m, 2000)
        int6 = np.random.normal(c6, m, 2000)
        int7 = np.random.normal(c7, m, 2000)

        list2 = list(range(len(int1)))

        for i1, j1, k1, l1, m1, n1, o1 in zip(list2, list2, list2, list2, list2, list2, list2):
            f1 = int1[i1] * dainp[:, 1] + int2[j1] * dainp[:, 3] + int3[k1] * dainp[:, 5] + int4[l1] * dainp[:, 7] + \
                 int5[m1] * dainp[:, 9] + int6[n1] * dainp[:, 11] + int7[o1] * dainp[:, 13]

            p = (1. / (len(ydata0) - 1. - w_values)) * np.sum(((ydata0 - f1) / err_y) ** 2)

            if int1[i1] >= 0. and int2[j1] >= 0. and int3[k1] >= 0. and int4[l1] >= 0. and int5[m1] >= 0. and int6[
                n1] >= 0. and int7[o1] >= 0. and p >= 0. and p <= 1058.:
                fp = open('Confidence_limits_2nd.dat', 'a')
                fp.write(
                    '{0:f} {1:f} {2:f} {3:f} {4:f} {5:f} {6:f} {7:f}\n'.format(int1[i1], int2[j1], int3[k1], int4[l1],
                                                                               int5[m1], int6[n1], int7[o1], p))
                fp.close()

        path = pathdir + 'Confidence_limits_2nd.dat'
        t = pd.read_csv(path, sep='\s+', header=None)  # astropy.io.ascii.read(path, format='no_header')
        w1 = t[0]
        w2 = t[1]
        w3 = t[2]
        w4 = t[3]
        w5 = t[4]
        w6 = t[5]
        w7 = t[6]
        y = t[7]
        delchi2 = y - min(y)

        Data1 = {'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4, 'w5': w5, 'w6': w6, 'w7': w7, 'y': y}
        Data2 = {'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4, 'w5': w5, 'w6': w6, 'w7': w7, 'delchi2': delchi2}
        df1 = DataFrame(Data1, columns=['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'y'])
        df2 = DataFrame(Data2, columns=['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'delchi2'])
        df1.sort_values(by=['y'], inplace=True)
        df2.sort_values(by=['delchi2'], inplace=True)
        np.savetxt('Confidence_limits_2nd.dat', df1, fmt='%1.4f')
        np.savetxt('Confidence_limits_Delta_2nd.dat', df2, fmt='%1.4f')

        # Statistical Plots
        fig = plt.figure()
        # from ENIIGMA.Stats import Stats_contour as stc
        from ENIIGMA.Stats import Stats_contour as stc
        file1 = pathdir + 'Confidence_limits_2nd.dat'
        stc.st_plot_contour7(file1, pathdir)
        plt.savefig(pp, format='pdf', bbox_inches='tight')

        from ENIIGMA.Stats import Stats_plot as stp

        fig = plt.figure()
        stp.min_max(xdata0, ydata0, e_min, e_max, pathdir)
        plt.savefig(pp, format='pdf', bbox_inches='tight')

        fig = plt.figure()
        file2 = pathdir + 'output_file.txt'
        fileqmin = pathdir + 'q_min.txt'
        fileqmax = pathdir + 'q_max.txt'
        for fn in glob.glob(pathdir + "Merge*"):  # Remove the Merge files
            os.remove(fn)
        stp.deconv_best(xdata0, ydata0, e_min, e_max, pathdir)
        plt.savefig(pp, format='pdf', bbox_inches='tight')

        try:
            fig = plt.figure()
            from ENIIGMA.Stats import Merge_colden as mc
            filename = pathdir + 'Column_density_*.csv'
            mc.mergecd(filename, pathdir)
            plt.savefig(pp, format='pdf')
        except:
            print('Analytical decomposition failed!')


    elif w_values == 8:
        c1, c2, c3, c4, c5, c6, c7, c8 = tcl[0][0], tcl[1][0], tcl[2][0], tcl[3][0], tcl[4][0], tcl[5][0], tcl[6][0], \
                                         tcl[7][0]
        m = f_sig * np.std(err_y)
        int1 = np.random.normal(c1, m, 5000)
        int2 = np.random.normal(c2, m, 5000)
        int3 = np.random.normal(c3, m, 5000)
        int4 = np.random.normal(c4, m, 5000)
        int5 = np.random.normal(c5, m, 5000)
        int6 = np.random.normal(c6, m, 5000)
        int7 = np.random.normal(c7, m, 5000)
        int8 = np.random.normal(c8, m, 5000)

        list2 = list(range(len(int1)))

        for i1, j1, k1, l1, m1, n1, o1, q1 in zip(list2, list2, list2, list2, list2, list2, list2, list2):
            f1 = int1[i1] * dainp[:, 1] + int2[j1] * dainp[:, 3] + int3[k1] * dainp[:, 5] + int4[l1] * dainp[:, 7] + \
                 int5[m1] * dainp[:, 9] + int6[n1] * dainp[:, 11] + int7[o1] * dainp[:, 13] + int8[q1] * dainp[:, 15]

            p = (1. / (len(ydata0) - 1. - w_values)) * np.sum(((ydata0 - f1) / err_y) ** 2)

            if int1[i1] >= 0. and int2[j1] >= 0. and int3[k1] >= 0. and int4[l1] >= 0. and int5[m1] >= 0. and int6[
                n1] >= 0. and int7[o1] >= 0. and int8[q1] >= 0. and p >= 0. and p <= 1000.:
                fp = open('Confidence_limits_2nd.dat', 'a')
                fp.write('{0:f} {1:f} {2:f} {3:f} {4:f} {5:f} {6:f} {7:f} {8:f}\n'.format(int1[i1], int2[j1], int3[k1],
                                                                                          int4[l1], int5[m1], int6[n1],
                                                                                          int7[o1], int8[q1], p))
                fp.close()

        path = pathdir + 'Confidence_limits_2nd.dat'
        t = pd.read_csv(path, sep='\s+', header=None)  # astropy.io.ascii.read(path, format='no_header')
        w1 = t[0]
        w2 = t[1]
        w3 = t[2]
        w4 = t[3]
        w5 = t[4]
        w6 = t[5]
        w7 = t[6]
        w8 = t[7]
        y = t[8]
        delchi2 = y - min(y)

        Data1 = {'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4, 'w5': w5, 'w6': w6, 'w7': w7, 'w8': w8, 'y': y}
        Data2 = {'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4, 'w5': w5, 'w6': w6, 'w7': w7, 'w8': w8, 'delchi2': delchi2}
        df1 = DataFrame(Data1, columns=['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8', 'y'])
        df2 = DataFrame(Data2, columns=['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8', 'delchi2'])
        df1.sort_values(by=['y'], inplace=True)
        df2.sort_values(by=['delchi2'], inplace=True)
        np.savetxt('Confidence_limits_2nd.dat', df1, fmt='%1.4f')
        np.savetxt('Confidence_limits_Delta_2nd.dat', df2, fmt='%1.4f')

        # Statistical Plots
        fig = plt.figure()
        # from ENIIGMA.Stats import Stats_contour as stc
        from ENIIGMA.Stats import Stats_contour as stc
        file1 = pathdir + 'Confidence_limits_2nd.dat'
        stc.st_plot_contour8(file1, pathdir)
        plt.savefig(pp, format='pdf', bbox_inches='tight')

        from ENIIGMA.Stats import Stats_plot as stp

        fig = plt.figure()
        stp.min_max(xdata0, ydata0, e_min, e_max, pathdir)
        plt.savefig(pp, format='pdf', bbox_inches='tight')

        fig = plt.figure()
        file2 = pathdir + 'output_file.txt'
        fileqmin = pathdir + 'q_min.txt'
        fileqmax = pathdir + 'q_max.txt'
        for fn in glob.glob(pathdir + "Merge*"):  # Remove the Merge files
            os.remove(fn)
        stp.deconv_best(xdata0, ydata0, e_min, e_max, pathdir)
        plt.savefig(pp, format='pdf', bbox_inches='tight')

        try:
            fig = plt.figure()
            from ENIIGMA.Stats import Merge_colden as mc
            filename = pathdir + 'Column_density_*.csv'
            mc.mergecd(filename, pathdir)
            plt.savefig(pp, format='pdf')
        except:
            print('Analytical decomposition failed!')

    plt.clf()
    pp.close()
