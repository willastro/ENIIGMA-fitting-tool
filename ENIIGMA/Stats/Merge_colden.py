import matplotlib.pyplot as plt
import glob
from pandas import DataFrame
import numpy as np
import os
import pandas as pd


#####

# Convert the column densities from deconvolutions in a array to be read by the barplot_GA, rotine used to show the column densities in a vertical barplot.
# The pathdir is the directory where the deconvoluted column densities are placed.

####
def mergecd_min(filename, pathdir):
    """
	Merge column density values.

	Parameters
	-------------

	filecomp : 'str'
		Path column density files.

	"""
    # filename = '/Users/will_rocha_starplan/Downloads/Data_StarPlan_Project/Fitting/CRBR/CRBR_5_8/Workspace/Processing/Interp_proc/Colum_density_*.csv'
    file_csv = glob.glob(filename)

    dir = os.getcwd()

    n = 0
    for i in range(len(file_csv)):
        name = os.path.splitext(os.path.basename(file_csv[i][len(dir) + 1:]))[0]
        fi = file_csv[i]
        data = pd.read_csv(fi, index_col=0, header=None).T
        data.to_csv('trans_' + name + '_' + str(n) + '.csv', index=False)
        n = n + 1

    filetrans = pathdir + 'trans_*.csv'  # '/Users/will_rocha_starplan/Downloads/Data_StarPlan_Project/Fitting/CRBR/CRBR_5_8/Workspace/Processing/Interp_proc/trans_*.csv'
    file_csvt = glob.glob(filetrans)

    filenames = file_csvt  # ['file1.txt','file2.txt']
    with open('MergeCD_min.csv', 'w') as outfile:
        for x in filenames:
            with open(x) as infile:
                for line in infile:
                    outfile.write(line)

    t = pd.read_csv(pathdir + 'MergeCD_min.csv', sep=',', header=None)
    sp = t[0]
    Val = t[1]

    Data1 = {'Label': sp, 'Values': Val}
    df1 = DataFrame(Data1, columns=['Label', 'Values'])
    # print df1
    df2 = df1.groupby(['Label']).sum()
    # print df2
    df2.to_csv('MergeCD2_min.csv', index=True)

    t = pd.read_csv(pathdir + 'MergeCD2_min.csv', sep=',')
    sp = t['Label']
    Val = t['Values']

    lis = list(range(len(sp)))

    for i, j in zip(lis, lis):
        spe = sp[i]
        if sp[i].find('mix') != -1:  # word exist
            CDmix = Val[j]  # print Val[j]
            spec = spe.split('_in_mix')[0]
        else:
            CDmix = 0.  # print 0
            spec = spe.split('_pure')[0]

        if sp[i].find('mix') == -1:  # word does not exist
            CDpure = Val[j]  # print Val[j]
        else:
            CDpure = 0.  # print 0

        fp = open('MergeCD3_min.csv', 'a')
        fp.write('{0:s} {1:e} {2:e}\n'.format(spec, CDmix, CDpure))
        fp.close()

    t0 = pd.read_csv(pathdir + 'MergeCD3_min.csv', sep='\s+', header=None)
    t = t0.T.values.tolist()
    sp = t[0]
    CDmix = t[1]
    CDpure = t[2]

    Data1 = {'Label': sp, 'CDinmix': CDmix, 'CDpure': CDpure}
    df1 = DataFrame(Data1, columns=['Label', 'CDinmix', 'CDpure'])
    df2 = df1.groupby(['Label']).sum()
    df2.to_csv('MergeCD4_min.csv', index=True)

    t = pd.read_csv(pathdir + 'MergeCD4_min.csv', sep=',')
    sp = t['Label']
    CDmix = t['CDinmix']
    CDpure = t['CDpure']
    CDtot = CDmix + CDpure

    Data1 = {'Label': sp, 'CDtot': CDtot, 'CDinmix': CDmix, 'CDpure': CDpure}
    df1 = DataFrame(Data1, columns=['Label', 'CDtot', 'CDinmix', 'CDpure'])
    df1.to_csv('MergeCD5_min.csv', index=False)


def mergecd_max(filename, pathdir):
    """
	Merge column density values.

	Parameters
	-------------

	filecomp : 'str'
		Path column density files.

	"""
    # filename = '/Users/will_rocha_starplan/Downloads/Data_StarPlan_Project/Fitting/CRBR/CRBR_5_8/Workspace/Processing/Interp_proc/Colum_density_*.csv'
    file_csv = glob.glob(filename)

    dir = os.getcwd()

    n = 0
    for i in range(len(file_csv)):
        name = os.path.splitext(os.path.basename(file_csv[i][len(dir) + 1:]))[0]
        fi = file_csv[i]
        data = pd.read_csv(fi, index_col=0, header=None).T
        data.to_csv('trans_' + name + '_' + str(n) + '.csv', index=False)
        n = n + 1

    filetrans = pathdir + 'trans_*.csv'  # '/Users/will_rocha_starplan/Downloads/Data_StarPlan_Project/Fitting/CRBR/CRBR_5_8/Workspace/Processing/Interp_proc/trans_*.csv'
    file_csvt = glob.glob(filetrans)

    filenames = file_csvt  # ['file1.txt','file2.txt']
    with open('MergeCD_max.csv', 'w') as outfile:
        for x in filenames:
            with open(x) as infile:
                for line in infile:
                    outfile.write(line)

    t = pd.read_csv(pathdir + 'MergeCD_max.csv', sep=',', header=None)
    sp = t[0]
    Val = t[1]

    Data1 = {'Label': sp, 'Values': Val}
    df1 = DataFrame(Data1, columns=['Label', 'Values'])
    # print df1
    df2 = df1.groupby(['Label']).sum()
    # print df2
    df2.to_csv('MergeCD2_max.csv', index=True)

    t = pd.read_csv(pathdir + 'MergeCD2_max.csv', sep=',')
    sp = t['Label']
    Val = t['Values']

    lis = list(range(len(sp)))

    for i, j in zip(lis, lis):
        spe = sp[i]
        if sp[i].find('mix') != -1:  # word exist
            CDmix = Val[j]  # print Val[j]
            spec = spe.split('_in_mix')[0]
        else:
            CDmix = 0.  # print 0
            spec = spe.split('_pure')[0]

        if sp[i].find('mix') == -1:  # word does not exist
            CDpure = Val[j]  # print Val[j]
        else:
            CDpure = 0.  # print 0

        fp = open('MergeCD3_max.csv', 'a')
        fp.write('{0:s} {1:e} {2:e}\n'.format(spec, CDmix, CDpure))
        fp.close()

    t0 = pd.read_csv(pathdir + 'MergeCD3_max.csv', sep='\s+', header=None)
    t = t0.T.values.tolist()
    sp = t[0]
    CDmix = t[1]
    CDpure = t[2]

    Data1 = {'Label': sp, 'CDinmix': CDmix, 'CDpure': CDpure}
    df1 = DataFrame(Data1, columns=['Label', 'CDinmix', 'CDpure'])
    df2 = df1.groupby(['Label']).sum()
    df2.to_csv('MergeCD4_max.csv', index=True)

    t = pd.read_csv(pathdir + 'MergeCD4_max.csv', sep=',')
    sp = t['Label']
    CDmix = t['CDinmix']
    CDpure = t['CDpure']
    CDtot = CDmix + CDpure

    Data1 = {'Label': sp, 'CDtot': CDtot, 'CDinmix': CDmix, 'CDpure': CDpure}
    df1 = DataFrame(Data1, columns=['Label', 'CDtot', 'CDinmix', 'CDpure'])
    df1.to_csv('MergeCD5_max.csv', index=False)


def mergecd(filename, pathdir):
    """
	Merge column density values.

	Parameters
	-------------

	filecomp : 'str'
		Path column density files.

	"""
    # filename = '/Users/will_rocha_starplan/Downloads/Data_StarPlan_Project/Fitting/CRBR/CRBR_5_8/Workspace/Processing/Interp_proc/Colum_density_*.csv'
    file_csv = glob.glob(filename)

    dir = os.getcwd()

    n = 0
    for i in range(len(file_csv)):
        name = os.path.splitext(os.path.basename(file_csv[i][len(dir) + 1:]))[0]
        fi = file_csv[i]
        data = pd.read_csv(fi, index_col=0, header=None).T
        data.to_csv('trans_' + name + '_' + str(n) + '.csv', index=False)
        n = n + 1

    filetrans = pathdir + 'trans_*.csv'  # '/Users/will_rocha_starplan/Downloads/Data_StarPlan_Project/Fitting/CRBR/CRBR_5_8/Workspace/Processing/Interp_proc/trans_*.csv'
    file_csvt = glob.glob(filetrans)

    filenames = file_csvt  # ['file1.txt','file2.txt']
    with open('MergeCD.csv', 'w') as outfile:
        for x in filenames:
            with open(x) as infile:
                for line in infile:
                    outfile.write(line)

    t = pd.read_csv(pathdir + 'MergeCD.csv', sep=',', header=None)
    sp = t[0]
    Val = t[1]

    Data1 = {'Label': sp, 'Values': Val}
    df1 = DataFrame(Data1, columns=['Label', 'Values'])
    # print df1
    df2 = df1.groupby(['Label']).sum()
    # print df2
    df2.to_csv('MergeCD2.csv', index=True)

    t = pd.read_csv(pathdir + 'MergeCD2.csv', sep=',')
    sp = t['Label']
    Val = t['Values']

    lis = list(range(len(sp)))

    for i, j in zip(lis, lis):
        spe = sp[i]
        if sp[i].find('mix') != -1:  # word exist
            CDmix = Val[j]  # print Val[j]
            spec = spe.split('_in_mix')[0]
        else:
            CDmix = 0.  # print 0
            spec = spe.split('_pure')[0]

        if sp[i].find('mix') == -1:  # word does not exist
            CDpure = Val[j]  # print Val[j]
        else:
            CDpure = 0.  # print 0

        fp = open('MergeCD3.csv', 'a')
        fp.write('{0:s} {1:e} {2:e}\n'.format(spec, CDmix, CDpure))
        fp.close()

    t0 = pd.read_csv(pathdir + 'MergeCD3.csv', sep='\s+', header=None)
    t = t0.T.values.tolist()
    sp = t[0]
    CDmix = t[1]
    CDpure = t[2]

    Data1 = {'Label': sp, 'CDinmix': CDmix, 'CDpure': CDpure}
    df1 = DataFrame(Data1, columns=['Label', 'CDinmix', 'CDpure'])
    df2 = df1.groupby(['Label']).sum()
    df2.to_csv('MergeCD4.csv', index=True)

    t = pd.read_csv(pathdir + 'MergeCD4.csv', sep=',')
    sp = t['Label']
    CDmix = t['CDinmix']
    CDpure = t['CDpure']
    CDtot = CDmix + CDpure

    Data1 = {'Label': sp, 'CDtot': CDtot, 'CDinmix': CDmix, 'CDpure': CDpure}
    df1 = DataFrame(Data1, columns=['Label', 'CDtot', 'CDinmix', 'CDpure'])
    df1.to_csv('MergeCD5.csv', index=False)

    from ENIIGMA.Stats.barplot_GA import barp

    fig = plt.figure()
    file = pathdir + 'MergeCD5.csv'
    filemin = pathdir + 'MergeCD5_min.csv'
    filemax = pathdir + 'MergeCD5_max.csv'
    barp(file, filemin, filemax)


def mergecd_no_bp(filename, pathdir):
    """
	Merge column density values.

	Parameters
	-------------

	filecomp : 'str'
		Path column density files.

	"""
    # filename = '/Users/will_rocha_starplan/Downloads/Data_StarPlan_Project/Fitting/CRBR/CRBR_5_8/Workspace/Processing/Interp_proc/Colum_density_*.csv'
    file_csv = glob.glob(filename)

    dir = os.getcwd()

    n = 0
    for i in range(len(file_csv)):
        name = os.path.splitext(os.path.basename(file_csv[i][len(dir) + 1:]))[0]
        fi = file_csv[i]
        data = pd.read_csv(fi, index_col=0, header=None).T
        data.to_csv('trans_' + name + '_' + str(n) + '.csv', index=False)
        n = n + 1

    filetrans = pathdir + 'trans_*.csv'  # '/Users/will_rocha_starplan/Downloads/Data_StarPlan_Project/Fitting/CRBR/CRBR_5_8/Workspace/Processing/Interp_proc/trans_*.csv'
    file_csvt = glob.glob(filetrans)

    filenames = file_csvt  # ['file1.txt','file2.txt']
    with open('MergeCD.csv', 'w') as outfile:
        for x in filenames:
            with open(x) as infile:
                for line in infile:
                    outfile.write(line)

    try:
        # print 'here1'
        t = pd.read_csv(pathdir + 'MergeCD.csv', sep=',', header=None)
        # print t
        sp = t[0]
        Val = t[1]

        Data1 = {'Label': sp, 'Values': Val}
        df1 = DataFrame(Data1, columns=['Label', 'Values'])
        # print df1
        df2 = df1.groupby(['Label']).sum()
        # print df2
        df2.to_csv('MergeCD2.csv', index=True)

    except:
        pass

    from os import path

    if path.exists(pathdir + 'MergeCD2.csv') == True:
        t = pd.read_csv(pathdir + 'MergeCD2.csv', sep=',')
        sp = t['Label']
        Val = t['Values']

        lis = list(range(len(sp)))

        for i, j in zip(lis, lis):
            spe = sp[i]
            if sp[i].find('mix') != -1:  # word exist
                CDmix = Val[j]  # print Val[j]
                spec = spe.split('_in_mix')[0]
            else:
                CDmix = 0.  # print 0
                spec = spe.split('_pure')[0]

            if sp[i].find('mix') == -1:  # word does not exist
                CDpure = Val[j]  # print Val[j]
            else:
                CDpure = 0.  # print 0

            fp = open('MergeCD3.csv', 'a')
            fp.write('{0:s} {1:e} {2:e}\n'.format(spec, CDmix, CDpure))
            fp.close()

        t0 = pd.read_csv(pathdir + 'MergeCD3.csv', sep='\s+', header=None)
        t = t0.T.values.tolist()
        sp = t[0]
        CDmix = t[1]
        CDpure = t[2]

        Data1 = {'Label': sp, 'CDinmix': CDmix, 'CDpure': CDpure}
        df1 = DataFrame(Data1, columns=['Label', 'CDinmix', 'CDpure'])
        df2 = df1.groupby(['Label']).sum()
        df2.to_csv('MergeCD4.csv', index=True)

        t = pd.read_csv(pathdir + 'MergeCD4.csv', sep=',')
        sp = t['Label']
        CDmix = t['CDinmix']
        CDpure = t['CDpure']
        CDtot = CDmix + CDpure

        Data1 = {'Label': sp, 'CDtot': CDtot, 'CDinmix': CDmix, 'CDpure': CDpure}
        df1 = DataFrame(Data1, columns=['Label', 'CDtot', 'CDinmix', 'CDpure'])
        df1.to_csv('MergeCD5.csv', index=False)
    else:
        print('Skipping file')
