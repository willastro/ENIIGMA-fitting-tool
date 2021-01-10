import numpy
import matplotlib.pyplot as plt
import sh
import glob
import os
import os.path
from pandas import DataFrame
import pandas as pd
import multiprocessing
from pyevolve import G1DList, GSimpleGA, Selectors, Crossovers
from pyevolve import Initializators, Mutators, Consts
from pyevolve import Interaction
from pyevolve import Statistics
from pyevolve import DBAdapters
from pyevolve import Scaling
import sys


def gamod(new_tau, etau, DIR1, DIR2, Stats='False', freq_stat=10, gen=100, ga_min=0., ga_max=0.5, mutp=0.01, popsize=50,
          cc=1., fitness='rmse', initializator=Initializators.G1DListInitializatorReal,
          mutator=Mutators.G1DListMutatorRealGaussian, crossover=Crossovers.G1DBinaryStringXTwoPoint,
          scaling=Scaling.LinearScaling, selector=Selectors.GTournamentSelector,
          termination=GSimpleGA.RawScoreCriteria):
    """
	Genetic Algorithm Module dedicated to the optmiziation. All these parameters are automatically taken from the file 'optmize.py'

	Parameters
	-------------

	new_tau : 'numpy.ndarray'
		Optical depth array.

	etau : 'numpy.ndarray'
		Optical depth error.

	DIR1, DIR2 : 'str'
		Paths to directories where the files are stored.

	Stats : 'bool'
		If 'True' the genetic algorithm statistic is shown.

	freq_stats : 'int'
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

	cc : 'int'
		counter for combinations.


	Returns
	-------------
	File 'Name_list.txt' list the files according to the combination.

	File 'best_comb.csv' contains the chi-square values of all combination.

	File 'Best_score.txt' contains the best chi-square among all combination.

	File 'Best_values.txt' contains the best genes (weights) among all combination.

	File 'comb_score.txt' contains the chi-squares over all combination.

	File 'Pyevolve.db' contains the genetic algorithm statistics over all combination.



	"""
    home1 = DIR1  # +'Workspace/Processing/Interp_proc/'
    home2 = DIR2  # +'/Workspace/Processing/Store_interp/'
    mypath = home1 + '*.dat'
    files = glob.glob(mypath)

    # print 'SIZE:', len(mypath)

    i = len(files)
    for index in range(i):
        name = (os.path.splitext(os.path.basename(files[index][len(home1):]))[0])  # FILENAME WITHOUT EXTENTION
        df = pd.read_csv(files[index], sep='\s+', header=None)
        xj = df[0]  # tj['col1']
        yj = df[1]  # abs(tj['col2'])

        y0 = numpy.transpose([yj])
        y1 = numpy.reshape(y0, (1, len(xj)))

        numpy.savetxt(name + '.inp', y1)

    mypath = home1 + '*.inp'
    files_inp = glob.glob(mypath)

    f1 = open(home1 + 'Name_list.txt', 'w')
    f1.close()
    sh.rm(sh.glob(home1 + 'Name_list.txt'))

    for f in files_inp:
        name3 = (os.path.splitext(os.path.basename(f[len(mypath) - 5:]))[0])  # +'.dat' #PAREI AQUI!!!!
        fp = open('Name_list.txt', 'a')
        fp.write('{0:s}\n'.format(name3))
        fp.close()

        os.system("cat " + f + " >> OutFile.txt")
    sh.rm(sh.glob(home1 + '*.inp'))

    with open(home1 + 'OutFile.txt', 'r') as f2:
        lines = f2.readlines()

    data0 = []
    data0 = [line.split() for line in lines]
    datainp = numpy.asfarray(data0)

    equation_inputs = numpy.array(datainp)
    # print 'equation_inputs', equation_inputs
    # Number of the weights we are looking to optimize.
    num_weights = len(files)

    # print 'num_weights', num_weights

    def eqi(equa):
        return equa

    # exit()
    eq = equation_inputs

    def red_chi_square(pop):
        # Calculating the fitness value of each solution in the current population.
        # The fitness function caulcuates the sum of products between each input and its corresponding weight.
        C = eqi(eq)

        term = 0
        for i in range(len(C)):
            term += pop[i] * C[i]

        chi_square = numpy.sum(((new_tau - term) / etau) ** 2)
        red_chi2 = (1. / (term.shape[0] - 1. - len(C))) * chi_square

        return red_chi2

    def evolve_callback(ga_engine):
        generation = ga_engine.getCurrentGeneration()
        if generation % 100 == 0:
            print("Current generation: %d" % (generation,))
            print(ga_engine.getStatistics())
        return False

    """
	def cal_pop_fitness(pop):
	    # Calculating the fitness value of each solution in the current population.
	    # The fitness function caulcuates the sum of products between each input and its corresponding weight.
	    C = eqi(eq)
	    #A = numpy.array([pop[0]*C[0], pop[1]*C[1], pop[2]*C[2], pop[3]*C[3], pop[4]*C[4], pop[5]*C[5]])

	    term = 0
	    for i in range(len(C)):
	    	term += pop[i]*C[i]

	    #print 'term.shape', term.shape

	    squared_residual = (new_tau-term)**2
	    chi_square = numpy.sum(((new_tau-term)/etau)**2)
	    f = (1./(term.shape[0]-1.-len(C)))*chi_square

	    likelihood = -0.5 * numpy.sum(((new_tau-term)/etau)**2 + numpy.log(2*numpy.pi*etau**2))
	    rmse = numpy.sqrt(numpy.mean((new_tau-term)**2)) 

	    score0 = f
	    score1 = 1./f
	    score2 = numpy.sum((squared_residual)**2)
	    score3 = 1./numpy.sum(squared_residual)
	    score4 = likelihood
	    score5 = chi_square
	    score6 = 1./chi_square
	    score7 = rmse
	    score8 = 1./rmse

	    return score7
	"""

    def cal_pop_fitness_rmse(pop):
        # Calculating the fitness value of each solution in the current population.
        # The fitness function caulcuates the sum of products between each input and its corresponding weight.
        C = eqi(eq)
        # A = numpy.array([pop[0]*C[0], pop[1]*C[1], pop[2]*C[2], pop[3]*C[3], pop[4]*C[4], pop[5]*C[5]])

        term = 0
        for i in range(len(C)):
            term += pop[i] * C[i]

        rmse = numpy.sqrt(numpy.mean((new_tau - term) ** 2))

        return rmse

    def cal_pop_fitness_chi2red(pop):
        # Calculating the fitness value of each solution in the current population.
        # The fitness function caulcuates the sum of products between each input and its corresponding weight.
        C = eqi(eq)
        # A = numpy.array([pop[0]*C[0], pop[1]*C[1], pop[2]*C[2], pop[3]*C[3], pop[4]*C[4], pop[5]*C[5]])

        term = 0
        for i in range(len(C)):
            term += pop[i] * C[i]

        chi_square = numpy.sum(((new_tau - term) / etau) ** 2)
        f = (1. / (term.shape[0] - 1. - len(C))) * chi_square

        return f

    def run_main(num_weights, ga_min=ga_min, ga_max=ga_max, Stat=Stats, freq=freq_stat, gen=gen, mutp=mutp,
                 popsize=popsize, cc=cc, fitness=fitness, initializator=initializator, mutator=mutator,
                 crossover=crossover, scaling=scaling, selector=selector, termination=termination):
        # Genome instance
        genome = G1DList.G1DList(num_weights)
        # print 'genome', genome
        genome.setParams(rangemin=ga_min, rangemax=ga_max, bestrawscore=0.0000, rounddecimal=4)
        genome.initializator.set(initializator)
        genome.mutator.set(mutator)
        # genome.crossover.set(crossover)
        # genome.mutator.set(Mutators.G1DListMutatorSwap)

        # The evaluator function (objective function)
        if fitness == 'rmse':
            genome.evaluator.set(cal_pop_fitness_rmse)
        elif fitness == 'chi2red':
            genome.evaluator.set(cal_pop_fitness_chi2red)

        # Genetic Algorithm Instance
        ga = GSimpleGA.GSimpleGA(genome)
        # ga.setMultiProcessing(flag=False, full_copy=False)
        pop = ga.getPopulation()
        # pop.scaleMethod.set(Scaling.SigmaTruncScaling)
        pop.scaleMethod.set(scaling)

        ga.selector.set(selector)
        # ga.selector.set(Selectors.GRouletteWheel)

        ga.setMinimax(Consts.minimaxType["minimize"])
        ga.setGenerations(gen)
        ga.setMutationRate(mutp)
        ga.setPopulationSize(popsize)
        ga.terminationCriteria.set(termination)
        # ga.setCrossoverRate(0.95)
        # ga.stepCallback.set(evolve_callback)
        # ga.terminationCriteria.set(GSimpleGA.FitnessStatsCriteria)
        # ga.terminationCriteria.set(GSimpleGA.ConvergenceCriteria)
        # ga.setInteractiveGeneration(2)
        # Sets the DB Adapter, the resetDB flag will make the Adapter recreate
        # the database and erase all data every run, you should use this flag
        # just in the first time, after the pyevolve.db was created, you can
        # omit it.
        # sqlite_adapter = DBAdapters.DBSQLite(identify="eniigma_"+str(cc), resetDB=True)
        # ga.setDBAdapter(sqlite_adapter)

        if os.path.exists(home1 + 'pyevolve.db') == False:
            sqlite_adapter = DBAdapters.DBSQLite(identify="eniigma_" + str(cc), resetDB=True)
            ga.setDBAdapter(sqlite_adapter)
        else:
            sqlite_adapter = DBAdapters.DBSQLite(identify="eniigma_" + str(cc), resetDB=False)
            ga.setDBAdapter(sqlite_adapter)

        # Do the evolution, with stats dump
        # frequency of 10 generations
        if Stat == 'True':
            ga.evolve(freq_stats=freq)
        else:
            ga.evolve(freq_stats=0)

        # for i in xrange(len(pop)):
        #	print(pop[i].fitness)
        # Best individual
        best = ga.bestIndividual()
        # print 'Best weights (genes) in the last combination', best
        numpy.savetxt(home1 + 'Best_values.txt', best)
        # print "Best individual score in the last combination: %.2e" % best.getRawScore()
        numpy.savetxt(home1 + 'Best_score.txt', [best.getRawScore()], fmt='%1.4e')
        f = open('comb_score.txt', 'a')
        f.write('{0:f} {1:f}\n'.format(best.getRawScore(), cc))
        f.close()
        f2 = open('comb_score0.txt', 'w')
        f2.write('{0:f} {1:f}\n'.format(best.getRawScore(), cc))
        f2.close()

    run_main(num_weights)

    t = pd.read_csv(home1 + 'Name_list.txt', sep='\s+', header=None)
    sp = t[0]
    # sp.to_csv('w_'+str(num_weights + 1)+'.csv',index=False)

    tbest = pd.read_csv(home1 + 'Best_values.txt', sep='\s+', header=None)
    best = tbest[0]

    tscore = pd.read_csv(home1 + 'Best_score.txt', sep='\s+', header=None)
    score = tscore[0]

    ti = pd.read_csv(home1 + 'i_var.txt', sep='\s+', header=None)
    iv = ti[0][0] - 1
    # print('iv is')
    # print(iv)

    Best_solution_fitness = score[0]
    Best_solution_fitness_chi = red_chi_square(best)

    header = []
    for h in range(num_weights):
        header.append('w' + str(h + 1))

    count = 1
    for j in range(num_weights):
        b = []
        s = []
        for i in range(num_weights):
            b.append(best[i])
            s.append(score[0])

        Data1 = {'w_' + str(count): b}
        df1 = DataFrame(Data1, columns=['w_' + str(count)])
        dfT = df1.T
        dfT.columns = header

        dfT.to_csv('w_' + str(count) + '.csv', index=False)
        count = count + 1

    df = pd.concat(map(pd.read_csv, glob.glob(os.path.join('', home1 + 'w*.csv'))))
    # df.to_csv('Merged_0_w.csv',index=False)
    sh.rm(sh.glob(home1 + 'w*'))

    name = {'name': sp}
    spn = DataFrame(name, columns=['name'])

    sc = {'best_chi': s}
    scc = DataFrame(sc, columns=['best_chi'])

    with open('Merged_0_w.csv', 'a') as f:
        df.to_csv(f, header=True, index=False)

    with open('Merged_1_name.csv', 'a') as f1:
        spn.to_csv(f1, header=True, index=False)

    with open('Merged_2_chi.csv', 'a') as f2:
        scc.to_csv(f2, header=True, index=False)

    all_filenames = [i for i in sorted(glob.glob(home1 + 'Merged*.csv'))]
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames], axis=1)
    combined_csv['index'] = combined_csv.index
    combined_csv.sort_values(by=['best_chi', 'index'], inplace=True)
    # print(combined_csv['best_chi'].min())
    combined_csv.to_csv("Best_comb.csv", index=False, encoding='utf-8-sig')

    if iv != 0.:
        combined_csv = combined_csv[:-iv]
        combined_csv.to_csv("Best_comb.csv", index=False, encoding='utf-8-sig')
    else:
        combined_csv = combined_csv
        combined_csv.to_csv("Best_comb.csv", index=False, encoding='utf-8-sig')

    from ENIIGMA.GA import sortpandas as s
    s.sortt('Best_comb.csv')


