import numpy as np
import pandas as pd
import os


def top_five_raw(savepdf = False):
	"""
	Compares the non-scaled fitness function among the five best combinations after the genetic algorithm optimization.
	
	Parameters
	-------------
	
	savepdf : 'bool'
		If 'True' an pdf output file will be saved. If 'False', the graph will be shown in a matplotlib interface.
	
	Returns
	-------------
	GA statistic comparison.
	
	    
	"""
	#DIR = os.getcwd()
	#DIR2 = '/Workspace/Processing/Interp_proc/'
	#os.chdir(DIR+DIR2)
	
	#file = 'comb_score.txt'
	#t = pd.read_csv(DIR+DIR2+file,sep='\s+', header=None)
	
	ipy_str = str(type(get_ipython()))
	if 'zmqshell' in ipy_str:
	  print('jupyter')
	  DIR = os.getcwd()
	  DIR2 = '/Workspace/Processing/Interp_proc/'
	  if 'Interp_proc' in DIR:
	    os.chdir(DIR)
	    file = '/comb_score.txt'
	    t = pd.read_csv(DIR+file,sep='\s+', header=None)
	  else:
	    os.chdir(DIR + DIR2)
	    file = 'comb_score.txt'
	    t = pd.read_csv(DIR+DIR2+file,sep='\s+', header=None)
	  print(DIR)
	  #file = 'comb_score.txt'
	  #t = pd.read_csv(DIR+DIR2+file,sep='\s+', header=None)
	else:
	 #print('terminal')
	 DIR = os.getcwd()
	 DIR2 = '/Workspace/Processing/Interp_proc/'
	 os.chdir(DIR+DIR2)
	 file = 'comb_score.txt'
	 t = pd.read_csv(DIR+DIR2+file,sep='\s+', header=None)
	
	t = t.sort_values(t.columns[0])
	best = t.T.values.tolist()[1]
	
	b1 = best[0]
	b2 = best[1]
	b3 = best[2]
	b4 = best[3]
	b5 = best[4]
	
	if savepdf == False:
		string_c = 'pyevolve_graph.py -i'+' '+'eniigma_'+str(int(b1))+','+'eniigma_'+str(int(b2))+',''eniigma_'+str(int(b3))+',''eniigma_'+str(int(b4))+',''eniigma_'+str(int(b5))+' '+'-6'	
		os.system(string_c)
	else:
		string_c = 'pyevolve_graph.py -i'+' '+'eniigma_'+str(int(b1))+','+'eniigma_'+str(int(b2))+',''eniigma_'+str(int(b3))+',''eniigma_'+str(int(b4))+',''eniigma_'+str(int(b5))+' '+'-6'+' '+'-o'+' '+'graph_eniigma_top_five_raw'+' '+'-e'+' '+'pdf'	
		os.system(string_c)

def top_five_scaled(savepdf=False):
	"""
	Compares the scaled fitness function among the five best combinations after the genetic algorithm optimization.
	
	Parameters
	-------------
	
	savepdf : 'bool'
		If 'True' an pdf output file will be saved. If 'False', the graph will be shown in a matplotlib interface.
	
	Returns
	-------------
	GA statistic comparison.
	
	    
	"""
	
	ipy_str = str(type(get_ipython()))
	if 'zmqshell' in ipy_str:
	  print('jupyter')
	  DIR = os.getcwd()
	  DIR2 = '/Workspace/Processing/Interp_proc/'
	  if 'Interp_proc' in DIR:
	    os.chdir(DIR)
	    file = '/comb_score.txt'
	    t = pd.read_csv(DIR+file,sep='\s+', header=None)
	  else:
	    os.chdir(DIR + DIR2)
	    file = 'comb_score.txt'
	    t = pd.read_csv(DIR+DIR2+file,sep='\s+', header=None)
	  print(DIR)
	  #file = 'comb_score.txt'
	  #t = pd.read_csv(DIR+DIR2+file,sep='\s+', header=None)
	else:
	 print('terminal')
	 DIR = os.getcwd()
	 DIR2 = '/Workspace/Processing/Interp_proc/'
	 os.chdir(DIR+DIR2)
	 file = 'comb_score.txt'
	 t = pd.read_csv(DIR+DIR2+file,sep='\s+', header=None)
	
	#file = 'comb_score.txt'
	#t = pd.read_csv(DIR+DIR2+file,sep='\s+', header=None)
	t = t.sort_values(t.columns[0])
	best = t.T.values.tolist()[1]
	
	b1 = best[0]
	b2 = best[1]
	b3 = best[2]
	b4 = best[3]
	b5 = best[4]
	
	if savepdf == False:
		string_c = 'pyevolve_graph.py -i'+' '+'eniigma_'+str(int(b1))+','+'eniigma_'+str(int(b2))+',''eniigma_'+str(int(b3))+',''eniigma_'+str(int(b4))+',''eniigma_'+str(int(b5))+' '+'-7'	
		print(string_c)
		os.system(string_c)
	else:
		string_c = 'pyevolve_graph.py -i'+' '+'eniigma_'+str(int(b1))+','+'eniigma_'+str(int(b2))+',''eniigma_'+str(int(b3))+',''eniigma_'+str(int(b4))+',''eniigma_'+str(int(b5))+' '+'-7'+' '+'-o'+' '+'graph_eniigma_top_five_scaled'+' '+'-e'+' '+'pdf'	
		os.system(string_c)

def check(combination=1, option=-1, savepdf = False):
	"""
	Show the fitness function statistic after the genetic algorithm optimization.
	
	Parameters
	-------------
	
	combination : 'int'
		Set the combination to check the GA optmization.
	
	option : 'int'
		
    	-1                  Error bars graph (raw scores).
    	-2                  Error bars graph (fitness scores).
    	-3                  Max/min/avg/std. dev. graph (raw scores).
    	-4                  Max/min/avg graph (fitness scores).
    	-5                  Raw and Fitness min/max difference graph.
    	-6                  Compare best raw score of two or more evolutions (you
    	                    must specify the identify comma-separed list with
    	                    --identify (-i) parameter, like 'one, two, three'),
    	                    the maximum is 6 items.
    	-7                  Compare best fitness score of two or more evolutions
    						(you must specify the identify comma-separed list with
    						--identify (-i) parameter, like 'one, two, three'),
    						the maximum is 6 items.
    	-8                  Show a heat map of population raw score distribution
    						between generations.
    	-9                  Show a heat map of population fitness score
    						distribution between generations.
	
	savepdf : 'bool'
		If 'True' an pdf output file will be saved. If 'False', the graph will be shown in a matplotlib interface.
	
	Returns
	-------------
	GA statistic for a given combination.
	
	    
	"""
	ipy_str = str(type(get_ipython()))
	if 'zmqshell' in ipy_str:
	  #print('jupyter')
	  DIR = os.getcwd()
	  DIR2 = '/Workspace/Processing/Interp_proc/'
	  if 'Interp_proc' in DIR:
	    os.chdir(DIR)
	  else:
	    os.chdir(DIR + DIR2)
	
	#DIR = os.getcwd()
	#DIR2 = '/Workspace/Processing/Interp_proc/'
	#os.chdir(DIR+DIR2)
	
	if savepdf == False:
		string_c = 'pyevolve_graph.py -i'+' '+'eniigma_'+str(combination)+' '+str(option)+' '+'-m'	
		os.system(string_c)
	else:
		string_c = 'pyevolve_graph.py -i'+' '+'eniigma_'+str(combination)+' '+str(option)+' '+'-m'+' '+'-o'+' '+'graph_eniigma'+' '+'-e'+' '+'pdf'
		os.system(string_c)

