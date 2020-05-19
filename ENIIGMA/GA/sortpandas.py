import pandas as pd

def sortt(file):
	"""
	Function to sort the 'Best_comb.csv file by the chi_square value.'
	
	Parameters
	-------------
	
	file : 'str'
		Path to the 'Best_comb.csv' file. Taken automatically.
	
	Returns
	-------------
	'Best_comb.csv' file sorted by the chi_square.
	
	    
	"""
	df0 = pd.read_csv(file,sep=',')
	df0.sort_values(by=['best_chi','index'],  inplace=True)
	df0.to_csv("Best_comb.csv", index=False, encoding='utf-8-sig')


