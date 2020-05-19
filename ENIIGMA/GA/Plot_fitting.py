import matplotlib.pyplot as plt
import pandas as pd
import sys

def plot(DIR):
	"""
	Function to read the global minumum outputs and plot the result
	
	Parameters
	-------------
	
	filename : 'str'
		Path to the files. Taken automatically.
	
	    
	"""
	#DIR = '/Users/will_rocha_starplan/Downloads/Data_StarPlan_Project/Fitting/WLT0/WL17_5_8/Workspace/Processing/Interp_proc/'
	df = pd.read_csv(DIR+'Interp_proc/Best_comb.csv',sep=',', header=1)
	n_genes = df.shape[1] - 3 # number of genes
		
	data = pd.read_csv(DIR+'Interp_proc/Best_comb.csv',sep=',', usecols=['name'], nrows=n_genes)#pd.read_csv((DIR+'Interp_proc/Best_comb.csv', delimiter=",", low_memory=True, usecols=[2], nrows=n_genes)
	spn = DIR+'Store_interp/'+data+'.dat'
	list = spn.T.values.tolist()[0]
		
	if sys.version_info[0] == 3:
		from ENIIGMA.GA import create3
		create3.create_file3f(list)
	else:
		import create
		create.create_file2f(list)
	
	header = []
	for h in range(n_genes):
		header.append('w'+str(h+1))
		
	data = pd.read_csv(DIR+'Interp_proc/Best_comb.csv', sep=',', low_memory=True, usecols=header, nrows=1)
	cmin = data.T.values.tolist()
		
	t1 = pd.read_csv(DIR+'Interp_proc/output_file_final.txt',sep='\s+', header=None)
	Ysp = pd.read_csv(DIR+'Interp_proc/output_file_final.txt',sep='\s+', header=None, usecols=range(1,t1.shape[1],2))
		
	yff = 0.
	crange = range(n_genes)
	ysprange = range(1,t1.shape[1],2)
	for i,j in zip(crange,ysprange):
		#print i,j
		yff += cmin[i]*Ysp[j]
	
	plt.plot(t1[0], yff, color='limegreen', label='Model')
	plt.legend()
