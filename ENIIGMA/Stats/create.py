import pandas as pd
import numpy as np
import os
import glob
import sys

def create_file2(file_list):
	"""
	Function to merge ascii files.
	
	Parameters
	-------------
	
	file_list : 'str'
		Path to the files. Taken automatically.
	
	Returns
	-------------
	Merged file: output_file.txt
	
	    
	"""
	with open('output_file.txt', 'w') as file3:
		readers = [open(file) for file in file_list]
		#print readers
		for lines in zip(*readers):
			print >>file3, ' '.join([line.strip() for line in lines])

def create_file2f(file_list):
	"""
	Function to merge ascii files.
	
	Parameters
	-------------
	
	file_list : 'str'
		Path to the files. Taken automatically.
	
	Returns
	-------------
	Merged file: output_file_final.txt
	
	    
	"""
	with open('output_file_final.txt', 'w') as file3:
		readers = [open(file) for file in file_list]
		#print readers
		for lines in zip(*readers):
			print >>file3, ' '.join([line.strip() for line in lines])

def create_file2four(file_list):
	"""
	Function to merge ascii files.
	
	Parameters
	-------------
	
	file_list : 'str'
		Path to the files. Taken automatically.
	
	Returns
	-------------
	Merged file: output_file4.txt
	
	    
	"""
	with open('output_file4.txt', 'w') as file3:
		readers = [open(file) for file in file_list]
		#print readers
		for lines in zip(*readers):
			print >>file3, ' '.join([line.strip() for line in lines])


def create_interp2(file_list):
	"""
	Function to merge ascii files.
	
	Parameters
	-------------
	
	file_list : 'str'
		Path to the files. Taken automatically.
	
	Returns
	-------------
	Merged file: interp_all.txt
	
	    
	"""
	with open('interp_all.txt', 'w') as file3:
		readers = [open(file) for file in file_list]
		for lines in zip(*readers):
			print >>file3, ' '.join([line.strip() for line in lines])
			
def create_R2(file_list):
	"""
	Function to merge ascii files.
	
	Parameters
	-------------
	
	file_list : 'str'
		Path to the files. Taken automatically.
	
	Returns
	-------------
	Merged file: all_T.txt
	
	    
	"""
	with open('all_R.txt', 'w') as file3:
		readers = [open(file) for file in file_list]
		for lines in zip(*readers):
			print >>file3, ' '.join([line.strip() for line in lines])