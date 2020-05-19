import pandas as pd
import numpy as np
import os
import glob
import sys

def create_file3(file_list):
	"""
	Function to merge ascii files in Python 3.
	
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
			print(' '.join([line.strip() for line in lines]), file=file3)

def create_file3f(file_list):
	"""
	Function to merge ascii files in Python 3.
	
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
			print(' '.join([line.strip() for line in lines]), file=file3)

def create_file3four(file_list):
	"""
	Function to merge ascii files in Python 3.
	
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
			print(' '.join([line.strip() for line in lines]), file=file3)


def create_interp3(file_list):
	"""
	Function to merge ascii files in Python 3.
	
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
			print(' '.join([line.strip() for line in lines]), file=file3)

def create_R3(file_list):
	"""
	Function to merge ascii files in Python 3.
	
	Parameters
	-------------
	
	file_list : 'str'
		Path to the files. Taken automatically.
	
	Returns
	-------------
	Merged file: all_R.txt
	
	    
	"""
	with open('all_R.txt', 'w') as file3:
		readers = [open(file) for file in file_list]
		for lines in zip(*readers):
			print(' '.join([line.strip() for line in lines]), file=file3)