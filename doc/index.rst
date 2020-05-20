.. ENIIGMA code documentation master file, created by
   sphinx-quickstart on Wed May  6 10:15:43 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ENIIGMA code's documentation!
========================================
Introduction
------------

ENIIGMA is a Python code for the d(**E**)compositio(**N**) of (**I**)nfrared (**I**)ce spectra using (**G**)enetic (**M**)odelling Algorithm.

An internal library containing infrared spectrum of ice data taken from public databases is used to deompose the input spectra.

The genetic algorithm module run the Pyevolve Python package and give the global minimum solution for the spectral decomposition after several generations. More about `Pyevolve <http://pyevolve.sourceforge.net/0_6rc1/>`_.

Once the ENIIGMA code has finished the spectral decomposition, an statistical can be performed to calculate: 1. the confidence interval, 2. the ice column density,
and 3. the degeneracy using pie chart and histograms.

Other functionalities of the ENIIGMA code regards to the spectral continuum calculation by polynomial and blackbody fitting.  

Features:
-----------

The ENIIGMA code is focused on the spectral decomposition of the observational IR spectra containing absorption ice features. So far, the *Spitzer* and VLT/ISAAC spectra
have been successfully tested. In this regard, this code will be useful to provide an unbiased analysis of the upcoming *James Webb Space Telescope - JWST* data. Nevertheless,
laboratory ice spectra can be decomposed using the ENIIGMA code.

Citing ENIIGMA
------------

If you use ENIIGMA code in your work, please cite Rocha et al. (in prep).

Citing ENIIGMA
------------

If you have questions or concerns regarding the code, please open an issue at https://github.com/willastro/ENIIGMA-code/issues.

The ENIIGMA team
----------------

* Will Rocha
* Giulia Perotti
* Lars E. Kristensen
* Jes K. J(\/o)rgensen



.. toctree::
   :maxdepth:2
   :hidden:
   
   beforeyoubegin
   
   Import_modules
   
   Examples 




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
