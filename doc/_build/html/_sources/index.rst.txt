.. ENIIGMA code documentation master file, created by
   sphinx-quickstart on Wed May  6 10:15:43 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ENIIGMA code's documentation!
========================================
Introduction
------------

ENIIGMA is a python code focused on the dEcomposition of Infrared Ice Spectra using Genetic Modelling Algorithm.

An infrared ice spectra library is included in the code directory and used to decompose either observational or experimental ice spectra.

The genetic algorithm module run the Pyevolve Python package and give the global minimum solution for the spectral decomposition after several generations.

Once the ENIIGMA code has finished the spectral decomposition, an statistical is called to perform 1. Confidence analysis interval, 2. Ice column density calculation,
and 3. Degeneracy analysys by Pie chart and histograms.

Other functionalities of the ENIIGMA code regards to the spectral continuum calculation by polynomial and blackbody fitting.  

Features:
-----------

The ENIIGMA code is focused on the spectral decomposition of the observational IR spectra containing absorption ice features. So far, the *Spitzer* and VLT/ISAAC spectra
have been successfully tested. In this regard, this code will be useful to provide an unbiased analysis of the upcoming *James Webb Space Telescope - JWST* data. Nevertheless,
laboratory ice spectra can be decomposed using the ENIIGMA code. The current and planned technical features are listed below:




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
