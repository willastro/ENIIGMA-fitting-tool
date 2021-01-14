.. _settingup:


Before you start
===============================


Required Python packages:
-------------------------

* Python (http://www.python.org/): tested with 2.7.10 and 3.7.7\
* Numpy (http://www.numpy.org): tested with 1.16+\
* Pandas (https://pandas.pydata.org): tested with 0.25+\
* LMFIT (https://lmfit.github.io/lmfit-py/index.html): tested with 0.9+\
* Pyevolve - Python 2.7 (http://pyevolve.sourceforge.net/0_6rc1/index.html): tested with 0.6+\
* Pyevolve - Python 3+ (https://github.com/BubaVV/Pyevolve): tested with 0.6+\
* Matplotlib (https://matplotlib.org): tested with 2.2.3 and 3.0.2\
* Scipy (https://www.scipy.org): tested with 1.1+\
* sh (https://pypi.org/project/sh/): tested with 1.12.14


Installation:
-------------------------

The installation can be done by typing the following command in shell:
::

	$pip install -i https://test.pypi.org/simple/ eniigma-try007==0.0.1

.. note:: Obs: Pyevolve in Python 3+ must be downloaded and installed via the following commands:
::

	$git clone https://github.com/BubaVV/Pyevolve.git
	$cd Pyevolve
	$pip3 install future
	$sudo python setup.py install --

Development version:
-------------------------

The ENIIGMA code can also be installed from GitHub:
::
	$git clone https://github.com/willastro/ENIIGMA-code.git
	$cd ENIIGMA-code
	$python setup.py install
	$python setup.py install --user (for non-root privilegies)
