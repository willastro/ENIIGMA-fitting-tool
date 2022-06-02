import glob
from pandas import DataFrame
import numpy as np
import os
from scipy.optimize import curve_fit
import scipy.optimize as optimize
import math
import operator
import scipy
from scipy.integrate import trapz, simps
import csv


################################LORENTZIAN FUNCTION##############################
def asym_peak_L(t, pars):
    """
    Lorentizian function.

    Parameters
    -------------

    t : 'array'

    pars : 'array'
        Parameters used in the decomposition. E.g. heigh, center, width.

    Returns
    -------------
    function
    """
    a0 = pars[0]  # height
    a1 = pars[1]  # center
    a2 = pars[2]  # width of gaussian
    # f = a0*np.exp(-(t - a1)**2/(2*a2**2)) #GAUSSIAN
    # f = (a0/np.pi)*(0.5*a2)/((t-a1)**2 + (0.5*a2)**2) #LORENTZIAN
    f = a0 * ((a2 ** 2) / ((t - a1) ** 2 + (1.0 * a2) ** 2))  # LORENTZIAN
    return f


def two_peaks_L(t, *pars):
    'function of two overlapping peaks'
    a10 = pars[0]
    a11 = pars[1]
    a12 = pars[2]
    a20 = pars[3]
    a21 = pars[4]
    a22 = pars[5]
    p1 = asym_peak_L(t, [a10, a11, a12])
    p2 = asym_peak_L(t, [a20, a21, a22])
    return p1 + p2


def three_peaks_L(t, *pars):
    'function of three overlapping peaks'
    a10 = pars[0]
    a11 = pars[1]
    a12 = pars[2]
    a20 = pars[3]
    a21 = pars[4]
    a22 = pars[5]
    a30 = pars[6]
    a31 = pars[7]
    a32 = pars[8]
    p1 = asym_peak_L(t, [a10, a11, a12])
    p2 = asym_peak_L(t, [a20, a21, a22])
    p3 = asym_peak_L(t, [a30, a31, a32])
    return p1 + p2 + p3


def four_peaks_L(t, *pars):
    'function of four overlapping peaks'
    a10 = pars[0]
    a11 = pars[1]
    a12 = pars[2]
    a20 = pars[3]
    a21 = pars[4]
    a22 = pars[5]
    a30 = pars[6]
    a31 = pars[7]
    a32 = pars[8]
    a40 = pars[9]
    a41 = pars[10]
    a42 = pars[11]
    p1 = asym_peak_L(t, [a10, a11, a12])
    p2 = asym_peak_L(t, [a20, a21, a22])
    p3 = asym_peak_L(t, [a30, a31, a32])
    p4 = asym_peak_L(t, [a40, a41, a42])
    return p1 + p2 + p3 + p4


def five_peaks_L(t, *pars):
    'function of five overlapping peaks'
    a10 = pars[0]
    a11 = pars[1]
    a12 = pars[2]
    a20 = pars[3]
    a21 = pars[4]
    a22 = pars[5]
    a30 = pars[6]
    a31 = pars[7]
    a32 = pars[8]
    a40 = pars[9]
    a41 = pars[10]
    a42 = pars[11]
    a50 = pars[12]
    a51 = pars[13]
    a52 = pars[14]
    p1 = asym_peak_L(t, [a10, a11, a12])
    p2 = asym_peak_L(t, [a20, a21, a22])
    p3 = asym_peak_L(t, [a30, a31, a32])
    p4 = asym_peak_L(t, [a40, a41, a42])
    p5 = asym_peak_L(t, [a50, a51, a52])
    return p1 + p2 + p3 + p4 + p5


def six_peaks_L(t, *pars):
    'function of six overlapping peaks'
    a10 = pars[0]
    a11 = pars[1]
    a12 = pars[2]
    a20 = pars[3]
    a21 = pars[4]
    a22 = pars[5]
    a30 = pars[6]
    a31 = pars[7]
    a32 = pars[8]
    a40 = pars[9]
    a41 = pars[10]
    a42 = pars[11]
    a50 = pars[12]
    a51 = pars[13]
    a52 = pars[14]
    a60 = pars[15]
    a61 = pars[16]
    a62 = pars[17]
    p1 = asym_peak_L(t, [a10, a11, a12])
    p2 = asym_peak_L(t, [a20, a21, a22])
    p3 = asym_peak_L(t, [a30, a31, a32])
    p4 = asym_peak_L(t, [a40, a41, a42])
    p5 = asym_peak_L(t, [a50, a51, a52])
    p6 = asym_peak_L(t, [a60, a61, a62])
    return p1 + p2 + p3 + p4 + p5 + p6


def seven_peaks_L(t, *pars):
    'function of seven overlapping peaks'
    a10 = pars[0]
    a11 = pars[1]
    a12 = pars[2]
    a20 = pars[3]
    a21 = pars[4]
    a22 = pars[5]
    a30 = pars[6]
    a31 = pars[7]
    a32 = pars[8]
    a40 = pars[9]
    a41 = pars[10]
    a42 = pars[11]
    a50 = pars[12]
    a51 = pars[13]
    a52 = pars[14]
    a60 = pars[15]
    a61 = pars[16]
    a62 = pars[17]
    a70 = pars[18]
    a71 = pars[19]
    a72 = pars[20]
    p1 = asym_peak_L(t, [a10, a11, a12])
    p2 = asym_peak_L(t, [a20, a21, a22])
    p3 = asym_peak_L(t, [a30, a31, a32])
    p4 = asym_peak_L(t, [a40, a41, a42])
    p5 = asym_peak_L(t, [a50, a51, a52])
    p6 = asym_peak_L(t, [a60, a61, a62])
    p7 = asym_peak_L(t, [a70, a71, a72])
    return p1 + p2 + p3 + p4 + p5 + p6 + p7


def eight_peaks_L(t, *pars):
    'function of eight overlapping peaks'
    a10 = pars[0]
    a11 = pars[1]
    a12 = pars[2]
    a20 = pars[3]
    a21 = pars[4]
    a22 = pars[5]
    a30 = pars[6]
    a31 = pars[7]
    a32 = pars[8]
    a40 = pars[9]
    a41 = pars[10]
    a42 = pars[11]
    a50 = pars[12]
    a51 = pars[13]
    a52 = pars[14]
    a60 = pars[15]
    a61 = pars[16]
    a62 = pars[17]
    a70 = pars[18]
    a71 = pars[19]
    a72 = pars[20]
    a80 = pars[21]
    a81 = pars[22]
    a82 = pars[23]
    p1 = asym_peak_L(t, [a10, a11, a12])
    p2 = asym_peak_L(t, [a20, a21, a22])
    p3 = asym_peak_L(t, [a30, a31, a32])
    p4 = asym_peak_L(t, [a40, a41, a42])
    p5 = asym_peak_L(t, [a50, a51, a52])
    p6 = asym_peak_L(t, [a60, a61, a62])
    p7 = asym_peak_L(t, [a70, a71, a72])
    p8 = asym_peak_L(t, [a80, a81, a82])
    return p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8


def nine_peaks_L(t, *pars):
    'function of nine overlapping peaks'
    a10 = pars[0]
    a11 = pars[1]
    a12 = pars[2]
    a20 = pars[3]
    a21 = pars[4]
    a22 = pars[5]
    a30 = pars[6]
    a31 = pars[7]
    a32 = pars[8]
    a40 = pars[9]
    a41 = pars[10]
    a42 = pars[11]
    a50 = pars[12]
    a51 = pars[13]
    a52 = pars[14]
    a60 = pars[15]
    a61 = pars[16]
    a62 = pars[17]
    a70 = pars[18]
    a71 = pars[19]
    a72 = pars[20]
    a80 = pars[21]
    a81 = pars[22]
    a82 = pars[23]
    a90 = pars[24]
    a91 = pars[25]
    a92 = pars[26]
    p1 = asym_peak_L(t, [a10, a11, a12])
    p2 = asym_peak_L(t, [a20, a21, a22])
    p3 = asym_peak_L(t, [a30, a31, a32])
    p4 = asym_peak_L(t, [a40, a41, a42])
    p5 = asym_peak_L(t, [a50, a51, a52])
    p6 = asym_peak_L(t, [a60, a61, a62])
    p7 = asym_peak_L(t, [a70, a71, a72])
    p8 = asym_peak_L(t, [a80, a81, a82])
    p9 = asym_peak_L(t, [a90, a91, a92])
    return p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9


def ten_peaks_L(t, *pars):
    'function of ten overlapping peaks'
    a10 = pars[0]
    a11 = pars[1]
    a12 = pars[2]
    a20 = pars[3]
    a21 = pars[4]
    a22 = pars[5]
    a30 = pars[6]
    a31 = pars[7]
    a32 = pars[8]
    a40 = pars[9]
    a41 = pars[10]
    a42 = pars[11]
    a50 = pars[12]
    a51 = pars[13]
    a52 = pars[14]
    a60 = pars[15]
    a61 = pars[16]
    a62 = pars[17]
    a70 = pars[18]
    a71 = pars[19]
    a72 = pars[20]
    a80 = pars[21]
    a81 = pars[22]
    a82 = pars[23]
    a90 = pars[24]
    a91 = pars[25]
    a92 = pars[26]
    a100 = pars[27]
    a101 = pars[28]
    a102 = pars[29]
    p1 = asym_peak_L(t, [a10, a11, a12])
    p2 = asym_peak_L(t, [a20, a21, a22])
    p3 = asym_peak_L(t, [a30, a31, a32])
    p4 = asym_peak_L(t, [a40, a41, a42])
    p5 = asym_peak_L(t, [a50, a51, a52])
    p6 = asym_peak_L(t, [a60, a61, a62])
    p7 = asym_peak_L(t, [a70, a71, a72])
    p8 = asym_peak_L(t, [a80, a81, a82])
    p9 = asym_peak_L(t, [a90, a91, a92])
    p10 = asym_peak_L(t, [a100, a101, a102])
    return p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9 + p10


####################################################################################
####################################################################################
####################################################################################

###############################GAUSSIAN FUNCTION#####################################################

def asym_peak_G(t, pars):
    """
        Gaussian function.

        Parameters
        -------------

        t : 'array'

        pars : 'array'
                Parameters used in the decomposition. E.g. heigh, center, width.

        Returns
        -------------
        function
        """
    a0 = pars[0]  # height
    a1 = pars[1]  # center
    a2 = pars[2]  # width of gaussian
    f = a0 * np.exp(-(t - a1) ** 2 / (2 * a2 ** 2))  # GAUSSIAN
    # f = a0*((a2**2)/((t-a1)**2 + (1.0*a2)**2)) #LORENTZIAN
    return f


def two_peaks_G(t, *pars):
    'function of two overlapping peaks'
    a10 = pars[0]
    a11 = pars[1]
    a12 = pars[2]
    a20 = pars[3]
    a21 = pars[4]
    a22 = pars[5]
    p1 = asym_peak_G(t, [a10, a11, a12])
    p2 = asym_peak_G(t, [a20, a21, a22])
    return p1 + p2


def three_peaks_G(t, *pars):
    'function of three overlapping peaks'
    a10 = pars[0]
    a11 = pars[1]
    a12 = pars[2]
    a20 = pars[3]
    a21 = pars[4]
    a22 = pars[5]
    a30 = pars[6]
    a31 = pars[7]
    a32 = pars[8]
    p1 = asym_peak_G(t, [a10, a11, a12])
    p2 = asym_peak_G(t, [a20, a21, a22])
    p3 = asym_peak_G(t, [a30, a31, a32])
    return p1 + p2 + p3


def four_peaks_G(t, *pars):
    'function of four overlapping peaks'
    a10 = pars[0]
    a11 = pars[1]
    a12 = pars[2]
    a20 = pars[3]
    a21 = pars[4]
    a22 = pars[5]
    a30 = pars[6]
    a31 = pars[7]
    a32 = pars[8]
    a40 = pars[9]
    a41 = pars[10]
    a42 = pars[11]
    p1 = asym_peak_G(t, [a10, a11, a12])
    p2 = asym_peak_G(t, [a20, a21, a22])
    p3 = asym_peak_G(t, [a30, a31, a32])
    p4 = asym_peak_G(t, [a40, a41, a42])
    return p1 + p2 + p3 + p4


def five_peaks_G(t, *pars):
    'function of five overlapping peaks'
    a10 = pars[0]
    a11 = pars[1]
    a12 = pars[2]
    a20 = pars[3]
    a21 = pars[4]
    a22 = pars[5]
    a30 = pars[6]
    a31 = pars[7]
    a32 = pars[8]
    a40 = pars[9]
    a41 = pars[10]
    a42 = pars[11]
    a50 = pars[12]
    a51 = pars[13]
    a52 = pars[14]
    p1 = asym_peak_G(t, [a10, a11, a12])
    p2 = asym_peak_G(t, [a20, a21, a22])
    p3 = asym_peak_G(t, [a30, a31, a32])
    p4 = asym_peak_G(t, [a40, a41, a42])
    p5 = asym_peak_G(t, [a50, a51, a52])
    return p1 + p2 + p3 + p4 + p5


def six_peaks_G(t, *pars):
    'function of six overlapping peaks'
    a10 = pars[0]
    a11 = pars[1]
    a12 = pars[2]
    a20 = pars[3]
    a21 = pars[4]
    a22 = pars[5]
    a30 = pars[6]
    a31 = pars[7]
    a32 = pars[8]
    a40 = pars[9]
    a41 = pars[10]
    a42 = pars[11]
    a50 = pars[12]
    a51 = pars[13]
    a52 = pars[14]
    a60 = pars[15]
    a61 = pars[16]
    a62 = pars[17]
    p1 = asym_peak_G(t, [a10, a11, a12])
    p2 = asym_peak_G(t, [a20, a21, a22])
    p3 = asym_peak_G(t, [a30, a31, a32])
    p4 = asym_peak_G(t, [a40, a41, a42])
    p5 = asym_peak_G(t, [a50, a51, a52])
    p6 = asym_peak_G(t, [a60, a61, a62])
    return p1 + p2 + p3 + p4 + p5 + p6


def seven_peaks_G(t, *pars):
    'function of seven overlapping peaks'
    a10 = pars[0]
    a11 = pars[1]
    a12 = pars[2]
    a20 = pars[3]
    a21 = pars[4]
    a22 = pars[5]
    a30 = pars[6]
    a31 = pars[7]
    a32 = pars[8]
    a40 = pars[9]
    a41 = pars[10]
    a42 = pars[11]
    a50 = pars[12]
    a51 = pars[13]
    a52 = pars[14]
    a60 = pars[15]
    a61 = pars[16]
    a62 = pars[17]
    a70 = pars[18]
    a71 = pars[19]
    a72 = pars[20]
    p1 = asym_peak_G(t, [a10, a11, a12])
    p2 = asym_peak_G(t, [a20, a21, a22])
    p3 = asym_peak_G(t, [a30, a31, a32])
    p4 = asym_peak_G(t, [a40, a41, a42])
    p5 = asym_peak_G(t, [a50, a51, a52])
    p6 = asym_peak_G(t, [a60, a61, a62])
    p7 = asym_peak_G(t, [a70, a71, a72])
    return p1 + p2 + p3 + p4 + p5 + p6 + p7


def eight_peaks_G(t, *pars):
    'function of eight overlapping peaks'
    a10 = pars[0]
    a11 = pars[1]
    a12 = pars[2]
    a20 = pars[3]
    a21 = pars[4]
    a22 = pars[5]
    a30 = pars[6]
    a31 = pars[7]
    a32 = pars[8]
    a40 = pars[9]
    a41 = pars[10]
    a42 = pars[11]
    a50 = pars[12]
    a51 = pars[13]
    a52 = pars[14]
    a60 = pars[15]
    a61 = pars[16]
    a62 = pars[17]
    a70 = pars[18]
    a71 = pars[19]
    a72 = pars[20]
    a80 = pars[21]
    a81 = pars[22]
    a82 = pars[23]
    p1 = asym_peak_G(t, [a10, a11, a12])
    p2 = asym_peak_G(t, [a20, a21, a22])
    p3 = asym_peak_G(t, [a30, a31, a32])
    p4 = asym_peak_G(t, [a40, a41, a42])
    p5 = asym_peak_G(t, [a50, a51, a52])
    p6 = asym_peak_G(t, [a60, a61, a62])
    p7 = asym_peak_G(t, [a70, a71, a72])
    p8 = asym_peak_G(t, [a80, a81, a82])
    return p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8


def nine_peaks_G(t, *pars):
    'function of nine overlapping peaks'
    a10 = pars[0]
    a11 = pars[1]
    a12 = pars[2]
    a20 = pars[3]
    a21 = pars[4]
    a22 = pars[5]
    a30 = pars[6]
    a31 = pars[7]
    a32 = pars[8]
    a40 = pars[9]
    a41 = pars[10]
    a42 = pars[11]
    a50 = pars[12]
    a51 = pars[13]
    a52 = pars[14]
    a60 = pars[15]
    a61 = pars[16]
    a62 = pars[17]
    a70 = pars[18]
    a71 = pars[19]
    a72 = pars[20]
    a80 = pars[21]
    a81 = pars[22]
    a82 = pars[23]
    a90 = pars[24]
    a91 = pars[25]
    a92 = pars[26]
    p1 = asym_peak_G(t, [a10, a11, a12])
    p2 = asym_peak_G(t, [a20, a21, a22])
    p3 = asym_peak_G(t, [a30, a31, a32])
    p4 = asym_peak_G(t, [a40, a41, a42])
    p5 = asym_peak_G(t, [a50, a51, a52])
    p6 = asym_peak_G(t, [a60, a61, a62])
    p7 = asym_peak_G(t, [a70, a71, a72])
    p8 = asym_peak_G(t, [a80, a81, a82])
    p9 = asym_peak_G(t, [a90, a91, a92])
    return p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9


def ten_peaks_G(t, *pars):
    'function of ten overlapping peaks'
    a10 = pars[0]
    a11 = pars[1]
    a12 = pars[2]
    a20 = pars[3]
    a21 = pars[4]
    a22 = pars[5]
    a30 = pars[6]
    a31 = pars[7]
    a32 = pars[8]
    a40 = pars[9]
    a41 = pars[10]
    a42 = pars[11]
    a50 = pars[12]
    a51 = pars[13]
    a52 = pars[14]
    a60 = pars[15]
    a61 = pars[16]
    a62 = pars[17]
    a70 = pars[18]
    a71 = pars[19]
    a72 = pars[20]
    a80 = pars[21]
    a81 = pars[22]
    a82 = pars[23]
    a90 = pars[24]
    a91 = pars[25]
    a92 = pars[26]
    a100 = pars[27]
    a101 = pars[28]
    a102 = pars[29]
    p1 = asym_peak_G(t, [a10, a11, a12])
    p2 = asym_peak_G(t, [a20, a21, a22])
    p3 = asym_peak_G(t, [a30, a31, a32])
    p4 = asym_peak_G(t, [a40, a41, a42])
    p5 = asym_peak_G(t, [a50, a51, a52])
    p6 = asym_peak_G(t, [a60, a61, a62])
    p7 = asym_peak_G(t, [a70, a71, a72])
    p8 = asym_peak_G(t, [a80, a81, a82])
    p9 = asym_peak_G(t, [a90, a91, a92])
    p10 = asym_peak_G(t, [a100, a101, a102])
    return p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9 + p10


####################################################################################

def get_line_number2(value, matrix):
    """
    Function used to get the line number of a value in a array.

    Parameters
    -------------

    value : 'float'
        Value of interest.

    matrix : 'numpy.ndarray'
        Vector with the value of interest.

    Returns
    ------------

    Index

    """
    for i, line in enumerate(matrix, 1):
        if line == value:
            return i


def find_nearest(array, value):
    """
        Find nearest value of a given number.

        Parameters
        -------------

        array : 'numpy.ndarray'
                Vector with the value of interest.

        value : 'float'
                Value of interest.

        Returns
        -------------
        Nearest value

        """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


####################################################################################
####################################################################################

def deconv_all(filecomp):
    """
    Decomposition of the laboratory data after the optimization.

    Parameters
    -------------

    filecomp : 'str'
        Path to the file containing the components of the fitting.

    A_H2O = 1.1e-17 #6.0 microns taken from Bouilloud+2015
    A_H2O_2 = 3.2e-17 #13.2 microns taken from Bouilloud+2015
    A_NH3 = 2.1e-17 #nu_2 v.mode (Umbrella - 9.34 microns) taken from Bouilloud+2015
    A_NH3_2 = 5.6e-18 #6.15 microns taken from Bouilloud+2015
    A_NH4 = 3.0e-17 #(Schutte & Khanna 2003)
    A_HCOOH = 2.9e-17#8.22 microns taken from Bouilloud+2015
    A_H2CO = 9.6e-18# 5.81 Schutte+1993
    A_CH3OH = 1.78e-17#9.74 microns taken from Bouilloud+2015
    A_CH3CN = 2.2e-18#Acetonitrile (Paper dHendecourt 1986)
    A_CH3COOH = 4.77e-17#Acetic Acid (Paper for Methyl formate Modica & Palumbo 2010)
    A_C3H6O = 1.50e-17#Acetone(Tabela)
    A_CH3CH2OH = 1.41e-17##Ethanol - Value for Ethanol mixed with CO(Terwisscha van Scheltinga 2018): 9.17 microns
    A_CH3OCH3 = 9.8e-18# Terwisscha van Scheltinga 2018): 8.58 microns
    A_C6H12 = 1.00e-17#cyclohexane - Assuming
    A_CH3CHO = 1.3e-17#Acetaldehyde - Terwisscha_van_Scheltinga2018 (Table 1)
    A_CH4 = 8.4e-18#7.68 microns taken from Bouilloud+2015
    A_COO = 6e-17 # Peak in 7.42 ----> Band strengh assumed to be the same of the feature in 6.03 microns (Taken from Caro&Schutte2003)


    Returns
    -------------
    Column densities

    """
    # Band strength values for the range between 1000 and 2000 cm-1.
    A_H2O = 1.1e-17  # 6.0 microns taken from Bouilloud+2015
    A_H2O_2 = 3.2e-17  # 13.2 microns taken from Bouilloud+2015
    A_H2O_3mic = 2.0e-16  # Peak at 3.0 ----> Taken from Gerakines1995
    # nu_2 v.mode (Umbrella - 9.34 microns) taken from Bouilloud+2015
    A_NH3 = 2.1e-17
    A_NH3_2 = 5.6e-18  # 6.15 microns taken from Bouilloud+2015
    A_NH4 = 4.0e-17  # (Schutte & Khanna 2003)
    A_HCOOH = 2.9e-17  # 8.22 microns taken from Bouilloud+2015
    A_H2CO = 9.6e-18  # 5.81 Schutte+1993
    A_CH3OH = 1.78e-17  # 9.74 microns taken from Bouilloud+2015
    A_CH3CN = 2.2e-18  # Acetonitrile (Paper dHendecourt 1986)
    # Acetic Acid (Paper for Methyl formate Modica & Palumbo 2010)
    A_CH3COOH = 4.77e-17
    A_C3H6O = 1.50e-17  # Acetone(Tabela)
    # Ethanol - Value for Ethanol mixed with CO(Terwisscha van Scheltinga 2018): 9.17 microns
    A_CH3CH2OH = 1.41e-17
    A_CH3OCH3 = 9.8e-18  # Terwisscha van Scheltinga 2018): 8.58 microns
    A_C6H12 = 1.00e-17  # cyclohexane - Assuming
    # Acetaldehyde - Terwisscha_van_Scheltinga2018 (Table 1)
    A_CH3CHO = 1.3e-17
    A_CH4 = 8.4e-18  # 7.68 microns taken from Bouilloud+2015
    # Peak in 7.42 ----> Band strengh assumed to be the same of the feature in 6.03 microns (Taken from Caro&Schutte2003)
    A_COO = 6e-17
    A_CO2 = 7.6e-17  # Peak at 4.26 ----> Taken from Gerakines1995
    A_CO = 1.1e-17  # Peak at 4.67 ----> Taken from Gerakines1995
    ###########################

    fil = filecomp  # '/Users/will_rocha_starplan/Downloads/Data_StarPlan_Project/Fitting/CRBR/CRBR_5_8/Workspace/Processing/Interp_proc/Components.csv'
    f = open(fil, 'r')
    reader = csv.reader(f)
    spn = next(reader, None)

    dat = np.loadtxt(fil, skiprows=1, delimiter=',')

    try:

        for j in range(1, len(spn) - 1):
            ntt = dat[:, 0]
            y = dat[:, j]

            ntt_cm = 1e4 * (1. / ntt)

            if spn[j] == 'HCOOH':
                P = scipy.integrate.trapz(y, ntt_cm)
                N_HCOOH = -P / A_HCOOH
                Data1 = {'HCOOH_pure': N_HCOOH}
                df1 = DataFrame(Data1, columns=['HCOOH_pure'], index=[0])
                df1.to_csv('Column_density_' +
                           'HCOOH_pure' + '.csv', index=False)
            elif spn[j] == 'H2CO':
                P = scipy.integrate.trapz(y, ntt_cm)
                N_H2CO = -P / A_H2CO
                Data1 = {'H2CO_pure': N_H2CO}
                df1 = DataFrame(Data1, columns=['H2CO_pure'], index=[0])
                df1.to_csv('Column_density_' +
                           'H2CO_pure' + '.csv', index=False)
            elif spn[j] == 'HCOOH_10K':
                P = scipy.integrate.trapz(y, ntt_cm)
                N_HCOOH = -P / A_HCOOH
                Data1 = {'HCOOH_pure': N_HCOOH}
                df1 = DataFrame(Data1, columns=['HCOOH_pure'], index=[0])
                df1.to_csv('Column_density_' +
                           'HCOOH_pure' + '.csv', index=False)
            elif spn[j] == 'NH3':
                P = scipy.integrate.trapz(y, ntt_cm)
                N_NH3 = -P / A_NH3
                Data1 = {'NH3_pure': N_NH3}
                df1 = DataFrame(Data1, columns=['NH3_pure'], index=[0])
                df1.to_csv('Column_density_' +
                           'NH3_pure' + '.csv', index=False)
            elif spn[j] == 'H2Oc':
                P = scipy.integrate.trapz(y, ntt_cm)
                N_H2Oc = -P / A_H2O
                Data1 = {'H2O_pure': N_H2Oc}
                df1 = DataFrame(Data1, columns=['H2O_pure'], index=[0])
                df1.to_csv('Column_density_' +
                           'H2O_pure' + '.csv', index=False)
            elif spn[j] == 'C3H6O':
                P = scipy.integrate.trapz(y, ntt_cm)
                N_C3H6O = -P / A_C3H6O
                Data1 = {'C3H6O_pure': N_C3H6O}
                df1 = DataFrame(Data1, columns=['C3H6O_pure'], index=[0])
                df1.to_csv('Column_density_' +
                           'C3H6O_pure' + '.csv', index=False)
            elif spn[j] == 'CH3COOH':
                P = scipy.integrate.trapz(y, ntt_cm)
                N_CH3COOH = -P / A_CH3COOH
                Data1 = {'CH3COOH_pure': N_CH3COOH}
                df1 = DataFrame(Data1, columns=['CH3COOH_pure'], index=[0])
                df1.to_csv('Column_density_' +
                           'CH3COOH_pure' + '.csv', index=False)
            elif spn[j] == 'CH3CH2OH':
                P = scipy.integrate.trapz(y, ntt_cm)
                N_CH3CH2OH = -P / A_CH3CH2OH
                Data1 = {'CH3CH2OH_pure': N_CH3CH2OH}
                df1 = DataFrame(Data1, columns=['CH3CH2OH_pure'], index=[0])
                df1.to_csv('Column_density_' +
                           'CH3CH2OH_pure' + '.csv', index=False)
            elif spn[j] == 'c-C6H12':
                P = scipy.integrate.trapz(y, ntt_cm)
                N_C6H12 = -P / A_C6H12
                Data1 = {'C6H12_pure': N_C6H12}
                df1 = DataFrame(Data1, columns=['C6H12_pure'], index=[0])
                df1.to_csv('Column_density_' +
                           'c-C6H12_pure' + '.csv', index=False)
            elif spn[j] == 'CH3OH':
                P = scipy.integrate.trapz(y, ntt_cm)
                N_CH3OH = -P / A_CH3OH
                Data1 = {'CH3OH_pure': N_CH3OH}
                df1 = DataFrame(Data1, columns=['CH3OH_pure'], index=[0])
                df1.to_csv('Column_density_' +
                           'CH3OH_pure' + '.csv', index=False)

            elif spn[j] == 'CH3OH_120K_NASA':
                P = scipy.integrate.trapz(y, ntt_cm)
                N_CH3OH = -P / A_CH3OH
                Data1 = {'CH3OH_pure': N_CH3OH}
                df1 = DataFrame(Data1, columns=['CH3OH_pure'], index=[0])
                df1.to_csv('Column_density_' +
                           'CH3OH_pure_120K_NASA' + '.csv', index=False)

            elif spn[j] == 'CH3OCH3':
                P = scipy.integrate.trapz(y, ntt_cm)
                N_CH3OCH3 = -P / A_CH3OCH3
                Data1 = {'CH3OCH3_pure': N_CH3OCH3}
                df1 = DataFrame(Data1, columns=['CH3OCH3_pure'], index=[0])
                df1.to_csv('Column_density_' +
                           'CH3OCH3_pure' + '.csv', index=False)

            elif spn[j] == 'H2Oa':
                print('Decomposing H2Oa')
                if ntt[0] < 4.:
                    lam1, lam2, lam3 = 2.96, 3.07, 3.16
                    x1, x2, x3 = find_nearest(ntt, lam1), find_nearest(
                        ntt, lam2), find_nearest(ntt, lam3)
                    y1, y2, y3 = y[get_line_number2(x1, ntt)], y[get_line_number2(x2, ntt)], y[
                        get_line_number2(x3, ntt)]
                    guess = (y1, x1, 0.02, y2, x2, 0.02, y3, x3, 0.01)
                    popt, pcov = curve_fit(
                        three_peaks_G, ntt, y, guess, maxfev=5000)
                    pars10b = popt[0:3]
                    pars20b = popt[3:6]
                    pars30b = popt[6:9]
                    peak10b = asym_peak_G(ntt, pars10b)
                    peak20b = asym_peak_G(ntt, pars20b)
                    peak30b = asym_peak_G(ntt, pars30b)
                    tot = peak10b + peak20b + peak30b
                    P1 = scipy.integrate.trapz(peak10b, ntt_cm)
                    P2 = scipy.integrate.trapz(peak20b, ntt_cm)
                    P3 = scipy.integrate.trapz(peak30b, ntt_cm)
                    N1 = -P1 / (A_H2O_3mic)
                    N2 = -P2 / (A_H2O_3mic)
                    N3 = -P3 / (A_H2O_3mic)
                    N_H2O = N1 + N2 + N3
                    Data1 = {'H2O_pure': N_H2O}
                    df1 = DataFrame(Data1, columns=['H2O_pure'], index=[0])
                    df1.to_csv('Column_density_' +
                               'H2O_pure' + '.csv', index=False)
                    np.savetxt('Analytic_deconv_H2O_pure', np.transpose(
                        [ntt, peak10b, peak20b, peak30b, tot, y]))

                elif ntt[0] > 4. and ntt[0] < 8.:
                    lam1, lam2, lam3 = 5.98, 6.24, 7.02
                    x1, x2, x3 = find_nearest(ntt, lam1), find_nearest(
                        ntt, lam2), find_nearest(ntt, lam3)
                    y1, y2, y3 = y[get_line_number2(x1, ntt)], y[get_line_number2(x2, ntt)], y[
                        get_line_number2(x3, ntt)]
                    guess = (y1, x1, 0.3, y2, x2, 0.7, y3, x3, 1.0)
                    popt, pcov = curve_fit(
                        three_peaks_G, ntt, y, guess, maxfev=5000)
                    pars10b = popt[0:3]
                    pars20b = popt[3:6]
                    pars30b = popt[6:9]
                    peak10b = asym_peak_G(ntt, pars10b)
                    peak20b = asym_peak_G(ntt, pars20b)
                    peak30b = asym_peak_G(ntt, pars30b)
                    tot = peak10b + peak20b + peak30b
                    P1 = scipy.integrate.trapz(peak10b, ntt_cm)
                    P2 = scipy.integrate.trapz(peak20b, ntt_cm)
                    P3 = scipy.integrate.trapz(peak30b, ntt_cm)
                    N1 = -P1 / (A_H2O)
                    N2 = -P2 / (A_H2O)
                    N3 = -P3 / (A_H2O)
                    N_H2O = N1 + N2 + N3
                    Data1 = {'H2O_pure': N_H2O}
                    df1 = DataFrame(Data1, columns=['H2O_pure'], index=[0])
                    df1.to_csv('Column_density_' +
                               'H2O_pure' + '.csv', index=False)
                    np.savetxt('Analytic_deconv_H2O_pure', np.transpose(
                        [ntt, peak10b, peak20b, peak30b, tot, y]))

                elif ntt[0] > 8. and ntt[0] < 15.:
                    lam1, lam2, lam3 = 12.4, 14.4, 17.0
                    x1, x2, x3 = find_nearest(ntt, lam1), find_nearest(
                        ntt, lam2), find_nearest(ntt, lam3)
                    y1, y2, y3 = y[get_line_number2(x1, ntt)], y[get_line_number2(x2, ntt)], y[
                        get_line_number2(x3, ntt)]
                    guess = (y1, x1, 0.3, y2, x2, 0.7, y3, x3, 1.0)
                    popt, pcov = curve_fit(
                        three_peaks_G, ntt, y, guess, maxfev=5000)
                    pars10b = popt[0:3]
                    pars20b = popt[3:6]
                    pars30b = popt[6:9]
                    peak10b = asym_peak_G(ntt, pars10b)
                    peak20b = asym_peak_G(ntt, pars20b)
                    peak30b = asym_peak_G(ntt, pars30b)
                    tot = peak10b + peak20b + peak30b
                    P1 = scipy.integrate.trapz(peak10b, ntt_cm)
                    P2 = scipy.integrate.trapz(peak20b, ntt_cm)
                    P3 = scipy.integrate.trapz(peak30b, ntt_cm)
                    N1 = -P1 / (A_H2O_2)
                    N2 = -P2 / (A_H2O_2)
                    N3 = -P3 / (A_H2O_2)
                    N_H2O = N1 + N2 + N3
                    Data1 = {'H2O_pure': N_H2O}
                    df1 = DataFrame(Data1, columns=['H2O_pure'], index=[0])
                    df1.to_csv('Column_density_' +
                               'H2O_pure' + '.csv', index=False)
                    np.savetxt('Analytic_deconv_H2O_pure', np.transpose(
                        [ntt, peak10b, peak20b, peak30b, tot, y]))

            elif spn[j] == 'H2O_NH3_CO2_CH4_10_1_1_1_35K_b':
                print('Decomposing H2O_NH3_CO2_CH4_10_1_1_1_35K_b')
                lam1, lam2, lam3, lam4, lam5, lam6, lam7, lam8, lam9, lam10 = 3.04, 3.38, 4.26, 4.36, 4.67, 6.04, 6.75, 7.68, 8.92, 9.8
                x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = find_nearest(ntt, lam1), find_nearest(ntt,
                                                                                                lam2), find_nearest(ntt,
                                                                                                                    lam3), find_nearest(
                    ntt, lam4), find_nearest(ntt, lam5), find_nearest(ntt, lam6), find_nearest(ntt, lam7), find_nearest(
                    ntt, lam8), find_nearest(ntt, lam9), find_nearest(ntt, lam10)
                y1, y2, y3, y4, y5, y6, y7, y8, y9, y10 = y[get_line_number2(x1, ntt)], y[get_line_number2(x2, ntt)], y[
                    get_line_number2(x3, ntt)], y[get_line_number2(x4, ntt)], y[get_line_number2(x5, ntt)], y[
                    get_line_number2(x6, ntt)], y[get_line_number2(x7, ntt)], \
                    y[get_line_number2(x8, ntt)], y[get_line_number2(x9, ntt)], y[
                    get_line_number2(x10, ntt)]
                guess = (
                    y1, x1, 0.1, y2, x2, 0.1, y3, x3, 0.01, y4, x4, 0.1, y5, x5, 0.05, y6, x6, 0.1, y7, x7, 0.1, y8, x8,
                    0.05, y9, x9, 0.1, y10, x10, 0.1)
                popt, pcov = curve_fit(ten_peaks_G, ntt, y, guess, maxfev=5000)
                pars10b = popt[0:3]
                pars20b = popt[3:6]
                pars30b = popt[6:9]
                pars40b = popt[9:12]
                pars50b = popt[12:15]
                pars60b = popt[15:18]
                pars70b = popt[18:21]
                pars80b = popt[21:24]
                pars90b = popt[24:27]
                pars100b = popt[27:30]
                peak10b = asym_peak_G(ntt, pars10b)
                peak20b = asym_peak_G(ntt, pars20b)
                peak30b = asym_peak_G(ntt, pars30b)
                peak40b = asym_peak_G(ntt, pars40b)
                peak50b = asym_peak_G(ntt, pars50b)
                peak60b = asym_peak_G(ntt, pars60b)
                peak70b = asym_peak_G(ntt, pars70b)
                peak80b = asym_peak_G(ntt, pars80b)
                peak90b = asym_peak_G(ntt, pars90b)
                peak100b = asym_peak_G(ntt, pars100b)
                tot = peak10b + peak20b + peak30b + peak40b + peak50b + \
                    peak60b + peak70b + peak80b + peak90b + peak100b
                P1 = scipy.integrate.trapz(peak10b, ntt_cm)
                P2 = scipy.integrate.trapz(peak20b, ntt_cm)
                P3 = scipy.integrate.trapz(peak30b, ntt_cm)
                P4 = scipy.integrate.trapz(peak40b, ntt_cm)
                P5 = scipy.integrate.trapz(peak50b, ntt_cm)
                P6 = scipy.integrate.trapz(peak60b, ntt_cm)
                P7 = scipy.integrate.trapz(peak70b, ntt_cm)
                P8 = scipy.integrate.trapz(peak80b, ntt_cm)
                P9 = scipy.integrate.trapz(peak90b, ntt_cm)
                P10 = scipy.integrate.trapz(peak100b, ntt_cm)
                N1 = -P1 / A_H2O_3mic
                N2 = -P2 / A_H2O_3mic
                N3 = -P3 / A_CO2
                N4 = 0.
                N5 = -P5 / A_CO
                N6 = -P6 / A_H2O
                N7 = 0.  # -P6/A_CH3OH
                N8 = -P8 / A_CH4
                N9 = -P9 / A_NH3
                N10 = -P10 / A_CH3OH
                N_H2O = N1 + N2
                N_CO2 = N3
                N_CO = N5
                N_CH4 = N8
                N_NH3 = N9
                N_CH3OH = N10
                Data1 = {'H2O_in_mix': N_H2O, 'NH3_in_mix': N_NH3, 'CH3OH_in_mix': N_CH3OH, 'CH4_in_mix': N_CH4,
                         'CO2_in_mix': N_CO2, 'CO_in_mix': N_CO}
                df1 = DataFrame(Data1, columns=['H2O_in_mix', 'NH3_in_mix', 'CH3OH_in_mix', 'CH4_in_mix', 'CO2_in_mix',
                                                'CO_in_mix'], index=[0])
                df1.to_csv('Column_density_' +
                           'H2O_NH3_CO2_CH4_10_1_1_1_35K_b' + '.csv', index=False)
                np.savetxt('Analytic_deconv_H2O_NH3_CO2_CH4_10_1_1_1_35K_b.dat', np.transpose(
                    [ntt, peak10b, peak20b, peak30b, peak40b, peak50b, peak60b, peak70b, peak80b, peak90b, peak100b,
                     tot, y]))

            elif spn[j] == 'Fig6_OR1t':
                print('Decomposing Fig6_OR1t')
                lam1, lam2, lam3, lam4, lam5, lam6, lam7, lam8, lam9 = 5.81, 5.92, 6.1, 6.27, 6.67, 6.81, 7.41, 7.64, 8.01
                x1, x2, x3, x4, x5, x6, x7, x8, x9 = find_nearest(ntt, lam1), find_nearest(ntt, lam2), find_nearest(ntt,
                                                                                                                    lam3), find_nearest(
                    ntt, lam4), find_nearest(ntt, lam5), find_nearest(ntt, lam6), find_nearest(ntt, lam7), find_nearest(
                    ntt, lam8), find_nearest(ntt, lam9)
                y1, y2, y3, y4, y5, y6, y7, y8, y9 = y[get_line_number2(x1, ntt)], y[get_line_number2(x2, ntt)], y[
                    get_line_number2(x3, ntt)], y[get_line_number2(x4, ntt)], y[get_line_number2(x5, ntt)], y[
                    get_line_number2(x6, ntt)], y[get_line_number2(x7, ntt)], y[
                    get_line_number2(x8, ntt)], y[get_line_number2(x9, ntt)]
                guess = (
                    y1, x1, 0.01, y2, x2, 0.05, y3, x3, 0.1, y4, x4, 0.05, y5, x5, 0.01, y6, x6, 0.1, y7, x7, 0.1, y8, x8,
                    0.1, y9, x9, 0.1)
                popt, pcov = curve_fit(
                    nine_peaks_G, ntt, y, guess, maxfev=5000)
                pars10b = popt[0:3]
                pars20b = popt[3:6]
                pars30b = popt[6:9]
                pars40b = popt[9:12]
                pars50b = popt[12:15]
                pars60b = popt[15:18]
                pars70b = popt[18:21]
                pars80b = popt[21:24]
                pars90b = popt[24:27]
                peak10b = asym_peak_G(ntt, pars10b)
                peak20b = asym_peak_G(ntt, pars20b)
                peak30b = asym_peak_G(ntt, pars30b)
                peak40b = asym_peak_G(ntt, pars40b)
                peak50b = asym_peak_G(ntt, pars50b)
                peak60b = asym_peak_G(ntt, pars60b)
                peak70b = asym_peak_G(ntt, pars70b)
                peak80b = asym_peak_G(ntt, pars80b)
                peak90b = asym_peak_G(ntt, pars90b)
                tot = peak10b + peak20b + peak30b + peak40b + \
                    peak50b + peak60b + peak70b + peak80b + peak90b
                P1 = scipy.integrate.trapz(peak10b, ntt_cm)
                P2 = scipy.integrate.trapz(peak20b, ntt_cm)
                P3 = scipy.integrate.trapz(peak30b, ntt_cm)
                P4 = scipy.integrate.trapz(peak40b, ntt_cm)
                P5 = scipy.integrate.trapz(peak50b, ntt_cm)
                P6 = scipy.integrate.trapz(peak60b, ntt_cm)
                P7 = scipy.integrate.trapz(peak70b, ntt_cm)
                P8 = scipy.integrate.trapz(peak80b, ntt_cm)
                P9 = scipy.integrate.trapz(peak90b, ntt_cm)
                N1 = -P1 / A_H2CO
                N2 = -P2 / A_H2CO  # CONH2
                N3 = -P3 / A_H2O
                N4 = 0.
                N5 = 0.
                N6 = -P6 / A_NH4
                N7 = 0.
                N8 = -P8 / A_CH4
                N9 = 0.
                N_H2CO = N1
                N_HCONH2 = N2
                N_H2O = N3
                N_NH4 = N6
                N_CH4 = N8
                Data1 = {'H2O_in_mix': N_H2O, 'H2CO_in_mix': N_H2CO, 'HCONH2_in_mix': N_HCONH2, 'CH4_in_mix': N_CH4,
                         'NH4_in_mix': N_NH4}
                df1 = DataFrame(Data1,
                                columns=['H2O_in_mix', 'H2CO_in_mix',
                                         'HCONH2_in_mix', 'CH4_in_mix', 'NH4_in_mix'],
                                index=[0])
                df1.to_csv('Column_density_' +
                           'Fig6_OR1t' + '.csv', index=False)
                np.savetxt('Analytic_deconv_Fig6_OR1t.dat', np.transpose(
                    [ntt, peak10b, peak20b, peak30b, peak40b, peak50b, peak60b, peak70b, peak80b, peak90b, tot, y]))

            elif spn[j] == 'H2O_CH3CH2OH_15.0K':
                print('Decomposing H2O_CH3CH2OH_15.0K')
                lam1, lam2, lam3, lam4, lam5, lam6, lam7, lam8, lam9 = 3.04, 3.38, 4.36, 6.04, 6.75, 9.2, 9.58, 12.0, 13.8
                x1, x2, x3, x4, x5, x6, x7, x8, x9 = find_nearest(ntt, lam1), find_nearest(ntt, lam2), find_nearest(ntt,
                                                                                                                    lam3), find_nearest(
                    ntt, lam4), find_nearest(ntt, lam5), find_nearest(ntt, lam6), find_nearest(ntt, lam7), find_nearest(
                    ntt, lam8), find_nearest(ntt, lam9)
                y1, y2, y3, y4, y5, y6, y7, y8, y9 = y[get_line_number2(x1, ntt)], y[get_line_number2(x2, ntt)], y[
                    get_line_number2(x3, ntt)], y[get_line_number2(x4, ntt)], y[get_line_number2(x5, ntt)], y[
                    get_line_number2(x6, ntt)], y[get_line_number2(x7, ntt)], y[
                    get_line_number2(x8, ntt)], y[get_line_number2(x9, ntt)]
                guess = (
                    y1, x1, 0.1, y2, x2, 0.1, y3, x3, 0.01, y4, x4, 0.1, y5, x5, 0.1, y6, x6, 0.1, y7, x7, 0.1, y8, x8, 0.1,
                    y9, x9, 0.1)
                popt, pcov = curve_fit(
                    nine_peaks_G, ntt, y, guess, maxfev=5000)
                pars10b = popt[0:3]
                pars20b = popt[3:6]
                pars30b = popt[6:9]
                pars40b = popt[9:12]
                pars50b = popt[12:15]
                pars60b = popt[15:18]
                pars70b = popt[18:21]
                pars80b = popt[21:24]
                pars90b = popt[24:27]
                peak10b = asym_peak_G(ntt, pars10b)
                peak20b = asym_peak_G(ntt, pars20b)
                peak30b = asym_peak_G(ntt, pars30b)
                peak40b = asym_peak_G(ntt, pars40b)
                peak50b = asym_peak_G(ntt, pars50b)
                peak60b = asym_peak_G(ntt, pars60b)
                peak70b = asym_peak_G(ntt, pars70b)
                peak80b = asym_peak_G(ntt, pars80b)
                peak90b = asym_peak_G(ntt, pars90b)
                tot = peak10b + peak20b + peak30b + peak40b + \
                    peak50b + peak60b + peak70b + peak80b + peak90b
                P1 = scipy.integrate.trapz(peak10b, ntt_cm)
                P2 = scipy.integrate.trapz(peak20b, ntt_cm)
                P3 = scipy.integrate.trapz(peak30b, ntt_cm)
                P4 = scipy.integrate.trapz(peak40b, ntt_cm)
                P5 = scipy.integrate.trapz(peak50b, ntt_cm)
                P6 = scipy.integrate.trapz(peak60b, ntt_cm)
                P7 = scipy.integrate.trapz(peak70b, ntt_cm)
                P8 = scipy.integrate.trapz(peak80b, ntt_cm)
                P9 = scipy.integrate.trapz(peak90b, ntt_cm)
                N1 = -P1 / A_H2O_3mic
                N2 = -P2 / A_H2O_3mic
                N3 = 0.
                N4 = 0.
                N5 = 0.
                N6 = -P7 / A_CH3CH2OH
                N7 = -P8 / A_CH3CH2OH
                N8 = -P8 / A_H2O_2
                N9 = -P9 / A_H2O_2
                N_H2O = N1 + N2
                N_CH3CH2OH = N6
                Data1 = {'H2O_in_mix': N_H2O, 'CH3CH2OH_in_mix': N_CH3CH2OH}
                df1 = DataFrame(
                    Data1, columns=['H2O_in_mix', 'CH3CH2OH_in_mix'], index=[0])
                df1.to_csv('Column_density_' +
                           'H2O_CH3CH2OH_15.0K' + '.csv', index=False)
                np.savetxt('Analytic_deconv_H2O_CH3CH2OH_15.0K.dat', np.transpose(
                    [ntt, peak10b, peak20b, peak30b, peak40b, peak50b, peak60b, peak70b, peak80b, peak90b, tot, y]))

            elif spn[j] == 'CH3CHO':
                guess = (0.01, 5.8, 0.05, 0.01, 7.4, 0.05)
                popt, pcov = curve_fit(two_peaks, ntt, y, guess, maxfev=5000)
                pars10b = popt[0:3]
                pars20b = popt[3:6]
                peak10b = asym_peak_L(ntt, pars10b)
                peak20b = asym_peak_L(ntt, pars20b)
                P1 = scipy.integrate.trapz(peak10b, ntt_cm)
                P2 = scipy.integrate.trapz(peak20b, ntt_cm)
                N1 = -P1 / A_CH3CHO
                N_CH3CHO = N1
                Data1 = {'CH3CHO_pure': N_CH3CHO}
                df1 = DataFrame(Data1, columns=['CH3CHO_pure'], index=[0])
                df1.to_csv('Column_density_' +
                           'CH3CHO_pure' + '.csv', index=False)

            elif spn[j] == 'CO':
                print('Decomposing CO')
                lam1, lam2 = 4.671, 4.675
                x1, x2 = find_nearest(ntt, lam1), find_nearest(ntt, lam2)
                y1, y2 = y[get_line_number2(
                    x1, ntt)], y[get_line_number2(x2, ntt)]
                # (0.02, 4.2, 0.05, 0.01, 4.26, 0.01)
                guess = (y1, x1, 0.005, y2, x2, 0.01,)
                popt, pcov = curve_fit(two_peaks_G, ntt, y, guess, maxfev=5000)
                pars10b = popt[0:3]
                pars20b = popt[3:6]
                peak10b = asym_peak_G(ntt, pars10b)
                peak20b = asym_peak_G(ntt, pars20b)
                tot = peak10b + peak20b
                P1 = scipy.integrate.trapz(peak10b, ntt_cm)
                P2 = scipy.integrate.trapz(peak20b, ntt_cm)
                N1 = -P1 / A_CO
                N2 = -P2 / A_CO
                N_CO = N1 + N2
                Data1 = {'CO_pure': N_CO}
                df1 = DataFrame(Data1, columns=['CO_pure'], index=[0])
                df1.to_csv('Column_density_' + 'CO_pure' + '.csv', index=False)
                np.savetxt('Analytic_deconv_CO.dat', np.transpose(
                    [ntt, peak10b, peak20b, tot, y]))

            elif spn[j] == 'CO2':
                print('Decomposing CO2')
                lam1, lam2 = 4.2, 4.26
                x1, x2 = find_nearest(ntt, lam1), find_nearest(ntt, lam2)
                y1, y2 = y[get_line_number2(
                    x1, ntt)], y[get_line_number2(x2, ntt)]
                # (0.02, 4.2, 0.05, 0.01, 4.26, 0.01)
                guess = (y1, x1, 0.05, y2, x2, 0.01,)
                popt, pcov = curve_fit(two_peaks_G, ntt, y, guess, maxfev=5000)
                pars10b = popt[0:3]
                pars20b = popt[3:6]
                peak10b = asym_peak_G(ntt, pars10b)
                peak20b = asym_peak_G(ntt, pars20b)
                tot = peak10b + peak20b
                P1 = scipy.integrate.trapz(peak10b, ntt_cm)
                P2 = scipy.integrate.trapz(peak20b, ntt_cm)
                N1 = -P1 / A_CO2
                N2 = -P2 / A_CO2
                N_CO2 = N1 + N2
                Data1 = {'CO2_pure': N_CO2}
                df1 = DataFrame(Data1, columns=['CO2_pure'], index=[0])
                df1.to_csv('Column_density_' +
                           'CO2_pure' + '.csv', index=False)
                np.savetxt('Analytic_deconv_CO2.dat', np.transpose(
                    [ntt, peak10b, peak20b, tot, y]))

            elif spn[j] == 'CH3CN':
                print('Decomposing CH3CN')
                lam1, lam2, lam3, lam4, lam5, lam6 = 6.9, 7.1, 7.29, 9.6, 10.86, 13.17
                x1, x2, x3, x4, x5, x6 = find_nearest(ntt, lam1), find_nearest(ntt, lam2), find_nearest(ntt,
                                                                                                        lam3), find_nearest(
                    ntt, lam4), find_nearest(ntt, lam5), find_nearest(ntt, lam6)
                y1, y2, y3, y4, y5, y6 = y[get_line_number2(x1, ntt)], y[get_line_number2(x2, ntt)], y[
                    get_line_number2(x3, ntt)], y[get_line_number2(x4, ntt)], y[get_line_number2(x5, ntt)], y[
                    get_line_number2(x6, ntt)]
                guess = (y1, x1, 0.1, y2, x2, 0.08, y3, x3, 0.08,
                         y4, x4, 0.1, y5, x5, 0.1, y6, x6, 0.1)
                popt, pcov = curve_fit(six_peaks_G, ntt, y, guess, maxfev=5000)
                pars10b = popt[0:3]
                pars20b = popt[3:6]
                pars30b = popt[6:9]
                pars40b = popt[9:12]
                pars50b = popt[12:15]
                pars60b = popt[15:18]
                peak10b = asym_peak_G(ntt, pars10b)
                peak20b = asym_peak_G(ntt, pars20b)
                peak30b = asym_peak_G(ntt, pars30b)
                peak40b = asym_peak_G(ntt, pars40b)
                peak50b = asym_peak_G(ntt, pars50b)
                peak60b = asym_peak_G(ntt, pars60b)
                tot = peak10b + peak20b + peak30b + peak40b + peak50b + peak60b
                P1 = scipy.integrate.trapz(peak10b, ntt_cm)
                P2 = scipy.integrate.trapz(peak20b, ntt_cm)
                P3 = scipy.integrate.trapz(peak30b, ntt_cm)
                P4 = scipy.integrate.trapz(peak40b, ntt_cm)
                P5 = scipy.integrate.trapz(peak50b, ntt_cm)
                P6 = scipy.integrate.trapz(peak60b, ntt_cm)
                N1 = 0.  # -P1/A_H2O
                N2 = 0.  # -P2/A_H2O
                N3 = 0.  # -P3/A_H2O
                N4 = -1.29 * P4 / A_CH3CN
                N5 = 0.  # -P5/A_H2O_2
                N6 = 0.  # -P6/A_H2O_2
                N_CH3CN = N4 / 1e2
                Data1 = {'CH3CN_pure': N_CH3CN}
                df1 = DataFrame(Data1, columns=['CH3CN_pure'], index=[0])
                df1.to_csv('Column_density_' +
                           'CH3CN_pure' + '.csv', index=False)
                np.savetxt('Analytic_deconv_CH3CN_pure.dat',
                           np.transpose([ntt, peak10b, peak20b, peak30b, peak40b, peak50b, peak60b, tot, y]))

            elif spn[j] == 'H2O_NH3_1_05_c':
                print('Decomposing H2O_NH3_1_0.5_c')
                lam1, lam2, lam3, lam4, lam5, lam6 = 5.92, 6.03, 6.75, 6.84, 6.90, 7.03
                x1, x2, x3, x4, x5, x6 = find_nearest(ntt, lam1), find_nearest(ntt, lam2), find_nearest(ntt,
                                                                                                        lam3), find_nearest(
                    ntt, lam4), find_nearest(ntt, lam5), find_nearest(ntt, lam6)
                y1, y2, y3, y4, y5, y6 = y[get_line_number2(x1, ntt)], y[get_line_number2(x2, ntt)], y[
                    get_line_number2(x3, ntt)], y[get_line_number2(x4, ntt)], y[get_line_number2(x5, ntt)], y[
                    get_line_number2(x6, ntt)]
                guess = (y1, x1, 0.05, y2, x2, 0.05, y3, x3, 0.05,
                         y4, x4, 0.05, y5, x5, 0.05, y6, x6, 0.05)
                popt, pcov = curve_fit(six_peaks_G, ntt, y, guess, maxfev=5000)
                pars10b = popt[0:3]
                pars20b = popt[3:6]
                pars30b = popt[6:9]
                pars40b = popt[9:12]
                pars50b = popt[12:15]
                pars60b = popt[15:18]
                peak10b = asym_peak_G(ntt, pars10b)
                peak20b = asym_peak_G(ntt, pars20b)
                peak30b = asym_peak_G(ntt, pars30b)
                peak40b = asym_peak_G(ntt, pars40b)
                peak50b = asym_peak_G(ntt, pars50b)
                peak60b = asym_peak_G(ntt, pars60b)
                tot = peak10b + peak20b + peak30b + peak40b + peak50b + peak60b
                P1 = scipy.integrate.trapz(peak10b, ntt_cm)
                P2 = scipy.integrate.trapz(peak20b, ntt_cm)
                P3 = scipy.integrate.trapz(peak30b, ntt_cm)
                P4 = scipy.integrate.trapz(peak40b, ntt_cm)
                P5 = scipy.integrate.trapz(peak50b, ntt_cm)
                P6 = scipy.integrate.trapz(peak60b, ntt_cm)
                N1 = -P1 / A_H2O
                N2 = -P2 / A_H2O
                N3 = -P3 / A_H2O
                N4 = -P4 / A_NH4
                N5 = -P5 / A_NH4
                N6 = -P6 / A_NH4
                N_H2O = N1 + N2 + N3
                N_NH4 = N4 + N5 + N6
                N_NH3 = 0.05 * N_H2O
                Data1 = {'H2O_in_mix': N_H2O,
                         'NH3_in_mix': N_NH3, 'NH4_in_mix': N_NH4}
                df1 = DataFrame(
                    Data1, columns=['H2O_in_mix', 'NH3_in_mix', 'NH4_in_mix'], index=[0])
                df1.to_csv('Column_density_' +
                           'H2O_NH3_1_0.5_c' + '.csv', index=False)
                np.savetxt('Analytic_deconv_H2O_NH3_1_0.5_c.dat',
                           np.transpose([ntt, peak10b, peak20b, peak30b, peak40b, peak50b, peak60b, tot, y]))

            elif spn[j] == 'H2O_CH3CH2OH_20_1_15K_NASA_base':
                guess = (
                    0.003, 5.92, 0.05, 0.003, 6.21, 0.05, 0.003, 6.52, 0.05, 0.003, 6.73, 0.01, 0.001, 6.85, 0.01, 0.001,
                    7.00, 0.01)
                popt, pcov = curve_fit(six_peaks, ntt, y, guess, maxfev=5000)
                pars10b = popt[0:3]
                pars20b = popt[3:6]
                pars30b = popt[6:9]
                pars40b = popt[9:12]
                pars50b = popt[12:15]
                pars60b = popt[15:18]
                peak10b = asym_peak_L(ntt, pars10b)
                peak20b = asym_peak_L(ntt, pars20b)
                peak30b = asym_peak_L(ntt, pars30b)
                peak40b = asym_peak_L(ntt, pars40b)
                peak50b = asym_peak_L(ntt, pars50b)
                peak60b = asym_peak_L(ntt, pars60b)
                # plt.plot(ntt, peak10b, ntt, peak20b, ntt, peak30b, ntt, peak40b, ntt, peak50b, ntt, peak60b)
                P1 = scipy.integrate.trapz(peak10b, ntt_cm)
                P2 = scipy.integrate.trapz(peak20b, ntt_cm)
                P3 = scipy.integrate.trapz(peak30b, ntt_cm)
                P4 = scipy.integrate.trapz(peak40b, ntt_cm)
                P5 = scipy.integrate.trapz(peak50b, ntt_cm)
                P6 = scipy.integrate.trapz(peak60b, ntt_cm)
                N1 = -P1 / A_H2O
                N2 = -P2 / A_H2O
                N3 = -P3 / A_H2O
                N4 = -P4 / A_CH3CH2OH
                N5 = -P5 / A_CH3CH2OH
                N6 = -P6 / A_CH3CH2OH
                N_H2O = N1 + N2 + N3
                N_CH3CH2OH = N4
                Data1 = {'H2O_in_mix': N_H2O, 'CH3CH2OH_in_mix': N_CH3CH2OH}
                df1 = DataFrame(
                    Data1, columns=['H2O_in_mix', 'CH3CH2OH_in_mix'], index=[0])
                df1.to_csv(
                    'Column_density_' + 'H2O_CH3CH2OH_20_1_15K_NASA_base' + '.csv', index=False)

            elif spn[j] == 'CO_CH3CH2OH_30.0K':
                lam1, lam2, lam3, lam4, lam5 = 9.1, 9.2, 9.38, 9.45, 9.51
                x1, x2, x3, x4, x5 = find_nearest(ntt, lam1), find_nearest(ntt, lam2), find_nearest(ntt,
                                                                                                    lam3), find_nearest(
                    ntt, lam4), find_nearest(ntt, lam5)
                y1, y2, y3, y4, y5 = y[get_line_number2(x1, ntt)], y[get_line_number2(x2, ntt)], y[
                    get_line_number2(x3, ntt)], y[get_line_number2(x4, ntt)], y[get_line_number2(x5, ntt)]
                guess = (y1, x1, 0.01, y2, x2, 0.01, y3, x3,
                         0.03, y4, x4, 0.01, y5, x5, 0.03)
                popt, pcov = curve_fit(
                    five_peaks_L, ntt, y, guess, maxfev=5000)
                pars10b = popt[0:3]
                pars20b = popt[3:6]
                pars30b = popt[6:9]
                pars40b = popt[9:12]
                pars50b = popt[12:15]
                peak10b = asym_peak_L(ntt, pars10b)
                peak20b = asym_peak_L(ntt, pars20b)
                peak30b = asym_peak_L(ntt, pars30b)
                peak40b = asym_peak_L(ntt, pars40b)
                peak50b = asym_peak_L(ntt, pars50b)
                tot = peak10b + peak20b + peak30b + peak40b + peak50b
                P1 = scipy.integrate.trapz(peak10b, ntt_cm)
                P2 = scipy.integrate.trapz(peak20b, ntt_cm)
                P3 = scipy.integrate.trapz(peak30b, ntt_cm)
                P4 = scipy.integrate.trapz(peak40b, ntt_cm)
                P5 = scipy.integrate.trapz(peak50b, ntt_cm)
                N1 = -P1 / A_CH3CH2OH
                print('P1, N1:', P1, N1)
                N2 = -P2 / A_CH3CH2OH
                N3 = -P3 / A_CH3CH2OH
                N4 = -P4 / A_CH3CH2OH
                N5 = -P5 / A_CH3CH2OH
                N_CH3CH2OH = (N4 + N5) / 1e1
                Data1 = {'CH3CH2OH_in_mix': N_CH3CH2OH}
                df1 = DataFrame(Data1, columns=['CH3CH2OH_in_mix'], index=[0])
                df1.to_csv('Column_density_' +
                           'CO_CH3CH2OH_30.0K' + '.csv', index=False)
                # plt.plot(ntt,y, ntt,peak10b,ntt,peak20b,ntt,peak30b,ntt,peak40b, ntt,tot)
                np.savetxt('Analytic_deconv_CO_CH3CH2OH_30.0K.dat',
                           np.transpose([ntt, peak10b, peak20b, peak30b, peak40b, peak50b, tot, y]))

            elif spn[j] == 'CO_CH3CH2OH_15.0K':
                lam1, lam2, lam3, lam4, lam5 = 9.1, 9.2, 9.38, 9.45, 9.51
                x1, x2, x3, x4, x5 = find_nearest(ntt, lam1), find_nearest(ntt, lam2), find_nearest(ntt,
                                                                                                    lam3), find_nearest(
                    ntt, lam4), find_nearest(ntt, lam5)
                y1, y2, y3, y4, y5 = y[get_line_number2(x1, ntt)], y[get_line_number2(x2, ntt)], y[
                    get_line_number2(x3, ntt)], y[get_line_number2(x4, ntt)], y[get_line_number2(x5, ntt)]
                guess = (y1, x1, 0.01, y2, x2, 0.01, y3, x3,
                         0.01, y4, x4, 0.01, y5, x5, 0.01)
                popt, pcov = curve_fit(
                    five_peaks_L, ntt, y, guess, maxfev=5000)
                pars10b = popt[0:3]
                pars20b = popt[3:6]
                pars30b = popt[6:9]
                pars40b = popt[9:12]
                pars50b = popt[12:15]
                peak10b = asym_peak_L(ntt, pars10b)
                peak20b = asym_peak_L(ntt, pars20b)
                peak30b = asym_peak_L(ntt, pars30b)
                peak40b = asym_peak_L(ntt, pars40b)
                peak50b = asym_peak_L(ntt, pars50b)
                tot = peak10b + peak20b + peak30b + peak40b + peak50b
                P1 = scipy.integrate.trapz(peak10b, ntt_cm)
                P2 = scipy.integrate.trapz(peak20b, ntt_cm)
                P3 = scipy.integrate.trapz(peak30b, ntt_cm)
                P4 = scipy.integrate.trapz(peak40b, ntt_cm)
                P5 = scipy.integrate.trapz(peak50b, ntt_cm)
                N1 = -P1 / A_CH3CH2OH
                N2 = -P2 / A_CH3CH2OH
                N3 = -P3 / A_CH3CH2OH
                N4 = -P4 / A_CH3CH2OH
                N4 = -P5 / A_CH3CH2OH
                N_CH3CH2OH = (N4 + N5) / 1e2
                Data1 = {'CH3CH2OH_in_mix': N_CH3CH2OH}
                df1 = DataFrame(Data1, columns=['CH3CH2OH_in_mix'], index=[0])
                df1.to_csv('Column_density_' +
                           'CO_CH3CH2OH_15.0K' + '.csv', index=False)
                # plt.plot(ntt,y, ntt,peak10b,ntt,peak20b,ntt,peak30b,ntt,peak40b, ntt,tot)
                np.savetxt('Analytic_deconv_CO_CH3CH2OH_15.0K.dat',
                           np.transpose([ntt, peak10b, peak20b, peak30b, peak40b, tot, y]))

            elif spn[j] == 'CO_CH3OH_CH3OCH3_15.0K':
                lam1, lam2, lam3, lam4, lam5 = 8.58, 8.84, 9.19, 9.68, 9.90
                x1, x2, x3, x4, x5 = find_nearest(ntt, lam1), find_nearest(ntt, lam2), find_nearest(ntt,
                                                                                                    lam3), find_nearest(
                    ntt, lam4), find_nearest(ntt, lam5)
                y1, y2, y3, y4, y5 = y[get_line_number2(x1, ntt)], y[get_line_number2(x2, ntt)], y[
                    get_line_number2(x3, ntt)], y[get_line_number2(x4, ntt)], y[get_line_number2(x5, ntt)]
                guess = (y1, x1, 0.01, y2, x2, 0.01, y3, x3,
                         0.01, y4, x4, 0.01, y5, x5, 0.05)
                popt, pcov = curve_fit(
                    five_peaks_G, ntt, y, guess, maxfev=5000)
                pars10b = popt[0:3]
                pars20b = popt[3:6]
                pars30b = popt[6:9]
                pars40b = popt[9:12]
                pars50b = popt[12:15]
                # pars60b = popt[15:18]
                peak10b = asym_peak_G(ntt, pars10b)
                peak20b = asym_peak_G(ntt, pars20b)
                peak30b = asym_peak_G(ntt, pars30b)
                peak40b = asym_peak_G(ntt, pars40b)
                peak50b = asym_peak_G(ntt, pars50b)
                # peak60b = asym_peak_G(ntt, pars60b)
                tot = peak10b + peak20b + peak30b + peak40b + peak50b  # + peak60b
                P1 = scipy.integrate.trapz(peak10b, ntt_cm)
                P2 = scipy.integrate.trapz(peak20b, ntt_cm)
                P3 = scipy.integrate.trapz(peak30b, ntt_cm)
                P4 = scipy.integrate.trapz(peak40b, ntt_cm)
                P5 = scipy.integrate.trapz(peak50b, ntt_cm)
                # P6 = scipy.integrate.trapz(peak60b,ntt_cm)
                N1 = -P1 / A_CH3OCH3
                N2 = 0.0  # -P2/A_H2O
                N3 = 0.0  # -P4/A_H2O_2
                N4 = -P4 / A_CH3OH
                N5 = -P5 / A_CH3OH
                N_CH3OCH3 = N1
                N_CH3OH = N4 + N5
                Data1 = {'CH3OCH3_in_mix': N_CH3OCH3, 'CH3OH_in_mix': N_CH3OH}
                df1 = DataFrame(
                    Data1, columns=['CH3OCH3_in_mix', 'CH3OH_in_mix'], index=[0])
                df1.to_csv('Column_density_' +
                           'CO_CH3OH_CH3OCH3_15.0K' + '.csv', index=False)

            elif spn[j] == 'CO_CH3CHO_30.0K':
                lam1, lam2, lam3 = 5.66, 5.8, 7.4
                x1, x2, x3 = find_nearest(ntt, lam1), find_nearest(
                    ntt, lam2), find_nearest(ntt, lam3)
                y1, y2, y3 = y[get_line_number2(x1, ntt)], y[get_line_number2(
                    x2, ntt)], y[get_line_number2(x3, ntt)]
                guess = (y1, x1, 0.01, y2, x2, 0.01, y3, x3, 0.01)
                popt, pcov = curve_fit(
                    three_peaks_L, ntt, y, guess, maxfev=5000)
                pars10b = popt[0:3]
                pars20b = popt[3:6]
                pars30b = popt[6:9]
                peak10b = asym_peak_L(ntt, pars10b)
                peak20b = asym_peak_L(ntt, pars20b)
                peak30b = asym_peak_L(ntt, pars30b)
                tot = peak10b + peak20b + peak30b
                P1 = scipy.integrate.trapz(peak10b, ntt_cm)
                P2 = scipy.integrate.trapz(peak20b, ntt_cm)
                P3 = scipy.integrate.trapz(peak30b, ntt_cm)
                N2 = -P2 / (
                    A_CH3CHO * 0.8)  # Term 0.8 takes into account the decreasing in the band strength of CH3CHO in CO ice (See Fig C.3 in Terwisscha van Scheltinga+2018)
                N_CH3CHO = N2
                Data1 = {'CH3CHO_in_mix': N_CH3CHO}
                df1 = DataFrame(Data1, columns=['CH3CHO_in_mix'], index=[0])
                df1.to_csv('Column_density_' +
                           'CO_CH3CHO_30.0K' + '.csv', index=False)
                np.savetxt('Analytic_deconv_CO_CH3CHO_30.0K', np.transpose(
                    [ntt, peak10b, peak20b, peak30b, tot, y]))

            elif spn[j] == 'H2O_CH3OH_CO_NH3_10K_NASA':
                guess = (
                    0.03, 5.92, 0.05, 0.03, 6.03, 0.05, 0.03, 6.75, 0.05, 0.03, 6.84, 0.05, 0.03, 6.90, 0.05, 0.03, 7.03,
                    0.05)
                popt, pcov = curve_fit(six_peaks_G, ntt, y, guess, maxfev=5000)
                pars10b = popt[0:3]
                pars20b = popt[3:6]
                pars30b = popt[6:9]
                pars40b = popt[9:12]
                pars50b = popt[12:15]
                pars60b = popt[15:18]
                peak10b = asym_peak_L(ntt, pars10b)
                peak20b = asym_peak_L(ntt, pars20b)
                peak30b = asym_peak_L(ntt, pars30b)
                peak40b = asym_peak_L(ntt, pars40b)
                peak50b = asym_peak_L(ntt, pars50b)
                peak60b = asym_peak_L(ntt, pars60b)
                # plt.plot(ntt, peak10b, ntt, peak20b, ntt, peak30b, ntt, peak40b, ntt, peak50b, ntt, peak60b)
                P1 = scipy.integrate.trapz(peak10b, ntt_cm)
                P2 = scipy.integrate.trapz(peak20b, ntt_cm)
                P3 = scipy.integrate.trapz(peak30b, ntt_cm)
                P4 = scipy.integrate.trapz(peak40b, ntt_cm)
                P5 = scipy.integrate.trapz(peak50b, ntt_cm)
                P6 = scipy.integrate.trapz(peak60b, ntt_cm)
                N1 = -P1 / A_H2O
                N2 = -P2 / A_H2O
                N3 = -P3 / A_CH3OH
                N4 = -P4 / A_CH3OH
                N5 = -P5 / A_CH3OH
                N6 = -P6 / A_CH3OH
                N_H2O = N1 + N2
                N_NH3 = 0.05 * N_H2O
                N_CH3OH = N3 + N4 + N5 + N6
                Data1 = {'H2O_in_mix': N_H2O,
                         'NH3_in_mix': N_NH3, 'CH3OH_in_mix': N_CH3OH}
                df1 = DataFrame(
                    Data1, columns=['H2O_in_mix', 'NH3_in_mix', 'CH3OH_in_mix'], index=[0])
                df1.to_csv('Column_density_' +
                           'H2O_CH3OH_CO_NH3_10K_NASA' + '.csv', index=False)

            elif spn[j] == 'H2O_CH3OH_CO_NH3_40K_NASA':
                guess = (
                    0.03, 5.92, 0.05, 0.03, 6.03, 0.05, 0.03, 6.75, 0.05, 0.03, 6.84, 0.05, 0.03, 6.90, 0.05, 0.03, 7.03,
                    0.05)
                popt, pcov = curve_fit(six_peaks_G, ntt, y, guess, maxfev=5000)
                pars10b = popt[0:3]
                pars20b = popt[3:6]
                pars30b = popt[6:9]
                pars40b = popt[9:12]
                pars50b = popt[12:15]
                pars60b = popt[15:18]
                peak10b = asym_peak_L(ntt, pars10b)
                peak20b = asym_peak_L(ntt, pars20b)
                peak30b = asym_peak_L(ntt, pars30b)
                peak40b = asym_peak_L(ntt, pars40b)
                peak50b = asym_peak_L(ntt, pars50b)
                peak60b = asym_peak_L(ntt, pars60b)
                # plt.plot(ntt, peak10b, ntt, peak20b, ntt, peak30b, ntt, peak40b, ntt, peak50b, ntt, peak60b)
                P1 = scipy.integrate.trapz(peak10b, ntt_cm)
                P2 = scipy.integrate.trapz(peak20b, ntt_cm)
                P3 = scipy.integrate.trapz(peak30b, ntt_cm)
                P4 = scipy.integrate.trapz(peak40b, ntt_cm)
                P5 = scipy.integrate.trapz(peak50b, ntt_cm)
                P6 = scipy.integrate.trapz(peak60b, ntt_cm)
                N1 = -P1 / A_H2O
                N2 = -P2 / A_H2O
                N3 = -P3 / A_CH3OH
                N4 = -P4 / A_CH3OH
                N5 = -P5 / A_CH3OH
                N6 = -P6 / A_CH3OH
                N_H2O = N1 + N2
                N_NH3 = 0.05 * N_H2O
                N_CH3OH = N3 + N4 + N5 + N6
                Data1 = {'H2O_in_mix': N_H2O,
                         'NH3_in_mix': N_NH3, 'CH3OH_in_mix': N_CH3OH}
                df1 = DataFrame(
                    Data1, columns=['H2O_in_mix', 'NH3_in_mix', 'CH3OH_in_mix'], index=[0])
                df1.to_csv('Column_density_' +
                           'H2O_CH3OH_CO_NH3_40K_NASA' + '.csv', index=False)

            elif spn[j] == 'H2O_CH3OH_CO_NH3_80K_NASA':
                guess = (
                    0.05, 5.92, 0.05, 0.05, 6.03, 0.05, 0.05, 6.75, 0.05, 0.05, 6.84, 0.05, 0.05, 6.90, 0.05, 0.05, 7.03,
                    0.05)
                popt, pcov = curve_fit(six_peaks_G, ntt, y, guess, maxfev=5000)
                pars10b = popt[0:3]
                pars20b = popt[3:6]
                pars30b = popt[6:9]
                pars40b = popt[9:12]
                pars50b = popt[12:15]
                pars60b = popt[15:18]
                peak10b = asym_peak_L(ntt, pars10b)
                peak20b = asym_peak_L(ntt, pars20b)
                peak30b = asym_peak_L(ntt, pars30b)
                peak40b = asym_peak_L(ntt, pars40b)
                peak50b = asym_peak_L(ntt, pars50b)
                peak60b = asym_peak_L(ntt, pars60b)
                # plt.plot(ntt, peak10b, ntt, peak20b, ntt, peak30b, ntt, peak40b, ntt, peak50b, ntt, peak60b)
                P1 = scipy.integrate.trapz(peak10b, ntt_cm)
                P2 = scipy.integrate.trapz(peak20b, ntt_cm)
                P3 = scipy.integrate.trapz(peak30b, ntt_cm)
                P4 = scipy.integrate.trapz(peak40b, ntt_cm)
                P5 = scipy.integrate.trapz(peak50b, ntt_cm)
                P6 = scipy.integrate.trapz(peak60b, ntt_cm)
                N1 = -P1 / A_H2O
                N2 = -P2 / A_H2O
                N3 = -P3 / A_CH3OH
                N4 = -P4 / A_CH3OH
                N5 = -P5 / A_CH3OH
                N6 = -P6 / A_CH3OH
                N_H2O = N1 + N2
                N_NH3 = 0.05 * N_H2O
                N_CH3OH = N3 + N4 + N5 + N6
                Data1 = {'H2O_in_mix': N_H2O,
                         'NH3_in_mix': N_NH3, 'CH3OH_in_mix': N_CH3OH}
                df1 = DataFrame(
                    Data1, columns=['H2O_in_mix', 'NH3_in_mix', 'CH3OH_in_mix'], index=[0])
                df1.to_csv('Column_density_' +
                           'H2O_CH3OH_CO_NH3_80K_NASA' + '.csv', index=False)

            elif spn[j] == 'H2O_CH3OH_CO_NH3_100K_NASA':
                guess = (
                    0.03, 5.92, 0.05, 0.03, 6.03, 0.05, 0.03, 6.75, 0.05, 0.03, 6.84, 0.05, 0.03, 6.90, 0.05, 0.03, 7.03,
                    0.05)
                popt, pcov = curve_fit(six_peaks_G, ntt, y, guess, maxfev=5000)
                pars10b = popt[0:3]
                pars20b = popt[3:6]
                pars30b = popt[6:9]
                pars40b = popt[9:12]
                pars50b = popt[12:15]
                pars60b = popt[15:18]
                peak10b = asym_peak_L(ntt, pars10b)
                peak20b = asym_peak_L(ntt, pars20b)
                peak30b = asym_peak_L(ntt, pars30b)
                peak40b = asym_peak_L(ntt, pars40b)
                peak50b = asym_peak_L(ntt, pars50b)
                peak60b = asym_peak_L(ntt, pars60b)
                # plt.plot(ntt, peak10b, ntt, peak20b, ntt, peak30b, ntt, peak40b, ntt, peak50b, ntt, peak60b)
                P1 = scipy.integrate.trapz(peak10b, ntt_cm)
                P2 = scipy.integrate.trapz(peak20b, ntt_cm)
                P3 = scipy.integrate.trapz(peak30b, ntt_cm)
                P4 = scipy.integrate.trapz(peak40b, ntt_cm)
                P5 = scipy.integrate.trapz(peak50b, ntt_cm)
                P6 = scipy.integrate.trapz(peak60b, ntt_cm)
                N1 = -P1 / A_H2O
                N2 = -P2 / A_H2O
                N3 = -P3 / A_CH3OH
                N4 = -P4 / A_CH3OH
                N5 = -P5 / A_CH3OH
                N6 = -P6 / A_CH3OH
                N_H2O = N1 + N2
                N_NH3 = 0.05 * N_H2O
                N_CH3OH = N3 + N4 + N5 + N6
                Data1 = {'H2O_in_mix': N_H2O,
                         'NH3_in_mix': N_NH3, 'CH3OH_in_mix': N_CH3OH}
                df1 = DataFrame(
                    Data1, columns=['H2O_in_mix', 'NH3_in_mix', 'CH3OH_in_mix'], index=[0])
                df1.to_csv('Column_density_' +
                           'H2O_CH3OH_CO_NH3_100K_NASA' + '.csv', index=False)

            elif spn[j] == 'H2O_CH3OH_CO_NH3_H2_10K_NASA':
                guess = (
                    0.03, 5.92, 0.05, 0.03, 6.03, 0.05, 0.03, 6.75, 0.05, 0.03, 6.84, 0.05, 0.03, 6.90, 0.05, 0.03, 7.03,
                    0.05)
                popt, pcov = curve_fit(six_peaks_G, ntt, y, guess, maxfev=5000)
                pars10b = popt[0:3]
                pars20b = popt[3:6]
                pars30b = popt[6:9]
                pars40b = popt[9:12]
                pars50b = popt[12:15]
                pars60b = popt[15:18]
                peak10b = asym_peak_L(ntt, pars10b)
                peak20b = asym_peak_L(ntt, pars20b)
                peak30b = asym_peak_L(ntt, pars30b)
                peak40b = asym_peak_L(ntt, pars40b)
                peak50b = asym_peak_L(ntt, pars50b)
                peak60b = asym_peak_L(ntt, pars60b)
                # plt.plot(ntt, peak10b, ntt, peak20b, ntt, peak30b, ntt, peak40b, ntt, peak50b, ntt, peak60b)
                P1 = scipy.integrate.trapz(peak10b, ntt_cm)
                P2 = scipy.integrate.trapz(peak20b, ntt_cm)
                P3 = scipy.integrate.trapz(peak30b, ntt_cm)
                P4 = scipy.integrate.trapz(peak40b, ntt_cm)
                P5 = scipy.integrate.trapz(peak50b, ntt_cm)
                P6 = scipy.integrate.trapz(peak60b, ntt_cm)
                N1 = -P1 / A_H2O
                N2 = -P2 / A_H2O
                N3 = -P3 / A_CH3OH
                N4 = -P4 / A_CH3OH
                N5 = -P5 / A_CH3OH
                N6 = -P6 / A_CH3OH
                N_H2O = N1 + N2
                N_NH3 = 0.06 * N_H2O
                N_CH3OH = N3 + N4 + N5 + N6
                Data1 = {'H2O_in_mix': N_H2O,
                         'NH3_in_mix': N_NH3, 'CH3OH_in_mix': N_CH3OH}
                df1 = DataFrame(
                    Data1, columns=['H2O_in_mix', 'NH3_in_mix', 'CH3OH_in_mix'], index=[0])
                df1.to_csv('Column_density_' +
                           'H2O_CH3OH_CO_NH3_H2_10K_NASA' + '.csv', index=False)

            elif spn[j] == 'H2O_CH3OH_CO_NH3_H2_40K_NASA':
                guess = (
                    0.03, 5.92, 0.05, 0.03, 6.03, 0.05, 0.03, 6.75, 0.05, 0.03, 6.84, 0.05, 0.03, 6.90, 0.05, 0.03, 7.03,
                    0.05)
                popt, pcov = curve_fit(six_peaks_G, ntt, y, guess, maxfev=5000)
                pars10b = popt[0:3]
                pars20b = popt[3:6]
                pars30b = popt[6:9]
                pars40b = popt[9:12]
                pars50b = popt[12:15]
                pars60b = popt[15:18]
                peak10b = asym_peak_L(ntt, pars10b)
                peak20b = asym_peak_L(ntt, pars20b)
                peak30b = asym_peak_L(ntt, pars30b)
                peak40b = asym_peak_L(ntt, pars40b)
                peak50b = asym_peak_L(ntt, pars50b)
                peak60b = asym_peak_L(ntt, pars60b)
                # plt.plot(ntt, peak10b, ntt, peak20b, ntt, peak30b, ntt, peak40b, ntt, peak50b, ntt, peak60b)
                P1 = scipy.integrate.trapz(peak10b, ntt_cm)
                P2 = scipy.integrate.trapz(peak20b, ntt_cm)
                P3 = scipy.integrate.trapz(peak30b, ntt_cm)
                P4 = scipy.integrate.trapz(peak40b, ntt_cm)
                P5 = scipy.integrate.trapz(peak50b, ntt_cm)
                P6 = scipy.integrate.trapz(peak60b, ntt_cm)
                N1 = -P1 / A_H2O
                N2 = -P2 / A_H2O
                N3 = -P3 / A_CH3OH
                N4 = -P4 / A_CH3OH
                N5 = -P5 / A_CH3OH
                N6 = -P6 / A_CH3OH
                N_H2O = N1 + N2
                N_NH3 = 0.06 * N_H2O
                N_CH3OH = N3 + N4 + N5 + N6
                Data1 = {'H2O_in_mix': N_H2O,
                         'NH3_in_mix': N_NH3, 'CH3OH_in_mix': N_CH3OH}
                df1 = DataFrame(
                    Data1, columns=['H2O_in_mix', 'NH3_in_mix', 'CH3OH_in_mix'], index=[0])
                df1.to_csv('Column_density_' +
                           'H2O_CH3OH_CO_NH3_H2_40K_NASA' + '.csv', index=False)

            elif spn[j] == 'H2O_CH3OH_CO_NH3_H2_80K_NASA':
                guess = (
                    0.01, 5.92, 0.05, 0.01, 6.03, 0.05, 0.02, 6.75, 0.05, 0.02, 6.84, 0.05, 0.02, 6.90, 0.05, 0.02, 7.03,
                    0.05)
                popt, pcov = curve_fit(six_peaks_G, ntt, y, guess, maxfev=5000)
                pars10b = popt[0:3]
                pars20b = popt[3:6]
                pars30b = popt[6:9]
                pars40b = popt[9:12]
                pars50b = popt[12:15]
                pars60b = popt[15:18]
                peak10b = asym_peak_L(ntt, pars10b)
                peak20b = asym_peak_L(ntt, pars20b)
                peak30b = asym_peak_L(ntt, pars30b)
                peak40b = asym_peak_L(ntt, pars40b)
                peak50b = asym_peak_L(ntt, pars50b)
                peak60b = asym_peak_L(ntt, pars60b)
                # plt.plot(ntt, peak10b, ntt, peak20b, ntt, peak30b, ntt, peak40b, ntt, peak50b, ntt, peak60b)
                P1 = scipy.integrate.trapz(peak10b, ntt_cm)
                P2 = scipy.integrate.trapz(peak20b, ntt_cm)
                P3 = scipy.integrate.trapz(peak30b, ntt_cm)
                P4 = scipy.integrate.trapz(peak40b, ntt_cm)
                P5 = scipy.integrate.trapz(peak50b, ntt_cm)
                P6 = scipy.integrate.trapz(peak60b, ntt_cm)
                N1 = -P1 / A_H2O
                N2 = -P2 / A_H2O
                N3 = -P3 / A_CH3OH
                N4 = -P4 / A_CH3OH
                N5 = -P5 / A_CH3OH
                N6 = -P6 / A_CH3OH
                N_H2O = N1 + N2
                N_NH3 = 0.06 * N_H2O
                N_CH3OH = N3 + N4 + N5 + N6
                Data1 = {'H2O_in_mix': N_H2O,
                         'NH3_in_mix': N_NH3, 'CH3OH_in_mix': N_CH3OH}
                df1 = DataFrame(
                    Data1, columns=['H2O_in_mix', 'NH3_in_mix', 'CH3OH_in_mix'], index=[0])
                df1.to_csv('Column_density_' +
                           'H2O_CH3OH_CO_NH3_H2_80K_NASA' + '.csv', index=False)

            elif spn[j] == 'H2O_CH3OH_CO_NH3_H2_100K_NASA':
                guess = (
                    0.03, 5.92, 0.05, 0.03, 6.03, 0.05, 0.03, 6.75, 0.05, 0.03, 6.84, 0.05, 0.03, 6.90, 0.05, 0.03, 7.03,
                    0.05)
                popt, pcov = curve_fit(six_peaks_G, ntt, y, guess, maxfev=5000)
                pars10b = popt[0:3]
                pars20b = popt[3:6]
                pars30b = popt[6:9]
                pars40b = popt[9:12]
                pars50b = popt[12:15]
                pars60b = popt[15:18]
                peak10b = asym_peak_L(ntt, pars10b)
                peak20b = asym_peak_L(ntt, pars20b)
                peak30b = asym_peak_L(ntt, pars30b)
                peak40b = asym_peak_L(ntt, pars40b)
                peak50b = asym_peak_L(ntt, pars50b)
                peak60b = asym_peak_L(ntt, pars60b)
                # plt.plot(ntt, peak10b, ntt, peak20b, ntt, peak30b, ntt, peak40b, ntt, peak50b, ntt, peak60b)
                P1 = scipy.integrate.trapz(peak10b, ntt_cm)
                P2 = scipy.integrate.trapz(peak20b, ntt_cm)
                P3 = scipy.integrate.trapz(peak30b, ntt_cm)
                P4 = scipy.integrate.trapz(peak40b, ntt_cm)
                P5 = scipy.integrate.trapz(peak50b, ntt_cm)
                P6 = scipy.integrate.trapz(peak60b, ntt_cm)
                N1 = -P1 / A_H2O
                N2 = -P2 / A_H2O
                N3 = -P3 / A_CH3OH
                N4 = -P4 / A_CH3OH
                N5 = -P5 / A_CH3OH
                N6 = -P6 / A_CH3OH
                N_H2O = N1 + N2
                N_NH3 = 0.05 * N_H2O
                N_CH3OH = N3 + N4 + N5 + N6
                Data1 = {'H2O_in_mix': N_H2O,
                         'NH3_in_mix': N_NH3, 'CH3OH_in_mix': N_CH3OH}
                df1 = DataFrame(
                    Data1, columns=['H2O_in_mix', 'NH3_in_mix', 'CH3OH_in_mix'], index=[0])
                df1.to_csv('Column_density_' +
                           'H2O_CH3OH_CO_NH3_H2_100K_NASA' + '.csv', index=False)

            elif spn[j] == 'H2O_CH3OH_CO2_CH4_87K_V2':
                guess = (
                    0.1, 5.94, 0.1, 0.03, 6.77, 0.05, 0.03, 6.84, 0.05, 0.03, 7.01, 0.05, 0.01, 7.39, 0.1, 0.02, 7.67, 0.1)
                popt, pcov = curve_fit(six_peaks_G, ntt, y, guess, maxfev=5000)
                pars10b = popt[0:3]
                pars20b = popt[3:6]
                pars30b = popt[6:9]
                pars40b = popt[9:12]
                pars50b = popt[12:15]
                pars60b = popt[15:18]
                peak10b = asym_peak_L(ntt, pars10b)
                peak20b = asym_peak_L(ntt, pars20b)
                peak30b = asym_peak_L(ntt, pars30b)
                peak40b = asym_peak_L(ntt, pars40b)
                peak50b = asym_peak_L(ntt, pars50b)
                peak60b = asym_peak_L(ntt, pars60b)
                # plt.plot(ntt, peak10b, ntt, peak20b, ntt, peak30b, ntt, peak40b, ntt, peak50b, ntt, peak60b)
                P1 = scipy.integrate.trapz(peak10b, ntt_cm)
                P2 = scipy.integrate.trapz(peak20b, ntt_cm)
                P3 = scipy.integrate.trapz(peak30b, ntt_cm)
                P4 = scipy.integrate.trapz(peak40b, ntt_cm)
                P5 = scipy.integrate.trapz(peak50b, ntt_cm)
                P6 = scipy.integrate.trapz(peak60b, ntt_cm)
                N1 = -P1 / A_H2O
                N2 = -P2 / A_CH3OH
                N3 = -P3 / A_CH3OH
                N4 = -P4 / A_CH3OH
                N5 = -P5 / A_H2O
                N6 = -P6 / A_CH4
                N_H2O = N1 + N5
                N_CH3OH = N2 + N3 + N4
                N_CH4 = N6
                Data1 = {'H2O_in_mix': N_H2O,
                         'CH4_in_mix': N_CH4, 'CH3OH_in_mix': N_CH3OH}
                df1 = DataFrame(
                    Data1, columns=['H2O_in_mix', 'CH4_in_mix', 'CH3OH_in_mix'], index=[0])
                df1.to_csv('Column_density_' +
                           'H2O_CH3OH_CO2_CH4_87K_V2' + '.csv', index=False)

            elif spn[j] == 'CO_NH3_10K':
                lam1, lam2 = 6.0, 6.56
                x1, x2 = find_nearest(ntt, lam1), find_nearest(ntt, lam2)
                y1, y2 = y[get_line_number2(
                    x1, ntt)], y[get_line_number2(x2, ntt)]
                guess = (y1, x1, 0.05, y2, x2, 0.05)
                popt, pcov = curve_fit(two_peaks_L, ntt, y, guess, maxfev=5000)
                pars10b = popt[0:3]
                pars20b = popt[3:6]
                peak10b = asym_peak_L(ntt, pars10b)
                peak20b = asym_peak_L(ntt, pars20b)
                tot = peak10b + peak20b
                P1 = scipy.integrate.trapz(peak10b, ntt_cm)
                P2 = scipy.integrate.trapz(peak20b, ntt_cm)
                N1 = -P1 / (
                    A_NH3)  # Term 0.8 takes into account the decreasing in the band strength of CH3CHO in CO ice (See Fig C.3 in Terwisscha van Scheltinga+2018)
                N2 = -P2 / (A_NH3)
                N_NH3 = N1 + N2
                Data1 = {'NH3_in_mix': N_NH3}
                df1 = DataFrame(Data1, columns=['NH3_in_mix'], index=[0])
                df1.to_csv('Column_density_' +
                           'CO_NH3_10K' + '.csv', index=False)
                np.savetxt('Analytic_deconv_CO_NH3_10K.dat',
                           np.transpose([ntt, peak10b, peak20b, tot, y]))

            elif spn[j] == 'H2O_CH4_10_0.6_a_V3':
                lam1, lam2, lam3, lam4, lam5, lam6, lam7 = 6.0, 6.56, 7.18, 7.69, 11.7, 12.84, 14.22
                x1, x2, x3, x4, x5, x6, x7 = find_nearest(ntt, lam1), find_nearest(ntt, lam2), find_nearest(ntt,
                                                                                                            lam3), find_nearest(
                    ntt, lam4), find_nearest(ntt, lam5), find_nearest(ntt, lam6), find_nearest(ntt, lam7)
                y1, y2, y3, y4, y5, y6, y7 = y[get_line_number2(x1, ntt)], y[get_line_number2(x2, ntt)], y[
                    get_line_number2(x3, ntt)], y[get_line_number2(x4, ntt)], y[get_line_number2(x5, ntt)], y[
                    get_line_number2(x6, ntt)], y[get_line_number2(x7, ntt)]
                guess = (y1, x1, 0.1, y2, x2, 0.08, y3, x3, 0.08, y4,
                         x4, 0.01, y5, x5, 0.2, y6, x6, 0.2, y7, x7, 0.2)
                popt, pcov = curve_fit(
                    seven_peaks_G, ntt, y, guess, maxfev=5000)
                pars10b = popt[0:3]
                pars20b = popt[3:6]
                pars30b = popt[6:9]
                pars40b = popt[9:12]
                pars50b = popt[12:15]
                pars60b = popt[15:18]
                pars70b = popt[18:21]
                peak10b = asym_peak_G(ntt, pars10b)
                peak20b = asym_peak_G(ntt, pars20b)
                peak30b = asym_peak_G(ntt, pars30b)
                peak40b = asym_peak_G(ntt, pars40b)
                peak50b = asym_peak_G(ntt, pars50b)
                peak60b = asym_peak_G(ntt, pars60b)
                peak70b = asym_peak_G(ntt, pars70b)
                tot = peak10b + peak20b + peak30b + peak40b + peak50b + peak60b + peak70b
                P1 = scipy.integrate.trapz(peak10b, ntt_cm)
                P2 = scipy.integrate.trapz(peak20b, ntt_cm)
                P3 = scipy.integrate.trapz(peak30b, ntt_cm)
                P4 = scipy.integrate.trapz(peak40b, ntt_cm)
                P5 = scipy.integrate.trapz(peak50b, ntt_cm)
                P6 = scipy.integrate.trapz(peak60b, ntt_cm)
                P7 = scipy.integrate.trapz(peak70b, ntt_cm)
                N1 = -P1 / A_H2O
                N2 = -P2 / A_H2O
                N3 = -P3 / A_H2O
                N4 = -P4 / A_CH4
                N5 = -P5 / A_H2O_2
                N6 = -P6 / A_H2O_2
                N7 = -P7 / A_H2O_2
                N_H2O = N1 + N2 + N3
                N_CH4 = N4
                Data1 = {'H2O_in_mix': N_H2O, 'CH4_in_mix': N_CH4}
                df1 = DataFrame(
                    Data1, columns=['H2O_in_mix', 'CH4_in_mix'], index=[0])
                df1.to_csv('Column_density_' +
                           'H2O_CH4_10_0.6_a_V3' + '.csv', index=False)
                np.savetxt('Analytic_deconv_H2O_CH4_a_V3.dat',
                           np.transpose([ntt, peak10b, peak20b, peak30b, peak40b, peak50b, peak60b, peak70b, tot, y]))

            elif spn[j] == 'H2O_CH4_10_0.6_a':
                lam1, lam2, lam3, lam4, lam5, lam6, lam7 = 6.0, 6.56, 7.18, 7.69, 11.7, 12.84, 14.22
                x1, x2, x3, x4, x5, x6, x7 = find_nearest(ntt, lam1), find_nearest(ntt, lam2), find_nearest(ntt,
                                                                                                            lam3), find_nearest(
                    ntt, lam4), find_nearest(ntt, lam5), find_nearest(ntt, lam6), find_nearest(ntt, lam7)
                y1, y2, y3, y4, y5, y6, y7 = y[get_line_number2(x1, ntt)], y[get_line_number2(x2, ntt)], y[
                    get_line_number2(x3, ntt)], y[get_line_number2(x4, ntt)], y[get_line_number2(x5, ntt)], y[
                    get_line_number2(x6, ntt)], y[get_line_number2(x7, ntt)]
                guess = (y1, x1, 0.1, y2, x2, 0.08, y3, x3, 0.08, y4,
                         x4, 0.01, y5, x5, 0.2, y6, x6, 0.2, y7, x7, 0.2)
                popt, pcov = curve_fit(
                    seven_peaks_G, ntt, y, guess, maxfev=5000)
                pars10b = popt[0:3]
                pars20b = popt[3:6]
                pars30b = popt[6:9]
                pars40b = popt[9:12]
                pars50b = popt[12:15]
                pars60b = popt[15:18]
                pars70b = popt[18:21]
                peak10b = asym_peak_G(ntt, pars10b)
                peak20b = asym_peak_G(ntt, pars20b)
                peak30b = asym_peak_G(ntt, pars30b)
                peak40b = asym_peak_G(ntt, pars40b)
                peak50b = asym_peak_G(ntt, pars50b)
                peak60b = asym_peak_G(ntt, pars60b)
                peak70b = asym_peak_G(ntt, pars70b)
                tot = peak10b + peak20b + peak30b + peak40b + peak50b + peak60b + peak70b
                P1 = scipy.integrate.trapz(peak10b, ntt_cm)
                P2 = scipy.integrate.trapz(peak20b, ntt_cm)
                P3 = scipy.integrate.trapz(peak30b, ntt_cm)
                P4 = scipy.integrate.trapz(peak40b, ntt_cm)
                P5 = scipy.integrate.trapz(peak50b, ntt_cm)
                P6 = scipy.integrate.trapz(peak60b, ntt_cm)
                P7 = scipy.integrate.trapz(peak70b, ntt_cm)
                N1 = -P1 / A_H2O
                N2 = -P2 / A_H2O
                N3 = -P3 / A_H2O
                N4 = -P4 / A_CH4
                N5 = -P5 / A_H2O_2
                N6 = -P6 / A_H2O_2
                N7 = -P7 / A_H2O_2
                N_H2O = N1 + N2 + N3
                N_CH4 = N4
                Data1 = {'H2O_in_mix': N_H2O, 'CH4_in_mix': N_CH4}
                df1 = DataFrame(
                    Data1, columns=['H2O_in_mix', 'CH4_in_mix'], index=[0])
                df1.to_csv('Column_density_' +
                           'H2O_CH4_10_0.6_a_V3' + '.csv', index=False)
                np.savetxt('Analytic_deconv_H2O_CH4_a.dat',
                           np.transpose([ntt, peak10b, peak20b, peak30b, peak40b, peak50b, peak60b, peak70b, tot, y]))

            elif spn[j] == 'H2O_NH3_CO_10_06_04_a':
                print('Decomposing H2O_NH3_CO_10_06_04_a')
                lam1, lam2, lam3, lam4, lam5, lam6, lam7 = 4.65, 6.67, 8.73, 9.05, 10.86, 11.92, 13.29
                x1, x2, x3, x4, x5, x6, x7 = find_nearest(ntt, lam1), find_nearest(ntt, lam2), find_nearest(ntt,
                                                                                                            lam3), find_nearest(
                    ntt, lam4), find_nearest(ntt, lam5), find_nearest(ntt, lam6), find_nearest(ntt, lam7)
                y1, y2, y3, y4, y5, y6, y7 = y[get_line_number2(x1, ntt)], y[get_line_number2(x2, ntt)], y[
                    get_line_number2(x3, ntt)], y[get_line_number2(x4, ntt)], y[get_line_number2(x5, ntt)], y[
                    get_line_number2(x6, ntt)], y[get_line_number2(x7, ntt)]
                guess = (
                    y1, x1, 0.01, y2, x2, 0.02, y3, x3, 0.36, y4, x4, 0.46, y5, x5, 0.84, y6, x6, 1.47, y7, x7, 1.72)
                popt, pcov = curve_fit(
                    seven_peaks_G, ntt, y, guess, maxfev=5000)
                pars10b = popt[0:3]
                pars20b = popt[3:6]
                pars30b = popt[6:9]
                pars40b = popt[9:12]
                pars50b = popt[12:15]
                pars60b = popt[15:18]
                pars70b = popt[18:21]
                peak10b = asym_peak_G(ntt, pars10b)
                peak20b = asym_peak_G(ntt, pars20b)
                peak30b = asym_peak_G(ntt, pars30b)
                peak40b = asym_peak_G(ntt, pars40b)
                peak50b = asym_peak_G(ntt, pars50b)
                peak60b = asym_peak_G(ntt, pars60b)
                peak70b = asym_peak_G(ntt, pars70b)
                tot = peak10b + peak20b + peak30b + peak40b + peak50b + peak60b + peak70b
                P1 = scipy.integrate.trapz(peak10b, ntt_cm)
                P2 = scipy.integrate.trapz(peak20b, ntt_cm)
                P3 = scipy.integrate.trapz(peak30b, ntt_cm)
                P4 = scipy.integrate.trapz(peak40b, ntt_cm)
                P5 = scipy.integrate.trapz(peak50b, ntt_cm)
                P6 = scipy.integrate.trapz(peak60b, ntt_cm)
                P7 = scipy.integrate.trapz(peak70b, ntt_cm)
                N1 = -P1 / A_CO
                N2 = -P2 / A_CO
                N3 = -P3 / A_NH3
                N4 = -P4 / A_NH3
                N5 = -P5 / A_H2O_2
                N6 = -P6 / A_H2O_2
                N7 = -P7 / A_H2O_2
                N_CO = N1 + N2
                N_NH3 = N3 + N4
                N_H2O = N5 + N6 + N7
                Data1 = {'H2O_in_mix': N_H2O,
                         'NH3_in_mix': N_NH3, 'CO_in_mix': N_CO}
                df1 = DataFrame(
                    Data1, columns=['H2O_in_mix', 'NH3_in_mix', 'CO_in_mix'], index=[0])
                df1.to_csv('Column_density_' +
                           'H2O_NH3_CO_10_06_04_a' + '.csv', index=False)
                np.savetxt('Analytic_deconv_H2O_NH3_CO_10_06_04_a.dat',
                           np.transpose([ntt, peak10b, peak20b, peak30b, peak40b, peak50b, peak60b, peak70b, tot, y]))

            elif spn[j] == 'H2O_NH3_CO_10_06_04_b':
                lam1, lam2, lam3, lam4, lam5, lam6, lam7 = 4.65, 6.67, 8.73, 9.05, 10.86, 11.92, 13.29
                x1, x2, x3, x4, x5, x6, x7 = find_nearest(ntt, lam1), find_nearest(ntt, lam2), find_nearest(ntt,
                                                                                                            lam3), find_nearest(
                    ntt, lam4), find_nearest(ntt, lam5), find_nearest(ntt, lam6), find_nearest(ntt, lam7)
                y1, y2, y3, y4, y5, y6, y7 = y[get_line_number2(x1, ntt)], y[get_line_number2(x2, ntt)], y[
                    get_line_number2(x3, ntt)], y[get_line_number2(x4, ntt)], y[get_line_number2(x5, ntt)], y[
                    get_line_number2(x6, ntt)], y[get_line_number2(x7, ntt)]
                guess = (
                    y1, x1, 0.01, y2, x2, 0.02, y3, x3, 0.36, y4, x4, 0.46, y5, x5, 0.84, y6, x6, 1.47, y7, x7, 1.72)
                popt, pcov = curve_fit(
                    seven_peaks_G, ntt, y, guess, maxfev=5000)
                pars10b = popt[0:3]
                pars20b = popt[3:6]
                pars30b = popt[6:9]
                pars40b = popt[9:12]
                pars50b = popt[12:15]
                pars60b = popt[15:18]
                pars70b = popt[18:21]
                peak10b = asym_peak_G(ntt, pars10b)
                peak20b = asym_peak_G(ntt, pars20b)
                peak30b = asym_peak_G(ntt, pars30b)
                peak40b = asym_peak_G(ntt, pars40b)
                peak50b = asym_peak_G(ntt, pars50b)
                peak60b = asym_peak_G(ntt, pars60b)
                peak70b = asym_peak_G(ntt, pars70b)
                tot = peak10b + peak20b + peak30b + peak40b + peak50b + peak60b + peak70b
                P1 = scipy.integrate.trapz(peak10b, ntt_cm)
                P2 = scipy.integrate.trapz(peak20b, ntt_cm)
                P3 = scipy.integrate.trapz(peak30b, ntt_cm)
                P4 = scipy.integrate.trapz(peak40b, ntt_cm)
                P5 = scipy.integrate.trapz(peak50b, ntt_cm)
                P6 = scipy.integrate.trapz(peak60b, ntt_cm)
                P7 = scipy.integrate.trapz(peak70b, ntt_cm)
                N1 = -P1 / A_CO
                N2 = -P2 / A_CO
                N3 = -P3 / A_NH3
                N4 = -P4 / A_NH3
                N5 = -P5 / A_H2O_2
                N6 = -P6 / A_H2O_2
                N7 = -P7 / A_H2O_2
                N_CO = N1 + N2
                N_NH3 = N3 + N4
                N_H2O = N5 + N6 + N7
                Data1 = {'H2O_in_mix': N_H2O,
                         'NH3_in_mix': N_NH3, 'CO_in_mix': N_CO}
                df1 = DataFrame(
                    Data1, columns=['H2O_in_mix', 'NH3_in_mix', 'CO_in_mix'], index=[0])
                df1.to_csv('Column_density_' +
                           'H2O_NH3_CO_10_06_04_b' + '.csv', index=False)
                np.savetxt('Analytic_deconv_H2O_NH3_CO_10_06_04_b.dat',
                           np.transpose([ntt, peak10b, peak20b, peak30b, peak40b, peak50b, peak60b, peak70b, tot, y]))

            elif spn[j] == 'H2O_NH3_CO_10_06_04_c':
                lam1, lam2, lam3, lam4, lam5, lam6, lam7 = 4.65, 6.67, 8.73, 9.05, 10.86, 11.92, 13.29
                x1, x2, x3, x4, x5, x6, x7 = find_nearest(ntt, lam1), find_nearest(ntt, lam2), find_nearest(ntt,
                                                                                                            lam3), find_nearest(
                    ntt, lam4), find_nearest(ntt, lam5), find_nearest(ntt, lam6), find_nearest(ntt, lam7)
                y1, y2, y3, y4, y5, y6, y7 = y[get_line_number2(x1, ntt)], y[get_line_number2(x2, ntt)], y[
                    get_line_number2(x3, ntt)], y[get_line_number2(x4, ntt)], y[get_line_number2(x5, ntt)], y[
                    get_line_number2(x6, ntt)], y[get_line_number2(x7, ntt)]
                guess = (
                    y1, x1, 0.01, y2, x2, 0.02, y3, x3, 0.36, y4, x4, 0.46, y5, x5, 0.84, y6, x6, 1.47, y7, x7, 1.72)
                popt, pcov = curve_fit(
                    seven_peaks_G, ntt, y, guess, maxfev=5000)
                pars10b = popt[0:3]
                pars20b = popt[3:6]
                pars30b = popt[6:9]
                pars40b = popt[9:12]
                pars50b = popt[12:15]
                pars60b = popt[15:18]
                pars70b = popt[18:21]
                peak10b = asym_peak_G(ntt, pars10b)
                peak20b = asym_peak_G(ntt, pars20b)
                peak30b = asym_peak_G(ntt, pars30b)
                peak40b = asym_peak_G(ntt, pars40b)
                peak50b = asym_peak_G(ntt, pars50b)
                peak60b = asym_peak_G(ntt, pars60b)
                peak70b = asym_peak_G(ntt, pars70b)
                tot = peak10b + peak20b + peak30b + peak40b + peak50b + peak60b + peak70b
                P1 = scipy.integrate.trapz(peak10b, ntt_cm)
                P2 = scipy.integrate.trapz(peak20b, ntt_cm)
                P3 = scipy.integrate.trapz(peak30b, ntt_cm)
                P4 = scipy.integrate.trapz(peak40b, ntt_cm)
                P5 = scipy.integrate.trapz(peak50b, ntt_cm)
                P6 = scipy.integrate.trapz(peak60b, ntt_cm)
                P7 = scipy.integrate.trapz(peak70b, ntt_cm)
                N1 = -P1 / A_CO
                N2 = -P2 / A_CO
                N3 = -P3 / A_NH3
                N4 = -P4 / A_NH3
                N5 = -P5 / A_H2O_2
                N6 = -P6 / A_H2O_2
                N7 = -P7 / A_H2O_2
                N_CO = N1 + N2
                N_NH3 = N3 + N4
                N_H2O = N5 + N6 + N7
                Data1 = {'H2O_in_mix': N_H2O,
                         'NH3_in_mix': N_NH3, 'CO_in_mix': N_CO}
                df1 = DataFrame(
                    Data1, columns=['H2O_in_mix', 'NH3_in_mix', 'CO_in_mix'], index=[0])
                df1.to_csv('Column_density_' +
                           'H2O_NH3_CO_10_06_04_c' + '.csv', index=False)
                np.savetxt('Analytic_deconv_H2O_NH3_CO_10_06_04_c.dat',
                           np.transpose([ntt, peak10b, peak20b, peak30b, peak40b, peak50b, peak60b, peak70b, tot, y]))

            elif spn[j] == 'H2O_CH4_1_0.6_a':
                lam1, lam2, lam3, lam4, lam5, lam6, lam7 = 6.0, 6.56, 7.18, 7.69, 11.7, 12.84, 14.22
                x1, x2, x3, x4, x5, x6, x7 = find_nearest(ntt, lam1), find_nearest(ntt, lam2), find_nearest(ntt,
                                                                                                            lam3), find_nearest(
                    ntt, lam4), find_nearest(ntt, lam5), find_nearest(ntt, lam6), find_nearest(ntt, lam7)
                y1, y2, y3, y4, y5, y6, y7 = y[get_line_number2(x1, ntt)], y[get_line_number2(x2, ntt)], y[
                    get_line_number2(x3, ntt)], y[get_line_number2(x4, ntt)], y[get_line_number2(x5, ntt)], y[
                    get_line_number2(x6, ntt)], y[get_line_number2(x7, ntt)]
                guess = (y1, x1, 0.1, y2, x2, 0.08, y3, x3, 0.08, y4,
                         x4, 0.01, y5, x5, 0.2, y6, x6, 0.2, y7, x7, 0.2)
                popt, pcov = curve_fit(
                    seven_peaks_G, ntt, y, guess, maxfev=5000)
                pars10b = popt[0:3]
                pars20b = popt[3:6]
                pars30b = popt[6:9]
                pars40b = popt[9:12]
                pars50b = popt[12:15]
                pars60b = popt[15:18]
                pars70b = popt[18:21]
                peak10b = asym_peak_G(ntt, pars10b)
                peak20b = asym_peak_G(ntt, pars20b)
                peak30b = asym_peak_G(ntt, pars30b)
                peak40b = asym_peak_G(ntt, pars40b)
                peak50b = asym_peak_G(ntt, pars50b)
                peak60b = asym_peak_G(ntt, pars60b)
                peak70b = asym_peak_G(ntt, pars70b)
                tot = peak10b + peak20b + peak30b + peak40b + peak50b + peak60b + peak70b
                P1 = scipy.integrate.trapz(peak10b, ntt_cm)
                P2 = scipy.integrate.trapz(peak20b, ntt_cm)
                P3 = scipy.integrate.trapz(peak30b, ntt_cm)
                P4 = scipy.integrate.trapz(peak40b, ntt_cm)
                P5 = scipy.integrate.trapz(peak50b, ntt_cm)
                P6 = scipy.integrate.trapz(peak60b, ntt_cm)
                P7 = scipy.integrate.trapz(peak70b, ntt_cm)
                N1 = -P1 / A_H2O
                N2 = -P2 / A_H2O
                N3 = -P3 / A_H2O
                N4 = -P4 / A_CH4
                N5 = -P5 / A_H2O_2
                N6 = -P6 / A_H2O_2
                N7 = -P7 / A_H2O_2
                N_H2O = N1 + N2 + N3
                N_CH4 = N4
                Data1 = {'H2O_in_mix': N_H2O, 'CH4_in_mix': N_CH4}
                df1 = DataFrame(
                    Data1, columns=['H2O_in_mix', 'CH4_in_mix'], index=[0])
                df1.to_csv('Column_density_' +
                           'H2O_CH4_1_0.6_a' + '.csv', index=False)
                np.savetxt('Analytic_deconv_H2O_CH4_a.dat',
                           np.transpose([ntt, peak10b, peak20b, peak30b, peak40b, peak50b, peak60b, peak70b, tot, y]))

            elif spn[j] == 'H2O_40K':
                lam1, lam2, lam3, lam4, lam5, lam6 = 6.0, 6.56, 7.18, 11.7, 12.84, 14.22
                x1, x2, x3, x4, x5, x6 = find_nearest(ntt, lam1), find_nearest(ntt, lam2), find_nearest(ntt,
                                                                                                        lam3), find_nearest(
                    ntt, lam4), find_nearest(ntt, lam5), find_nearest(ntt, lam6)
                y1, y2, y3, y4, y5, y6 = y[get_line_number2(x1, ntt)], y[get_line_number2(x2, ntt)], y[
                    get_line_number2(x3, ntt)], y[get_line_number2(x4, ntt)], y[get_line_number2(x5, ntt)], y[
                    get_line_number2(x6, ntt)]
                guess = (y1, x1, 0.1, y2, x2, 0.08, y3, x3, 0.08,
                         y4, x4, 0.1, y5, x5, 0.1, y6, x6, 0.1)
                popt, pcov = curve_fit(six_peaks_G, ntt, y, guess, maxfev=5000)
                pars10b = popt[0:3]
                pars20b = popt[3:6]
                pars30b = popt[6:9]
                pars40b = popt[9:12]
                pars50b = popt[12:15]
                pars60b = popt[15:18]
                peak10b = asym_peak_G(ntt, pars10b)
                peak20b = asym_peak_G(ntt, pars20b)
                peak30b = asym_peak_G(ntt, pars30b)
                peak40b = asym_peak_G(ntt, pars40b)
                peak50b = asym_peak_G(ntt, pars50b)
                peak60b = asym_peak_G(ntt, pars60b)
                tot = peak10b + peak20b + peak30b + peak40b + peak50b + peak60b
                P1 = scipy.integrate.trapz(peak10b, ntt_cm)
                P2 = scipy.integrate.trapz(peak20b, ntt_cm)
                P3 = scipy.integrate.trapz(peak30b, ntt_cm)
                P4 = scipy.integrate.trapz(peak40b, ntt_cm)
                P5 = scipy.integrate.trapz(peak50b, ntt_cm)
                P6 = scipy.integrate.trapz(peak60b, ntt_cm)
                N1 = -P1 / A_H2O
                N2 = -P2 / A_H2O
                N3 = -P3 / A_H2O
                N4 = -P4 / A_H2O_2
                N5 = -P5 / A_H2O_2
                N6 = -P6 / A_H2O_2
                N_H2O = N1 + N2 + N3
                Data1 = {'H2O_pure': N_H2O}
                df1 = DataFrame(Data1, columns=['H2O_pure'], index=[0])
                df1.to_csv('Column_density_' +
                           'H2O_pure' + '.csv', index=False)
                np.savetxt('Analytic_deconv_H2O_40K.dat',
                           np.transpose([ntt, peak10b, peak20b, peak30b, peak40b, peak50b, peak60b, tot, y]))

            elif spn[j] == 'H2O_80K':
                lam1, lam2, lam3, lam4, lam5, lam6 = 6.0, 6.56, 7.18, 11.7, 12.84, 14.22
                x1, x2, x3, x4, x5, x6 = find_nearest(ntt, lam1), find_nearest(ntt, lam2), find_nearest(ntt,
                                                                                                        lam3), find_nearest(
                    ntt, lam4), find_nearest(ntt, lam5), find_nearest(ntt, lam6)
                y1, y2, y3, y4, y5, y6 = y[get_line_number2(x1, ntt)], y[get_line_number2(x2, ntt)], y[
                    get_line_number2(x3, ntt)], y[get_line_number2(x4, ntt)], y[get_line_number2(x5, ntt)], y[
                    get_line_number2(x6, ntt)]
                guess = (y1, x1, 0.1, y2, x2, 0.08, y3, x3, 0.08,
                         y4, x4, 0.1, y5, x5, 0.1, y6, x6, 0.1)
                popt, pcov = curve_fit(six_peaks_G, ntt, y, guess, maxfev=5000)
                pars10b = popt[0:3]
                pars20b = popt[3:6]
                pars30b = popt[6:9]
                pars40b = popt[9:12]
                pars50b = popt[12:15]
                pars60b = popt[15:18]
                peak10b = asym_peak_G(ntt, pars10b)
                peak20b = asym_peak_G(ntt, pars20b)
                peak30b = asym_peak_G(ntt, pars30b)
                peak40b = asym_peak_G(ntt, pars40b)
                peak50b = asym_peak_G(ntt, pars50b)
                peak60b = asym_peak_G(ntt, pars60b)
                tot = peak10b + peak20b + peak30b + peak40b + peak50b + peak60b
                P1 = scipy.integrate.trapz(peak10b, ntt_cm)
                P2 = scipy.integrate.trapz(peak20b, ntt_cm)
                P3 = scipy.integrate.trapz(peak30b, ntt_cm)
                P4 = scipy.integrate.trapz(peak40b, ntt_cm)
                P5 = scipy.integrate.trapz(peak50b, ntt_cm)
                P6 = scipy.integrate.trapz(peak60b, ntt_cm)
                N1 = -P1 / A_H2O
                N2 = -P2 / A_H2O
                N3 = -P3 / A_H2O
                N4 = -P4 / A_H2O_2
                N5 = -P5 / A_H2O_2
                N6 = -P6 / A_H2O_2
                N_H2O = N1 + N2 + N3
                Data1 = {'H2O_pure': N_H2O}
                df1 = DataFrame(Data1, columns=['H2O_pure'], index=[0])
                df1.to_csv('Column_density_' +
                           'H2O_pure' + '.csv', index=False)
                np.savetxt('Analytic_deconv_H2O_80K.dat',
                           np.transpose([ntt, peak10b, peak20b, peak30b, peak40b, peak50b, peak60b, tot, y]))

            elif spn[j] == 'H2O_100K':
                lam1, lam2, lam3, lam4, lam5, lam6 = 6.0, 6.56, 7.18, 11.7, 12.84, 14.22
                x1, x2, x3, x4, x5, x6 = find_nearest(ntt, lam1), find_nearest(ntt, lam2), find_nearest(ntt,
                                                                                                        lam3), find_nearest(
                    ntt, lam4), find_nearest(ntt, lam5), find_nearest(ntt, lam6)
                y1, y2, y3, y4, y5, y6 = y[get_line_number2(x1, ntt)], y[get_line_number2(x2, ntt)], y[
                    get_line_number2(x3, ntt)], y[get_line_number2(x4, ntt)], y[get_line_number2(x5, ntt)], y[
                    get_line_number2(x6, ntt)]
                guess = (y1, x1, 0.1, y2, x2, 0.08, y3, x3, 0.08,
                         y4, x4, 0.1, y5, x5, 0.1, y6, x6, 0.1)
                popt, pcov = curve_fit(six_peaks_G, ntt, y, guess, maxfev=5000)
                pars10b = popt[0:3]
                pars20b = popt[3:6]
                pars30b = popt[6:9]
                pars40b = popt[9:12]
                pars50b = popt[12:15]
                pars60b = popt[15:18]
                peak10b = asym_peak_G(ntt, pars10b)
                peak20b = asym_peak_G(ntt, pars20b)
                peak30b = asym_peak_G(ntt, pars30b)
                peak40b = asym_peak_G(ntt, pars40b)
                peak50b = asym_peak_G(ntt, pars50b)
                peak60b = asym_peak_G(ntt, pars60b)
                tot = peak10b + peak20b + peak30b + peak40b + peak50b + peak60b
                P1 = scipy.integrate.trapz(peak10b, ntt_cm)
                P2 = scipy.integrate.trapz(peak20b, ntt_cm)
                P3 = scipy.integrate.trapz(peak30b, ntt_cm)
                P4 = scipy.integrate.trapz(peak40b, ntt_cm)
                P5 = scipy.integrate.trapz(peak50b, ntt_cm)
                P6 = scipy.integrate.trapz(peak60b, ntt_cm)
                N1 = -P1 / A_H2O
                N2 = -P2 / A_H2O
                N3 = -P3 / A_H2O
                N4 = -P4 / A_H2O_2
                N5 = -P5 / A_H2O_2
                N6 = -P6 / A_H2O_2
                N_H2O = N1 + N2 + N3
                Data1 = {'H2O_pure': N_H2O}
                df1 = DataFrame(Data1, columns=['H2O_pure'], index=[0])
                df1.to_csv('Column_density_' +
                           'H2O_pure' + '.csv', index=False)
                np.savetxt('Analytic_deconv_H2O_100K.dat',
                           np.transpose([ntt, peak10b, peak20b, peak30b, peak40b, peak50b, peak60b, tot, y]))

            elif spn[j] == 'H2O_120K':
                lam1, lam2, lam3, lam4, lam5, lam6 = 6.0, 6.56, 7.18, 11.7, 12.84, 14.22
                x1, x2, x3, x4, x5, x6 = find_nearest(ntt, lam1), find_nearest(ntt, lam2), find_nearest(ntt,
                                                                                                        lam3), find_nearest(
                    ntt, lam4), find_nearest(ntt, lam5), find_nearest(ntt, lam6)
                y1, y2, y3, y4, y5, y6 = y[get_line_number2(x1, ntt)], y[get_line_number2(x2, ntt)], y[
                    get_line_number2(x3, ntt)], y[get_line_number2(x4, ntt)], y[get_line_number2(x5, ntt)], y[
                    get_line_number2(x6, ntt)]
                guess = (y1, x1, 0.1, y2, x2, 0.08, y3, x3, 0.08,
                         y4, x4, 0.1, y5, x5, 0.1, y6, x6, 0.1)
                popt, pcov = curve_fit(six_peaks_G, ntt, y, guess, maxfev=5000)
                pars10b = popt[0:3]
                pars20b = popt[3:6]
                pars30b = popt[6:9]
                pars40b = popt[9:12]
                pars50b = popt[12:15]
                pars60b = popt[15:18]
                peak10b = asym_peak_G(ntt, pars10b)
                peak20b = asym_peak_G(ntt, pars20b)
                peak30b = asym_peak_G(ntt, pars30b)
                peak40b = asym_peak_G(ntt, pars40b)
                peak50b = asym_peak_G(ntt, pars50b)
                peak60b = asym_peak_G(ntt, pars60b)
                tot = peak10b + peak20b + peak30b + peak40b + peak50b + peak60b
                P1 = scipy.integrate.trapz(peak10b, ntt_cm)
                P2 = scipy.integrate.trapz(peak20b, ntt_cm)
                P3 = scipy.integrate.trapz(peak30b, ntt_cm)
                P4 = scipy.integrate.trapz(peak40b, ntt_cm)
                P5 = scipy.integrate.trapz(peak50b, ntt_cm)
                P6 = scipy.integrate.trapz(peak60b, ntt_cm)
                N1 = -P1 / A_H2O
                N2 = -P2 / A_H2O
                N3 = -P3 / A_H2O
                N4 = -P4 / A_H2O_2
                N5 = -P5 / A_H2O_2
                N6 = -P6 / A_H2O_2
                N_H2O = N1 + N2 + N3
                Data1 = {'H2O_pure': N_H2O}
                df1 = DataFrame(Data1, columns=['H2O_pure'], index=[0])
                df1.to_csv('Column_density_' +
                           'H2O_pure' + '.csv', index=False)
                np.savetxt('Analytic_deconv_H2O_120K.dat',
                           np.transpose([ntt, peak10b, peak20b, peak30b, peak40b, peak50b, peak60b, tot, y]))

            elif spn[j] == 'H2O_140K':
                lam1, lam2, lam3, lam4, lam5, lam6 = 6.0, 6.56, 7.18, 11.7, 12.84, 14.22
                x1, x2, x3, x4, x5, x6 = find_nearest(ntt, lam1), find_nearest(ntt, lam2), find_nearest(ntt,
                                                                                                        lam3), find_nearest(
                    ntt, lam4), find_nearest(ntt, lam5), find_nearest(ntt, lam6)
                y1, y2, y3, y4, y5, y6 = y[get_line_number2(x1, ntt)], y[get_line_number2(x2, ntt)], y[
                    get_line_number2(x3, ntt)], y[get_line_number2(x4, ntt)], y[get_line_number2(x5, ntt)], y[
                    get_line_number2(x6, ntt)]
                guess = (y1, x1, 0.1, y2, x2, 0.08, y3, x3, 0.08,
                         y4, x4, 0.1, y5, x5, 0.1, y6, x6, 0.1)
                popt, pcov = curve_fit(six_peaks_G, ntt, y, guess, maxfev=5000)
                pars10b = popt[0:3]
                pars20b = popt[3:6]
                pars30b = popt[6:9]
                pars40b = popt[9:12]
                pars50b = popt[12:15]
                pars60b = popt[15:18]
                peak10b = asym_peak_G(ntt, pars10b)
                peak20b = asym_peak_G(ntt, pars20b)
                peak30b = asym_peak_G(ntt, pars30b)
                peak40b = asym_peak_G(ntt, pars40b)
                peak50b = asym_peak_G(ntt, pars50b)
                peak60b = asym_peak_G(ntt, pars60b)
                tot = peak10b + peak20b + peak30b + peak40b + peak50b + peak60b
                P1 = scipy.integrate.trapz(peak10b, ntt_cm)
                P2 = scipy.integrate.trapz(peak20b, ntt_cm)
                P3 = scipy.integrate.trapz(peak30b, ntt_cm)
                P4 = scipy.integrate.trapz(peak40b, ntt_cm)
                P5 = scipy.integrate.trapz(peak50b, ntt_cm)
                P6 = scipy.integrate.trapz(peak60b, ntt_cm)
                N1 = -P1 / A_H2O
                N2 = -P2 / A_H2O
                N3 = -P3 / A_H2O
                N4 = -P4 / A_H2O_2
                N5 = -P5 / A_H2O_2
                N6 = -P6 / A_H2O_2
                N_H2O = N1 + N2 + N3
                Data1 = {'H2O_pure': N_H2O}
                df1 = DataFrame(Data1, columns=['H2O_pure'], index=[0])
                df1.to_csv('Column_density_' +
                           'H2O_pure' + '.csv', index=False)
                np.savetxt('Analytic_deconv_H2O_140K.dat',
                           np.transpose([ntt, peak10b, peak20b, peak30b, peak40b, peak50b, peak60b, tot, y]))

            elif spn[j] == 'HCOOH_30.0K':
                print('Decomposing HCOOH at 30K')
                lam1, lam2, lam3, lam4, lam5, lam6, lam7, lam8, lam9 = 5.85, 6.12, 7.23, 8.2, 8.53, 9.35, 10.7, 11.7, 14.6
                x1, x2, x3, x4, x5, x6, x7, x8, x9 = find_nearest(ntt, lam1), find_nearest(ntt, lam2), find_nearest(ntt,
                                                                                                                    lam3), find_nearest(
                    ntt, lam4), find_nearest(ntt, lam5), find_nearest(ntt, lam6), find_nearest(ntt, lam7), find_nearest(
                    ntt, lam8), find_nearest(ntt, lam9)
                y1, y2, y3, y4, y5, y6, y7, y8, y9 = y[get_line_number2(x1, ntt)], y[get_line_number2(x2, ntt)], y[
                    get_line_number2(x3, ntt)], y[get_line_number2(x4, ntt)], y[get_line_number2(x5, ntt)], y[
                    get_line_number2(x6, ntt)], y[get_line_number2(x7, ntt)], y[
                    get_line_number2(x8, ntt)], y[get_line_number2(x9, ntt)]
                guess = (
                    y1, x1, 0.01, y2, x2, 0.05, y3, x3, 0.03, y4, x4, 0.03, y5, x5, 0.05, y6, x6, 0.01, y7, x7, 0.05, y8,
                    x8, 0.05, y9, x9, 0.05)
                plt.plot(ntt, nine_peaks_G(ntt, *guess), 'g-')
                plt.plot(ntt, y)
                plt.show()

                popt, pcov = curve_fit(
                    nine_peaks_G, ntt, y, guess, maxfev=5000)
                pars10b = popt[0:3]
                pars20b = popt[3:6]
                pars30b = popt[6:9]
                pars40b = popt[9:12]
                pars50b = popt[12:15]
                pars60b = popt[15:18]
                pars70b = popt[18:21]
                pars80b = popt[21:24]
                pars90b = popt[24:27]
                peak10b = asym_peak_G(ntt, pars10b)
                peak20b = asym_peak_G(ntt, pars20b)
                peak30b = asym_peak_G(ntt, pars30b)
                peak40b = asym_peak_G(ntt, pars40b)
                peak50b = asym_peak_G(ntt, pars50b)
                peak60b = asym_peak_G(ntt, pars60b)
                peak70b = asym_peak_G(ntt, pars70b)
                peak80b = asym_peak_G(ntt, pars80b)
                peak90b = asym_peak_G(ntt, pars90b)
                tot = peak10b + peak20b + peak30b + peak40b + \
                    peak50b + peak60b + peak70b + peak80b + peak90b
                P1 = scipy.integrate.trapz(peak10b, ntt_cm)
                P2 = scipy.integrate.trapz(peak20b, ntt_cm)
                P3 = scipy.integrate.trapz(peak30b, ntt_cm)
                P4 = scipy.integrate.trapz(peak40b, ntt_cm)
                P5 = scipy.integrate.trapz(peak50b, ntt_cm)
                P6 = scipy.integrate.trapz(peak60b, ntt_cm)
                P7 = scipy.integrate.trapz(peak70b, ntt_cm)
                P8 = scipy.integrate.trapz(peak80b, ntt_cm)
                P9 = scipy.integrate.trapz(peak90b, ntt_cm)
                N1 = 0.
                N2 = 0.
                N3 = 0.
                N4 = -P4 / A_HCOOH
                N5 = -P5 / A_HCOOH
                N_HCOOH = N4 + N5
                Data1 = {'HCOOH_pure': N_HCOOH}
                df1 = DataFrame(Data1, columns=['HCOOH_pure'], index=[0])
                df1.to_csv('Column_density_' +
                           'HCOOH_pure' + '.csv', index=False)
                np.savetxt('Analytic_deconv_HCOOH_30K.dat', np.transpose(
                    [ntt, peak10b, peak20b, peak30b, peak40b, peak50b, peak60b, peak70b, peak80b, peak90b, tot, y]))

            elif spn[j] == 'HCOOH_15.0K':
                lam1, lam2, lam3, lam4, lam5, lam6, lam7, lam8, lam9 = 5.85, 6.12, 7.23, 8.2, 8.53, 9.35, 10.7, 11.7, 14.6
                x1, x2, x3, x4, x5, x6, x7, x8, x9 = find_nearest(ntt, lam1), find_nearest(ntt, lam2), find_nearest(ntt,
                                                                                                                    lam3), find_nearest(
                    ntt, lam4), find_nearest(ntt, lam5), find_nearest(ntt, lam6), find_nearest(ntt, lam7), find_nearest(
                    ntt, lam8), find_nearest(ntt, lam9)
                y1, y2, y3, y4, y5, y6, y7, y8, y9 = y[get_line_number2(x1, ntt)], y[get_line_number2(x2, ntt)], y[
                    get_line_number2(x3, ntt)], y[get_line_number2(x4, ntt)], y[get_line_number2(x5, ntt)], y[
                    get_line_number2(x6, ntt)], y[get_line_number2(x7, ntt)], y[
                    get_line_number2(x8, ntt)], y[get_line_number2(x9, ntt)]
                guess = (
                    y1, x1, 0.01, y2, x2, 0.05, y3, x3, 0.03, y4, x4, 0.03, y5, x5, 0.05, y6, x6, 0.01, y7, x7, 0.05, y8,
                    x8, 0.05, y9, x9, 0.05)
                popt, pcov = curve_fit(
                    nine_peaks_G, ntt, y, guess, maxfev=5000)
                pars10b = popt[0:3]
                pars20b = popt[3:6]
                pars30b = popt[6:9]
                pars40b = popt[9:12]
                pars50b = popt[12:15]
                pars60b = popt[15:18]
                pars70b = popt[18:21]
                pars80b = popt[21:24]
                pars90b = popt[24:27]
                peak10b = asym_peak_G(ntt, pars10b)
                peak20b = asym_peak_G(ntt, pars20b)
                peak30b = asym_peak_G(ntt, pars30b)
                peak40b = asym_peak_G(ntt, pars40b)
                peak50b = asym_peak_G(ntt, pars50b)
                peak60b = asym_peak_G(ntt, pars60b)
                peak70b = asym_peak_G(ntt, pars70b)
                peak80b = asym_peak_G(ntt, pars80b)
                peak90b = asym_peak_G(ntt, pars90b)
                tot = peak10b + peak20b + peak30b + peak40b + \
                    peak50b + peak60b + peak70b + peak80b + peak90b
                P1 = scipy.integrate.trapz(peak10b, ntt_cm)
                P2 = scipy.integrate.trapz(peak20b, ntt_cm)
                P3 = scipy.integrate.trapz(peak30b, ntt_cm)
                P4 = scipy.integrate.trapz(peak40b, ntt_cm)
                P5 = scipy.integrate.trapz(peak50b, ntt_cm)
                P6 = scipy.integrate.trapz(peak60b, ntt_cm)
                P7 = scipy.integrate.trapz(peak70b, ntt_cm)
                P8 = scipy.integrate.trapz(peak80b, ntt_cm)
                P9 = scipy.integrate.trapz(peak90b, ntt_cm)
                N1 = 0.
                N2 = 0.
                N3 = 0.
                N4 = -P4 / A_HCOOH
                N5 = -P5 / A_HCOOH
                N_HCOOH = N4 + N5
                Data1 = {'HCOOH_pure': N_HCOOH}
                df1 = DataFrame(Data1, columns=['HCOOH_pure'], index=[0])
                df1.to_csv('Column_density_' +
                           'HCOOH_pure' + '.csv', index=False)
                np.savetxt('Analytic_deconv_HCOOH_15K.dat', np.transpose(
                    [ntt, peak10b, peak20b, peak30b, peak40b, peak50b, peak60b, peak70b, peak80b, peak90b, tot, y]))

            elif spn[j] == 'HCOOH_14K':
                lam1, lam2, lam3, lam4, lam5, lam6, lam7, lam8, lam9 = 5.85, 6.12, 7.23, 8.2, 8.53, 9.35, 10.7, 11.7, 14.6
                x1, x2, x3, x4, x5, x6, x7, x8, x9 = find_nearest(ntt, lam1), find_nearest(ntt, lam2), find_nearest(ntt,
                                                                                                                    lam3), find_nearest(
                    ntt, lam4), find_nearest(ntt, lam5), find_nearest(ntt, lam6), find_nearest(ntt, lam7), find_nearest(
                    ntt, lam8), find_nearest(ntt, lam9)
                y1, y2, y3, y4, y5, y6, y7, y8, y9 = y[get_line_number2(x1, ntt)], y[get_line_number2(x2, ntt)], y[
                    get_line_number2(x3, ntt)], y[get_line_number2(x4, ntt)], y[get_line_number2(x5, ntt)], y[
                    get_line_number2(x6, ntt)], y[get_line_number2(x7, ntt)], y[
                    get_line_number2(x8, ntt)], y[get_line_number2(x9, ntt)]
                guess = (
                    y1, x1, 0.01, y2, x2, 0.05, y3, x3, 0.03, y4, x4, 0.03, y5, x5, 0.05, y6, x6, 0.01, y7, x7, 0.05, y8,
                    x8, 0.05, y9, x9, 0.05)
                popt, pcov = curve_fit(
                    nine_peaks_G, ntt, y, guess, maxfev=5000)
                pars10b = popt[0:3]
                pars20b = popt[3:6]
                pars30b = popt[6:9]
                pars40b = popt[9:12]
                pars50b = popt[12:15]
                pars60b = popt[15:18]
                pars70b = popt[18:21]
                pars80b = popt[21:24]
                pars90b = popt[24:27]
                peak10b = asym_peak_G(ntt, pars10b)
                peak20b = asym_peak_G(ntt, pars20b)
                peak30b = asym_peak_G(ntt, pars30b)
                peak40b = asym_peak_G(ntt, pars40b)
                peak50b = asym_peak_G(ntt, pars50b)
                peak60b = asym_peak_G(ntt, pars60b)
                peak70b = asym_peak_G(ntt, pars70b)
                peak80b = asym_peak_G(ntt, pars80b)
                peak90b = asym_peak_G(ntt, pars90b)
                tot = peak10b + peak20b + peak30b + peak40b + \
                    peak50b + peak60b + peak70b + peak80b + peak90b
                P1 = scipy.integrate.trapz(peak10b, ntt_cm)
                P2 = scipy.integrate.trapz(peak20b, ntt_cm)
                P3 = scipy.integrate.trapz(peak30b, ntt_cm)
                P4 = scipy.integrate.trapz(peak40b, ntt_cm)
                P5 = scipy.integrate.trapz(peak50b, ntt_cm)
                P6 = scipy.integrate.trapz(peak60b, ntt_cm)
                P7 = scipy.integrate.trapz(peak70b, ntt_cm)
                P8 = scipy.integrate.trapz(peak80b, ntt_cm)
                P9 = scipy.integrate.trapz(peak90b, ntt_cm)
                N1 = 0.
                N2 = 0.
                N3 = 0.
                N4 = -P4 / A_HCOOH
                N5 = -P5 / A_HCOOH
                N_HCOOH = N4 + N5
                Data1 = {'HCOOH_pure': N_HCOOH}
                df1 = DataFrame(Data1, columns=['HCOOH_pure'], index=[0])
                df1.to_csv('Column_density_' +
                           'HCOOH_pure' + '.csv', index=False)
                np.savetxt('Analytic_deconv_HCOOH_14K.dat', np.transpose(
                    [ntt, peak10b, peak20b, peak30b, peak40b, peak50b, peak60b, peak70b, peak80b, peak90b, tot, y]))

            elif spn[j] == 'CO_CH3OH_10K':
                print('Decomposing CO_CH3OH_10K')
                lam1, lam2 = 9.27, 9.82
                x1, x2 = find_nearest(ntt, lam1), find_nearest(ntt, lam2)
                y1, y2 = y[get_line_number2(
                    x1, ntt)], y[get_line_number2(x2, ntt)]
                guess = (y1, x1, 0.01, y2, x2, 0.03)
                popt, pcov = curve_fit(two_peaks_G, ntt, y, guess, maxfev=5000)
                pars10b = popt[0:3]
                pars20b = popt[3:6]
                peak10b = asym_peak_G(ntt, pars10b)
                peak20b = asym_peak_G(ntt, pars20b)
                tot = peak10b + peak20b
                P1 = scipy.integrate.trapz(peak10b, ntt_cm)
                P2 = scipy.integrate.trapz(peak20b, ntt_cm)
                N1 = 0.
                N2 = -P1 / A_CH3OH
                N_CH3OH = N2
                Data1 = {'CH3OH_in_mix': N_CH3OH}
                df1 = DataFrame(Data1, columns=['CH3OH_in_mix'], index=[0])
                df1.to_csv('Column_density_' +
                           'CO_CH3OH_10K' + '.csv', index=False)
                np.savetxt('Analytic_deconv_CO_CH3OH_10K.dat',
                           np.transpose([ntt, peak10b, peak20b, tot, y]))

            elif spn[j] == 'HNCO_NH3':
                lam1, lam2, lam3, lam4, lam5 = 5.9, 6.80, 7.53, 8.08, 8.92
                x1, x2, x3, x4, x5 = find_nearest(ntt, lam1), find_nearest(ntt, lam2), find_nearest(ntt,
                                                                                                    lam3), find_nearest(
                    ntt, lam4), find_nearest(ntt, lam5)
                y1, y2, y3, y4, y5 = y[get_line_number2(x1, ntt)], y[get_line_number2(x2, ntt)], y[
                    get_line_number2(x3, ntt)], y[get_line_number2(x4, ntt)], y[get_line_number2(x5, ntt)]
                guess = (y1, x1, 0.2, y2, x2, 0.2, y3, x3,
                         0.1, y4, x4, 0.1, y5, x5, 0.2)
                # popt, pcov = curve_fit(five_peaks_G, ntt, y, guess, maxfev=5000)
                # plt.plot(ntt, five_peaks_G(ntt, *guess),'g-')
                # plt.plot(ntt,y)
                # plt.show()
                """
                pars10b = popt[0:3]
                pars20b = popt[3:6]
                pars30b = popt[6:9]
                pars40b = popt[9:12]
                pars50b = popt[12:15]
                peak10b = asym_peak_G(ntt, pars10b)
                peak20b = asym_peak_G(ntt, pars20b)
                peak30b = asym_peak_G(ntt, pars30b)
                peak40b = asym_peak_G(ntt, pars40b)
                peak50b = asym_peak_G(ntt, pars50b)
                tot = peak10b + peak20b + peak30b + peak40b + peak50b
                P1 = scipy.integrate.trapz(peak10b,ntt_cm)
                P2 = scipy.integrate.trapz(peak20b,ntt_cm)
                P3 = scipy.integrate.trapz(peak30b,ntt_cm)
                P4 = scipy.integrate.trapz(peak40b,ntt_cm)
                P5 = scipy.integrate.trapz(peak50b,ntt_cm)
                N1 = 0.#-P1/A_H2O
                N2 = -P2/A_NH4
                N3 = 0.#-P3/A_CH3OH
                N4 = 0.#-P4/A_CH3OH
                N5 = -P5/A_NH3
                N_NH4 = N2
                N_NH3 = N5
                Data1 = {'NH3_in_mix': N_NH3, 'NH4+_in_mix': N_NH4}
                df1 = DataFrame(Data1, columns= ['NH3_in_mix', 'NH4+_in_mix'], index=[0])
                df1.to_csv('Colum_density_'+'HNCO_NH3'+'.csv',index=False)
                np.savetxt('Analytic_deconv_HNCO_NH3.dat', np.transpose([ntt, peak10b,peak20b,peak30b,peak40b,peak50b,tot,y]))
                """

            elif spn[j] == 'Fig1d_SK':
                print('Decomposing Fig1d_SK')
                lam1, lam2, lam3, lam4, lam5, lam6 = 6.70, 6.85, 7.06, 8.68, 9.02, 9.39
                x1, x2, x3, x4, x5, x6 = find_nearest(ntt, lam1), find_nearest(ntt, lam2), find_nearest(ntt,
                                                                                                        lam3), find_nearest(
                    ntt, lam4), find_nearest(ntt, lam5), find_nearest(ntt, lam6)
                y1, y2, y3, y4, y5, y6 = y[get_line_number2(x1, ntt)], y[get_line_number2(x2, ntt)], y[
                    get_line_number2(x3, ntt)], y[get_line_number2(x4, ntt)], y[get_line_number2(x5, ntt)], y[
                    get_line_number2(x6, ntt)]
                guess = (y1, x1, 0.18, y2, x2, 0.21, y3, x3, 0.17,
                         y4, x4, 0.33, y5, x5, 0.48, y6, x6, 0.27)
                popt, pcov = curve_fit(six_peaks_G, ntt, y, guess, maxfev=5000)
                pars10b = popt[0:3]
                pars20b = popt[3:6]
                pars30b = popt[6:9]
                pars40b = popt[9:12]
                pars50b = popt[12:15]
                pars60b = popt[15:18]
                peak10b = asym_peak_G(ntt, pars10b)
                peak20b = asym_peak_G(ntt, pars20b)
                peak30b = asym_peak_G(ntt, pars30b)
                peak40b = asym_peak_G(ntt, pars40b)
                peak50b = asym_peak_G(ntt, pars50b)
                peak60b = asym_peak_G(ntt, pars60b)
                tot = peak10b + peak20b + peak30b + peak40b + peak50b + peak60b
                P1 = scipy.integrate.trapz(peak10b, ntt_cm)
                P2 = scipy.integrate.trapz(peak20b, ntt_cm)
                P3 = scipy.integrate.trapz(peak30b, ntt_cm)
                P4 = scipy.integrate.trapz(peak40b, ntt_cm)
                P5 = scipy.integrate.trapz(peak50b, ntt_cm)
                P6 = scipy.integrate.trapz(peak60b, ntt_cm)
                N1 = -P1 / A_NH4
                N2 = -P2 / A_NH4
                N3 = -P3 / A_NH4
                N4 = -P1 / A_NH3
                N5 = -P2 / A_NH3
                N6 = -P3 / A_NH3

                N_NH4 = N1 + N2 + N3
                N_NH3 = N4 + N5 + N6

                Data1 = {'NH4_in_mix': N_NH4, 'NH3_in_mix': N_NH3}
                df1 = DataFrame(
                    Data1, columns=['NH4_in_mix', 'NH3_in_mix'], index=[0])
                df1.to_csv('Column_density_' +
                           'Fig1d_SK' + '.csv', index=False)
                np.savetxt('Analytic_deconv_Fig1d_SK.dat',
                           np.transpose([ntt, peak10b, peak20b, peak30b, peak40b, peak50b, peak60b, tot, y]))

            elif spn[j] == 'NH4NO3_200K':
                print('Decomposing NH4NO3_200K')
                lam1, lam2, lam3, lam4, lam5, lam6, lam7 = 5.96, 6.16, 6.89, 7.47, 8.34, 11.19, 12.3
                x1, x2, x3, x4, x5, x6, x7 = find_nearest(ntt, lam1), find_nearest(ntt, lam2), find_nearest(ntt,
                                                                                                            lam3), find_nearest(
                    ntt, lam4), find_nearest(ntt, lam5), find_nearest(ntt, lam6), find_nearest(ntt, lam7)
                y1, y2, y3, y4, y5, y6, y7 = y[get_line_number2(x1, ntt)], y[get_line_number2(x2, ntt)], y[
                    get_line_number2(x3, ntt)], y[get_line_number2(x4, ntt)], y[get_line_number2(x5, ntt)], y[
                    get_line_number2(x6, ntt)], y[get_line_number2(x7, ntt)]
                guess = (y1, x1, 0.2, y2, x2, 0.2, y3, x3, 0.2, y4,
                         x4, 0.2, y5, x5, 0.2, y7, x7, 0.2, y7, x7, 0.2)
                # plt.plot(ntt, seven_peaks_G(ntt, *guess),'g-')
                # plt.plot(ntt,y)
                # plt.show()
                # """
                popt, pcov = curve_fit(
                    seven_peaks_G, ntt, y, guess, maxfev=5000)
                pars10b = popt[0:3]
                pars20b = popt[3:6]
                pars30b = popt[6:9]
                pars40b = popt[9:12]
                pars50b = popt[12:15]
                pars60b = popt[15:18]
                pars70b = popt[18:21]
                peak10b = asym_peak_G(ntt, pars10b)
                peak20b = asym_peak_G(ntt, pars20b)
                peak30b = asym_peak_G(ntt, pars30b)
                peak40b = asym_peak_G(ntt, pars40b)
                peak50b = asym_peak_G(ntt, pars50b)
                peak60b = asym_peak_G(ntt, pars60b)
                peak70b = asym_peak_G(ntt, pars70b)
                tot = peak10b + peak20b + peak30b + peak40b + peak50b + peak60b + peak70b
                P1 = scipy.integrate.trapz(peak10b, ntt_cm)
                P2 = scipy.integrate.trapz(peak20b, ntt_cm)
                P3 = scipy.integrate.trapz(peak30b, ntt_cm)
                P4 = scipy.integrate.trapz(peak40b, ntt_cm)
                P5 = scipy.integrate.trapz(peak50b, ntt_cm)
                P6 = scipy.integrate.trapz(peak60b, ntt_cm)
                P7 = scipy.integrate.trapz(peak70b, ntt_cm)
                N1 = 0.  # -P1/A_H2O
                N2 = -P2 / A_H2O
                N3 = -P3 / A_NH4
                N4 = -P4 / A_COO
                N5 = 0.
                N6 = 0.  # -P5/A_NH3
                N7 = 0.  # -P5/A_NH3
                N_H2O = N2
                N_NH4 = N3
                N_COO = N4
                Data1 = {'H2O_in_mix': N_H2O,
                         'COO_in_mix': N_COO, 'NH4_in_mix': N_NH4}
                df1 = DataFrame(
                    Data1, columns=['COO_in_mix', 'COO_in_mix', 'NH4+_in_mix'], index=[0])
                df1.to_csv('Column_density_' +
                           'NH4NO3_200K' + '.csv', index=False)
                np.savetxt('Analytic_deconv_NH4NO3_200K.dat',
                           np.transpose([ntt, peak10b, peak20b, peak30b, peak40b, peak50b, peak60b, peak70b, tot, y]))
            # """

            elif spn[j] == 'NH3_CH3OH_S':
                print('Decomposing NH3_CH3OH_S')
                lam1, lam2, lam3, lam4, lam5, lam6, lam7, lam8 = 6.1, 6.6, 6.86, 7.08, 8.85, 9.51, 9.72, 9.91
                x1, x2, x3, x4, x5, x6, x7, x8 = find_nearest(ntt, lam1), find_nearest(ntt, lam2), find_nearest(ntt,
                                                                                                                lam3), find_nearest(
                    ntt, lam4), find_nearest(ntt, lam5), find_nearest(ntt, lam6), find_nearest(ntt, lam7), find_nearest(
                    ntt, lam8)
                y1, y2, y3, y4, y5, y6, y7, y8 = y[get_line_number2(x1, ntt)], y[get_line_number2(x2, ntt)], y[
                    get_line_number2(x3, ntt)], y[get_line_number2(x4, ntt)], y[get_line_number2(x5, ntt)], y[
                    get_line_number2(x6, ntt)], y[get_line_number2(x7, ntt)], y[
                    get_line_number2(x8, ntt)]
                guess = (
                    y1, x1, 0.04, y2, x2, 0.04, y3, x3, 0.04, y4, x4, 0.04, y5, x5, 0.01, y6, x6, 0.01, y7, x7, 0.01, y8, x8,
                    0.01)
                popt, pcov = curve_fit(
                    eight_peaks_G, ntt, y, guess, maxfev=5000)
                # plt.plot(ntt, eight_peaks_L(ntt, *guess),'g-')
                # plt.plot(ntt,y)
                # plt.show()
                # """
                pars10b = popt[0:3]
                pars20b = popt[3:6]
                pars30b = popt[6:9]
                pars40b = popt[9:12]
                pars50b = popt[12:15]
                pars60b = popt[15:18]
                pars70b = popt[18:21]
                pars80b = popt[21:24]
                peak10b = asym_peak_G(ntt, pars10b)
                peak20b = asym_peak_G(ntt, pars20b)
                peak30b = asym_peak_G(ntt, pars30b)
                peak40b = asym_peak_G(ntt, pars40b)
                peak50b = asym_peak_G(ntt, pars50b)
                peak60b = asym_peak_G(ntt, pars60b)
                peak70b = asym_peak_G(ntt, pars70b)
                peak80b = asym_peak_G(ntt, pars80b)
                tot = peak10b + peak20b + peak30b + peak40b + \
                    peak50b + peak60b + peak70b + peak80b
                P1 = scipy.integrate.trapz(peak10b, ntt_cm)
                P2 = scipy.integrate.trapz(peak20b, ntt_cm)
                P3 = scipy.integrate.trapz(peak30b, ntt_cm)
                P4 = scipy.integrate.trapz(peak40b, ntt_cm)
                P5 = scipy.integrate.trapz(peak50b, ntt_cm)
                P6 = scipy.integrate.trapz(peak60b, ntt_cm)
                P7 = scipy.integrate.trapz(peak70b, ntt_cm)
                P8 = scipy.integrate.trapz(peak80b, ntt_cm)
                N1 = 0.  # -P1 / A_NH3_2
                N2 = 0.  # -P2/A_NH4
                N3 = 0.  # -P3/A_CH3OH
                N4 = 0.  # -P4/A_CH3OH
                N5 = -P5/A_NH3
                N6 = -P6/A_CH3OH
                N7 = -P7/A_CH3OH
                N8 = -P8/A_CH3OH
                N_NH3 = N5+0.4*N8
                N_CH3OH = N6+N7+0.6*N8
                Data1 = {'NH3_in_mix': N_NH3, 'CH3OH_in_mix': N_CH3OH}
                df1 = DataFrame(
                    Data1, columns=['NH3_in_mix', 'CH3OH_in_mix'], index=[0])
                df1.to_csv('Column_density_' +
                           'NH3_CH3OH_S' + '.csv', index=False)
                np.savetxt('Analytic_deconv_NH3_CH3OH_S.dat', np.transpose(
                    [ntt, peak10b, peak20b, peak30b, peak40b, peak50b, peak60b, peak70b, peak80b, tot, y]))

            elif spn[j] == 'NH3_CH3OH_50_10K':
                print('Decomposing NH3_CH3OH_50_10K')
                lam1, lam2, lam3, lam4, lam5, lam6, lam7, lam8 = 6.13, 6.65, 6.86, 6.95, 8.82, 9.24, 9.72, 10.88
                x1, x2, x3, x4, x5, x6, x7, x8 = find_nearest(ntt, lam1), find_nearest(ntt, lam2), find_nearest(ntt,
                                                                                                                lam3), find_nearest(
                    ntt, lam4), find_nearest(ntt, lam5), find_nearest(ntt, lam6), find_nearest(ntt, lam7), find_nearest(
                    ntt, lam8)
                y1, y2, y3, y4, y5, y6, y7, y8 = y[get_line_number2(x1, ntt)], y[get_line_number2(x2, ntt)], y[
                    get_line_number2(x3, ntt)], y[get_line_number2(x4, ntt)], y[get_line_number2(x5, ntt)], y[
                    get_line_number2(x6, ntt)], y[get_line_number2(x7, ntt)], y[
                    get_line_number2(x8, ntt)]
                guess = (
                    y1, x1, 0.04, y2, x2, 0.04, y3, x3, 0.04, y4, x4, 0.04, y5, x5, 0.2, y6, x6, 0.2, y7, x7, 0.2, y8, x8,
                    0.2)
                popt, pcov = curve_fit(
                    eight_peaks_G, ntt, y, guess, maxfev=5000)
                # plt.plot(ntt, eight_peaks_L(ntt, *guess),'g-')
                # plt.plot(ntt,y)
                # plt.show()
                # """
                pars10b = popt[0:3]
                pars20b = popt[3:6]
                pars30b = popt[6:9]
                pars40b = popt[9:12]
                pars50b = popt[12:15]
                pars60b = popt[15:18]
                pars70b = popt[18:21]
                pars80b = popt[21:24]
                peak10b = asym_peak_G(ntt, pars10b)
                peak20b = asym_peak_G(ntt, pars20b)
                peak30b = asym_peak_G(ntt, pars30b)
                peak40b = asym_peak_G(ntt, pars40b)
                peak50b = asym_peak_G(ntt, pars50b)
                peak60b = asym_peak_G(ntt, pars60b)
                peak70b = asym_peak_G(ntt, pars70b)
                peak80b = asym_peak_G(ntt, pars80b)
                tot = peak10b + peak20b + peak30b + peak40b + \
                    peak50b + peak60b + peak70b + peak80b
                P1 = scipy.integrate.trapz(peak10b, ntt_cm)
                P2 = scipy.integrate.trapz(peak20b, ntt_cm)
                P3 = scipy.integrate.trapz(peak30b, ntt_cm)
                P4 = scipy.integrate.trapz(peak40b, ntt_cm)
                P5 = scipy.integrate.trapz(peak50b, ntt_cm)
                P6 = scipy.integrate.trapz(peak60b, ntt_cm)
                P7 = scipy.integrate.trapz(peak70b, ntt_cm)
                P8 = scipy.integrate.trapz(peak80b, ntt_cm)
                N1 = -P1 / A_NH3_2
                N2 = 0.  # -P2/A_NH4
                N3 = 0.  # -P3/A_CH3OH
                N4 = 0.  # -P4/A_CH3OH
                N5 = 0.  # -P5/A_NH3
                N6 = 0.  # -P5/A_NH3
                N7 = -P5 / A_CH3OH
                N8 = 0.  # -P5/A_NH3
                N_NH3 = N1
                N_CH3OH = N7
                Data1 = {'NH3_in_mix': N_NH3, 'CH3OH_in_mix': N_CH3OH}
                df1 = DataFrame(
                    Data1, columns=['NH3_in_mix', 'CH3OH_in_mix'], index=[0])
                df1.to_csv('Column_density_' +
                           'NH3_CH3OH_50_10K' + '.csv', index=False)
                np.savetxt('Analytic_deconv_NH3_CH3OH_50_10K.dat', np.transpose(
                    [ntt, peak10b, peak20b, peak30b, peak40b, peak50b, peak60b, peak70b, peak80b, tot, y]))
            # """

            elif spn[j] == 'd_NH3_CH3OH_50_10K_I10m_Baselined':
                print('Decomposing d_NH3_CH3OH_50_10K_I10m_Baselined')
                lam1, lam2, lam3, lam4, lam5, lam6, lam7, lam8 = 6.13, 6.65, 6.86, 6.95, 8.82, 9.24, 9.72, 10.88
                x1, x2, x3, x4, x5, x6, x7, x8 = find_nearest(ntt, lam1), find_nearest(ntt, lam2), find_nearest(ntt,
                                                                                                                lam3), find_nearest(
                    ntt, lam4), find_nearest(ntt, lam5), find_nearest(ntt, lam6), find_nearest(ntt, lam7), find_nearest(
                    ntt, lam8)
                y1, y2, y3, y4, y5, y6, y7, y8 = y[get_line_number2(x1, ntt)], y[get_line_number2(x2, ntt)], y[
                    get_line_number2(x3, ntt)], y[get_line_number2(x4, ntt)], y[get_line_number2(x5, ntt)], y[
                    get_line_number2(x6, ntt)], y[get_line_number2(x7, ntt)], y[
                    get_line_number2(x8, ntt)]
                guess = (
                    y1, x1, 0.01, y2, x2, 0.01, y3, x3, 0.01, y4, x4, 0.01, y5, x5, 0.1, y6, x6, 0.1, y7, x7, 0.1, y8, x8,
                    0.1)
                popt, pcov = curve_fit(
                    eight_peaks_G, ntt, y, guess, maxfev=5000)
                pars10b = popt[0:3]
                pars20b = popt[3:6]
                pars30b = popt[6:9]
                pars40b = popt[9:12]
                pars50b = popt[12:15]
                pars60b = popt[15:18]
                pars70b = popt[18:21]
                pars80b = popt[21:24]
                peak10b = asym_peak_L(ntt, pars10b)
                peak20b = asym_peak_L(ntt, pars20b)
                peak30b = asym_peak_L(ntt, pars30b)
                peak40b = asym_peak_L(ntt, pars40b)
                peak50b = asym_peak_L(ntt, pars50b)
                peak60b = asym_peak_L(ntt, pars60b)
                peak70b = asym_peak_L(ntt, pars70b)
                peak80b = asym_peak_L(ntt, pars80b)
                tot = peak10b + peak20b + peak30b + peak40b + \
                    peak50b + peak60b + peak70b + peak80b
                P1 = scipy.integrate.trapz(peak10b, ntt_cm)
                P2 = scipy.integrate.trapz(peak20b, ntt_cm)
                P3 = scipy.integrate.trapz(peak30b, ntt_cm)
                P4 = scipy.integrate.trapz(peak40b, ntt_cm)
                P5 = scipy.integrate.trapz(peak50b, ntt_cm)
                P6 = scipy.integrate.trapz(peak60b, ntt_cm)
                P7 = scipy.integrate.trapz(peak70b, ntt_cm)
                P8 = scipy.integrate.trapz(peak80b, ntt_cm)
                N1 = -P1 / A_NH3_2
                N2 = 0.  # -P2/A_NH4
                N3 = 0.  # -P3/A_CH3OH
                N4 = 0.  # -P4/A_CH3OH
                N5 = 0.  # -P5/A_NH3
                N6 = 0.  # -P5/A_NH3
                N7 = -P5 / A_CH3OH
                N8 = 0.  # -P5/A_NH3
                N_NH3 = N1
                N_CH3OH = N7
                Data1 = {'NH3_in_mix': N_NH3, 'CH3OH_in_mix': N_CH3OH}
                df1 = DataFrame(
                    Data1, columns=['NH3_in_mix', 'CH3OH_in_mix'], index=[0])
                df1.to_csv(
                    'Column_density_' + 'd_NH3_CH3OH_50_10K_I10m_Baselined' + '.csv', index=False)
                np.savetxt('Analytic_deconv_d_NH3_CH3OH_50_10K_I10m_Baselined.dat', np.transpose(
                    [ntt, peak10b, peak20b, peak30b, peak40b, peak50b, peak60b, peak70b, peak80b, tot, y]))

            elif spn[j] == 'H2O_NH3_CO_1.0_0.6_0.4_c':
                print('Decomposing H2O_NH3_CO_1.0_0.6_0.4_c')
                lam1, lam2, lam3, lam4, lam5, lam6, lam7, lam8, lam9, lam10 = 5.94, 6.11, 6.73, 6.96, 7.25, 7.50, 8.95, 10.97, 12.21, 13.7
                x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = find_nearest(ntt, lam1), find_nearest(ntt,
                                                                                                lam2), find_nearest(ntt,
                                                                                                                    lam3), find_nearest(
                    ntt, lam4), find_nearest(ntt, lam5), find_nearest(ntt, lam6), find_nearest(ntt, lam7), find_nearest(
                    ntt, lam8), find_nearest(ntt, lam9), find_nearest(ntt, lam10)
                y1, y2, y3, y4, y5, y6, y7, y8, y9, y10 = y[get_line_number2(x1, ntt)], y[get_line_number2(x2, ntt)], y[
                    get_line_number2(x3, ntt)], y[get_line_number2(x4, ntt)], y[get_line_number2(x5, ntt)], y[
                    get_line_number2(x6, ntt)], y[get_line_number2(x7, ntt)], \
                    y[get_line_number2(x8, ntt)], y[get_line_number2(x9, ntt)], y[
                    get_line_number2(x10, ntt)]
                guess = (
                    y1, x1, 0.01, y2, x2, 0.01, y3, x3, 0.01, y4, x4, 0.01, y5, x5, 0.01, y6, x6, 0.01, y7, x7, 0.05, y8,
                    x8, 0.05, y9, x9, 0.05, y10, x10, 0.05)
                popt, pcov = curve_fit(ten_peaks_G, ntt, y, guess, maxfev=5000)
                pars10b = popt[0:3]
                pars20b = popt[3:6]
                pars30b = popt[6:9]
                pars40b = popt[9:12]
                pars50b = popt[12:15]
                pars60b = popt[15:18]
                pars70b = popt[18:21]
                pars80b = popt[21:24]
                pars90b = popt[24:27]
                pars100b = popt[27:30]
                peak10b = asym_peak_G(ntt, pars10b)
                peak20b = asym_peak_G(ntt, pars20b)
                peak30b = asym_peak_G(ntt, pars30b)
                peak40b = asym_peak_G(ntt, pars40b)
                peak50b = asym_peak_G(ntt, pars50b)
                peak60b = asym_peak_G(ntt, pars60b)
                peak70b = asym_peak_G(ntt, pars70b)
                peak80b = asym_peak_G(ntt, pars80b)
                peak90b = asym_peak_G(ntt, pars90b)
                peak100b = asym_peak_G(ntt, pars100b)
                tot = peak10b + peak20b + peak30b + peak40b + peak50b + \
                    peak60b + peak70b + peak80b + peak90b + peak100b
                P1 = scipy.integrate.trapz(peak10b, ntt_cm)
                P2 = scipy.integrate.trapz(peak20b, ntt_cm)
                P3 = scipy.integrate.trapz(peak30b, ntt_cm)
                P4 = scipy.integrate.trapz(peak40b, ntt_cm)
                P5 = scipy.integrate.trapz(peak50b, ntt_cm)
                P6 = scipy.integrate.trapz(peak60b, ntt_cm)
                P7 = scipy.integrate.trapz(peak70b, ntt_cm)
                P8 = scipy.integrate.trapz(peak80b, ntt_cm)
                P9 = scipy.integrate.trapz(peak90b, ntt_cm)
                P10 = scipy.integrate.trapz(peak100b, ntt_cm)
                N1 = 0.
                N2 = 0.
                N3 = -P3 / A_NH4
                N4 = -P4 / A_NH4
                N5 = -P5 / A_NH4
                N6 = -P6 / A_NH4
                N7 = -P7 / A_NH3
                N8 = -P8 / A_H2O_2
                N9 = -P9 / A_H2O_2
                N10 = -P10 / A_H2O_2
                N_NH4 = N3 + N4 + N5 + N6
                N_NH3 = N7
                N_H2O = N8 + N9 + N10
                Data1 = {'H2O_in_mix': N_H2O,
                         'NH4_in_mix': N_NH4, 'NH3_in_mix': N_NH3}
                df1 = DataFrame(
                    Data1, columns=['H2O_in_mix', 'NH4+_in_mix', 'NH3_in_mix'], index=[0])
                df1.to_csv('Column_density_' +
                           'H2O_NH3_CO_1.0_0.6_0.4_c' + '.csv', index=False)
                np.savetxt('Analytic_deconv_H2O_NH3_CO_1.0_0.6_0.4_c.dat', np.transpose(
                    [ntt, peak10b, peak20b, peak30b, peak40b, peak50b, peak60b, peak70b, peak80b, peak90b, peak100b,
                     tot, y]))

            elif spn[j] == 'H2O_CH3OH_1_1_c':
                lam1, lam2, lam3, lam4, lam5, lam6, lam7, lam8, lam9 = 5.94, 6.73, 7.22, 7.94, 9.13, 9.84, 11.19, 12.50, 14.0
                x1, x2, x3, x4, x5, x6, x7, x8, x9 = find_nearest(ntt, lam1), find_nearest(ntt, lam2), find_nearest(ntt,
                                                                                                                    lam3), find_nearest(
                    ntt, lam4), find_nearest(ntt, lam5), find_nearest(ntt, lam6), find_nearest(ntt, lam7), find_nearest(
                    ntt, lam8), find_nearest(ntt, lam9)
                y1, y2, y3, y4, y5, y6, y7, y8, y9 = y[get_line_number2(x1, ntt)], y[get_line_number2(x2, ntt)], y[
                    get_line_number2(x3, ntt)], y[get_line_number2(x4, ntt)], y[get_line_number2(x5, ntt)], y[
                    get_line_number2(x6, ntt)], y[get_line_number2(x7, ntt)], y[
                    get_line_number2(x8, ntt)], y[get_line_number2(x9, ntt)]
                guess = (
                    y1, x1, 0.08, y2, x2, 0.05, y3, x3, 0.05, y4, x4, 0.05, y5, x5, 0.08, y6, x6, 0.05, y7, x7, 0.1, y8, x8,
                    0.1, y9, x9, 0.1)
                # plt.plot(ntt, nine_peaks_G(ntt, *guess),'g-')
                # plt.plot(ntt,y)
                # plt.show()
                """
                popt, pcov = curve_fit(nine_peaks_G, ntt, y, guess, maxfev=5000)
                pars10b = popt[0:3]
                pars20b = popt[3:6]
                pars30b = popt[6:9]
                pars40b = popt[9:12]
                pars50b = popt[12:15]
                pars60b = popt[15:18]
                pars70b = popt[18:21]
                pars80b = popt[21:24]
                pars90b = popt[24:27]
                peak10b = asym_peak_G(ntt, pars10b)
                peak20b = asym_peak_G(ntt, pars20b)
                peak30b = asym_peak_G(ntt, pars30b)
                peak40b = asym_peak_G(ntt, pars40b)
                peak50b = asym_peak_G(ntt, pars50b)
                peak60b = asym_peak_G(ntt, pars60b)
                peak70b = asym_peak_G(ntt, pars70b)
                peak80b = asym_peak_G(ntt, pars80b)
                peak90b = asym_peak_G(ntt, pars90b)
                tot = peak10b + peak20b + peak30b + peak40b + peak50b + peak60b + peak70b + peak80b + peak90b
                P1 = scipy.integrate.trapz(peak10b,ntt_cm)
                P2 = scipy.integrate.trapz(peak20b,ntt_cm)
                P3 = scipy.integrate.trapz(peak30b,ntt_cm)
                P4 = scipy.integrate.trapz(peak40b,ntt_cm)
                P5 = scipy.integrate.trapz(peak50b,ntt_cm)
                P6 = scipy.integrate.trapz(peak60b,ntt_cm)
                P7 = scipy.integrate.trapz(peak70b,ntt_cm)
                P8 = scipy.integrate.trapz(peak80b,ntt_cm)
                P9 = scipy.integrate.trapz(peak90b,ntt_cm)
                N1 = 0.
                N2 = 0.
                N3 = 0.
                N4 = 0.
                N5 = 0.
                N6 = -P6/A_CH3OH
                N7 = -P7/A_H2O_2
                N8 = -P8/A_H2O_2
                N9 = -P9/A_H2O_2
                N_CH3OH = N6
                N_H2O = N7+N8+N9
                Data1 = {'H2O_in_mix': N_H2O, 'CH3OH_in_mix': N_CH3OH}
                df1 = DataFrame(Data1, columns= ['H2O_in_mix', 'CH3OH_in_mix'], index=[0])
                df1.to_csv('Colum_density_'+'H2O_CH3OH_1_1_c'+'.csv',index=False)
                np.savetxt('Analytic_deconv_H2O_CH3OH_1_1_c', np.transpose([ntt, peak10b,peak20b,peak30b,peak40b,peak50b,peak60b,peak70b,peak80b,peak90b,tot,y]))
                """
    except:
        pass
        print('Failed to decompose mixture!')
