from .filepaths import HPGe_eff_file as default_file_path
import pandas as pd
import numpy as np
from numpy import sqrt
from numpy import array as ary, log as ln
import uncertainties.unumpy as unpy
import uncertainties as unc
from foilselector.constants import MeV

def exp(numbers):
    if isinstance(numbers, unc.core.AffineScalarFunc):
        return unpy.exp(numbers)
    else:
        return np.exp(numbers)

def HPGe_efficiency_curve_generator(file_location=default_file_path, deg=4, cov=True):
    '''
    Polynomial fit in log-log space is the best for emulating the efficiency curve of a HPGe detector.* (see references below)
    This is a function factory that fits a known list of data points,
    thus generating an efficiency function that accepts scalars or numpy arrays.
    this efficiency function is returned as the output of this function factory.

    Ref 1: Knoll Radiation Detection (equation 12.32),
    Ref 2:
        @article{kis1998comparison,
        title={Comparison of efficiency functions for Ge gamma-ray detectors in a wide energy range},
        author={Kis, Zs and Fazekas, B and {\"O}st{\"o}r, J and R{\'e}vay, Zs and Belgya, T and Moln{\'a}r, GL and Koltay, L},
        journal={Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment},
        volume={418},
        number={2-3},
        pages={374--386},
        year={1998},
        publisher={Elsevier}
        }
    '''
    datapoints = pd.read_csv(file_location)
    assert "energ" in datapoints.columns[0].lower(), "The file must contain a header. Energy (eV/MeV) has to be placed in the first column"
    E, eff = datapoints.values.T[:2]
    if 'MeV' in datapoints.columns[0] or 'MeV' in file_location:
        E = MeV*E
    print("Using the gamma detection efficiency curve read from {}".format(file_location))
    if datapoints.values.T[2:].shape[0]>0: # assume that column is the error
        sigma = datapoints.values.T[2]
        print("Using the 3rd column of ", file_location, r" as error on the efficiency measurements (\sigma)")
        cov = 'unscaled'
        w = 1/sigma**2
    else:
        w = None
    from numpy import polyfit
    p = polyfit(ln(E), ln(eff), deg, w=w, cov=cov) #make sure that the curve points downwards at high energy 
    if cov:
        p, pcov = p
    #iteratively try increasing degree of polynomial fits.
    #Currently the termination condition is via making sure that the curve points downwards at higher energy (leading coefficient is negative).
    #However, I intend to change the termination condition to Bayesian Information Criterion instead.
    if not p[0]<0:
        mindeg, maxdeg = 2,6
        for i in range(mindeg, maxdeg+1):
            p = polyfit(ln(E), ln(eff), deg, w=w, cov=cov)            
            if cov:
                p, pcov = p
            if p[0]<0:
                print("a {0} order polynomial fit to the log(energy)-log(efficiency) curve is used instead of the default {1} order".format(i, deg))
                break
            elif i==maxdeg:#if we've reached the max degree tested and still haven't broken the loop, it means none of them fits.
                print("None of the polynomial fit in log-log space is found to extrapolate properly! using the default {0} order fit...".format(deg) )
                p = polyfit(ln(E), ln(eff), deg)
    if cov: 
        print("The covariance matrix is\n", pcov)

    def efficiency_curve(E):
        if isinstance(E, unc.core.AffineScalarFunc):
            lnE = ln(E.n)
        else:
            lnE = ln(E)
        lneff = np.sum( [p[::-1][i]* lnE**i for i in range(len(p))], axis=0) #coefficient_i * x ** i
        
        if cov:
            lnE_powvector = [lnE**i for i in range(len(p))][::-1]
            variance_on_lneff = (lnE_powvector @ pcov @ lnE_powvector) # variance on lneff
            if isinstance(E, unc.core.AffineScalarFunc):
                error_of_lnE = E.s/E.n
                variance_from_E = sum([p[::-1][i]*i*lnE**(i-1) for i in range(1, len(p))])**2 * (error_of_lnE)**2
                variance_on_lneff += variance_from_E
            lneff_variance = exp(lneff)**2 * variance_on_lneff
            return unc.core.Variable( exp(lneff), sqrt(lneff_variance) )
        else:
            return exp(lneff)
    return efficiency_curve