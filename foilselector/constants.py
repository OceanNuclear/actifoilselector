"""
Constants for converting to and from cgs units.
cgs (centimeter, grams, second) is used throughout the entire foilselector package.
"""

import math

MeV = 1e6  # express [MeV] in [eV]
keV = 1e3  # express [keV] in [eV]

BARN = 1e-24  # express [barn] in [cm2]

MM_CM = 0.1  # express [mm] in [cm]
amu = 1.660538921e-24  # express [atomic mass unit] in [gram]
me_eV = 510.9989461e3  # express mass of electron in [eV]
FWHM_SIGMA = 2 * math.sqrt(2 * math.log(2))  # express one FWHM in sigma.
ZERO_E_THRESHOLD = 1.0  # energy [eV]
# we consider radiation with less than this energy to have effectively "zero-energy".
