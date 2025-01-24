"""
Constants for converting to and from cgs units.
cgs (centimeter, grams, second) is used throughout the entire foilselector package.
"""

import math

# convert values in MeV/keV to eV
MeV = 1e6
keV = 1e3

# convert values in barn to cm^2
BARN = 1e-24

# convert values in mm to cm
MM_CM = 0.1
amu = 1.660538921e-24
me_eV = 510.9989461e3
FWHM_SIGMA = 2 * math.sqrt(2 * math.log(2))
