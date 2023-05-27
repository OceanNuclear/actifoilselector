"""
    []Kim MCNP:/home/ocean/Documents/PhD/ExperimentalVerification/data/efficiency/ocean-*detector/*.o
    []Carlo: /home/ocean/Documents/PhD/ExperimentalVerification/data/efficiency/*.dat
        *contact*.dat
    []ISOCS point: /media/ocean/OS/GENIE2K/isocs/data/GEOMETRY/Laboratory/POINT/*.ecc
        /media/ocean/OS/GENIE2K/isocs/data/GEOMETRY/Laboratory/POINT/3mm.ecc
        # /media/ocean/OS/GENIE2K/isocs/data/GEOMETRY/Laboratory/POINT/contact.ecc
    []ISOCS extended: /media/ocean/OS/GENIE2K/isocs/data/GEOMETRY/In-Situ/RECTANGULAR_PLANE/*.ecc
        /media/ocean/OS/GENIE2K/isocs/data/GEOMETRY/In-Situ/RECTANGULAR_PLANE/*3mm.ecc
    experimental data:
    []H=0: /home/ocean/Documents/PhD/ExperimentalVerification/data/efficiency/H=0_efficiency.csv
    []H=H1:/home/ocean/Documents/PhD/ExperimentalVerification/data/efficiency/H1_efficiency.csv

    python plotcurve.py carlo-simpledetector/ptsource-singlebin.o /home/ocean/Documents/PhD/ExperimentalVerification/data/efficiency/data_compare/pt_source_0/output_simulation_contact_pointlikeefficiency.dat 
    OR
    data_compare/foil_source/*
    data_compare/pt_source_0/*
    data_compare/pt_source_H1/*
    python plotcurve.py /home/ocean/Documents/PhD/ExperimentalVerification/data/efficiency/data_compare/foil_source/*simple*.o

MCNP
Carlo
ISOCS pt
ISOCS extend
"""
import numpy as np
from numpy import sqrt, array as ary, log as ln
from uncertainties import nominal_value as nom
import pandas as pd
k=1000; M=1000000
PLOTLY = True

from collections import namedtuple
MCNPOut = namedtuple('MCNPOut', ['El', 'Eu', 'lc1', 'lc2', 'uc1', 'uc2', 'tc1', 'tc2'])
ISOCSOut = namedtuple('ISOCSOut', ['E', 'eff', 'integer', 'e1', 'deviation', 'e2', 'ID'])
EffCurve = namedtuple('EffCurve', ['E', 'eff', 'unc'])

def read_mcnp_output(fname):
    with open(fname) as f:
        data = f.readlines()
    llim_lines = data[1::4]
    ulim_lines = data[2::4]
    total_lines= data[3::4]
    El, Eu = [float(l.split()[0]) for l in llim_lines], [float(l.split()[0]) for l in ulim_lines]
    low_col1, low_col2 = [float(l.split()[1]) for l in llim_lines], [float(l.split()[2]) for l in llim_lines]
    upp_col1, upp_col2 = [float(l.split()[1]) for l in ulim_lines], [float(l.split()[2]) for l in ulim_lines]
    tot_col1, tot_col2 = [float(l.split()[1]) for l in total_lines],[float(l.split()[2]) for l in total_lines]
    return MCNPOut(
    ary(El)*M,
    ary(Eu)*M,
    ary(low_col1), # number of counts depositing E= 0 to El per source particle
    ary(low_col2), # relative uncertainty on the ^ number
    ary(upp_col1), # number of counts depositing E=El to Eu per source particle
    ary(upp_col2), # relative uncertainty on the ^ number
    ary(tot_col1), # number of counts depositing E= 0 to Eu per source particle
    ary(tot_col2)
    ) # relative uncertainty on the ^ number

def read_dat(fname):
    with open(fname) as f:
        data = f.readlines()[1:]
    E = [float(line.split()[0]) for line in data]
    eff = [float(line.split()[1]) for line in data]
    return ary(E)*k, ary(eff)

def read_ecc(fname):
    with open(fname) as f:
        file = f.readlines()[11:]
    table = []
    for line in file:
        table.append([float(i) for i in line.split()[1:]])
    tabularized = ary(table).T
    return ISOCSOut(*tabularized)

def read_csv(fname):
    df = pd.read_csv(fname)
    if "MeV" in df.columns[0]:
        df[df.columns[0]] = df[df.columns[0]] * M
    elif "keV" in df.columns[0]:
        df[df.columns[0]] = df[df.columns[0]] * k
    return df

def efficiency_curve_factory(fname):
    if fname.endswith(".o"):
        out = read_mcnp_output(fname)
        E = np.mean([out.El, out.Eu], axis=0)
        return EffCurve(E, out.uc1, out.uc1*out.uc2)

    elif fname.endswith(".dat"):
        E, eff = read_dat(fname)
        return EffCurve(E, eff, None)

    elif fname.endswith(".ecc"):
        isocs_output = read_ecc(fname)
        return EffCurve(isocs_output[0], isocs_output[1], None)

    elif fname.endswith(".csv"):
        dataframe = read_csv(fname)
        # cols = dataframe.columns
        if dataframe.shape[1]==2:
            return EffCurve(*dataframe.values.T, None)
        elif dataframe.shape[1]==3:
            return EffCurve(*dataframe.values.T)
        else:
            raise ValueError("Wrong shape of .csv file!")

    else:
        raise TypeError(f"{fname} not an accepted filetype")

class EfficiencyCurve:
    def __init__(self, eff_curve_object):
        self.E = ary(eff_curve_object.E)
        self.eff = ary(eff_curve_object.eff)
        self.unc = ary(eff_curve_object.unc) if eff_curve_object.unc is not None else None
        # create a fit for >100 keV.
        self._extrapolation_inference_threshold = 800*k # we deduce the slope of the extrapolation using datapoints above 800 keV
        # used for calculating the interpolated curve:
        self._log_E = ln(self.E)
        self._log_eff = ln(self.eff)
        self._log_fit_thres = ln(self._extrapolation_inference_threshold)
        self._log_max_E = max(self._log_E)
        fitting_region = self._log_E>=self._log_fit_thres
        fitted_slope, fitted_offset = np.polyfit(self._log_E[fitting_region],
                                  self._log_eff[fitting_region],
                                  1,
                                  w=None if self.unc is None else 1/ary(self.unc)[fitting_region])
        eff_at_max_E = self._log_eff[self._log_E==self._log_max_E][0] # dirty hack to find the efficiency at the largest recorded energy.
        self._extrapolate = lambda x: eff_at_max_E + fitted_slope * (x-log_max_E)


    def _fitted_func_in_loglog_space(self, scalar_or_vector):
        """
        Calculate the efficiencies using the stored list of energies and efficiencies.
        Below the min stored E value : 0.
        Between min E stored E and max stored E : interpolate in log-log space.
        Above the max stored E value : extrapolate linearly in log-log space.
        Parameters
        ----------
        log of energy(ies) which is a scalar (or a vector) at which we want to find the efficiencies

        Returns
        -------
        scalar efficiency if input is scalar; vector efficiency if input is vector.
        """
        # extrapolate by drawing a line with slope = fitted slope, crossing the rightmost stored point, to ensure continuity.
        # fastest implementation is to NOT use np.vectorize, even though it's a bit uglier.
        if np.ndim(scalar_or_vector)==0: # scalar
            scalar = scalar_or_vector
            if scalar>=self._log_max_E:
                return self._extrapolate(scalar)
            else:
                return np.interp(scalar, self._log_E, self._log_eff, left=-np.inf)
        else: # vector
            vector = scalar_or_vector
            extrapolated_part = vector>=self._log_max_E
            output = np.zeros_like(vector)
            output[extrapolated_part] = self._extrapolate(vector[extrapolated_part])
            output[~extrapolated_part] = np.interp(vector[~extrapolated_part], self._log_E, self._log_eff, left=-np.inf)
            return output

    @classmethod
    def from_file(cls, fname):
        return cls(efficiency_curve_factory(fname))

    def __call__(self, required_E_in_eV):
        return np.exp(self._fitted_func_in_loglog_space(ln(nom(required_E_in_eV))))

    def plot(self, ax=None):
        """Plot to examine how well the fit is.
        Even though the underlying constants are in eV,
        everything that the user interacts with (smoothline_lower and _upper, and ax.xlabels) are in keV"""
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.axes()
        smoothline_lower, smoothline_upper = min(self.E), max(self.E)
        energy_keV = np.geomspace(smoothline_lower, smoothline_upper, 300)
        energy_eV = energy_keV*k
        smooth_eff = self.__call__(energy_eV)
        ax.plot(energy_eV/k, smooth_eff)
        ax.scatter(self.E/k, self.eff)
        ax.set_xlabel("E (keV)")
        ax.set_ylabel("Efficiency (fraction)")
        ax.set_xscale('log'), ax.set_yscale('log')
        plt.show()