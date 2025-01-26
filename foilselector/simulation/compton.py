"""
Module to simulate Compton continua.
This include functions to calculate the shape of the compton continuum, as well as the
function to calculate the compton-to-peak ratio.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from uncertainties import nominal_value as nom
from uncertainties.core import AffineScalarFunc

from foilselector.constants import MeV, keV, me_eV


class ComptonToPeakRatioCurve:
    """Also known as the Compton-from-Peak curve, because it is a curve with the
    following input and output:
    curve(input: # counts in peaks) -> ouput: # counts in the compton continuum.
    """

    @classmethod
    def fit_data(
        cls,
        energy: np.ndarray[float],
        compton_to_peak_ratio: np.ndarray[float],
        degree_of_fit: int = 6,
    ):
        """Create a fit using the data given.

        Parameters
        ----------
        energy:
            Energy of each data point in [eV]
        compton_to_peak_ratio:
            Compton-to-peak ratio of each data point [dimensionless].
        degree_of_fit:
            Degree of fit in log-log space,
            from log(energy) to log(compton_to_peak_ratio).

        Returns
        -------
        self:
            A new instance created by fitting the data provided.

        Raises
        ------
        ValueError
            Raised when fitting is not done correctly.
        """
        with warnings.catch_warnings(record=True) as w:
            loglog_coefficients = np.polyfit(
                np.log(energy),
                np.log(compton_to_peak_ratio),
                degree_of_fit,
            )
        if w:
            raise ValueError(f"{w[0].category}: {w[0].message}")
        self = cls(loglog_coefficients)
        self.E = energy
        self.compton_to_peak_ratio = compton_to_peak_ratio
        self.degree_of_fit = degree_of_fit
        return self

    def __init__(self, coefficients: np.ndarray[float]):
        self.coefficients = coefficients
        self._fitted_func_in_loglog_space = np.poly1d(coefficients)

    def __call__(
        self,
        required_E_in_eV: float | np.ndarray[float],
    ) -> float | np.ndarray[float]:
        """
        Parameters
        ----------
        required_E_in_eV:
            input in the x-coordinates

        Returns
        -------
        :
            output in the y-coordinates
        """
        return np.exp(self._fitted_func_in_loglog_space(np.log(nom(required_E_in_eV))))

    @classmethod
    def from_file(cls, filename: Path, degree_of_fit: int = 6):
        """Create a compton-to-peak-ratio curve object from a .csv file.

        Parameters
        ----------
        filename:
            A .csv file storing the data points to be fitted, in 2 columns.
            The first column is the x-data (energy) column, which must have eV/MeV/keV
            as its unit, present in the file name (e.g. "energy (eV)").
            The second column has the y-data column, which is the Compton-to-peak ratio
            [dimensionless].
        degree_of_fit:
            parsed onto :meth:`~ComptonToPeakRatioCurve.fit_data`.

        Returns
        -------
        :
            An object of :class:`~ComptonToPeakRatioCurve` created by fitting the data
            points in filename.

        Raises
        ------
        ValueError
            A unit error
        """
        ratio_csv = pd.read_csv(filename, comment="#")
        energy_name = ratio_csv.columns[0]
        if "MeV" in energy_name:
            energy = ratio_csv[energy_name] * MeV
        elif "keV" in energy_name:
            energy = ratio_csv[energy_name] * keV
        elif "eV" in energy_name:
            energy = ratio_csv[energy_name]
        else:
            raise ValueError(
                f"UnitError: column 0 of {filename} (gamma-energy column)"
                "has ambiguous unit!",
            )

        ratio = ratio_csv[ratio_csv.columns[1]]
        return cls.fit_data(np.array(energy), np.array(ratio), degree_of_fit)


def get_default_peak_to_Compton_file() -> Path:
    """Get the file path of the peak-to-Compton-ratio file.

    Returns
    -------
    :
        The absolute path to the default peak-to-Compton-ratio file.
    """
    return Path(
        # relative path, relative to
        Path(__file__).parent,  # THIS particular file, compton.py right here.
        "..",
        "physicalparameters",
        "efficiency",
        "Compton_to_peak_ratio.csv",
    ).resolve()


def compton_edge(peak_energy: float) -> float:
    """Calculate the Compton edge energy corresponding to a photopeak, where a the photon
    recoils 180° and loses as much energy to the electron as possible.

    Parameters
    ----------
    peak_energy:
        energy of the photopeak, in eV.

    Returns
    -------
    Compton_edge_energy:
        energy of the compton edge corresponding to the thing.
    """
    factor = 1.0 + 2 * peak_energy / me_eV
    return peak_energy * (1 - 1 / factor)


def make_sharp_compton_distribution(
    photopeak_energy: AffineScalarFunc,
    test_energies: np.ndarray[float],
) -> np.ndarray[float]:
    r"""
    Create a normalized distribution that represents the Compton continuum.

    Parameters
    ----------
    photopeak_energy:
        the energy of the gamma ray that's causing this Compton continuum.
    test_energies:
        The array of energies for which we have to compute the Compton continuum for.

    Returns
    -------
    energy_deposited:
        The amount (per unit [eV]) of Compton scattered into each of the test_energies,
        for every gamma-ray counted at the photopeak of photopeak_energy.

    Formulae
    --------
    The Compton distribution is given as below:


    .. math::

        C(E_{dep}) = \left(\frac{E_{\gamma}-E_{dep}}{E_{\gamma}}\right)^2 \left[
        \frac{E_{\gamma}-E_{dep}}{E_{\gamma}} + \frac{E_{\gamma}}{E_{\gamma}-E_{dep}}
        - \frac{2 E_{dep} (E_{\gamma} -2 E_{dep})}{\epsilon (E_{\gamma}-E_{dep})^2}
        \right]

    where the distribution is valid between $0\leq E_{dep}\leq E_{Comp}$, and

    .. math::

        E_{Comp} = E_{\gamma} - \frac{E_{\gamma}}{1+2\epsilon}

        \epsilon = \frac{E_{\gamma}}{m_e c^2} = \frac{E_{\gamma}}{511 keV}.

    This is obtained by rewriting the Klein-Nishina formula for the
    differential cross-section (See Wikipedia:
    https://en.wikipedia.org/wiki/Klein%E2%80%93Nishina_formula) by parametrising theta
    in terms of E_dep = E_gamma - E_gamma', i.e. energy deposited by photon through
    Compton scattering.
    For an interactive demo of this distribution (w.r.t. changing photopeak energy)
    please see https://www.desmos.com/calculator/8zksn1wtya

    The integral of this area under the curve is given by:

    .. math::

        N = \frac{E_{\gamma}}{4} - \frac{E_{\gamma}}{4(1+2\epsilon)^4}
        - \frac{E_{Comp}^2}{2E_{\gamma}} + E_{Comp} + \frac{E_{Comp}^2(E_{Comp}
        - 3\frac{E_{\gamma}}{1+2\epsilon})}{3E_{\gamma}^2\epsilon}


    such that the normalized Compton distribution is given by $\frac{C(E_{dep})}{N}$.

    Usage
    -----
    This distribution shall be rescaled to the appropriate height to become the Compton
    continuum, by assuming that all counts in the Compton continuum are formed by single-
    Compton scattering events, i.e. the scattered Compton photon immediately exit the
    gamma-ray detector and never interacts with it again.
    """
    energy_deposited = np.zeros(len(test_energies))
    Eg = nom(photopeak_energy)
    if np.isclose(Eg, 0, atol=1, rtol=0):  # anything between 0 - 1 eV -> no Compton.
        return energy_deposited
    Ecomp = compton_edge(Eg)
    affected_bins = test_energies <= Ecomp
    Ed = test_energies[affected_bins]  # alias for energy deposited by photon.
    me_ratio = Eg / me_eV  # ratio to electron mass

    # calculating the differential cross-section (only the part that varies w.r.t. E_dep)
    big_bracket = (
        (Eg - Ed) / Eg
        + Eg / (Eg - Ed)
        - 2 * Ed * (Eg - 2 * Ed) / (me_ratio * (Eg - Ed) ** 2)
    )
    unnormed_dist = ((Eg - Ed) / Eg) ** 2 * big_bracket
    # Normalization factor
    norm_factor = (
        Eg / 4
        - Eg / (4 * (1 + 2 * me_ratio) ** 4)
        - Ecomp**2 / 2 / Eg
        + Ecomp
        + Ecomp**2 * (Ecomp - 3 * Eg / (1 + 2 * me_ratio)) / (3 * Eg**2 * me_ratio)
    )

    energy_deposited[affected_bins] = unnormed_dist / norm_factor
    return energy_deposited
