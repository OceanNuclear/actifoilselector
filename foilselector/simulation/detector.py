"""
Function used to simulate the properties of the gamma-ray detector.

Specifically, 1. the resolution, which decides the width of a peak as a function of the
mean energy of that peak (E) using the equation R(E)=√(x_0+x_1*E+x_2*E^2+...); and 2.
the Compton-to-photopeak ratio, which decides the ratio of the
Compton continuum:mean energy of the peak as a function of energy of the peak, expressed
as 
"""
import numpy as np
from typing import Callable, Iterable
from numpy import typing as npt

__all__= ["get_default_resolution_coefficients", "fit_fwhms", "resolution_curve_factory",
        "get_default_peak_to_Compton_coefficients", "fit_peak_to_Compton",
        "Compton_to_peak_curve_factory"]

def get_default_resolution_coefficients():
    """
    Placeholder values.
    This specific set of values is obtained by fitting
    01_Cu_001.Spe from
    https://github.com/OceanNuclear/PeakFinding/commit/a386d484420d8efd2f5b3132f17fa5a66cc9988c
    using `fit_fwhm_cal_interactively`
    https://github.com/OceanNuclear/PeakFinding/commit/dd17a8d9cbbb80ce62f3bd10f4a24c5c6594b217
    .
    """
    return np.array([0.5212873549440453, 0.0024969943490051713])

def fit_fwhms(E: npt.NDArray[float], fwhm: npt.NDArray[float], degree_of_fit: int=2) -> npt.NDArray[float]:
    """
    Parameters
    ----------
    E:
        mean energy of the peaks in keV
    fwhm:
        full-width half-maximum of the peaks in keV.

    Returns
    -------
    coefficients:
        a list of coefficients in ascending degrees.
    """
    return np.polyfit(E, fwhm**2, degree_of_fit)[::-1]

def resolution_curve_factory(coefficients: Iterable[float], min_fwhm=0.1) -> Callable[[float | npt.NDArray], float | npt.NDArray]:
    """
    Parameters
    ----------
    coefficients:
        an iterable of coefficients in ascending degrees.
    min_fwhm:
        The, minimum resolution of the gamma-ray detector. It is not allowed to have resolution better than this.

    Returns
    -------
    resolution_curve
    """
    raw_curve = np.poly1d(coefficients[::-1])
    def resolution_curve(E: float| npt.NDArray) -> float | npt.NDArray:
        """A function that clamps the resolution output from below."""
        return np.clip(raw_curve(E), min_fwhm, np.inf)
    return resolution_curve

def get_default_peak_to_Compton_coefficients():
    """
    Placeholder value for the peak-to-Comption ratio.
    This specific number is obtained from summarizing
    https://doi.org/10.1016/j.nima.2023.168826 (P/C = 62:1) and
    https://doi.org/10.1016/j.nima.2018.08.048, (P/C = 60.8:1).
    """
    return np.array([61.4]) # assume constant.

def fit_peak_to_Compton(E: npt.NDArray[float], pc_ratio: npt.NDArray[float], degree_of_fit: int=1) -> npt.NDArray[float]:
    """
    Parameters
    ----------
    E:
        mean energy of the peaks in keV
    pc_ratio:
        Peak-to-Comptoin ratio of the peaks. [dimensionless]

    Returns
    -------
    coefficients:
        a list of coefficients in ascending degrees.
    """
    return np.polyfit(E, pc_ratio, degree_of_fit)[::-1]

def Compton_to_peak_curve_factory(coefficients: Iterable[float], min_peak_to_comp_ratio = 1.0) -> Callable[[float | npt.NDArray], float | npt.NDArray]:
    """
    Parameters
    ----------
    coefficients:
        An iterable of coefficients in ascending degrees.
    min_peak_to_comp_ratio:
        The peak-to-Comptoin ratio is not allowed to drop below this number.
        This limit is implemented to prevent infinite Compton continua to be created at
        high energies due to poorly fitted peak-to-Compton curves.

    Returns
    -------
    Compton_to_peak_curve
    """
    raw_curve = np.poly1d(coefficients[::-1])
    def Compton_to_peak_curve(E: float| npt.NDArray) -> float | npt.NDArray:
        return 1/np.clip(raw_curve(E), min_peak_to_comp_ratio, np.infty)
    return Compton_to_peak_curve