"""
Function used to simulate the properties of the gamma-ray detector.

Specifically, 1. the resolution, which decides the width of a peak as a function of the
mean energy of that peak (E) using the equation R(E)=√(x_0+x_1*E+x_2*E^2+...); and 2.
the Compton-to-photopeak ratio, which decides the ratio of the
Compton continuum:mean energy of the peak as a function of energy of the peak, expressed
as
"""

from collections.abc import Callable, Iterable

import numpy as np

__all__ = [
    "fit_fwhms",
    "get_default_resolution_coefficients",
    "resolution_curve_factory",
]


def get_default_resolution_coefficients():
    """
    Give a set of default resolution coefficient values.

    This specific set of values is obtained by fitting
    01_Cu_001.Spe from
    https://github.com/OceanNuclear/PeakFinding/commit/a386d484420d8efd2f5b3132f17fa5a66cc9988c
    using `fit_fwhm_cal_interactively`
    https://github.com/OceanNuclear/PeakFinding/commit/dd17a8d9cbbb80ce62f3bd10f4a24c5c6594b217
    .
    """
    return np.array([5.212873549440453 * 1e5, 2.4969943490051713])  # noqa: DOC201


def fit_fwhms(
    E: np.ndarray[float],
    fwhm: np.ndarray[float],
    degree_of_fit: int = 2,
) -> np.ndarray[float]:
    """
    Fit the FWHM curve R(E) = √(x_0 + x_1 * E + x_2 * E^2 + ...) where E has units eV.

    Parameters
    ----------
    E:
        mean energy of the peaks in eV
    fwhm:
        full-width half-maximum of the peaks in eV.

    Returns
    -------
    coefficients:
        a list of coefficients in ascending degrees.
    """
    return np.polyfit(E, fwhm**2, degree_of_fit)[::-1]


def resolution_curve_factory(
    coefficients: Iterable[float],
    min_fwhm: float = 10,
) -> Callable[[float | np.ndarray], float | np.ndarray]:
    """
    Parameters
    ----------
    coefficients:
        An iterable of coefficients in ascending degrees.
    min_fwhm:
        The resolution of the gamma-ray detector is bounded below by this limit[eV].
        The FWHM (at low E) is clamped to above this limit.

    Returns
    -------
    resolution_curve:
        A function that returns the FWHM (eV) when given an energy input (eV), but the
        returned FWHM will never be smaller than min_fwhm.
    """
    raw_curve = np.poly1d(coefficients[::-1])
    min_fwhm_squared = min_fwhm**2

    def resolution_curve(E: float | np.ndarray) -> float | np.ndarray:
        """Clamp the resolution output from below, and transform it by taking sqrt."""
        return np.clip(raw_curve(E), min_fwhm_squared, np.inf) ** 0.5  # noqa: DOC201

    return resolution_curve
