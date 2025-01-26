"""
Function used to simulate the properties of the gamma-ray detector.

Specifically, 1. the resolution, which decides the width of a peak as a function of the
mean energy of that peak (E) using the equation R(E)=√(x_0+x_1*E+x_2*E^2+...); and 2.
the Compton-to-photopeak ratio, which decides the ratio of the
Compton continuum:mean energy of the peak as a function of energy of the peak, expressed
as
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

__all__ = [
    "ResolutionMaxCountRate",
    "fit_fwhms",
    "get_default_resolution_coefficients",
    "resolution_curve_factory",
]

GAMMA_RES_AND_COUNT_RATE_FILENAME = ".gamma-resolution-count-rate-coefs.txt"


class ResolutionMaxCountRate:
    """Data on the resolution curve of the gamma-ray detector, and the maximum count rate
    at which this resolution can be achieved without degredation.
    """

    def __init__(self, resolution_coefficients: Iterable[float], max_count_rate: float):
        """
        Initialize from an Iterable of resolution curve coefficients and the maximum
        count rate.

        Paraemters
        ----------
        resolution_coefficients:
            The polynomial coefficients that get the resolution (expressed as FWHM) as
            FWHM = sqrt(polynomial(energy)).
            See :func:`~resolution_curve_factory` for more details.
        """
        self.resolution_coefficients = resolution_coefficients
        self.max_count_rate = max_count_rate

    def save(self, directory: Path | str = ".") -> None:
        """Store data as plain text file."""
        with Path(directory, GAMMA_RES_AND_COUNT_RATE_FILENAME).open("w") as f:
            for i, coef in enumerate(self.resolution_coefficients):
                f.write(f"x_{i}={coef}\n")
            f.write(f"max. count rate={self.max_count_rate}\n")

    @staticmethod
    def load(directory: Path | str = ".") -> tuple[list[float], float]:
        """Load data back from the GAMMA_RES_AND_COUNT_RATE_FILENAME file.

        Returns
        -------
        Directly return the two objects:
            resolution_coefficients, max_count_rate [float].

        Raises
        ------
        ValueError
            Raised when the text inside the GAMMA_RES_AND_COUNT_RATE_FILENAME does not
            match the expected text.
        """
        with Path(directory, GAMMA_RES_AND_COUNT_RATE_FILENAME).open() as f:
            text = f.readlines()
        resolution_coefficients = []
        while text:
            if text[0].startswith("x"):
                resolution_coefficients.append(float(text.pop(0).split("=")[1]))
            else:
                break
        if not text[0].startswith("max"):
            raise ValueError("Expected x_0=...\nx_1=...\n...\nmax. count rate=...")
        max_count_rate = float(text.pop(0).split("=")[1])
        return resolution_coefficients, max_count_rate


def get_default_resolution_coefficients() -> np.ndarray[float]:
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

    Raises
    ------
    ValueError

    """
    with warnings.catch_warnings(record=True) as w:
        polynomial = np.polyfit(E, fwhm**2, degree_of_fit)[::-1]
    if w:
        raise ValueError(f"{w[0].category}: {w[0].message}")
    return polynomial


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
