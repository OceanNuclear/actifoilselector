"""Module to load any gamma-ray detection absolute efficiency related functions and
classes.
"""

import itertools
from collections import namedtuple
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import array as ary
from numpy import log as ln
from uncertainties import nominal_value as nom
from uncertainties.core import AffineScalarFunc

from foilselector.constants import MeV, keV

MCNPOut = namedtuple("MCNPOut", ["El", "Eu", "lc1", "lc2", "uc1", "uc2", "tc1", "tc2"])
ISOCSOut = namedtuple("ISOCSOut", ["E", "eff", "integer", "e1", "deviation", "e2", "ID"])
EffCurve = namedtuple("EffCurve", ["E", "eff", "unc"])


APPROVED_EFFICIENCY_FILE_EXTENSIONS = {
    ".o": "MCNP simulation output",
    ".ecc": "GENIE/ISOCS simulation output",
    ".csv": (
        "Generic comma-separated file, energy in MeV in first column, "
        "efficiency in second column, uncertainty potentially in third column."
    ),
    ".dat": "Plain text file, same as .csv, but space delimited instead",
}


def list_dir_eff_files(directory: Path) -> list[Path]:
    """Pretty print the list of all efficiency files in a specified directory.

    Returns
    -------
    fname:
        A list of directories that
    """
    fnames = list(
        itertools.chain(
            Path(directory).glob("*.o"),
            Path(directory).glob("*.ecc"),
            Path(directory).glob("*.dat"),
            Path(directory).glob("*.csv"),
        ),
    )
    print("########################")
    print("------------------------")
    for f in fnames:
        print(f.name)
    print("------------------------")
    print("########################")
    return fnames


def read_mcnp_output(fname: Path) -> MCNPOut:  # noqa: D103
    with Path(fname).open() as f:
        data = f.readlines()
    llim_lines = data[1::4]
    ulim_lines = data[2::4]
    total_lines = data[3::4]
    El, Eu = (
        [float(line.split()[0]) for line in llim_lines],
        [float(line.split()[0]) for line in ulim_lines],
    )
    low_col1, low_col2 = (
        [float(line.split()[1]) for line in llim_lines],
        [float(line.split()[2]) for line in llim_lines],
    )
    upp_col1, upp_col2 = (
        [float(line.split()[1]) for line in ulim_lines],
        [float(line.split()[2]) for line in ulim_lines],
    )
    tot_col1, tot_col2 = (
        [float(line.split()[1]) for line in total_lines],
        [float(line.split()[2]) for line in total_lines],
    )
    return MCNPOut(
        ary(El) * MeV,
        ary(Eu) * MeV,
        ary(low_col1),  # number of counts depositing E= 0 to El per source particle
        ary(low_col2),  # relative uncertainty on the ^ number
        ary(upp_col1),  # number of counts depositing E=El to Eu per source particle
        ary(upp_col2),  # relative uncertainty on the ^ number
        ary(tot_col1),  # number of counts depositing E= 0 to Eu per source particle
        ary(tot_col2),
    )  # relative uncertainty on the ^ number


def read_dat(fname: Path) -> tuple[np.ndarray[float], np.ndarray[float]]:  # noqa: D103
    with Path(fname).open() as f:
        data = f.readlines()[1:]
    E = [float(line.split()[0]) for line in data]
    eff = [float(line.split()[1]) for line in data]
    return ary(E) * keV, ary(eff)


def read_ecc(fname: Path) -> ISOCSOut:  # noqa: D103
    with Path(fname).open() as f:
        file = f.readlines()[11:]
    table = [[float(i) for i in line.split()[1:]] for line in file]
    tabularized = ary(table).T
    return ISOCSOut(*tabularized)


def read_csv(fname: Path) -> pd.DataFrame:  # noqa: D103
    df = pd.read_csv(fname)  # noqa: PD901
    if "MeV" in df.columns[0]:
        df[df.columns[0]] = df[df.columns[0]] * MeV  # noqa: PLR6104
    elif "keV" in df.columns[0]:
        df[df.columns[0]] = df[df.columns[0]] * keV  # noqa: PLR6104
    else:
        raise ValueError(
            "Please include the energy unit (keV/MeV) in the title of the first column "
            f"of {fname}",
        )
    return df


def efficiency_curve_from_file(fname: Path) -> EffCurve:
    """Create the efficiency curve from a file.

    Parameters
    ----------
    fname:
        file in which the efficiency curve fitting data is stored.

    Returns
    -------
    :
        EffCurve namedtuple, containing the error on each of the efficiency data point
        where present.

    Raises
    ------
    TypeError
        When an invalid file extension is given, this error is rased.
    ValueError
        When the provided file is a .csv file, but it has too many/ too few columns
        to contain just the energy, efficiency, and optionally the error on efficiency,
        data, for each data point.
    """
    fname = Path(fname)
    if fname.suffix == ".o":
        out = read_mcnp_output(fname)
        E = np.mean([out.El, out.Eu], axis=0)
        return EffCurve(E, out.uc1, out.uc1 * out.uc2)

    if fname.suffix == ".dat":
        E, eff = read_dat(fname)
        return EffCurve(E, eff, None)

    if fname.suffix == ".ecc":
        isocs_output = read_ecc(fname)
        return EffCurve(isocs_output[0], isocs_output[1], None)

    if fname.suffix == ".csv":
        dataframe = read_csv(fname)
        # cols = dataframe.columns
        if dataframe.shape[1] == 2:  # noqa: PLR2004
            return EffCurve(*dataframe.to_numpy().T, None)
        if dataframe.shape[1] == 3:  # noqa: PLR2004
            return EffCurve(*dataframe.to_numpy().T)
        raise ValueError("Wrong shape of .csv file!")

    raise TypeError(f"{fname} not an accepted filetype")


class EfficiencyCurve:
    """A Callable class which acts as the efficiency curve, turning incident gamma-ray
    energy input [eV] into efficiency [dimensionless].
    """

    def __init__(
        self,
        eff_curve_object: EffCurve,
        extrapolation_inference_threshold_keV: float = 800,
    ):
        self.E = ary(eff_curve_object.E)
        self.eff = ary(eff_curve_object.eff)
        self.unc = (
            ary(eff_curve_object.unc) if eff_curve_object.unc is not None else None
        )
        # create a fit for >100 keV.
        self._extrapolation_inference_threshold = (
            extrapolation_inference_threshold_keV * keV
        )  # we deduce the slope of the extrapolation using datapoints above 800 keV
        # used for calculating the interpolated curve:
        self._log_E = ln(self.E)
        self._log_eff = ln(self.eff)
        self._log_fit_thres = ln(self._extrapolation_inference_threshold)
        self._log_max_E = max(self._log_E)
        fitting_region = self._log_E >= self._log_fit_thres
        fitted_slope, _fitted_offset = np.polyfit(
            self._log_E[fitting_region],
            self._log_eff[fitting_region],
            1,
            w=None if self.unc is None else 1 / ary(self.unc)[fitting_region],
        )
        eff_at_max_E = self._log_eff[self._log_E == self._log_max_E][
            0
        ]  # dirty hack to find the efficiency at the largest recorded energy.
        self._extrapolate = lambda x: eff_at_max_E + fitted_slope * (x - self._log_max_E)

    def _fitted_func_in_loglog_space(
        self,
        scalar_or_vector: float | np.ndarray[float],
    ) -> float | np.ndarray[float]:
        """
        Calculate the efficiencies using the stored list of energies and efficiencies.
        Below the min stored E value : 0.
        Between min E stored E and max stored E : interpolate in log-log space.
        Above the max stored E value : extrapolate linearly in log-log space.

        Parameters
        ----------
        scalar_or_vector:
            log of energy(ies) which is a scalar (or a vector) at which we want to find
            the efficiencies.

        Returns
        -------
        output:
            scalar efficiency if input is scalar; vector efficiency if input is vector.
        """
        # extrapolate by drawing a line with slope = fitted slope, crossing the rightmost
        # stored point, to ensure continuity.
        # fastest implementation is to NOT use np.vectorize, even though it's uglier.
        if np.ndim(scalar_or_vector) == 0:  # scalar
            scalar = scalar_or_vector
            if scalar >= self._log_max_E:
                return self._extrapolate(scalar)
            return np.interp(scalar, self._log_E, self._log_eff, left=-np.inf)
        # vector
        vector = scalar_or_vector
        extrapolated_part = vector >= self._log_max_E
        output = np.zeros_like(vector)
        output[extrapolated_part] = self._extrapolate(vector[extrapolated_part])
        output[~extrapolated_part] = np.interp(
            vector[~extrapolated_part],
            self._log_E,
            self._log_eff,
            left=-np.inf,
        )
        return output

    @classmethod
    def from_file(cls, fname: Path):
        """Create an EfficiencyCurve by fitting from a file."""
        return cls(efficiency_curve_from_file(fname))  # noqa: DOC201

    def __call__(
        self,
        required_E_in_eV: float
        | AffineScalarFunc
        | np.ndarray[float]
        | np.ndarray[AffineScalarFunc],
    ) -> float | np.ndarray[float]:
        """Calculate the efficiency at the required energy [eV] according to the fitted
        efficiency curve.

        Parameters
        ----------
        required_E_in_eV:
            Incident gamma-ray energies [eV] for which we want the efficiency values for.

        Returns
        -------
        efficiency:
            The efficiency in the same shape, either as a single float (scalar) or as an
            1D array of floats [vector].
        """
        E = nom(required_E_in_eV)
        if np.isclose(E, 0):
            return 0.0
        return np.exp(self._fitted_func_in_loglog_space(ln(E)))

    def plot(self, ax: plt.Axes | None = None) -> None:
        """Plot to examine how well the fit is.

        Even though the underlying constants are in eV,
        everything that the user interacts with (smoothline_lower and _upper, and
        ax.xlabels) are in keV
        """
        ax = ax or plt.axes()
        smoothline_lower, smoothline_upper = min(self.E), max(self.E)
        energy_keV = np.geomspace(smoothline_lower, smoothline_upper, 300)
        energy_eV = energy_keV * keV
        smooth_eff = self(energy_eV)
        ax.plot(energy_eV / keV, smooth_eff)
        ax.scatter(self.E / keV, self.eff)
        ax.set_xlabel("E (keV)")
        ax.set_ylabel("Efficiency (fraction)")
        ax.set_xscale("log"), ax.set_yscale("log")
        plt.show()


def get_default_efficiency_curve_path() -> Path:
    """Get the file path of the efficiency file.

    Returns
    -------
    :
        The absolute path to the default efficiency file.
    """
    return Path(
        # relative path, relative to
        Path(__file__).parent,  # THIS particular file, compton.py right here.
        "..",
        "physicalparameters",
        "efficiency",
        "Absolute_photopeak_efficiencyMeV.csv",
    ).resolve()
