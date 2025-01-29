"""Functions to read the nucleat data libraries using openmc."""

# typical system/python stuff
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from uncertainties import nominal_value as nom

from foilselector.constants import keV
from foilselector.openmcextension.table import Integral, Tab1DExtended

if TYPE_CHECKING:
    from openmc.data import Tabulated1D
    from uncertainties.core import AffineScalarFunc, Variable

__all__ = [
    "ContinuousRadiationDistribution",
    "DiscreteRadiation",
    "collapse_single_xs",
    "flatten_photon_spectrum",
]


def collapse_single_xs(
    xs_entry: Tabulated1D | Tab1DExtended,
    gs_array: np.ndarray,
) -> np.ndarray[float]:
    """Collapse a cross-section (continuous function) into the provided group structure
    (a which yields a vector, i.e. a discrete list of floats).

    Parameters
    ----------
    xs_entry:
        A single openmc.data.Tabulated1D instance found in
        `openmc.data.IncidentNeutron.from_endf(file).reactions[...].xs["0K"]`
    gs_array:
        Group structure showing the lower and upper bound of each bin,
        shape = len(gs) * 2.

    Returns
    -------
    cross-section:
        The cross-section collapsed into the appropriate group structure.
    """
    return (
        Integral(xs_entry).definite_integral(*gs_array.T)
        / np.diff(gs_array, axis=1).flatten()
    )


class RadiationSource:
    """Contain all information that needs to be expressed about where the radiation came
    from.
    """

    def __init__(
        self,
        source: list[str] | list[RadiationSource],
        decay_mode: str,
        energy: float | AffineScalarFunc,
    ):
        """Create a RadiationSource corresponding to a single peak, containing all
        information about where it came from.

        Parameters
        ----------
        source:
            The isotopes in the decay chain that led to this radiation being emitted,
            starting with the reactant and ending with the daughter of the decay.
        decay_mode:
            The method of the final decay that released this gamma-ray.
            This data is stored but won't be presented to the user if the radiation
            origintated from a single decay pathway.
        energy:
            The energy of the isotope, given in eV.
            This data is stored but won't be presented to the user if the radiation
            origintated from a single decay pathway.
        """
        self.is_multiple_sources = self._check_multiple_source(source)
        self.source = source
        self.decay_mode = decay_mode
        self.energy = energy

    @staticmethod
    def _check_multiple_source(test_source: list[str] | list[RadiationSource]) -> None:
        """Validate that self.source is either a list of str (i.e. isotopes),
        or a list of sources.

        Raises
        ------
        ValueError
            Raised if the validation doesn't pass.
        """
        if all(isinstance(source, str) for source in test_source):
            return False
        if not any(source.is_multiple_sources for source in test_source):
            return True
        raise ValueError(
            f"Incorrect source type provided: {test_source} should be either a list of "
            "strings or list of RadiationSource whose .source is a list of strings",
        )
        # can never be deeper than 1 level.

    def __str__(self):
        """Return the str expression of the RadiationSource.

        Returns
        -------
        :
            if it's a single radiation: just the decay chain and decay mode.
            if it's multiple radiations: decay chains and decay modes of each, plus their
                true energies.
        """
        if self.is_multiple_sources:
            each_expression = [
                str(source) + " at " + str(round(nom(source.energy), 5))
                for source in self.source
            ]
            return "; ".join(each_expression)
        return (
            self.source[0] + "+n->" + "->".join(self.source[1]) + " " + self.decay_mode
        )

    def __repr__(self):
        return str(self)


def merge_sources_from_multiple_radiations(
    rad_list: list[DiscreteRadiation | ContinuousRadiationDistribution],
) -> str:
    """Combine multiple radiation's sources into one.

    Returns
    -------
    :
        An item that can be used as the .source part of a DiscreteRadiation or
        ContinuousRadiationDistribution.
    """
    return "; ".join(rad.source for rad in rad_list)  # to be replaced by the line below
    # return RadiationSource([rad.source for rad in rad_list])


class DiscreteRadiation(NamedTuple):
    """
    Attributes
    ----------
    energy: openmc.core.Variable
        mean energy of this discrete radiation line.
    intensity: openmc.core.Variable
        number of this radiation line released per decay of the immediate parent.
    source: RadiationSource
        description of where the radiation originated from.
        decay radiation type, immediate parent's name, and decay mode inducing the
        release of this radiation. e.g. "gamma from Y101 beta-"
    """

    energy: AffineScalarFunc | np.float64
    intensity: AffineScalarFunc | np.float64
    source: RadiationSource

    def __hash__(self) -> int:
        """Create a hash out of the contents."""
        return hash((  # noqa: DOC201
            (nom(self.energy), 0.0 if isinstance(self.energy, float) else self.energy.s),
            (
                nom(self.intensity),
                0.0 if isinstance(self.intensity, float) else self.intensity.s,
            ),
            self.source,
        ))

    def copy(self) -> DiscreteRadiation:
        """Shallow copy the discrete radiation.

        Returns
        -------
        :
            A copy of the discrete radiation with the same self.energy and
            self.intensity, i.e. referring to the same AffineScalarFunc object, but the
            self.source (which should be a string) is duplicated as it is immutable.
        """
        return self.__class__(self.energy, self.intensity, self.source)

    def deepcopy(self) -> DiscreteRadiation:
        """Shallow copy the discrete radiation.

        Returns
        -------
        :
            A copy of the discrete radiation with the identical self.energy and
            self.intensity, but (new.energy-self.energy) and
            (new.intensity-self.intensity) != 0.0+/-0.0.
            self.source (which should be a string) is duplicated as it is immutable.
        """
        return self.__class__(self.energy.copy(), self.intensity.copy(), self.source)

    def plot_label_format(self) -> str:
        """
        Return a str representation of itself that that can be used as a label for itself
        when plotting.
        """
        return "{} keV\n{} counts\nby {}".format(  # noqa: DOC201
            nom(self.energy) / keV,
            nom(self.intensity),
            self.source.replace("; ", ";\n"),
        )


class ContinuousRadiationDistribution(NamedTuple):
    """
    Attributes
    ----------
    distribution: `foilselector.openmcextension.table.Tab1DExtended`
        Distribution of gamma-lines(x=energy (eV), y=intensity) released per decay of
        the immediate parent.
        Area under the distribution integrates to the value given by
        `nom(openmc_decay_spectrum[radiation_type]["continuous_normalization"])`.
    source: RadiationSource
        description of where the radiation originated from.
        decay radiation type, immediate parent's name, and decay mode inducing the
        release of this radiation. e.g. "gamma from Y101 beta-"
    """

    distribution: Tab1DExtended
    source: RadiationSource

    def __hash__(self) -> int:
        """Overloads the definition of hash().

        Returns
        -------
        :
            A hash that corresponds to the data of the Radiation.
        """
        return hash((self.distribution, self.source))

    def copy(self) -> ContinuousRadiationDistribution:
        """Shallow copy.

        Returns
        -------
        :
            self.distribution is referring to the same object, while self.source is
            a copy of itself as it is immutable.
        """
        return self.__class__(self.distribution, self.source)

    def deepcopy(self) -> ContinuousRadiationDistribution:
        """Deep copy.

        Returns
        -------
        :
            self.distribution is copied, while self.source is
            a copy of itself as it is immutable.
        """
        return self.__class__(self.distribution.copy(), self.source)

    def __mul__(self, multiplier: float) -> ContinuousRadiationDistribution:
        """Allow the distribution itself to be scaled directly (with no effect to the
        source).

        Returns
        -------
        :
            A ContinuousRadiationDistribution with a scaled copy of the radiation
            distribution.
        """
        return self.__class__(self.distribution * multiplier, self.source)


def flatten_photon_spectrum(  # noqa: C901
    openmc_decay_spectrum: dict,
    isotope_name: str,
) -> tuple[list[DiscreteRadiation], list[ContinuousRadiationDistribution]]:
    """
    Turn a openmc.data.Decay.from_endf(...).spectra from a nested dictionary into
    a lists of photon (xray and gamma) lines.

    Parameters
    ----------
    openmc_decay_spectrum:
        dictionary obtained by openmc.data.Decay.from_endf(isotope).spectra.
    isotope_name:
        name of the isotope to which the spectrum belongs.

    Returns
    -------
    discrete_spec: list[3-tuple]
        DiscreteRadiation(energy (eV), intensity, source description)

    continuous_spec: list[2-tuple]
        ContinuousRadiationDistribution(photon energy distribution, source description)
    """
    discrete_spec, continuous_spec = [], []

    def extend_discrete_lines(
        discrete: list[dict],
        discrete_normalization: Variable,
        source: str,
    ) -> None:
        """Flatten a decay["spectrum"][radiation_type]["discrete"]."""
        if nom(discrete_normalization):
            for line in discrete:
                discrete_spec.append(
                    DiscreteRadiation(
                        line["energy"],
                        line["intensity"] * discrete_normalization,
                        f"{source} from {isotope_name} {','.join(line['from_mode'])}",
                    ),
                )

    if "xray" in openmc_decay_spectrum:
        if "discrete" in openmc_decay_spectrum["xray"]:
            extend_discrete_lines(
                openmc_decay_spectrum["xray"]["discrete"],
                openmc_decay_spectrum["xray"]["discrete_normalization"],
                "xray",
            )
        if "continuous" in openmc_decay_spectrum["xray"]:
            prob_table = openmc_decay_spectrum["xray"]["continuous"]["probability"]
            cont_norm = nom(openmc_decay_spectrum["xray"]["continuous_normalization"])
            xray_dist = Tab1DExtended.from_openmc(prob_table) * cont_norm

            if cont_norm:
                src = ",".join(openmc_decay_spectrum["xray"]["continuous"]["from_mode"])
                continuous_spec.append(
                    ContinuousRadiationDistribution(
                        xray_dist,
                        f"xray from {isotope_name} {src}",
                    ),
                )
    if "gamma" in openmc_decay_spectrum:
        if "discrete" in openmc_decay_spectrum["gamma"]:
            extend_discrete_lines(
                openmc_decay_spectrum["gamma"]["discrete"],
                openmc_decay_spectrum["gamma"]["discrete_normalization"],
                "gamma",
            )
        if "continuous" in openmc_decay_spectrum["gamma"]:
            prob_table = openmc_decay_spectrum["gamma"]["continuous"]["probability"]
            cont_norm = nom(openmc_decay_spectrum["gamma"]["continuous_normalization"])
            gamma_dist = Tab1DExtended.from_openmc(prob_table) * cont_norm

            if cont_norm:
                src = ",".join(openmc_decay_spectrum["gamma"]["continuous"]["from_mode"])
                continuous_spec.append(
                    ContinuousRadiationDistribution(
                        gamma_dist,
                        f"gamma from {isotope_name} {src}",
                    ),
                )

    return discrete_spec, continuous_spec
