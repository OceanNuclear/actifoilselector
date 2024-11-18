# typical system/python stuff
from collections import namedtuple

# typical python numerical stuff
import numpy as np
from numpy import typing as npt

# openmc stuff
from openmc.data import Tabulated1D

# uncertainties
from uncertainties.core import Variable
from uncertainties import nominal_value as nom

# local modules
from foilselector.openmcextension.table import Integrate, Tab1DExtended

__all__ = [
    "collapse_single_xs",
    "DiscreteRadiation",
    "ContinuousRadiationDistribution",
    "flatten_photon_spectrum",
]


def collapse_single_xs(xs_entry: Tabulated1D, gs_array: npt.NDArray):
    """
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

    """
    return (
        Integrate(xs_entry).definite_integral(*gs_array.T)
        / np.diff(gs_array, axis=1).flatten()
    )


class DiscreteRadiation(namedtuple("Radiation", ["energy", "intensity", "source"])):
    """
    Attributes
    ----------
    energy: openmc.core.Variable
        mean energy of this discrete radiation line.
    intensity: openmc.core.Variable
        number of this radiation line released per decay of the immediate parent.
    source: str
        description of where the radiation originated from.
        decay radiation type, immediate parent's name, and decay mode inducing the
        release of this radiation. e.g. "gamma from Y101 beta-"
    """

    def __hash__(self):
        return hash((
            (nom(self.energy), 0.0 if isinstance(self.energy, float) else self.energy.s),
            (
                nom(self.intensity),
                0.0 if isinstance(self.intensity, float) else self.intensity.s,
            ),
            self.source,
        ))


class ContinuousRadiationDistribution(
    namedtuple("RadiationDistribution", ["distribution", "source"])
):
    """
    Attributes
    ----------
    distribution: `foilselector.openmcextension.table.Tab1DExtended`
        Distribution of gamma-lines(x=energy (eV), y=intensity) released per decay of
        the immediate parent.
        Area under the distribution integrates to the value given by
        `nom(openmc_decay_spectrum[radiation_type]["continuous_normalization"])`.
    source: str
        description of where the radiation originated from.
        decay radiation type, immediate parent's name, and decay mode inducing the
        release of this radiation. e.g. "gamma from Y101 beta-"
    """


def flatten_photon_spectrum(
    openmc_decay_spectrum: dict, isotope_name: str
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
        discrete: list[dict], discrete_normalization: Variable, source: str
    ):
        """Flatten a decay["spectrum"][radiation_type]["discrete"]"""
        if nom(discrete_normalization):
            for line in discrete:
                discrete_spec.append(
                    DiscreteRadiation(
                        line["energy"],
                        line["intensity"] * discrete_normalization,
                        f"{source} from {isotope_name} {','.join(line['from_mode'])}",
                    )
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
            xray_dist = (Tab1DExtended.from_openmc(prob_table) * cont_norm,)

            if cont_norm:
                continuous_spec.append(
                    ContinuousRadiationDistribution(
                        xray_dist,
                        f"xray from {isotope_name} {','.join(openmc_decay_spectrum['xray']['continuous']['from_mode'])}",
                    )
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

            # num_counts = Integrate(gamma_dist).definite_integral(*minmax(gamma_dist))
            # warnings.warn(
            #     "Continuous gamma-ray distribution found in the decay gamma-ray spectrum"
            #     f"! This distribution sums up to {num_counts * nom()} gamma-rays "
            #     f"released per decay of {isotope_name}."
            # )
            if cont_norm:
                continuous_spec.append(
                    ContinuousRadiationDistribution(
                        gamma_dist,
                        f"gamma from {isotope_name} {','.join(openmc_decay_spectrum['gamma']['continuous']['from_mode'])}",
                    )
                )

    return discrete_spec, continuous_spec
