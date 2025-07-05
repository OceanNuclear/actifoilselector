"""Read the spectra saved from the previous step (simulate.py), and plot it
interactively.
"""

import json
from collections.abc import Iterable
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from foilselector.openmcextension.extended_io import deserialize_radiation_list
from foilselector.simulation.spectral_simulation import plot_spectrum

RED = (1.0, 0.0, 0.0, 1.0)


def convert_to_path(path_or_name: str | Path) -> Path:
    if not str(path_or_name).endswith(".json"):
        name = path_or_name
        return Path("gamma_spectra", f"{name}.json")
    return Path(path_or_name)


def fs_sensitivity_distribution_plot(
    summed_sensitivity_array: np.ndarray[float],
    apriori: np.ndarray[float],
    gs_array: np.ndarray,
    precision_unit: str,
) -> tuple[plt.Figure, plt.Axes]:
    bin_widths = np.diff(gs_array).flatten()
    flux_per_eV = apriori / bin_widths
    fig, ax = plt.subplots()
    ax.loglog(gs_array.flatten(), np.repeat(flux_per_eV, 2), label="a priori")
    ax.set_title("Sensitivity of this specific foil set")
    cmap = mpl.colormaps["Reds"]
    norm = mpl.colors.PowerNorm(vmin=0.0, vmax=summed_sensitivity_array.max(), gamma=0.2)
    for bin_bounds, flux, sens in zip(
        gs_array,
        flux_per_eV,
        summed_sensitivity_array,
        strict=False,
    ):
        ax.fill_between(bin_bounds, [flux, flux], color=cmap(norm(sens)))
    cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax)
    cbar.set_label(f"Precision ({precision_unit})")
    ax.legend()
    ax.set_xlabel("E (eV)")
    ax.set_ylabel("Neutron flux (cm^-2 eV^-1)")
    return fig, ax


def main(gamma_json_paths: Iterable[Path | str]):
    for path_or_name in gamma_json_paths:
        path = convert_to_path(path_or_name)
        with path.open() as j:
            data = json.load(j)
        energy, spectrum = np.array(data["energy (keV)"]), np.array(data["spectrum"])
        print(spectrum)
        reaction_info = deserialize_radiation_list(data["radiation"])
        ax = plot_spectrum(energy, spectrum, peak_labels=reaction_info)
        # ax = plot_in_sqrt_scale(energy, spectrum, peak_labels=reaction_info)
        ax.set_title(path.stem)
        ax.get_figure().set_size_inches(20, 12)
        print(f"All visible photopeaks of {path.stem}")
        peaks_df = pd.DataFrame(
            [[peak.energy, peak.intensity, peak.source] for peak in reaction_info],
            columns=["energy (eV)", "number of counts", "source"],
        )
        peaks_df.index.name = "peak #"
        print(peaks_df.to_markdown())
        plt.show()
        plt.close()
        print()
