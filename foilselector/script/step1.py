"""
Interactive pre-processing to convert the data into the formats that the program can use
in later steps.

Files saved
-----------
.gs.csv
    group structure (lower to upper bound) saved as a 2-column .csv file.
.integrated_apriori.csv
    neutron flux (group-flux, hence the name 'integrated') in each group of the a priori
    spectrum, saved as a .csv file with as many rows as .gs.csv. Potentially includes
    the error on each bin.
.continuous_apriori.csv
    A priori spectrum saved as a continuous function. This can be decoded as a thing.
.gamma-resolution-coefs.txt
    The list of Gamma-ray detector resolution coefficients, stored in ascending degrees,
    newline-delimited, such that we can express the resolution as
    R(E) = √(x_0 + x_1 * E + x_2 * E^2 + ...)
.gamma-peak-to-Compton-coefs.txt
    The list of Gamma-ray detector peak-to-Compton ratio coefficients, stored in
    ascending degrees, newline-delimited, such that we can express the peak-to-Compton
    ratio as P/C(E) = x_0 + x_1 * E + x_2 * E^2 + ...
"""

import numpy as np
from numpy import array as ary
from numpy import typing as npt
import pandas as pd
from openmc.data import Tabulated1D
from matplotlib import pyplot as plt
import os
from pathlib import Path
from os.path import join

from foilselector.fluxconversion import *
from foilselector.generic import minmax
from foilselector.constants import MeV, keV
from foilselector.openmcextension import Integrate, detabulate
from foilselector.openmcextension.warning import SilenceNumpyDivisionError
from foilselector.simulation.detector import *


def section_title(title: str):
    """Print the title for the step that follows."""
    window_width = 124
    pad_length = max([0, len(title) - window_width])
    print(title + "-" * pad_length)


def stage1_read_raw_ap_gs(directory: Path):
    """Read the apriori value"""
    section_title("1.1 Reading the a priori values.")
    apriori = get_column_interactive(
        directory,
        "a priori spectrum's values (ignoring the uncertainty)",
        first_time_use=True,
    )
    fmt_question = "What format was the a priori provided in?('per eV', 'per keV', 'per MeV', 'PUL', 'integrated')"
    in_unit = ask_question(
        fmt_question, ["PUL", "per MeV", "per keV", "per eV", "integrated"]
    )

    section_title("1.2 Conversion into continuous format.")
    group_or_point_question = "Is this a priori spectrum given in 'group-wise' (fluxes inside discrete bins) or 'point-wise' (continuous function requiring interpolation) format?"
    group_or_point = ask_question(group_or_point_question, ["group-wise", "point-wise"])

    return apriori, in_unit, group_or_point


def stage2_interpret_ap_energy(
    directory: Path, apriori: npt.NDArray, in_unit: str, *, group_or_point: str
):
    section_title("2. Reading the energy values associated with the a priori.")
    if group_or_point == "group-wise":
        apriori_gs = ask_for_gs(directory)
        apriori = flux_conversion(apriori, apriori_gs, in_unit, "per eV")

        E_values = np.hstack([apriori_gs[:, 0], apriori_gs[-1, 1]])
        scheme = histogramic

    elif group_or_point == "point-wise":
        E_values = get_column_interactive(
            directory, "energy paired to each data point of the a priori"
        )
        E_values = scale_to_eV_interactive(E_values)
        # convert to per eV format
        if in_unit == "PUL":
            apriori *= E_values
        elif in_unit == "per MeV":
            apriori /= MeV
        elif in_unit == "per keV":
            apriori /= keV
        elif in_unit == "per eV":
            pass  # nothing needs to change
        elif in_unit == "integrated":
            raise NotImplementedError(
                "Point-wise (continuous) data should never be an integrated flux!"
            )

        print(
            "The scheme available for interpolating between data points are\n",
            INTERPOLATION_SCHEME,
            "\ne.g. linear-log denotes linear in y, logarithmic in x",
        )
        scheme = int(
            ask_question(
                "What scheme should be used to interpolate between the two points? (type the index)",
                [str(i) for i in INTERPOLATION_SCHEME.keys()],
            )
        )

        if scheme == histogramic:
            E_values = np.hstack([E_values, E_values[-1] + np.diff(E_values)[-1]])
        apriori_gs = ary([E_values[:-1], E_values[1:]]).T

    if scheme == histogramic:
        apriori = np.hstack([apriori, apriori[-1]])
    continuous_apriori = Tabulated1D(
        E_values,
        apriori,
        breakpoints=[
            len(apriori),
        ],
        interpolation=[
            scheme,
        ],
    )
    return E_values, apriori, continuous_apriori, apriori_gs, scheme


def calculate_x_points(E_values):
    """
    Calculate the optimal list of points to be sampled to fully capture a histogramic
    function represented by a Tabulated1D file.
    """
    x = np.linspace(E_values[:-1], E_values[1:], 4, endpoint=True).T
    span = ary(E_values[1:]) - ary(E_values[:-1])
    x[:, -1] -= span * 0.0001
    x = x.flatten()
    x = np.hstack([x, E_values[-1]])
    return x


def stage2_plot_apriori(
    E_values: npt.NDArray[float],
    apriori: npt.NDArray[float],
    continuous_apriori: Tabulated1D,
    apriori_gs: npt.NDArray,
):
    # plot in per eV scale
    x = calculate_x_points(E_values)
    plt.plot(x, continuous_apriori(x))
    plt.title("Neutron flux per eV")
    (
        plt.xlabel("Neutron energy (eV)"),
        plt.ylabel("neutron flux per unit energy (cm^-2 s^-1 eV^-1)"),
    )
    plt.show()
    # and then the same thing, but in in log-log scale, because on some computers the matplotlib plot function doesn't show a button to see log-scale and it works
    plt.loglog(x, continuous_apriori(x))
    (
        plt.xlabel("Neutron energy (eV)"),
        plt.ylabel("neutron flux per unit energy (cm^-2 s^-1 eV^-1)"),
    )
    plt.title("Neutron flux per eV (log-log plot)")
    plt.show()

    # plot in lethargy scale
    ap_plot = flux_conversion(apriori[:-1], apriori_gs, "per eV", "PUL")
    plt.step(E_values, np.hstack([ap_plot, ap_plot[-1]]), where="post")
    plt.yscale("log"), plt.xscale("log")
    (
        plt.xlabel("Neutron energy (eV)"),
        plt.ylabel("neutron flux per unit energy (cm^-2 s^-1 eV^-1)"),
    )
    plt.title("Neutron flux per unit lethargy")
    plt.show()


def stage3_modify_apriori(
    E_values: npt.NDArray[float],
    apriori: npt.NDArray[float],
    continuous_apriori: Tabulated1D,
):
    section_title("3. [optional] Modifying the a priori.")
    # scale the peak up and down (while keeping the total flux the same)
    if ask_yn_question("Would you like to shift the energy scale up/down?"):
        while True:
            print("new energy scale = (scale_factor) * current energy scale + offset")
            try:
                # Setting the affine scaling coefficients
                scale_factor = float(input("scale_factor="))
                offset = float(input("offset="))
                # Scale E_values and apriori
                E_values = scale_factor * E_values + offset
                continuous_apriori = Tabulated1D(
                    E_values,
                    apriori,
                    breakpoints=[
                        len(apriori),
                    ],
                    interpolation=[
                        scheme,
                    ],
                )
                # plotting
                x = calculate_x_points(E_values)
                plt.loglog(x, continuous_apriori(x))
                plt.show()
                if ask_yn_question("Is this scaling satisfactory?"):
                    break
                print("Scaling further...")
            except ValueError as e:
                print(e, ", trying again")

    # increase the flux up to a set total flux
    total_flux = Integrate(continuous_apriori).definite_integral(*minmax(E_values))
    if ask_yn_question(f"{total_flux = }, would you like to scale it up/down?"):
        while True:
            try:
                new_total_flux = float(input("Please input the new total flux:"))
                break
            except ValueError as e:
                print(e, ", trying again")
        # scaling y values
        apriori = apriori * new_total_flux / total_flux
        continuous_apriori = Tabulated1D(
            E_values,
            apriori,
            breakpoints=[len(apriori)],
            interpolation=[
                scheme,
            ],
        )
        total_flux = Integrate(continuous_apriori).definite_integral(*minmax(E_values))
        # plotting
        x = calculate_x_points(E_values)
        plt.loglog(x, continuous_apriori(x))
        plt.show()
    print(f"{total_flux = }")
    return E_values, apriori, continuous_apriori


def stage4_add_uncertainty(
    directory: Path,
    apriori: npt.NDArray[float],
    continuous_apriori: Tabulated1D,
    apriori_copy: npt.NDArray[float],
    scheme: int,
    E_values: npt.NDArray[float],
) -> tuple[Tabulated1D, Tabulated1D] | None:
    section_title("4. [optional] adding an uncertainty to the a priori.")
    if ask_yn_question(
        "Does the a priori spectrum comes with an associated error (y-error bars) on itself?"
    ):
        error_series = get_column_interactive(
            directory,
            "error (which should be of the same shape as the a priori spectrum input)",
        )
        # allow the error to be inputted in either fractional error or absolute error.
        absolute_or_fractional = ask_question(
            "does this describe the 'fractional' or 'absolute' error?",
            ["fractional", "absolute"],
        )
        if absolute_or_fractional == "fractional":
            fractional_error = error_series
        else:  # absolute error
            with SilenceNumpyDivisionError:
                fractional_error = np.nan_to_num(error_series / apriori_copy)
        if scheme == histogramic:
            error = np.hstack([fractional_error, fractional_error[-1]]) * apriori
        else:
            error = fractional_error * apriori
        continuous_apriori_lower = Tabulated1D(
            E_values,
            apriori - error,
            breakpoints=[len(apriori)],
            interpolation=[
                scheme,
            ],
        )
        continuous_apriori_upper = Tabulated1D(
            E_values,
            apriori + error,
            breakpoints=[len(apriori)],
            interpolation=[
                scheme,
            ],
        )
        x = calculate_x_points(E_values)
        plt.fill_between(x, continuous_apriori_upper(x), continuous_apriori_lower(x))
        plt.loglog(x, continuous_apriori(x), color="orange")
        plt.show()
        return continuous_apriori_lower, continuous_apriori_upper


def stage5_load_group_structure(
    directory: Path,
    apriori_gs: npt.NDArray,
    E_values: npt.NDArray[float],
    continuous_apriori: Tabulated1D,
):
    section_title("5. Load in group structure.")
    if ask_yn_question(
        "Should a different group structure than the apriori_gs (entered above) be used?"
    ):  # same gs as the a priori
        gs_source = ask_question(
            "Would you like to read the gs_bounds 'from file' or manually create an 'evenly spaced' group structure?",
            ["from file", "evenly spaced"],
        )
        if gs_source == "evenly spaced":
            print(
                "Using the naïve approach of dividing the energy/lethargy axis into equally spaced bins."
            )
            print(
                "For reference, the current minimum and maximum of the a priori spectrum are",
                *minmax(continuous_apriori.x),
            )
            while True:
                try:
                    E_min = float(
                        ask_question(
                            "What is the desired minimum energy for the group structure used?",
                            [],
                            check=False,
                        )
                    )
                    E_max = float(
                        ask_question(
                            "What is the desired maximum energy for the group structure used?",
                            [],
                            check=False,
                        )
                    )
                    spacing_prompt = "Would you like to perform a 'log-space'(equal spacing in energy space) or 'lin-space'(equal spacing in lethargy space) interpolation between these two limits?"
                    E_interp = ask_question(spacing_prompt, ["log-space", "lin-space"])
                    E_num = int(
                        ask_question(
                            "How many bins would you like to have?", [], check=False
                        )
                    )
                    break
                except ValueError as e:
                    print(e, ", trying again...")
            if E_interp == "log-space":
                gs_bounds = np.geomspace(E_min, E_max, num=E_num + 1)
            elif E_interp == "lin-space":
                gs_bounds = np.linspace(E_min, E_max, num=E_num + 1)
            gs_min, gs_max = gs_bounds[:-1], gs_bounds[1:]
            gs_array = ary([gs_min, gs_max]).T
        else:  # gs_source=='from file'
            gs_array = ask_for_gs(directory)
    else:
        gs_array = apriori_gs

    # plot the histogramic version of it once
    fig, ax = plt.subplots()
    ax.set_xlabel("E(eV)")
    ax.set_ylabel("flux(per eV)")
    x = calculate_x_points(E_values)
    ax.plot(x, continuous_apriori(x))
    ybounds = ax.get_ybound()
    for limits in gs_array:
        ax.errorbar(
            x=np.mean(limits),
            y=np.percentile(ybounds, 10),
            xerr=abs(np.diff(limits) / 2),
            capsize=30,
            color="black",
        )
        # Draw the group structure at ~10% of the height of the graph.
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.show()
    return gs_array


def stage5_save_apriori_files(
    directory: Path,
    gs_array: npt.NDArray,
    continuous_apriori,
    error_present: tuple[Tabulated1D, Tabulated1D] | None,
):
    gs_df = pd.DataFrame(gs_array, columns=["min", "max"])
    gs_df.to_csv(join(directory, ".gs.csv"), index=False)

    integrated_flux = Integrate(continuous_apriori).definite_integral(*gs_array.T)

    if error_present:
        continuous_apriori_lower, continuous_apriori_upper = error_present
        uncertainty = (
            Integrate(continuous_apriori_upper).definite_integral(*gs_array.T)
            - Integrate(continuous_apriori_lower).definite_integral(*gs_array.T)
        ) / 2
        print(
            "An (inaccurate) estimate of the error is provided as well. If a different group structure than the input file's group structure is used, then this error likely overestimated (by a factor of ~ sqrt(2)) as it does not obey the rules of error propagation properly."
        )
        apriori_vector_df = pd.DataFrame(
            ary([integrated_flux, uncertainty]).T, columns=["value", "uncertainty"]
        )
    else:
        apriori_vector_df = pd.DataFrame(integrated_flux, columns=["value"])

    apriori_vector_df.to_csv(join(directory, ".integrated_apriori.csv"), index=False)

    # save the continuous a priori distribution (an openmc.data.Tabulated1D object) as a csv, by specifying the interpolation scheme as well.
    detabulated_apriori = detabulate(continuous_apriori)
    detabulated_apriori["interpolation"].append(
        0
    )  # 0 is a placeholder, it doesn't correspond to any interpolation scheme, but is an integer so that pandas wouldn't treat it differently; unlike using None, which would force the entire column to become floats.
    detabulated_apriori_df = pd.DataFrame(detabulated_apriori)
    detabulated_apriori_df["interpolation"] = detabulated_apriori_df[
        "interpolation"
    ].astype(int)
    detabulated_apriori_df.to_csv(
        join(directory, ".continuous_apriori.csv"), index=False
    )  # x already acts pretty well as the index.
    print(
        """Preprocessing completed. The outputs are saved to:
group structure                                         => .gs.csv,
apriori flux                                            => .integrated_apriori.csv,
continuous apriori (an openmc.data.Tabulated1D object)  => .continuous_apriori.csv"""
    )


def stage6_load_and_save_gamma_resolution(directory: Path):
    """
    Returns
    -------
    coefficients
        coefficients that can be used to reconstruct the resolution curve.
    """
    section_title("6. Save gamma-ray detector resolution")

    default_res_func = resolution_curve_factory(get_default_resolution_coefficients())
    fwhm_examples = ";\n".join(
        f"FWHM = {default_res_func(peak)} keV at E={peak} keV"
        for peak in [511, 662, 1173, 1332]
    )
    print(f"Default resolution is\n{fwhm_examples}.")
    if ask_yn_question(
        "Would you like to provide your own resolution curve instead of using the default resolutions? (no = use default)"
    ):
        if ask_yn_question(
            "Do you have the coefficients in the resolution curve R(E)=√(x_0+x_1*E+x_2*E^2+...) (where E and R(E) have unit keV)? (y/n)"
        ):
            while True:
                coef_str = input(
                    "Please enter the list of coefficients, separated by comma, in increasing degree of the coefficients: "
                )
                try:
                    coefficients = ary([float(c) for c in coef_str.split(",")])
                    break
                except ValueError as e:
                    print(e, ", trying again...")
        else:
            E_keV, full_path = get_column_interactive(
                directory,
                "Mean energy of the peaks (in keV)",
                output_full_file_path=True,
            )
            fwhm_keV = get_column_interactive(
                directory, "FWHM energy of the peaks (in keV)", file_path_given=full_path
            )
            while True:
                try:
                    degree_of_fit = ask_question(
                        "How many degrees of coefficient shall be fitted (i.e. how precise should the fitting polynomial be)? (Please enter number between [0-3])",
                        "0123",
                    )
                    coefficients = fit_fwhms(E_keV, fwhm_keV, int(degree_of_fit))
                except Exception as e:
                    print(e, ", trying again...")
    else:
        coefficients = get_default_resolution_coefficients()
    with open(".gamma-resolution-coefs.txt", "w") as f:
        for coef in coefficients:
            f.write(str(coef))
    return coefficients


def stage7_load_and_save_gamma_peak_to_Compton_ratio(directory: Path):
    """
    Returns
    -------
    coefficients
        coefficients that can be used to reconstruct the Compton-to-peak curve.
    """
    section_title("7. Save gamma-ray detector photopeak-to-Compton ratio")
    default_CS_func = Compton_to_peak_curve_factory(
        get_default_peak_to_Compton_coefficients()
    )
    cs_ratio_examples = ";\n".join(
        f"P/C = {1 / default_CS_func(peak)} keV at E={peak} keV"
        for peak in [511, 662, 1173, 1332]
    )
    print(f"Default peak-to-Compton ratio is\n{cs_ratio_examples}.")
    if ask_yn_question(
        "Would you like to provide your own peak-to-Compton curve instead of using the default photopeak-to-Compton ratios? (no = use default)"
    ):
        if ask_yn_question(
            "Do you have the coefficients for the peak-to-Compton ratios PC(E) = x_0+x_1*E+x_2*E^2+... (where E has unit keV)? (y/n)"
        ):
            while True:
                coef_str = input(
                    "Please enter the list of coefficients, separated by comma, in increasing degree of the coefficients: "
                )
                try:
                    coefficients = ary([float(c) for c in coef_str.split(",")])
                    break
                except ValueError as e:
                    print(e, ", trying again...")
        else:
            E_keV, full_path = get_column_interactive(
                directory,
                "Mean energy of the peaks (in keV)",
                output_full_file_path=True,
            )
            pc = get_column_interactive(
                directory,
                "Peak-to-Compton ratio of the peaks (dimensionless)",
                file_path_given=full_path,
            )
            while True:
                try:
                    degree_of_fit = ask_question(
                        "How many degree of coefficient shall be fitted (i.e. how precise should the fitting polynomial be)? (Please enter number between [0-2])",
                        "012",
                    )
                    coefficients = fit_peak_to_Compton(E_keV, pc, int(degree_of_fit))
                except Exception as e:
                    print(e, ", trying again...")
    else:
        coefficients = get_default_peak_to_Compton_coefficients()
    with open(".gamma-peak-to-Compton-coefs.txt", "w") as f:
        for coef in coefficients:
            f.write(str(coef))
    return coefficients


def main(directory: Path):
    print(
        """
The following inputs are needed:
1. The a priori spectrum, and the energies at which those measurements are taken.
2. The group structure to be used in the investigation that follows.

The relevant data will be retrieved from the following csv files.
In the provided directory {}, the following .csv files are found:""".format(directory)
    )

    assert os.path.exists(directory), f"Directory provided '{directory}' does not exist!"
    list_dir_csv(directory)

    apriori_raw, in_unit, group_or_point = stage1_read_raw_ap_gs(directory)
    orig_apriori_raw_values = apriori_raw.copy()  # leave a copy to be used in stage4.

    E_values, apriori, continuous_apriori, apriori_gs, scheme = (
        stage2_interpret_ap_energy(
            directory, apriori_raw, in_unit, group_or_point=group_or_point
        )
    )
    stage2_plot_apriori(E_values, apriori, continuous_apriori, apriori_gs)
    E_values, apriori, continuous_apriori = stage3_modify_apriori(
        E_values, apriori, continuous_apriori
    )

    error_present = stage4_add_uncertainty(
        directory, apriori, continuous_apriori, orig_apriori_raw_values, scheme, E_values
    )

    gs_array = stage5_load_group_structure(
        directory, apriori_gs, E_values, continuous_apriori
    )
    stage5_save_apriori_files(directory, gs_array, continuous_apriori, error_present)
    resolution_coefficients = stage6_load_and_save_gamma_resolution(directory)
    peak_to_Compton_coefficients = stage7_load_and_save_gamma_peak_to_Compton_ratio(
        directory
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Interact with the user to convert the neutorn spectrum into the desired input format and group structure."
    )

    parser.add_argument("", default=Path.cwd())

    args = parser.parse_args()
    print("Acting on directory {}".format(args.cwd))
    main(args.cwd)
