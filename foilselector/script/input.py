"""
Interactive pre-processing to convert the data into the formats that the program can use
in later steps.

Files saved
-----------
.gs.csv (foldermanagement.GROUP_STRUCTURE_FILENAME)
    group structure (lower to upper bound) saved as a 2-column .csv file.
.integrated_apriori.csv (foldermanagement.APRIORI_FILENAME)
    neutron flux (group-flux, hence the name 'integrated') in each group of the a priori
    spectrum, saved as a .csv file with as many rows as .gs.csv. Potentially includes
    the error on each bin.
.continuous_apriori.csv (foldermanagement.CONT_APRIORI_FILENAME)
    A priori spectrum saved as a continuous function. This can be decoded as a thing.
.gamma-resolution-coefs.txt (foldermanagement.GAMMA_RES_AND_COUNT_RATE_FILENAME)
    The list of Gamma-ray detector resolution coefficients, stored in ascending degrees,
    newline-delimited, such that we can express the resolution as
    R(E) = √(x_0 + x_1 * E + x_2 * E^2 + ...)
.gamma-Compton-to-peak-coefs.txt (foldermanagement.PEAK_TO_COMPTON_FILENAME)
    The list of Gamma-ray detector Compton-to-peak ratio coefficients, stored in
    ascending degrees, newline-delimited, such that we can express the Compton-to-peak
    ratio as P/C(E) = x_0 + x_1 * E + x_2 * E^2 + ...
.efficiency{.o,.ecc,.csv,.dat}
    A copy of the gamma-ray detector efficiency calibration file, where the file suffix
    is the same as the source file's suffix. See foilselector.simulation.efficiency for
    more details.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import array as ary
from numpy import typing as npt
from openmc.data import Tabulated1D

from foilselector.constants import MeV, keV
from foilselector.fluxconversion import (
    ask_for_gs,
    ask_question,
    ask_yn_question,
    flux_conversion,
    get_column_interactive,
    histogramic,
    scale_to_eV_interactive,
)
from foilselector.fluxconversion.schemes import INTERPOLATION_SCHEME
from foilselector.foldermanagement import (
    APRIORI_FILENAME,
    CONT_APRIORI_FILENAME,
    GROUP_STRUCTURE_FILENAME,
    save_apriori,
    save_gs,
)
from foilselector.generic import SilenceNumpyDivisionError, minmax
from foilselector.openmcextension import Integral, detabulate
from foilselector.openmcextension.table import Tab1DExtended
from foilselector.simulation.compton import (
    ComptonToPeakRatioCurve,
    get_default_peak_to_Compton_file,
)
from foilselector.simulation.efficiency import (
    APPROVED_EFFICIENCY_FILE_EXTENSIONS,
    EfficiencyCurve,
    get_default_efficiency_curve_path,
    list_dir_eff_files,
    save_as_efficiency_file,
)
from foilselector.simulation.resolution import (
    ResolutionMaxCountRate,
    fit_fwhms,
    get_default_resolution_coefficients,
    resolution_curve_factory,
)


def section_title(title: str) -> None:
    """Print the title for the step that follows."""
    window_width = 124
    pad_length = max([0, len(title) - window_width])
    print(title + "-" * pad_length)


def stage1_read_raw_ap_gs() -> tuple[npt.NDArray, str, str]:
    """Read the apriori value.

    Returns
    -------
    apriori:
        raw a priori spectrum, before unit conversion.
    in_unit:
        unit for each data point on the raw a priori spectrum.
    group_or_point:
        string of either "group-wise" or "point-wise".
    """
    section_title("1.1 Reading the a priori values.")
    apriori = get_column_interactive(
        Path.cwd(),
        "a priori spectrum's values (ignoring the uncertainty)",
        first_time_use=True,
    )
    fmt_question = (
        "What format was the a priori provided in?"
        "('per eV', 'per keV', 'per MeV', 'PUL', 'integrated')"
    )
    in_unit = ask_question(
        fmt_question,
        ["PUL", "per MeV", "per keV", "per eV", "integrated"],
    )

    section_title("1.2 Conversion into continuous format.")
    group_or_point_question = (
        "Is this a priori spectrum given in 'group-wise' "
        "(fluxes inside discrete bins) or 'point-wise' (continuous function requiring "
        "interpolation) format?"
    )
    group_or_point = ask_question(group_or_point_question, ["group-wise", "point-wise"])

    return apriori, in_unit, group_or_point


def stage2_interpret_ap_energy(
    apriori: npt.NDArray,
    in_unit: str,
    *,
    group_or_point: str,
) -> tuple[
    npt.NDArray[float],
    npt.NDArray[float],
    Tabulated1D,
    int,
]:
    """Put apriori into the right group-structure, by asking the user to find the file
    with the desired group structure.

    Parameters
    ----------
    apriori:
        The raw a priori spectrum outputted by :func:`~stage1_read_raw_ap_gs`.
    in_unit:
        The unit of the raw a priori spectrum as outputted by
        :func:`~stage1_read_raw_ap_gs`.
    group_or_point:
        str of either "group-wise" or "point-wise" as outputted by
        :func:`~stage1_read_raw_ap_gs`.

    Returns
    -------
    E_values:
        Flattend group structure, containing n+1 energy boundary values for the n bins.
        Sorted in ascending energies.
    apriori:
        The a priori, after being re-binned into the desired group structure.
        Each float in the array represents the integrated flux in that bin.
    continuous_apriori:
        The a priori represented as a continuous function.
    apriori_gs:
        Group structure, with shape (n, 2) listing the lower and upper bound of each of
        the n bins.
    scheme:
        integer representing which interpolation scheme does the continuous_apriori
        follow.
    """
    section_title("2. Reading the energy values associated with the a priori.")
    cwd = Path.cwd()
    if group_or_point == "group-wise":
        apriori_gs = ask_for_gs(cwd)
        apriori = flux_conversion(apriori, apriori_gs, in_unit, "per eV")

        E_values = np.hstack([apriori_gs[:, 0], apriori_gs[-1, 1]])
        scheme = histogramic

    elif group_or_point == "point-wise":
        E_values = get_column_interactive(
            cwd,
            "energy paired to each data point of the a priori",
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
                "Point-wise (continuous) data should never be an integrated flux!",
            )

        print(
            "The scheme available for interpolating between data points are\n",
            INTERPOLATION_SCHEME,
            "\ne.g. linear-log denotes linear in y, logarithmic in x",
        )
        scheme = int(
            ask_question(
                "What scheme should be used to interpolate between the two points? "
                "(type the index)",
                [str(i) for i in INTERPOLATION_SCHEME],
            ),
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


def _calculate_x_points(E_values: npt.NDArray[float]) -> npt.NDArray[float]:
    """
    Calculate the optimal list of points to be sampled to fully capture a histogramic
    function represented by a Tabulated1D file.

    Parameters
    ----------
    E_values:
        A 1D list of floats denoting the class mark of each energy bin.

    Returns
    -------
    x:
        a list of energy values useful for sampling and plotting the a priori at.
    """
    x = np.linspace(E_values[:-1], E_values[1:], 4, endpoint=True).T
    span = ary(E_values[1:]) - ary(E_values[:-1])
    x[:, -1] -= span * 0.0001
    x = x.flatten()
    return np.hstack([x, E_values[-1]])


def stage2_plot_apriori(
    E_values: npt.NDArray[float],
    apriori: npt.NDArray[float],
    continuous_apriori: Tabulated1D,
    apriori_gs: npt.NDArray,
) -> None:
    """Plot the a priori neutron spectrum."""
    # plot in per eV scale
    x = _calculate_x_points(E_values)
    plt.plot(x, continuous_apriori(x))
    plt.title("Neutron flux per eV")
    (
        plt.xlabel("Neutron energy (eV)"),
        plt.ylabel("neutron flux per unit energy (cm^-2 s^-1 eV^-1)"),
    )
    plt.show()
    # and then the same thing, but in in log-log scale, because on some computers the
    # matplotlib plot function doesn't show a button to see log-scale and it works
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
    continuous_apriori: Tab1DExtended,
) -> tuple[npt.NDArray[float], npt.NDArray[float], Tabulated1D | Tab1DExtended]:
    """Modify the a priori neutron spectrum to something that the user wants, by
    transforming the energy scale and scaling the y-axis.

    Parameters
    ----------
    E_values:
        Location where the a priori is defined from data.
    apriori:
        The a priori neutron spectrum at the energies specified before modification.
    continuous_apriori:
        A distribution of a priori, continuous in energy space.

    Returns
    -------
    E_values:
        Location where the a priori is defined from data, possibly after modification.
    apriori:
        The a priori neutron spectrum at the energies specified after modification.
    continuous_apriori:
        A distribution of a priori, continuous in energy space.
    """
    section_title("3. [optional] Modifying the a priori.")
    # scale the peak up and down (while keeping the total flux the same)
    if ask_yn_question("Would you like to shift the energy scale up/down?"):
        while True:
            print("new energy scale = (scale_factor * current energy scale) + offset")
            try:
                # Setting the affine scaling coefficients
                scale_factor = float(input("scale_factor="))
                offset = float(input("offset="))
                # Scale E_values and apriori
                if isinstance(continuous_apriori, Tabulated1D):
                    continuous_apriori = Tab1DExtended.from_openmc(continuous_apriori)
                continuous_apriori = continuous_apriori.scale_x(scale_factor)
                continuous_apriori = continuous_apriori.offset_x(offset)
                E_values = scale_factor * E_values + offset
                apriori *= scale_factor
                # plotting
                x = _calculate_x_points(E_values)
                plt.loglog(x, continuous_apriori(x))
                plt.show()
                if ask_yn_question("Is this scaling satisfactory?"):
                    break
                print("Scaling further...")
            except ValueError as e:
                print(e, ", trying again")

    # increase the flux up to a set total flux
    total_flux = sum(apriori)
    if ask_yn_question(f"{total_flux = }, would you like to scale it up/down?"):
        while True:
            try:
                new_total_flux = float(input("Please input the new total flux:"))
                if isinstance(continuous_apriori, Tabulated1D):
                    continuous_apriori = Tab1DExtended.from_openmc(continuous_apriori)
                break
            except ValueError as e:
                print(e, ", trying again")
        # scaling y values
        scale_factor = new_total_flux / total_flux
        apriori *= scale_factor
        continuous_apriori *= scale_factor
        total_flux = sum(apriori)
        # plotting
        x = _calculate_x_points(E_values)
        plt.loglog(x, continuous_apriori(x))
        plt.show()
    print(f"{total_flux = }")
    return E_values, apriori, continuous_apriori


def stage4_add_uncertainty(
    apriori: npt.NDArray[float],
    continuous_apriori: Tab1DExtended,
    apriori_copy: npt.NDArray[float],
    scheme: int,
    E_values: npt.NDArray[float],
) -> tuple[Tabulated1D, Tabulated1D] | None:
    """Add an uncertainty quantification to the a priori.

    Returns
    -------
    continuous_apriori_lower:
        the a priori neutron spectrum -1 sigma
    continuous_apriori_upper:
        the a priori neutron spectrum +1 sigma
    """
    section_title("4. [optional] adding an uncertainty to the a priori.")
    if ask_yn_question(
        "Does the a priori spectrum comes with an associated error (y-error bars) on "
        "itself?",
    ):
        error_series = get_column_interactive(
            Path.cwd(),
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
            with SilenceNumpyDivisionError():
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
        x = _calculate_x_points(E_values)
        plt.fill_between(x, continuous_apriori_upper(x), continuous_apriori_lower(x))
        plt.loglog(x, continuous_apriori(x), color="orange")
        plt.show()
        return continuous_apriori_lower, continuous_apriori_upper
    return None


def stage5_load_group_structure(
    apriori_gs: npt.NDArray,
    E_values: npt.NDArray[float],
    continuous_apriori: Tab1DExtended,
) -> npt.NDArray:
    """Choose a different group structure than what the a priori used.

    Returns
    -------
    gs_array:
        2D group structure array, with shape = (n,2) where n = number of neutron energy
        bins, and the two elements shows the lower and upper energy boundary of the bin
        respsectively.
    """
    section_title("5. [optional] Load in new group structure.")
    if ask_yn_question(
        "Should a different group structure than the apriori_gs (entered above) be used?",
    ):  # same gs as the a priori
        gs_source = ask_question(
            "Would you like to read the gs_bounds 'from file' or manually create an "
            "'evenly spaced' group structure?",
            ["from file", "evenly spaced"],
        )
        if gs_source == "evenly spaced":
            print(
                "Using the naïve approach of dividing the energy/lethargy axis into "
                "equally spaced bins.",
            )
            print(
                "For reference, the current minimum and maximum of the a priori "
                "spectrum are",
                *minmax(continuous_apriori.x),
            )
            while True:
                try:
                    E_min = float(
                        ask_question(
                            "What is the desired minimum energy for the group structure "
                            "used?",
                            [],
                            check=False,
                        ),
                    )
                    E_max = float(
                        ask_question(
                            "What is the desired maximum energy for the group structure "
                            "used?",
                            [],
                            check=False,
                        ),
                    )
                    spacing_prompt = "Would you like to perform a 'log-space'(equal "
                    "spacing in energy space) or 'lin-space'(equal spacing in lethargy "
                    "space) interpolation between these two limits?"
                    E_interp = ask_question(spacing_prompt, ["log-space", "lin-space"])
                    E_num = int(
                        ask_question(
                            "How many bins would you like to have?",
                            [],
                            check=False,
                        ),
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
            gs_array = ask_for_gs(Path.cwd())
    else:
        gs_array = apriori_gs

    # plot the histogramic version of it once
    _fig, ax = plt.subplots()
    ax.set_xlabel("E(eV)")
    ax.set_ylabel("flux(per eV)")
    x = _calculate_x_points(E_values)
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
    gs_array: npt.NDArray,
    continuous_apriori: Tab1DExtended,
    error_present: tuple[Tabulated1D, Tabulated1D] | None,
) -> None:
    """Save the a priori as a file to be used in the next step."""
    gs_df = pd.DataFrame(gs_array, columns=["min", "max"])
    save_gs(gs_df)

    integrated_flux = Integral(continuous_apriori).definite_integral(*gs_array.T)

    if error_present:
        continuous_apriori_lower, continuous_apriori_upper = error_present
        uncertainty = (
            Integral(continuous_apriori_upper).definite_integral(*gs_array.T)
            - Integral(continuous_apriori_lower).definite_integral(*gs_array.T)
        ) / 2
        print(
            "An (inaccurate) estimate of the error is provided as well. If a different "
            "group structure than the input file's group structure is used, then this "
            "error likely overestimated (by a factor of ~ sqrt(2)) as it does not obey "
            "the rules of error propagation properly.",
        )
        apriori_vector_df = pd.DataFrame(
            ary([integrated_flux, uncertainty]).T,
            columns=["value", "uncertainty"],
        )
    else:
        apriori_vector_df = pd.DataFrame(integrated_flux, columns=["value"])

    save_apriori(apriori_vector_df)

    # save the continuous a priori distribution (an openmc.data.Tabulated1D object) as a
    # csv, by specifying the interpolation scheme as well.
    detabulated_apriori = detabulate(continuous_apriori)
    detabulated_apriori["interpolation"].append(
        0,
    )  # 0 is a placeholder, it doesn't correspond to any interpolation scheme, but is an
    # integer so that pandas wouldn't treat it differently; unlike using None, which
    # would force the entire column to become floats.
    detabulated_apriori_df = pd.DataFrame(detabulated_apriori)
    detabulated_apriori_df["interpolation"] = detabulated_apriori_df[
        "interpolation"
    ].astype(int)
    detabulated_apriori_df.to_csv(
        CONT_APRIORI_FILENAME,
        index=False,
    )  # x already acts pretty well as the index.
    print(
        f"""Preprocessing completed. The outputs are saved to:
group structure                                         => {GROUP_STRUCTURE_FILENAME},
apriori flux                                            => {APRIORI_FILENAME},
continuous apriori (an openmc.data.Tabulated1D object)  => {CONT_APRIORI_FILENAME}""",
    )


def stage6_load_and_save_gamma_resolution() -> None:
    """
    Write to file the gamma-ray detector's resolution.

    Returns
    -------
    coefficients
        coefficients that can be used to reconstruct the resolution curve.
    """
    section_title("6. Save gamma-ray detector resolution")
    cwd = Path.cwd()

    default_res_func = resolution_curve_factory(get_default_resolution_coefficients())
    fwhm_examples = ";\n".join(
        f"FWHM = {default_res_func(peak * keV)} keV at E={peak} keV"
        for peak in [511, 662, 1173, 1332]
    )
    print(f"Default resolution is\n{fwhm_examples}.")
    if ask_yn_question(
        "Would you like to provide your own resolution curve instead of using the "
        "default resolutions? (no = use default)",
    ):
        if ask_yn_question(
            "Do you have the coefficients in the resolution curve "
            "R(E) = √(x_0 + x_1*E + x_2*E^2 + ...) (where E and R(E) have unit eV)? "
            "(y/n)",
        ):
            while True:
                try:
                    coef_str = input(
                        "Please enter the list of coefficients, separated by comma, in "
                        "increasing degree of the coefficients: ",
                    )
                    coefficients = ary([float(c) for c in coef_str.split(",")])
                    break
                except ValueError as e:
                    print(e, ", trying again...")
        else:
            print("Fitting to FWHM data:")
            while True:
                try:
                    E_keV, full_path = get_column_interactive(
                        cwd,
                        "Mean energy of the peaks (in keV)",
                        output_full_file_path=True,
                        first_time_use=True,
                    )
                    fwhm_keV = get_column_interactive(
                        cwd,
                        "FWHM energy of the peaks (in keV)",
                        file_path_given=full_path,
                    )
                    degree_of_fit = ask_question(
                        "How many degrees of coefficient shall be fitted (i.e. how "
                        "precise should the fitting polynomial be)? (Please enter "
                        "number between [0-3])",
                        "0123",
                    )
                    coefficients = fit_fwhms(
                        E_keV * keV,
                        fwhm_keV * keV,
                        int(degree_of_fit),
                    )
                    break
                except ValueError as e:
                    print(e, ", trying again...")
            ax = plt.axes()
            ax.scatter(E_keV, fwhm_keV)
            e_keV_range = np.linspace(0, max(E_keV))
            ax.plot(
                e_keV_range,
                np.sqrt(np.poly1d(coefficients[::-1])(e_keV_range * keV)) / keV,
            )
            ax.set_title(
                "Resolution of the gamma-ray detector (i.e.\npeak-width v.s. incident gamma-ray energy)",
            )
            ax.set_xlabel("Photopeak energy (keV)")
            ax.set_ylabel("FWHM (keV)")
            plt.show()
    else:
        coefficients = get_default_resolution_coefficients()

    while True:
        try:
            max_count_rate = float(
                input(
                    "What is the maximum count rate (pulse/s) that the gamma-ray "
                    "detector can be operated at without degrading this resolution?",
                ),
            )
            break
        except ValueError as e:
            print(
                e,
                ". Please enter the numeric value of the max. count rate in pulse/s.",
            )
    return ResolutionMaxCountRate(coefficients, max_count_rate).save()


def stage7_load_and_save_gamma_efficiency() -> None:
    """Write to file the gamma-ray detector's absolute efficiency curve."""
    cwd = Path.cwd()
    section_title("7. Save photopeak efficiency file")
    endings = list(APPROVED_EFFICIENCY_FILE_EXTENSIONS)

    def one_loop(eff_file_path: Path | str) -> EfficiencyCurve:
        """A single iteration of opening an efficiency file and plotting it."""  # noqa: D401
        eff_file_path = Path(eff_file_path)
        with eff_file_path.open() as f:
            print(f"Opened {eff_file_path.name}, which has `head` = ")
            for _ in range(3):
                print(f.readline()[:-1])
        print("...")

        efficiency_curve = EfficiencyCurve.from_file(eff_file_path)
        E, eff = efficiency_curve.E / keV, efficiency_curve.eff
        if efficiency_curve.unc is not None:
            plt.errorbar(
                E,
                eff,
                yerr=efficiency_curve.unc,
                linestyle="",
                capsize=2.5,
                marker="x",
                label="efficiency data-points",
            )
        else:
            plt.scatter(E, eff, label="efficiency data-points")
        x = np.geomspace(
            np.min(E),
            max([np.max(E), 2000]),
            300,
        )  # from the lowest E point to 2000 keV.
        plt.semilogy(
            x,
            efficiency_curve(x * keV),
            color="C1",
            label="efficiency curve fitted from this data",
        )
        plt.legend()
        plt.xlabel(r"$E_\gamma$ (keV)")
        plt.ylabel(r"efficiency $\epsilon$")
        plt.show()
        plt.close()
        return efficiency_curve  # noqa: DOC201

    while True:
        try:
            EfficiencyCurve.from_file(
                default_efficiency_file := get_default_efficiency_curve_path(),
            )
            chosen_eff_file = input(
                f"Please choose file from the list above (file must end in {endings});"
                "\nOr enter nothing to use the example efficiency file stored at "
                f"{default_efficiency_file}:",
            )
            if not chosen_eff_file:
                chosen_eff_file = default_efficiency_file
            eff_curve = one_loop(chosen_eff_file)
            if ask_yn_question("Is this curve satisfactory?"):
                save_as_efficiency_file(chosen_eff_file)
                break
            print(
                "Add/change datapoints/ use a different data file, and try again...",
            )
        except FileNotFoundError as e:
            print(
                e,
                f", please confirm that file name is correct and exists in {cwd}. "
                "Trying again...",
            )
        except ValueError as e:
            print(e, ", Perhaps not enough data points were given? Trying again...")

    return eff_curve


def stage8_load_and_save_gamma_peak_to_Compton_ratio() -> None:
    """
    Write to file the gamma-ray detector's Compton-to-peak ratio.

    Returns
    -------
    coefficients
        coefficients that can be used to reconstruct the Compton-to-peak curve.
    """
    cwd = Path.cwd()
    section_title("8. Save gamma-ray detector Compton-to-photopeak ratio")
    default_CS_func = ComptonToPeakRatioCurve.from_file(
        get_default_peak_to_Compton_file(),
    )
    cs_ratio_examples = ";\n".join(
        f"Compton : peak ratio = {default_CS_func(peak * keV)} keV at E={peak} keV"
        for peak in [511, 662, 1173, 1332]
    )
    print(f"Default Compton-to-peak ratio is\n{cs_ratio_examples}.")
    if ask_yn_question(
        "Would you like to provide your own Compton-to-peak curve instead of using the "
        "default Compton-to-photopeak ratios? (no = use default)",
    ):
        if ask_yn_question(
            "Do you have the coefficients for the Compton-to-peak ratios "
            "log(Compton-to-Peak(E)) = x_0 + x_1*log(E) + x_2*log(E)^2 + ... "
            "(where E has unit eV)? (y/n)",
        ):
            while True:
                try:
                    coef_str = input(
                        "Please enter the list of coefficients, separated by comma, in "
                        "increasing degree of the coefficients: ",
                    )
                    coefficients = ary([float(c) for c in coef_str.split(",")])
                    curve = ComptonToPeakRatioCurve(coefficients)
                    break
                except ValueError as e:
                    print(e, ", trying again...")
        else:
            print("Fitting to Compton-to-peak ratio data:")
            while True:
                try:
                    E_keV, full_path = get_column_interactive(
                        cwd,
                        "Mean energy of each photopeak (in keV)",
                        output_full_file_path=True,
                        first_time_use=True,
                    )
                    pc = get_column_interactive(
                        cwd,
                        "Peak-to-Compton ratio of each photopeak (dimensionless)",
                        file_path_given=full_path,
                    )
                    degree_of_fit = ask_question(
                        "How many degree of coefficient shall be fitted (i.e. how "
                        "precise should the fitting polynomial be)? (Please enter a "
                        "number between [0-11] inclusive.)",
                        list(range(12)),
                    )
                    curve = ComptonToPeakRatioCurve.fit_data(
                        E_keV * keV,
                        pc,
                        int(degree_of_fit),
                    )
                    break
                except ValueError as e:
                    print(e, ", trying again...")
    else:
        curve = default_CS_func
    return curve.save()


def main() -> None:
    """Main script of step1:input."""  # noqa: D401
    cwd = Path.cwd()
    print(
        f"""
The following inputs are needed:
1. The a priori spectrum, and the energies at which those measurements are taken.
2. The group structure to be used in the investigation that follows.

The relevant data will be retrieved from the following csv files.
In the current directory {cwd}, the following .csv files are found:""",
    )

    # stage 1
    apriori_raw, in_unit, group_or_point = stage1_read_raw_ap_gs()
    orig_apriori_raw_values = apriori_raw.copy()  # leave a copy to be used in stage4.

    # stage 2
    E_values, apriori, continuous_apriori, apriori_gs, scheme = (
        stage2_interpret_ap_energy(apriori_raw, in_unit, group_or_point=group_or_point)
    )
    stage2_plot_apriori(E_values, apriori, continuous_apriori, apriori_gs)

    # stage 3
    E_values, apriori, continuous_apriori = stage3_modify_apriori(
        E_values,
        apriori,
        Tab1DExtended.from_openmc(continuous_apriori),
    )

    # stage 4
    error_present = stage4_add_uncertainty(
        apriori,
        continuous_apriori,
        orig_apriori_raw_values,
        scheme,
        E_values,
    )

    # stage 5
    gs_array = stage5_load_group_structure(apriori_gs, E_values, continuous_apriori)
    stage5_save_apriori_files(gs_array, continuous_apriori, error_present)

    # stage 6
    _resolution_max_count_rate = stage6_load_and_save_gamma_resolution()

    # stage 7
    list_dir_eff_files(cwd)
    stage7_load_and_save_gamma_efficiency()

    # stage 8
    stage8_load_and_save_gamma_peak_to_Compton_ratio()
