from foilselector.fluxconversion import *
from foilselector.constants import MeV, keV
from foilselector.openmcextension import Integrate, detabulate
# openmc
from openmc.data import Tabulated1D
from matplotlib import pyplot as plt

import argparse
parg = argparse.ArgumentParser(description="Interact with the user to convert the neutorn spectrum into the desired input format and group structure.")

def main(directory):
    print("""
The following inputs are needed:
1. The a priori spectrum, and the energies at which those measurements are taken.
2. The group structure to be used in the investigation that follows.

The relevant data will be retrieved from the following csv files.
In the provided directory {}, the following .csv files are found:""".format(directory) )

    assert os.path.exists(directory), f"Directory {directory} does not exist!"
    list_dir_csv(directory)

    print("1.1 Reading the a priori values---------------------------------------------------------------------------------------------")
    apriori = get_column_interactive(directory, "a priori spectrum's values (ignoring the uncertainty)", first_time_use=True)
    apriori_copy = apriori.copy() # leave a copy to be used in stage [4.]
    fmt_question = "What format was the a priori provided in?('per eV', 'per keV', 'per MeV', 'PUL', 'integrated')"
    in_unit = ask_question(fmt_question, ['PUL','per MeV', 'per keV', 'per eV', 'integrated'])

    print("1.2 Conversion into continuous format---------------------------------------------------------------------------------------")
    group_or_point_question = "Is this a priori spectrum given in 'group-wise' (fluxes inside discrete bins) or 'point-wise' (continuous function requiring interpolation) format?"
    group_or_point = ask_question(group_or_point_question, ['group-wise','point-wise'])

    print("2. Reading the energy values associated with the a priori------------------------------------------------------------------")
    if group_or_point=='group-wise':
        apriori_gs = ask_for_gs(directory)
        apriori = flux_conversion(apriori, apriori_gs, in_unit, 'per eV')

        E_values = np.hstack([apriori_gs[:,0], apriori_gs[-1,1]])
        scheme = histogramic

    elif group_or_point=='point-wise':
        E_values = get_column_interactive(directory, 'energy associated with each data point of the a priori')
        E_values = scale_to_eV_interactive(E_values)
        #convert to per eV format
        if in_unit=='PUL':
            apriori *= E_values 
        elif in_unit=='per MeV':
            apriori /= MeV
        elif in_unit=='per keV':
            apriori /= keV
        elif in_unit=='per eV':
            pass # nothing needs to change
        elif in_unit=='integrated':
            raise NotImplementedError("Continuous data should never be an integrated flux!")

        print("The scheme available for interpolating between data points are\n", INTERPOLATION_SCHEME, "\ne.g. linear-log denotes linear in y, logarithmic in x")
        scheme = int(ask_question("What scheme should be used to interpolate between the two points? (type the index)", [str(i) for i in INTERPOLATION_SCHEME.keys()]))
        
        if scheme==histogramic:
            E_values = np.hstack([E_values, E_values[-1]+np.diff(E_values)[-1]])
        apriori_gs = ary([E_values[:-1], E_values[1:]]).T

    if scheme==histogramic:
        apriori = np.hstack([apriori, apriori[-1]])
    continuous_apriori = Tabulated1D(E_values, apriori, breakpoints=[len(apriori),], interpolation=[scheme,])

    # plot in per eV scale
    plt.plot(x:=np.linspace(min(E_values), max(E_values), num=200), continuous_apriori(x))
    plt.title("Neutron flux per eV")
    plt.xlabel("Neutron energy (eV)"), plt.ylabel("neutron flux per unit energy (cm^-2 s^-1 eV^-1)")
    plt.show()
    # and then the same thing, but in in log-log scale, because on some computers the matplotlib plot function doesn't show a button to see log-scale and it works
    plt.loglog(x:=np.geomspace(min(E_values), max(E_values), num=200), continuous_apriori(x))
    plt.xlabel("Neutron energy (eV)"), plt.ylabel("neutron flux per unit energy (cm^-2 s^-1 eV^-1)")
    plt.title("Neutron flux per eV (log-log plot)")
    plt.show()

    #plot in lethargy scale
    ap_plot = flux_conversion(apriori[:-1], apriori_gs, 'per eV', 'PUL')
    plt.step(E_values, np.hstack([ap_plot, ap_plot[-1]]), where='post')
    plt.yscale('log'), plt.xscale('log')
    plt.xlabel("Neutron energy (eV)"), plt.ylabel("neutron flux per unit energy (cm^-2 s^-1 eV^-1)")
    plt.title('Neutron flux per unit lethargy')
    plt.show()

    print("3. [optional] Modifying the a priori---------------------------------------------------------------------------------------")
    # scale the peak up and down (while keeping the total flux the same)
    flux_shifting_question = "Would you like to shift the energy scale up/down?"
    to_shift = ask_yn_question(flux_shifting_question)
    if not to_shift:
        pass
    if to_shift:
        print("new energy scale = (scale_factor) * old energy scale + offset")
        while True:
            try:
                scale_factor=float(input("scale_factor="))
                offset = float(input("offset="))
                E_values = scale_factor * E_values + offset
                continuous_apriori = Tabulated1D(E_values, apriori, breakpoints=[len(apriori),], interpolation=[scheme,])
                plt.plot(x:=np.linspace(*minmax(E_values), num=200), continuous_apriori(x))
                plt.show()
                can_stop = ask_yn_question("Is this satisfactory?")
                if can_stop:
                    break
                print("Retrying...")
            except ValueError as e:
                print(e)

    #increase the flux up to a set total flux
    total_flux = Integrate(continuous_apriori).definite_integral(min(E_values), max(E_values))
    flux_scaling_question = f"{total_flux = }, would you like to scale it up/down?"
    to_scale = ask_yn_question(flux_scaling_question)

    if not to_scale:
        pass
    elif to_scale:
        while True:
            try:
                new_total_flux = float(input("Please input the new total flux:"))
                break
            except ValueError as e:
                print(e)
        apriori = apriori * new_total_flux/total_flux
        continuous_apriori = Tabulated1D(E_values, apriori, breakpoints=[len(apriori)], interpolation=[scheme,])
        total_flux = Integrate(continuous_apriori).definite_integral(min(E_values), max(E_values))
        plt.plot(x, continuous_apriori(x))
        plt.show()
    print(f"{total_flux = }")

    print("4. [optional] adding an uncertainty to the a priori-------------------------------------------------------------------------")
    error_present = ask_yn_question("Does the a priori spectrum comes with an associated error (y-error bars) on itself?")
    if error_present:
        error_series = get_column_interactive(directory, "error (which should be of the same shape as the a priori spectrum input)")
        # allow the error to be inputted in either fractional error or absolute error.
        absolute_or_fractional = ask_question("does this describe the 'fractional' or 'absolute' error?", ['fractional', 'absolute'])
        if absolute_or_fractional=='fractional':
            fractional_error = error_series
        else:
            fractional_error = np.nan_to_num(error_series/apriori_copy)
        if scheme==histogramic:
            fractional_error_ext = np.hstack([fractional_error, fractional_error[-1]])
            error = fractional_error_ext * apriori
        else:
            error = fractional_error * apriori
        continuous_apriori_lower = Tabulated1D(E_values, apriori-error, breakpoints=[len(apriori)], interpolation=[scheme,])
        continuous_apriori_upper = Tabulated1D(E_values, apriori+error, breakpoints=[len(apriori)], interpolation=[scheme,])
        plt.fill_between(x, continuous_apriori_upper(x), continuous_apriori_lower(x))
        plt.plot(x, continuous_apriori(x), color='orange')
        plt.show()

    print("5. Load in group structure--------------------------------------------------------------------------------------------------")
    diff_gs_question = "Should a different group structure than the apriori_gs (entered above) be used?"
    diff_gs = ask_yn_question(diff_gs_question)

    if not diff_gs: # same gs as the a priori
        gs_array = apriori_gs
    elif diff_gs:
        gs_source_question = "Would you like to read the gs_bounds 'from file' or manually create an 'evenly spaced' group structure?"
        gs_source = ask_question(gs_source_question, ['from file', 'evenly spaced'])
        if gs_source=='evenly spaced':
            print("Using the naïve approach of dividing the energy/lethargy axis into equally spaced bins.")
            print("For reference, the current minimum and maximum of the a priori spectrum are", *minmax(continuous_apriori.x))
            while True:
                try:
                    E_min = float(ask_question("What is the desired minimum energy for the group structure used?", [], check=False))
                    E_max = float(ask_question("What is the desired maximum energy for the group structure used?", [], check=False))
                    spacing_prompt = "Would you like to perform a 'log-space'(equal spacing in energy space) or 'lin-space'(equal spacing in lethargy space) interpolation between these two limits?"
                    E_interp = ask_question(spacing_prompt, ['log-space', 'lin-space'])
                    E_num = int(ask_question("How many bins would you like to have?", [], check=False))
                    break
                except ValueError as e:
                    print(e)
            if E_interp=='log-space':
                gs_bounds = np.geomspace(E_min, E_max, num=E_num+1)
            elif E_interp=='lin-space':
                gs_bounds = np.linspace(E_min, E_max, num=E_num+1)
            gs_min, gs_max = gs_bounds[:-1], gs_bounds[1:]
            gs_array = ary([gs_min, gs_max]).T
        elif gs_source=='from file':
            gs_array = ask_for_gs(directory)

    fig, ax = plt.subplots()
    ax.set_ylabel("E(eV)")
    ax.set_xlabel("flux(per eV)")
    ax.plot(x, continuous_apriori(x))
    ybounds = ax.get_ybound()
    for limits in gs_array:
        ax.errorbar(x=np.mean(limits), y=np.percentile(ybounds, 10), xerr=abs(np.diff(limits)/2), capsize=30, color='black')
        # Draw the group structure at ~10% of the height of the graph.
    plt.show()

    # plot the histogramic version of it once
    integrated_flux = Integrate(continuous_apriori).definite_integral(*gs_array.T)
    if error_present:
        uncertainty = (Integrate(continuous_apriori_upper).definite_integral(*gs_array.T) -
                        Integrate(continuous_apriori_lower).definite_integral(*gs_array.T) )/2
        print("An (inaccurate) estimate of the error is provided as well. If a different group structure than the input file's group structure is used, then this error likely overestimated (by a factor of ~ sqrt(2)) as it does not obey the rules of error propagation properly.")

    gs_df = pd.DataFrame(gs_array, columns=['min', 'max'])
    gs_df.to_csv(join(directory, 'gs.csv'), index=False)

    if not error_present:
        apriori_vector_df = pd.DataFrame(integrated_flux, columns=['value'])
    elif error_present:
        apriori_vector_df = pd.DataFrame(ary([integrated_flux, uncertainty]).T, columns=['value', 'uncertainty'])

    apriori_vector_df.to_csv(join(directory, 'integrated_apriori.csv'), index=False)

    # save the continuous a priori distribution (an openmc.data.Tabulated1D object) as a csv, by specifying the interpolation scheme as well.
    detabulated_apriori = detabulate(continuous_apriori)
    detabulated_apriori["interpolation"].append(0) # 0 is a placeholder, it doesn't correspond to any interpolation scheme, but is an integer so that pandas wouldn't treat it differently; unlike using None, which would force the entire column to become floats.
    detabulated_apriori_df = pd.DataFrame(detabulated_apriori)
    detabulated_apriori_df["interpolation"] = detabulated_apriori_df["interpolation"].astype(int)
    detabulated_apriori_df.to_csv(join( directory, "continuous_apriori.csv"), index=False) # x already acts pretty well as the index.

    print(
    """Preprocessing completed. The outputs are saved to:
group structure                                         => gs.csv,
apriori flux                                            => integrated_apriori.csv,
continuous apriori (an openmc.data.Tabulated1D object)  => continuous_apriori.csv"""
    )
if __name__=="__main__":
    import sys
    if len(sys.argv)==1:
        cwd = "./"
    else:
        cwd = sys.argv[-1]
    print("Acting on directory {}".format(cwd))
    main(cwd)