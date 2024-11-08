import json
from pathlib import Path
from foldermanagement import get_apriori_from_folder, get_durations_from_csv

NUM_COPIES = 5

# 1. Load the a priori, load the durations from .counts.csv
apriori = get_apriori_from_folder(Path(".").resolve())
irradiation_duration, transit_duration, acquisition_duration = get_durations_from_csv(".counts.csv")
apriori_fluence = apriori * irradiation_duration

# 2. Get the max count rate, used for calculating the foil size.
max_count_rate = input("Max count rate per second of the gamma-ray detector?")

# 3. Calculate the foil size,
# 3.1 Include the foil mass into the name
# 3.2 save as a data object?

with open(".atomic_composition.json") as j:
    atomic_composition = json.load(j)
    # sigma_df (prev. version) is in barns
foil_choices = {}

for element_name, isotopes in atomic_composition.items():
    one_atom_response_matrix = ...
    num_reactants = calculate_max_num_reactants(one_atom_response_matrix, apriori_fluence, max_count_rate, acquisition_duration)
    foil_mass = mass_of_one_reactant_atom(isotopes) * num_reactants
    if isotopes:
        for foil_num in range(1, NUM_COPIES+1):
            # foil_choices[element_name+"_foil"+str(foil_num)] = {isotope: frac_isotope * num_elements for isotope, frac_isotope in isotopes.items()}
            foil_choices

for foil in foil_choices:
    foil