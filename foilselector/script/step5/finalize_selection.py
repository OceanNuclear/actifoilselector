"""
Contains support functions for simulating finalized foils gamma spectra
(and potentially effectiveness at unfolding as well).
"""
def get_num_reactants_dict(foil_dict):
    volume_cm3 = foil_dict["thickness (cm)"] * foil["area (cm^2)"]
    number_density = foil["number density (cm^-3)"]
    num_reactants = { species : num_density * volume_cm3 for species, num_density in number_density.items() }
    return num_reactants
