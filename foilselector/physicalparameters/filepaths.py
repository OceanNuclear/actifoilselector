"""default file paths to be used on functions"""
from os import path as _path
local_dir = _path.abspath(_path.dirname(__file__))

HPGe_eff_file = _path.join(local_dir, "photopeak_efficiency", "Absolute_photopeak_efficiencyMeV.csv")

PHYSICAL_PROP_FILE = _path.join(local_dir, "material_properties", "elemental_frac_isotopic_frac_physical_property.csv")

PRICE_FILE = _path.join(local_dir, "price", "goodfellow_selected_elements.csv")
