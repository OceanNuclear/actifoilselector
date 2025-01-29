"""default file paths to be used on functions."""

from pathlib import Path

__all__ = ["HPGE_EFF_FILE", "PEAK_TO_COMPTON_FILE", "PHYSICAL_PROP_FILE", "PRICE_FILE"]

_local_dir = Path(__file__).resolve().parent

HPGE_EFF_FILE = Path(
    _local_dir,
    "efficiency",
    "Absolute_photopeak_efficiencyMeV.csv",
)
PEAK_TO_COMPTON_FILE = Path(_local_dir, "efficiency", "Compton_to_peak_ratio.csv")

PHYSICAL_PROP_FILE = Path(
    _local_dir,
    "material_properties",
    "elemental_frac_isotopic_frac_physical_property.csv",
)

PRICE_FILE = Path(_local_dir, "price", "goodfellow_selected_elements.csv")
