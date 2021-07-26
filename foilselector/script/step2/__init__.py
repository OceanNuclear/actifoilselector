from .read_data import *

SORT_BY_REACTION_RATE = True # sort all reactions in descending order according to the reaction rates
SHOW_SEPARATE_MT_REACTION_RATES = False # show different MT pathways of getting the same products from the same parents in separate lines.

FULL_DECAY_INFO_FILE = "decay_radiation.json"
CONDENSED_DECAY_INFO_FILE = "decay_info.json"
MICROSCOPIC_XS_CSV = "microscopic_xs.csv"
MAX_XS_FILE = "max_xs.json"

##################################### Calculate count info of about each reactions #####################
IRRADIATION_DURATION = 3600 # spread the irradiation power over the entire course of this length
TRANSIT_DURATION = 86400*3
MEASUREMENT_DURATION = 3600*3
PRE_MEASUREMENT_POPULATION_FILE = "pre-measurement_population.json"
POST_MEASUREMENT_POPULATION_FILE = "post-measurement_population.json"
