"""frequently referenced table for interpreting openmc data"""
from openmc.data.reaction import REACTION_NAME
from openmc.data.endf import SUM_RULES
from openmc.data import AVOGADRO # scalar constant
from openmc.data import NATURAL_ABUNDANCE, ATOMIC_NUMBER, ATOMIC_SYMBOL # atomic data look up tables
from openmc.data import atomic_mass, isotopes # atomic data lookup
from openmc.data import INTERPOLATION_SCHEME

MT_to_nuc_num = {
    2:(0, 0),
    4:(0, 0),
    11:(-1, -3),
    16:(0, -1),
    17:(0, -2),
    22:(-2, -4),
    23:(-6, -12),
    24:(-2, -5),
    25:(-2, -6),
    28:(-1, -1),
    29:(-4, -8),
    30:(-4, -9),
    32:(-1, -2),
    33:(-1, -3),
    34:(-2, -3),
    35:(-5, -10),
    36:(-5, -11),
    37:(0, -3),
    41:(-1, -2),
    42:(-1, -3),
    44:(-2, -2),
    45:(-3, -5),
    102:(0, 1),
    103:(-1, 0),
    104:(-1, -1),
    105:(-1, -2),
    106:(-2, -2),
    107:(-2, -3),
    108:(-4, -7),
    109:(-6, -11),
    111:(-2, -1),
    112:(-3, -4),
    113:(-5, -10),
    114:(-5, -9),
    115:(-2, -2),
    116:(-2, -3),
    117:(-3, -5),
    152:( 0, -4),
    153:( 0, -5),
    154:(-1, -4),
    155:(-3, -6),
    156:(-1, -4),
    157:(-1, -4),
    158:(-3, -6),
    159:(-3, -6),
    160:( 0, -6),
    161:( 0, -7),
    162:(-1, -5),
    163:(-1, -6),
    164:(-1, -7),
    165:(-2, -7),
    166:(-2, -8),
    167:(-2, -9),
    168:( -2, -10),
    169:(-1, -5),
    170:(-1, -6),
    171:(-1, -7),
    172:(-1, -5),
    173:(-1, -6),
    174:(-1, -7),
    175:(-1, -8),
    176:(-2, -4),
    177:(-2, -5),
    178:(-2, -6),
    179:(-2, -4),
    180:(-4, -10),
    181:(-3, -7),
    182:(-2, -4),
    183:(-2, -3),
    184:(-2, -4),
    185:(-2, -5),
    186:(-3, -4),
    187:(-3, -5),
    188:(-3, -6),
    189:(-3, -7),
    190:(-2, -3),
    191:(-3, -3),
    192:(-3, -4),
    193:(-4, -6),
    194:(-2, -5),
    195:( -4, -11),
    196:(-3, -8),
    197:(-3, -2),
    198:(-3, -3),
    199:(-4, -8),
    200:(-2, -6),
    800:(-2, -3),
    801:(-2, -3)
}
for i in range(51, 92): # (z,n?)
    MT_to_nuc_num[i] = tuple([*MT_to_nuc_num[2], i-50])
for i in range(600, 649): # (z, p?)
    MT_to_nuc_num[i] = tuple([*MT_to_nuc_num[103], i-600])
MT_to_nuc_num[649] = MT_to_nuc_num[103]
for i in range(650, 699): # (z, d?)
    MT_to_nuc_num[i] = tuple([*MT_to_nuc_num[104], i-650])
MT_to_nuc_num[699] = MT_to_nuc_num[104]
for i in range(700, 749): # (z, t?)
    MT_to_nuc_num[i] = tuple([*MT_to_nuc_num[105], i-700])
MT_to_nuc_num[749] = MT_to_nuc_num[105]
for i in range(750, 799): # (z, He3?)
    MT_to_nuc_num[i] = tuple([*MT_to_nuc_num[106], i-750])
MT_to_nuc_num[799] = MT_to_nuc_num[106]
for i in range(800, 849):
    MT_to_nuc_num[i] = tuple([*MT_to_nuc_num[107], i-800])
MT_to_nuc_num[849] = MT_to_nuc_num[107]
for i in range(875, 891):
    MT_to_nuc_num[i] = tuple([*MT_to_nuc_num[16], i-875])
MT_to_nuc_num[891] = MT_to_nuc_num[16]

FISSION_MTS = (18, 19, 20, 21, 22, 38)
AMBIGUOUS_MT = (1, 3, 5, 18, 27, 101, 201, 202, 203, 204, 205, 206, 207, 649)

"""package all of the nuclear-data related variables into a single dictionary,
so that when testing, it becomes easier to import all of them at once from openmcextension.constants"""
# all variables in this file are listed below:
(REACTION_NAME,
SUM_RULES,
NATURAL_ABUNDANCE,
atomic_mass,
ATOMIC_NUMBER,
INTERPOLATION_SCHEME,
FISSION_MTS,
AMBIGUOUS_MT, 
MT_to_nuc_num)