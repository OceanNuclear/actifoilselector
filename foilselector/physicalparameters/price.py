import math
import pandas as pd
from .filepaths import PRICE_FILE as default_price_file

def get_price_df(price_df_file=default_price_file):
    return pd.read_csv(price_df_file, index_col=[0])

def get_price_from_price_df(price_df, material, desired_thickness):
    series = price_df.loc[material][1:]
    allowed_thick, allowed_price = [], []
    for thick, price in zip(series[::2], series[1::2]):
        # for all non-empty records
        if not math.isnan(thick):
            allowed_thick.append(thick)
            allowed_price.append(price)                            
    if allowed_thick: # if there are *ANY foils with recorded prices at all:
        ind = abs(ary(allowed_thick) - desired_thickness).argmin() # find the closest matched thickness
        return allowed_price[ind]
    else: # if there are no thicknesses with recorded prices
        return math.nan