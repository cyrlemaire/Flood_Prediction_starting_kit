import pandas as pd
import numpy as np

def parse_dates(band_str:str)->tuple:
    """handle the case where the band is a datetime64 or a string (e.g. "20030101-20030108") 
    conversion to a datetime64 (some xarry have datetime64 and some have string)

    Args:
        band_str (str): string of the form "20030101-20030108"

    Returns:
        tuple: start_date, end_date
    """
    if isinstance(band_str, np.datetime64):
        start_date = band_str
        end_date = None  
    else:
        start_str, end_str = band_str.split('-')
        start_date = pd.to_datetime(start_str)
        end_date = pd.to_datetime(end_str)
    return start_date, end_date