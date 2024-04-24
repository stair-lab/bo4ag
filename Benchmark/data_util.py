import torch
import pandas as pd
from scipy import stats

def stadardize(data):
    mean = data.mean()
    std = data.std()    
    standardized_data = (data - mean) / std
    return standardized_data

def getLookup(trait, transform=None):
    path = f"./data/{trait}_coh2.csv"
    lookup = pd.read_csv(path, header=0)
    # Replace nans with zero
    lookup_tensor = torch.tensor(lookup.values, dtype=torch.float64)
    no_nan_lookup = torch.nan_to_num(lookup_tensor)
    no_nan_lookup[no_nan_lookup > 1] = 0
    
    # Make zeros, non-zero
    mask = no_nan_lookup == 0
    no_nan_lookup[mask] = 10e-6
    
    # Apply input transforms
    if transform == "log":
        log_data = torch.log(no_nan_lookup)
        return stadardize(log_data) #standardize
    elif transform == "box_cox":
        flat_lookup = no_nan_lookup.reshape(-1)
        fitted_data, fitted_lambda = stats.boxcox(flat_lookup)
        unflat_lookup = torch.from_numpy(fitted_data).reshape(2151, 2151)
        return stadardize(unflat_lookup) #standardize
    elif transform == "standardized":
        return stadardize(no_nan_lookup) #standardize
    elif transform is not None:
        print(f"{transform} is not a valid input tranform...")
    return no_nan_lookup