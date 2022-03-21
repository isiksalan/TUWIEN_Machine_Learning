import numpy as np
import pandas as pd

def under_sample (pd_data, b = 100):
    value_keys = pd_data[pd_data.columns[0]].value_counts().to_dict()
    pd_new = pd.DataFrame()
    
    for key in value_keys.keys():
        loc = pd_data.loc[pd_data['class'] == key]

        if value_keys[key] > b:
            loc = loc.sample(n = b, replace = False)

        if len(pd_new) == 0:
            pd_new = loc
        else:
            pd_new = pd.concat((pd_new,loc),axis=0)
    
    pd_new = pd_new.sample(frac=1).reset_index(drop=True)

    
    pd_X = pd_new[pd_new.columns[1:]]
    pd_Y = pd_new[pd_new.columns[0]]
        
    return pd_X,pd_Y, pd_new


def over_sample(pd_data, b = 100):
    value_keys = pd_data[pd_data.columns[0]].value_counts().to_dict()
    pd_new = pd.DataFrame()
    
    for key in value_keys.keys():
        loc = pd_data.loc[pd_data['class'] == key]
        
        if value_keys[key] < b:
            
            app = loc.sample(n = (b - value_keys[key]), replace = False)
            
            loc = pd.concat((app,loc),axis=0)
            
        if len(pd_new) == 0:
            pd_new = loc
        else:
            pd_new = pd.concat((pd_new,loc),axis=0)
    
    pd_new = pd_new.sample(frac=1).reset_index(drop=True)

    
    pd_X = pd_new[pd_new.columns[1:]]
    pd_Y = pd_new[pd_new.columns[0]]
        
    return pd_X,pd_Y, pd_new

def min_max_scaling(pd_data, max_vals = [], min_vals = []):
    pd_data_modified = pd_data.copy()
    
    if len(max_vals) == 0:
        max_vals = pd_data.max()
        min_vals = pd_data.min()
    
    pd_data_modified = (pd_data_modified - min_vals)/(max_vals-min_vals)
    
    
    return max_vals, min_vals, pd_data_modified


