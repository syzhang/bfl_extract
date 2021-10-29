"""
matching subjects using bridge file
"""

import os
import pandas as pd

def convert_eid(df_bridge, in_csv, save=False):
    """convert eid to bmrc ones"""
    df_subjs = pd.read_csv('./subjs/'+in_csv, header=None)
    df_slice = df_subjs.merge(df_bridge, left_on=0, right_on='eid_45465')
    # df_slice = df_bridge[df_subjs[0].isin(df_bridge['eid_45465'])]
    df_bmrs = df_slice['eid_8107']
    if save:
        df_bmrs.to_csv('./bmrc_subjs/'+in_csv, index=None, header=None)
    return df_bmrs

def revert_eid(df_bridge, in_csv, save=False):
    """revert bmrc to project eid"""
    df_subjs = pd.read_csv('./bmrc_full/'+in_csv, header=None)
    df_slice = df_subjs.merge(df_bridge, left_on=0, right_on='eid_45465')
    # df_slice = df_bridge[df_subjs[0].isin(df_bridge['eid_45465'])]
    df_bmrs = df_slice['eid_8107']
    if save:
        df_bmrs.to_csv('./bmrc_subjs/'+in_csv, index=None, header=None)
    return df_bmrs

# run
if __name__=="__main__":
    df_bridge = pd.read_csv('../bridge_file/bridge_8107_45465.csv')
    
    # convert
    for f in os.listdir('./subjs'):
        if f.endswith('.csv'):
            print(f)
            _ = convert_eid(df_bridge, f, save=True)