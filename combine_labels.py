"""
combine labels from bmrc and japaleno
"""

import os
import numpy as np
import pandas as pd

def full_labels(csv_name, save=True):
    """getting full set of labels for features"""
    # load data
    base_dir = '/well/seymour/users/uhu195/python/extract_npy/'
    print(f'{base_dir}/labels/label_{csv_name}.csv')
    labels = pd.read_csv(f'{base_dir}/labels/label_{csv_name}.csv')
    sj_bmrc = pd.read_csv(f'{base_dir}/bmrc_subjs/subjs_{csv_name}.csv', header=None)
    sj_full = pd.read_csv(f'{base_dir}/bmrc_full/subjs_{csv_name}.csv')
    # rename columns
    sj_bmrc.rename(columns={0:'bmrc'}, inplace=True)
    sj_full.rename(columns={'eid':'bmrc_eid'}, inplace=True)
    print(sj_full.shape)
    # merge
    bmrc_full = sj_bmrc.merge(sj_full, left_on='bmrc', right_on='bmrc_eid', how='left',indicator=True)
    # concat with labels
    full_df = pd.concat([labels, bmrc_full], axis=1)
    df_out = full_df[full_df['_merge']=='both']
    df_out_clean = df_out.drop(columns=['bmrc_eid', '_merge']).reset_index(drop=True)
    print(df_out_clean.shape)
    # save
    if save:
        save_path = f'{base_dir}/labels_full/label_{csv_name}.csv'
        df_out_clean.to_csv(save_path, index=None)
    return df_out_clean


# run
if __name__=="__main__":
    # create list of full labels
    df_out = full_labels('painquestion', save=True)