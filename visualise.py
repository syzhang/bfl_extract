"""
visualisation
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_cv(cv_dir):
    """load cv results"""
    cv_ls = []
    for f in os.listdir(cv_dir):
        if f.endswith('.csv'):	
            tmp = pd.read_csv(os.path.join(cv_dir, f))
            cv_ls.append(tmp)
    df = pd.concat(cv_ls)
    print(df.head())
    return df
    
def plot_cv(df):
    """plot cv results"""
    df.plot(kind)
        
# run
if __name__=="__main__":
    curr_dir = '/well/seymour/users/uhu195/python/extract_npy'
    cv_dir = os.path.join(curr_dir, 'cv_results')
    
    load_cv(cv_dir)
