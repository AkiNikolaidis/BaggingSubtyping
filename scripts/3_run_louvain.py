# Runs louvain subtyping

# Input options: csv of either raw data, or stability matrix (need to add this stability matrix functionality)
# Bootstrapping options: 'none' (no bootstrapping), or an integer n_boots
    # If bootstrapping is chosen, will return n_boots csvs

# Paul Bloom
# June 20, 2020

import sys
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from time import time
import os
from louvain_functions import *

def run_louvain(data_path, n_straps, out_dir):
    # make output directory
    os.system(f'mkdir {out_dir}')

    # load input dataframe
    df = pd.read_csv(data_path)
    df = df.rename(columns={'URSI': 'Key'})

    # variable to cluster on
    subset = df.columns[df.columns != 'Key']

    #put subject ID into a list that will be matched with bootsrapped indicices. This important because this step will be used later on to match ID to cluster assignment.
    y = np.array(df['Key'])

    # non-boostrap version (just run pheno_clust once)
    if n_straps == 'none':
        # scale the data within each bootstrap
        X_data =np.array(df[subset]).astype(np.float64)
        X_data_scaled = sklearn.preprocessing.scale(X_data)

        # run pheno_clsut function on scaled data
        communities, Q = pheno_clust(X=X_data_scaled, plot=False, verbose=False)

        # put together subids, cluster assignments, and modularity values into dataframe together
        out_df = pd.DataFrame({'URSI':y,'cluster':communities, 'Q':Q})
        out_df.to_csv(f'{out_dir}/louvain_clusters.csv', index = False)

    # bootstrap version
    else:
        n_straps = int(n_straps)
        n = df.shape[0]
        b_idx = np.zeros((n_straps, n))

        # bootstrapping
        for i in range(n_straps):
            # fix random state loop so that results can be reproduced across runs
            random_state = np.random.RandomState(seed = i)
            b_idx[i] = random_state.randint(0, high=n - 1, size=n)

        b_idx = b_idx.astype(np.int)
        y_boot = np.zeros(b_idx.shape, dtype='object')
        for i in range(b_idx.shape[0]):
            y_boot[i] = y[b_idx[i]]

        bootstrap_split_subids = []
        bootstrap_split_communities = []
        bootstrap_split_Q = []

        for i in range(n_straps):
            print(f'Starting bootstrap iteration {i+1}/{n_straps}')
            X_split = df.iloc[b_idx[i],:]

            bootstrap_split_subids.append([y_boot[i]])

            # scale the data within each bootstrap
            X_data =np.array(X_split[subset]).astype(np.float64)
            X_data_scaled = sklearn.preprocessing.scale(X_data)

            # run pheno_clsut function on scaled data
            communities, Q = pheno_clust(X=X_data_scaled, plot=False, verbose=False)

            bootstrap_split_communities.append([communities])
            bootstrap_split_Q.append([Q])

        allcsvs = {}
        for i in range(n_straps):
            out_df = {'URSI':bootstrap_split_subids[i][0],'cluster':bootstrap_split_communities[i][0], 'Q':bootstrap_split_Q[i][0]}
            allcsvs[i] = pd.DataFrame(out_df)
            allcsvs[i].to_csv(f'{out_dir}/louvain_boot_cluster_{i}.csv', index = False)
