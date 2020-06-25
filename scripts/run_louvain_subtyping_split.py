# Runs louvain subtyping
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
from run_louvain import *
import multiprocessing as mp


# directory to where split1.csv and split2.csv files live
data_path = sys.argv[1]

# bootstrap or not
n_straps = sys.argv[2]

# output directory
out_dir = sys.argv[3]

# paths to csv files with split data in them
paths = [data_path + 'split1.csv', data_path + 'split2.csv']


# run with multiprocessing
os.system(f'mkdir {out_dir}')
if __name__ == '__main__':
    processes = []
    for i in [0,1]:
        #os.system(f'mkdir {out_dir}/split{i+1}')
        p = mp.Process(target=run_louvain, args=(paths[i], n_straps, f'{out_dir}/split{i+1}'))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()

    print()
