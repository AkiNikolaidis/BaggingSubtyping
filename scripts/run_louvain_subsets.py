import pandas as pd
from louvain_functions import *

def run_louvain_subsets(input_data, out_dir, n_subsets, split_pct, n_straps):
    """For n_subsets different subsets of the full datasets, run louvain subtyping both with bootstrapping and without
        Returns a directory of csv files with subtype labels for each participant, both bootstrapped and not

        Keyword arguments:
        input_data -- string specifying path to a .csv file with the raw data
        out_dir -- name for the output directory
        n_subsets -- integer specifying # of bootstrap iterations to run for bootstrapping
        split_pct -- integer from 1-100 specifying what proportion of the data is included
        n_straps -- number of bootstrap iterations to run for bootstrapped subtyping
    """
    # read in data from csv
    df = pd.read_csv(input_data)

    # make directory structure
    os.system(f'mkdir {out_dir}')
    os.system(f'mkdir {out_dir}/{split_pct}_pct')

    # split the data repeatedly in a loop (random state follows iterator variable)
    for i in range(n_subsets):
        # split the data
        split = df.sample(frac=split_pct/100, replace=False, random_state=i)

        # make a sub-directory for that split and save the data there
        os.system(f'mkdir {out_dir}/{split_pct}_pct/split_{i}')
        split_data_path = f'{out_dir}/{split_pct}_pct/split_{i}/split_sample.csv'
        split.to_csv(split_data_path, index = False)

        # run bootstrapped louvain for that split
        run_louvain(data_path = split_data_path, n_straps = n_straps, split_id = i, subset_proportion = split_pct, out_dir = out_dir)

        # run non-bootstrapped louvain for that split
        run_louvain(data_path = split_data_path, n_straps = 'none', split_id = i, subset_proportion = split_pct, out_dir = out_dir)
