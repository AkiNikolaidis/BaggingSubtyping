import pandas as pd
from louvain_functions import *
import multiprocessing
import sys
import scipy

"""6 command line arguments:

Command line args (order matters)
1 -- path to raw data csv
2 -- subset percentage (integer from 1-100)
3 -- number of subsets to do (integer)
4 -- number of bootstraps per subset (integer)
5 -- maximum processes (for parallel processing)
6 -- mri (True if mri data, false otherwise)
7 -- output directory
"""
data_path = sys.argv[1]
split_pct = int(sys.argv[2])
n_subsets = int(sys.argv[3])
n_straps = int(sys.argv[4])
max_processes = int(sys.argv[5])
mri = bool(sys.argv[6])
out_dir = sys.argv[7]

df = pd.read_csv(data_path)


# make directory structure if it doesn't exist already
if not os.path.isdir(f'{out_dir}/{split_pct}_pct'):
    if not os.path.isdir(out_dir):
        os.system(f'mkdir {out_dir}')
    os.system(f'mkdir {out_dir}/{split_pct}_pct')

# this function takes only one argument: the subset
# the function is passed into multiprocessing pool
def run_louvain_one_subset(i):
    print(f'starting subset {i}')
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

def run_louvain_one_subset_fmri(i):
    print(f'starting subset {i}')
    # split the data
    split = df.sample(frac=split_pct/100, replace=False, random_state=i)

    # taking only the top 5% of features (5% most connected edges)
    mean = split.mean().values
    thr=mean.shape[0]*0.95 # 
    rank=scipy.stats.rankdata(mean,method='min')
    ind = np.argwhere(rank>=thr)
    top5_col = split.columns[ind]
    split = split[top5_col.flatten()].dropna()

    # make a sub-directory for that split and save the data there
    os.system(f'mkdir {out_dir}/{split_pct}_pct/split_{i}')
    split_data_path = f'{out_dir}/{split_pct}_pct/split_{i}/split_sample.csv'
    split.to_csv(split_data_path, index = False)

    # run bootstrapped louvain for that split
    run_louvain(data_path = split_data_path, n_straps = n_straps, split_id = i, subset_proportion = split_pct, out_dir = out_dir)

    # run non-bootstrapped louvain for that split
    run_louvain(data_path = split_data_path, n_straps = 'none', split_id = i, subset_proportion = split_pct, out_dir = out_dir)    

if mri:
    if __name__ == '__main__':
        pool = multiprocessing.Pool(max_processes)
        results = pool.map_async(run_louvain_one_subset_fmri, range(n_subsets))
        pool.close()
        pool.join()
else:
    if __name__ == '__main__':
        pool = multiprocessing.Pool(max_processes)
        results = pool.map_async(run_louvain_one_subset, range(n_subsets))
        pool.close()
        pool.join()  