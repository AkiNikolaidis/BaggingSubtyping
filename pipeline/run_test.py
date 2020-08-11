from louvain_functions import *
import os


for pct in [20, 50, 90]:
    os.system(f'python run_louvain_subsets.py ../data/split1.csv {pct} 50 50 4 False temp_output_1')


# Full dataset
os.system('python run_louvain_subsets.py ../data/split1.csv 100 1 1 4 False temp_output_1')

for i in range(50):
    for j in [20, 50, 90]:
        bagging_adjacency_matrixes(csv_folder = f'temp_output_1/{j}_pct/split_{i}/boot/',
                                   data_frame_path =f'temp_output_1/{j}_pct/split_{i}/split_sample.csv',
                                   out_dir = f'temp_output_1/{j}_pct/split_{i}/split_sample_louvain.csv')
