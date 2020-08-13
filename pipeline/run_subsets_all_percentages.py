from louvain_functions import *
import os


def run_subsets_all_percentages(input_data_path, n_subsets, n_boots, mri, n_cpus, output_dir)
	# run boostrapping on each subset for each percentage
	for pct in [10,20,30,40,50,60,70,80,90]:
		os.system(f'python run_louvain_subsets.py ../data/split1.csv {pct} {n_subsets} {n_boots} {n_cpus} False {output_dir}')

	# Full dataset
	os.system(f'python run_louvain_subsets.py ../data/split1.csv 100 1 1 1 False {output_dir}')

	# Make the full stability matrix and generate final subtypes for the bagged pipelines for each subset for each subset percentage
	for i in range(50):
	    for j in [10,20,30,40,50,60,70,80,90]:
	        bagging_adjacency_matrixes(csv_folder = f'{output_dir}/{j}_pct/split_{i}/boot/',
	                                   data_frame_path =f'{output_dir}/{j}_pct/split_{i}/split_sample.csv',
	                                   out_dir = f'{output_dir}/{j}_pct/split_{i}/split_sample_louvain.csv')