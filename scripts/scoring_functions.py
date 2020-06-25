# scoring functions for evaluating similarity of subtyping a) across splits and b) between subtyping with splits and a single subtyping of the full sample
# Paul A. Bloom
# June 18, 2020

import sklearn
import pandas as pd
from sklearn.metrics import adjusted_rand_score
import numpy as np
from scipy.spatial import distance

def get_euclidean_distances(split_df):
    # group by split and matched cluster, get mean for each
    cluster_split_means = split_df.drop(columns = ['cluster']).groupby(['split', 'cluster_matched']).mean()
    
    # make columns for the grouping variables
    cluster_split_means['cluster'] = cluster_split_means.index.get_level_values('cluster_matched')
    cluster_split_means['split'] = cluster_split_means.index.get_level_values('split')
    
    # loop through cluster indices until we get to a cluster index not present in both splits
    clust_index = 1
    max_clust_lim = False
    euclidean_distances = []
    while not max_clust_lim:
        # filter by matched cluster index across splits (should always be 2 rows)
        matched_df = cluster_split_means[(cluster_split_means.cluster == clust_index)].drop(columns = ['cluster', 'split']).to_numpy()
        if matched_df.shape[0] == 2:
            # calculate euclidean distance between the 2 rows (i.e. same cluster, across splits)
            euclidean_distances.append(distance.euclidean(matched_df[0,], matched_df[1,]))
        else:
            max_clust_lim = True
        clust_index +=1
    
    # order and rank the euclidean distances and add to a dataframe
    df = pd.DataFrame({'euclidean_distance':np.sort(euclidean_distances),
                       'rank':np.arange(start = 1, stop = len(euclidean_distances) + 1)})
    return(df)


# get adjusted rand index for subtyping run on a split of the data, compared to those same participants if subtyping is run on the full sample
def rand_split_full(cluster_labels_bag, cluster_labels_full, split):
    merged_labels = cluster_labels_bag.merge(cluster_labels_full, on = 'URSI', how = 'left')
    merged_labels = merged_labels[merged_labels.split == split]
    rand = adjusted_rand_score(merged_labels.clust_bag, merged_labels.clust_no_bag)
    return([merged_labels, rand])

### test runs of each of the functions #####


# import data for testing out functions
louvain_bag_clust = pd.read_csv('../../clean_data/both_splits_bag_match_demo.csv')
full = pd.read_csv('../output/fullSamp_WISC-WIAT_LouClusterID+SubjID.csv')


# test euclidean distance function
print(get_euclidean_distances(louvain_bag_clust))

# clean data to be input to rand function
bag_labels = louvain_bag_clust[['URSI', 'cluster', 'split']].rename(columns = {'cluster':'clust_bag'})
full_labels = full[['URSI', 'louvain_community']].rename(columns = {'louvain_community':'clust_no_bag'})

# test rand function
print(rand_split_full(cluster_labels_bag = bag_labels, cluster_labels_full = full_labels, split = 'split1')[1])
print(rand_split_full(cluster_labels_bag = bag_labels, cluster_labels_full = full_labels, split = 'split2')[1])

