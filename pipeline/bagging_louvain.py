import pandas as pd
import numpy as np
from scipy import stats
import nilearn
import nibabel as nib
from nilearn.input_data import NiftiLabelsMasker
import os
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from igraph import *


def bagging_adjacency_matrixes(csv_folder, split):
    '''csv_folder: folder that contains all the csv output from python louvain.
       split: string in the csv files' names that specifies the split eg.Split_1 '''

    my_data = [pd.DataFrame(np.load(csv_folder+file, allow_pickle=True)) for file in os.listdir(csv_folder) if split in file] # list contains all the csv in dataframe format


    for i in range(len(my_data)): # loop over my_data
        my_data[i].drop(columns = [2], inplace = True)
        my_data[i].rename({0:'Key', 1:'cluster'}, inplace = True, axis = 1)
        boot=my_data[i].drop_duplicates() #remove duplicates
        df_merge = boot.merge(boot, on='cluster') # create adjacency matrix step1
        this_adj =pd.crosstab(df_merge.Key_x, df_merge.Key_y) # create adjacency matrix step2
        np.fill_diagonal(this_adj.values, 0) #set the diagonal of the adjacency matrix to 0

        if i ==0:
            adj_full=this_adj #if this is the first adjacency matrix set it as the initial adj_full
        else:
            adj_full=adj_full.add(this_adj,fill_value=0) # add this adjacency matrix to the full adjacency matrix


        boot_mask=my_data[i].drop_duplicates()
        boot_mask.cluster=boot_mask.cluster.replace(boot_mask.cluster.unique(),np.ones(boot_mask.cluster.unique().shape))#replace all clusters with 1
        df_merge2 = boot_mask.merge(boot_mask, on='cluster')
        mask_adj =pd.crosstab(df_merge2.Key_x, df_merge2.Key_y)
        np.fill_diagonal(mask_adj.values, 0) #set the diagonal of the adjacency matrix to 0
        if i ==0:
            mask_full= mask_adj  #if this is the first adjacency matrix set it as the initial mask_full
        else:
            mask_full=mask_full.add( mask_adj,fill_value=0) # add this mask adjacency matrix to the full mask adjacency matrix


    stab_full = adj_full.div(mask_full)
    np.fill_diagonal(stab_full.values, 0)

    return adj_full, mask_full, stab_full


csv_folder= '../temp_output/50_pct/split_0/boot/'

# Apply Baggin on Split 1
split = 'louvain'
adj_full, mask_full, stab_full = bagging_adjacency_matrixes(csv_folder, split)

# # Apply Bagging on Split 2
# split = 'Cluster_2'
# adj_full_2, mask_full_2, stab_full_2 = bagging_adjacency_matrixes(csv_folder, split)


#save to csv, can add this step in the function if needed
adj_full.to_csv('adj_full.csv')
mask_full.to_csv('mask_full.csv')
stab_full.to_csv('stab_full.csv')

#save to csv, can add this step in the function if needed
# adj_full_2.to_csv('adj_full_2.csv')
# mask_full_2.to_csv('mask_full_2.csv')
# stab_full_2.to_csv('stab_full_2.csv')


# Split 1 Summed Adj. Matrixes
#show as image
plt.imshow(adj_full)
plt.colorbar()

# Split 2 Summed Adj. Matrixes
#show as image
# plt.imshow(adj_full_2)
# plt.colorbar()

# Split 1 Mask Matrix
plt.imshow(mask_full)
plt.colorbar()

# Split 2 Mask Marix
# plt.imshow(mask_full_2)
# plt.colorbar()

# Split 1 Stability Matrix
plt.imshow(stab_full)
plt.colorbar()

# Split 2 Stability Matrix
# plt.imshow(stab_full_2)
# plt.colorbar()

#Create df Id from matrix column names
columnsNamesArr = pd.DataFrame(stab_full.columns.values)

#Create df Id from matrix column names
# columnsNamesArr_2 = pd.DataFrame(stab_full_2.columns.values)

# Final Louvain Clustering Split 1
graph = Graph.Weighted_Adjacency(stab_full.values.tolist(), mode=ADJ_UNDIRECTED, attr="weight")
Louvain = graph.community_multilevel(weights=graph.es['weight'])

#Louvain.membership
Q = graph.modularity(Louvain, weights=graph.es['weight'])
print(Q)

# Create dataframe of Subtypes for Split 1
Subtypes = pd.DataFrame(Louvain.membership)
Subtypes = Subtypes + 1

#
# # Final Louvain Clustering Split 2
# graph_2 = Graph.Weighted_Adjacency(stab_full_2.values.tolist(), mode=ADJ_UNDIRECTED, attr="weight")
# Louvain_2 = graph_2.community_multilevel(weights=graph_2.es['weight'])
# #Louvain.membership
# Q_2 = graph_2.modularity(Louvain_2, weights=graph_2.es['weight'])
# print(Q_2)
#
# # Create dataframe of Subtypes for Split 2
# Subtypes_2 = pd.DataFrame(Louvain_2.membership)
# Subtypes_2 = Subtypes_2 + 1

# create dataframe of Split 1 Subtypes
subs = pd.concat([columnsNamesArr.reset_index(drop=True), Subtypes], axis=1)
subs.columns = ['Key', 'Subtype']


# create dtaframe of Split 2 Subtypes
# subs_2 = pd.concat([columnsNamesArr_2.reset_index(drop=True), Subtypes_2], axis=1)
# subs_2.columns = ['Key', 'Subtype']

# read in Split 1 dataframe
df = pd.read_csv('../temp_output/50_pct/split_0/split_sample.csv') # Change Path
df = df.rename(columns={'URSI': 'Key'})

# read in Split 2 dataframe
# df_2 = pd.read_csv('C:/Users/jacob.derosa/Desktop/Scripts/Baggin_Subtyping/CBCL/Splits/CBCL_Split_2.csv') # Change Path
# df_2 = df_2.rename(columns={'Unnamed: 0': 'Key'})

# create Split 1
Split_1 = pd.merge(subs, df, on='Key')

# create Split 2
#Split_2 = pd.merge(subs_2, df_2, on='Key')

# write created dataframes to csv
Split_1.to_csv('../temp_output/50_pct/split_0/split_sample_merged.csv') # Change Path
#Split_2.to_csv('/Users/jacob.derosa/Desktop/Scripts/Baggin_Subtyping/Python_R_Test/CBCL_Split_2/Python_CBCL_Split_2.csv') # Change Path
