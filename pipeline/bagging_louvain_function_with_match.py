import pandas as pd 
import numpy as np
from scipy import stats
import nilearn
import nibabel as nib
from nilearn.input_data import NiftiLabelsMasker
import os
import warnings
from PyBASC import utils as ut
warnings.filterwarnings('ignore')
import seaborn as sns
from igraph import *



def bagging_adjacency_matrixes(csv_folder, split, data_frame_path, out_dir):
    
    '''csv_folder: folder that contains all the csv output from python louvain.
       split: string in the csv files' names that specifies the split eg.Split_1 
       data_frame_path: name of path to orginial dataframe that includes subject ID and variables used for louvain 
       out_dir: name of path where final dataframe with subtypes will be saved  
    '''
    
    my_data = [pd.read_csv(csv_folder+file,index_col=0) for file in os.listdir(csv_folder) if split in file] # list contains all the csv in dataframe format
    
    for i in range(len(my_data)): # loop over my_data
        boot=my_data[i].drop_duplicates() #remove duplicates
        a =  ut.adjacency_matrix(boot.cluster.values[:, np.newaxis]).astype('int') # create adjacency matrix
        this_adj=pd.DataFrame(a.todense(),columns=boot.Key, index=boot.Key).sort_index().sort_index(axis=1)# add columns and index name to adj
        np.fill_diagonal(this_adj.values, 0)
        
        if i ==0:
            adj_full=this_adj #if this is the first adjacency matrix set it as the initial adj_full 
        else:
            adj_full=adj_full.add(this_adj,fill_value=0) # add this adjacency matrix to the full adjacency matrix


        boot_mask=my_data[i].drop_duplicates()
        boot_mask.cluster=boot_mask.cluster.replace(boot_mask.cluster.unique(),np.ones(boot_mask.cluster.unique().shape))#replace all clusters with 1
        m =  ut.adjacency_matrix(boot_mask.cluster.values[:, np.newaxis]).astype('int') # create adjacency matrix
        mask_adj=pd.DataFrame(m.todense(),columns=boot.Key, index=boot.Key).sort_index().sort_index(axis=1)# add columns and index name to adj
        np.fill_diagonal(mask_adj.values, 0)
        if i ==0:
            mask_full= mask_adj  #if this is the first adjacency matrix set it as the initial mask_full 
        else:
            mask_full=mask_full.add( mask_adj,fill_value=0) # add this mask adjacency matrix to the full mask adjacency matrix
    
    
        stab_full = adj_full.div(mask_full)
        np.fill_diagonal(stab_full.values, 0)

    columnsNamesArr = pd.DataFrame(stab_full.columns.values) #extract the Key (ID) from the stability matrix 
    graph = Graph.Weighted_Adjacency(stab_full.values.tolist(), mode=ADJ_UNDIRECTED, attr="weight") #turn stability matrix into a weighted graph 
    Louvain = graph.community_multilevel(weights=graph.es['weight']) #apply louvain community detection 
    Q = graph.modularity(Louvain, weights=graph.es['weight']) #compute modularity 
    print(Q)
    # Create dataframe of Subtypes for Split 1 
    Subtypes = pd.DataFrame(Louvain.membership) #obtain subtype membership 
    Subtypes = Subtypes + 1
    
    # create dataframe of Split 1 Subtypes 
    subs = pd.concat([columnsNamesArr.reset_index(drop=True), Subtypes], axis=1) #concatenate subtype assignments and Key 
    subs.columns = ['Key', 'Subtype']
    
    # read in Split 1 dataframe 
    df = pd.read_csv(data_frame_path) #read in orginial dataframe that includes subject ID ans variables used for louvain 
    df = df.rename(columns={'Unnamed: 0': 'Key'})
    
    Split = pd.merge(subs, df, on='Key') #add subtype assignments to dataframe 

    Split.to_csv(out_dir) #save final dataframe to desired directory

    return adj_full, mask_full, stab_full, Split, Q



def match_corrlation(Split1,Split2,variables):
    '''
    Split1: dataframe/output from bagging_adjacency_matrixes
    Split2: dataframe/output from bagging_adjacency_matrixes
    variables: a list of variables(strings) to caculate the subtype correlation on 
    '''

    
    split1_mean=Split1.groupby('Subtype').mean()[variables]#take the mean of each variable
    split2_mean=Split2.groupby('Subtype').mean()[variables]
    
    full_splits=split1_mean.append(split2_mean) #row bind
    full_splits.index=np.array(['Split1.1','Split1.2','Split1.3','Split1.4',
                           'Split2.1','Split2.2','Split2.3','Split2.4']) #rename index
    
    full_splits=full_splits.T # transpose
    corr = full_splits.corr(method='pearson').replace(1, -np.inf )#caculate pearson correlation
    matchedcors = corr.max().values[4:]
    print(matchedcors)
    return matchedcors





#Example:

csv_folder = 'C:/Users/jacob.derosa/Desktop/Scripts/Baggin_Subtyping/CBCL/home/jderosa/Documents/Baggin_Subtyping/Output/data/csv/'
split = 'Cluster_1' 
out_dir = 'C:/Users/jacob.derosa/Documents/Split_1.csv'
data_frame_path = 'C:/Users/jacob.derosa/Desktop/Scripts/Baggin_Subtyping/CBCL/Splits/CBCL_Split_1.csv' # Change Path 
adj_full1, mask_full1, stab_full1, Split1, Q1 = bagging_adjacency_matrixes(csv_folder, split, data_frame_path, out_dir)

split = 'Cluster_2' 
out_dir = 'C:/Users/jacob.derosa/Documents/Split_2.csv'
data_frame_path = 'C:/Users/jacob.derosa/Desktop/Scripts/Baggin_Subtyping/CBCL/Splits/CBCL_Split_1.csv' # Change Path 
adj_full2, mask_full2, stab_full2, Split2, Q2 = bagging_adjacency_matrixes(csv_folder, split, data_frame_path, out_dir)

variables=['CBCL_AD_T','CBCL_WD_T','CBCL_SC_T','CBCL_SP_T','CBCL_TP_T','CBCL_RBB_T','CBCL_AP_T','CBCL_AB_T']
 matchedcors = match_corrlation(Split1,Split2,variables)