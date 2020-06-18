
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import train_test_split
from time import time

def match(C):
    """
    Replication of R's match function
    """
    
    u, ui = np.unique(C, return_index=True)
    ui = sorted(ui)
    newC = np.zeros(C.shape)
    for i in range(len(ui)):
        newC[np.where(C == C[ui[i]])[0]] = i + 1
        
    return newC


def outer_equal(x):
    """
    Replication of R's outer function with FUN param '=='
    """
    
    res = np.zeros((x.shape[0], x.shape[0]))
    for i in range(x.shape[0]):
        res[i,:] = x == x[i]
    
    return res


def hadamard(nearest):
    """
    Transform data into Hadamard distance matrix
    """
    
    common = nearest.T @ nearest
    ranks = np.outer(np.diag(common), np.ones(nearest.shape[0]))
    neighborUnion = ranks + ranks.T - common
    G = common / neighborUnion
    np.fill_diagonal(G, 0)
    
    return G


def topMax(x, N):
    """
    find Nth largest number in an array
    """
    
    L = len(x)
    assert N < L, 'Number of neighbors cannot be larger than length of data'
    
    while L != 1:
        initial_guess = x[0]
        top_list = x[x > initial_guess]
        bottom_list = x[x < initial_guess]
        
        topL = len(top_list)
        bottomL = len(bottom_list)
        
        if (topL < N) and (L - bottomL >= N):
            x = initial_guess
            break
        
        if topL >= N:
            x = top_list
        else:
            x = bottom_list
            N = N - L + bottomL
        L = len(x)
    
    return x


def bottomMin(x, N):
    """
    find Nth smallest number in an array
    """
    
    return np.round(topMax(x, len(x) - N + 1))


def find_neighbors(D, k):
    """
    Tranform distance matrix to binary k-nearest neighbors graph
    """
    
    nearest = np.zeros(D.shape)
    for i in range(nearest.shape[1]):
        nearest[:,i] = np.round(D[:,i]) <= bottomMin(np.round(D[:,i]), k)
    
    return nearest


def modularity(G, C):
    """
    Calculate graph's modularity
    """
    
    m = np.sum(G)
    Ki = np.repeat(np.sum(G, axis=1), G.shape[1]).reshape(G.shape)
    Kj = Ki.T
    delta_function = outer_equal(C)
    Q = (G - Ki * Kj / m) * delta_function / m
    return np.sum(Q)


def delta_modularity(G, C, i, m, Kj, newC=None):
    """
    Calculate change in modularity by adding a node to a cluster or by removing it
    """
    
    c = np.copy(C)
    
    if newC is None: # removing a node from C
        # removing a solitary node from a cluster does not change the modularity
        if np.sum(c == c[i]) == 1:
            return 0
        newC = c[i]
        c[i] = np.max(c) + 1
        
    I = c == newC
    Ki = np.sum(G[i,:]) # the sum of all the edges connected to node i
    Ki_in = np.sum(G[i,I]) # the sum of all the edges conneting node i to C
    Kjs = np.sum(Kj[I])
    deltaQ = Ki_in / m - Ki * Kjs / m**2
    
    return deltaQ * 2


def louvain_step(G, C, O, Q=None):
    """
    Run a single step of the Louvain algorithm
    """
    
    if not Q:
        Q = modularity(G, C)
        
    m = np.sum(G)
    Kj = np.sum(G, axis=1)
    
    for i in O:
        reassign = np.array(list(set(C[G[i,:] > 0]).difference(set([C[i]]))))
        
        if len(reassign) != 0:
            delta_remove = delta_modularity(G, C, i, m, Kj)

            deltaQ = np.array([delta_modularity(G, C, i, m, Kj, newC=x) for x in reassign])

            if np.max(deltaQ) - delta_remove > 0:
                idx = np.argmax(deltaQ)
                Q += np.max(deltaQ) - delta_remove
                if reassign[idx] > 2389:
                    print(i)
                C[i] = reassign[idx]
            if delta_remove < 0:
                C[i] = np.max(C) + 1
                Q -= delta_remove
    
    return C, Q


def louvain(G, C, maxreps=100):
    """
    Run Louvain algorithm
    """
    
    assert np.allclose(G, G.T), 'Input graph must be symmetric'
    
    order = np.random.permutation(G.shape[1])
    Q = modularity(G, C)
    
    for i in range(maxreps):
        C, Q = louvain_step(G, C, order, Q)
        C = match(C)
        
        # run the second phase, trying to combine each cluster
        n_clust = len(np.unique(C))
        metaG = np.zeros((n_clust, n_clust))
        for ci in range(1, n_clust + 1):
            G_tmp = G[C == ci]
            for cj in range(ci, n_clust + 1):
                G_tmp2 = G_tmp[:,C == cj]
                metaG[ci-1,cj-1] = np.sum(G_tmp2)
                metaG[cj-1,ci-1] = metaG[ci-1,cj-1]
                
        metaC, metaQ = louvain_step(metaG, np.arange(n_clust) + 1, np.random.permutation(n_clust))
        if metaQ - Q > 1e-15:
            tempC = np.copy(C)
            for ci in range(1, n_clust + 1):
                tempC[C == ci] = metaC[ci-1]
            C = match(tempC)
            Q = metaQ
        else:
            break
            
    return C, Q


def plot_community_profiles(X, communities, outfile=None):
    """
    Create radar plot of average profiles of each detected community
    """
    
    # number of variables
    categories = subset
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    fig=plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories)
    pi = np.pi
    for label,i in zip(ax.get_xticklabels(), range(0,len(angles))):
        angle_rad = angles[i]
        if angle_rad <= pi/2:
            ha = 'left'
            va = "bottom"
            angle_text = angle_rad * (-180/pi) + 90
        elif pi/2 < angle_rad <= pi:
            ha = 'left'
            va = "top"
            angle_text = angle_rad * (-180/pi) + 90
        elif pi < angle_rad <= (3*pi/2):
            ha = 'right'
            va = "top"  
            angle_text = angle_rad * (-180/pi) - 90
        else:
            ha = 'right'
            va = "bottom"
            angle_text = angle_rad * (-180/pi) - 90
        label.set_rotation(angle_text)
        label.set_verticalalignment(va)
        label.set_horizontalalignment(ha)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks(np.linspace(50,70,5), color="k", size=10)
    plt.ylim(50,70)

    # ------- PART 2: Add plots
    comms, counts = np.unique(communities, return_counts=True)
    for c, n in zip(comms, counts):
        comm_X = X[communities == c,:]
        comm_avg = np.mean(comm_X, axis=0)
        values = comm_avg.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1.5, linestyle='solid', label="Community %d (N = %d)" % (c, n))
#         ax.fill(angles, values, 'b', alpha=0.1)
    plt.legend(bbox_to_anchor=(1.1, 1.1), fontsize=10)
    if outfile:
        plt.savefig(outfile, bbox_inches='tight', dpi=300)
        
        
def pheno_clust(filepath=None, subset=None, X=None, plot=True, outfile=None, repeats=50, verbose=True):
    """
    Run entire phenoClust pipeline
    """
    
    assert filepath or X is not None, 'Either a matrix or a filepath to the data must be passed in'
    
    if verbose:
        tic = time()
    
    if filepath:
        assert subset, 'If filepath to dataframe is passed, you must also include the subset of columns that you intend to pass into the algorithm'
        
        df = pd.read_csv(filepath)
        df = df[subset]
        df.dropna(inplace=True)

        X = np.array(df)
        
    X_Z = (X - np.mean(X, axis=0, keepdims=True)) / np.std(X, axis=0, dtype=np.float64, ddof=1, keepdims=True)
    D, rho = spearmanr(X_Z, axis=1)
    D = np.round((X_Z.shape[1]**3 - X_Z.shape[1]) * (1 - D) / 6).astype(np.int)
    
    b = np.ceil(np.log2(X_Z.shape[0]) + 1)
    k = np.ceil(X_Z.shape[0] / b).astype(np.int)
    nearest = find_neighbors(D, k + 1)
    G = hadamard(nearest)
    C = np.arange(G.shape[1]) + 1
    
    Q = -np.inf
    for t in range(repeats):
        C, newQ = louvain(G, C)
        if newQ > Q:
            Q = newQ
            if verbose:
                print('iter: %d/%d; Q = %.5f; # of communities: %d' % (t+1, repeats, Q, len(np.unique(C))))
        else:
            break
    
    if verbose:
        toc = time()
        print('PyPhenoClust completed in: %.2f seconds' % (toc - tic))
    
    if plot:
        plot_community_profiles(X, C, outfile)
    
    return C, Q


#df = pd.read_csv("C:/Users/jacob.derosa/Downloads/CBCL.csv")
df = pd.read_csv('C:/Users/jacob.derosa/Desktop/Scripts/Full_CBCL_Splits/CBCL.csv')
df = df.rename(columns={'Unnamed: 0': 'Key'})
subset = ['CBCL_AD_T', 'CBCL_WD_T', 'CBCL_SC_T', 'CBCL_SP_T', 'CBCL_TP_T', 'CBCL_AP_T', 'CBCL_RBB_T', 'CBCL_AB_T']
y = np.array(df['Key'])

#Variables that will be split on. Note, the Key varialbe needs to be included because it will be used to match the split data back into the dataset with the CBCL scores 
#before the algorithm runs. 

Split = df[['Key','ADHD_I','ADHD_C','ODD', 'ADHD_H','ADHD','ASD','ANX', 'NT','DEP','Other','LD','Age','Sex']] 

#initialize empty lists that will store URSI (subject ID), Q (modularity = quantifies the quality of an assignment of nodes to communities), and cluster assignments for each subject
split_subids = []
split_communities = []
split_Q = []
split_subids2 = []
split_communities2 = []
split_Q2 = []

#_____________________________________________________________________________________________________________________________________________________________________

n_straps = 100 #number of iterations used

#split sample in half (splits results in even distrubutions of Age, Sex and Diagnosis)
for i in range(n_straps):
    X1, X2, y1, y2 = train_test_split(Split, y, test_size=.5) 
    
    #merge subscales back with split data -- you will need to change the variable names except for 
    result = pd.merge(X1,
                 df[['Key','CBCL_AD_T', 'CBCL_WD_T', 'CBCL_SC_T', 'CBCL_SP_T', 'CBCL_TP_T', 'CBCL_AP_T', 'CBCL_RBB_T', 'CBCL_AB_T']],
                 on='Key')
    result2 = pd.merge(X2,
                 df[['Key','CBCL_AD_T', 'CBCL_WD_T', 'CBCL_SC_T', 'CBCL_SP_T', 'CBCL_TP_T', 'CBCL_AP_T', 'CBCL_RBB_T', 'CBCL_AB_T']],
                 on='Key')
    
    split_subids.append([y1])
    split_subids2.append([y2])
    
    print('Lacing up left shoe bootstrap #%d' % (i + 1), end='\t')
    communities1, Q1 = pheno_clust(X=np.array(result[subset]).astype(np.float64), plot=False, verbose=False)
    
    print('Lacing up right shoe bootstrap #%d' % (i + 1))
    communities2, Q2 = pheno_clust(X=np.array(result2[subset]).astype(np.float64), plot=False, verbose=False)
    
    split_communities.append([communities1]) #append cluster assignments from each interation for split 1
    split_Q.append([Q1])  #append moduarity value from each for split 1
    split_communities2.append([communities2]) #append cluster assignments from each interation for split 2
    split_Q2.append([Q2])  #append moduarity value for split 2

#_____________________________________________________________________________________________________________________________________________________________________
#create dataframes of matched clusters and subject IDs and save outputs into CSVs 
#data split 1
allcsvs3 = {}
for i in range(n_straps):
    df2 = {'Key':split_subids[i][0],'cluster':split_communities[i][0]}
    allcsvs3[i] = pd.DataFrame(df2)
    allcsvs3[i].to_csv('C:/Users/jacob.derosa/Desktop/Scripts/Baggin_Subtyping/CBCL/Non_Boot_Data/CBCL_Split_1_csv_%s.csv' % i)
#data split 2     
allcsvs4 = {}
for i in range(n_straps):
    df3 = {'Key':split_subids2[i][0],'cluster':split_communities2[i][0]}
    allcsvs4[i] = pd.DataFrame(df3)
    allcsvs4[i].to_csv('C:/Users/jacob.derosa/Desktop/Scripts/Baggin_Subtyping/CBCL/Non_Boot_Data/CBCL_Split_2_csv_%s.csv' % i)
