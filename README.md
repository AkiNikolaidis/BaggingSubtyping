# BaggingSubtyping



Step One Assess Full Sample Subtyping (Build a reference)

1- Import the data

2- Run Louvain on the full sample and save cluster labels

3- Run bootstrapping and run LCD on each bootstrap sample and save cluster labels.

4- Aggregated Adjacency matrices into stability matrix and run LCD & Save cluster labels


STEP TWO: Assess impact of bagging on subtyping

1- Import the data.

2- Take X%-sized subsets of data *Parallelized

3a- Run LCD on subset & save cluster labels

	** Creates a folder of cluster labels and saves *non-bagged*
	** Keep split ID on the cluster labels
	
3b- Run bootstrapping on the subset *Parallelized

3b2- Run LCD on bootstrapped subset & save cluster labels

	** Creates a folder of cluster labels saves all bagged cluster labels
	** Keep split ID on the cluster labels
	
4- Aggregate all bootstrapped labels from 3b2 folder to create Stability Matrix

5- Run Stability Matrix through Louvain clustering and save FINAL cluster labels

6- Repeat steps 2-5 X times where X (100 – 500) is the number of splits desired

7- Review all splits:

	7a- Compare split labels for (final) bagged and non-bagged to full sample LCD (both bagged and non-bagged)
	7b- Compare split labels for (final) bagged & non-bagged with one another
	
8- Repeat steps 2-7 for 10%, 20%, 30%, 40%, 50%, [60% 70% 80%?] of data.

Cluster Labels: SplitID_SubsetID_Bagging(Y/N)_baggednumber (1 for nonbagged, 1-X for bagged).npy
	

Functions- 
Run LCD- to be used in STEP ONE, part 2 & 3, and Step TWO 3a & 3b.

