import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from scipy import stats

def chi_loop(df, Split_Criteria, ID,  split_pct, binary_vars = None, continuous_vars = None):
    '''
    df: loaded input dataframe
    Split_Criteria: subset dataframe of desired variables that will be used to split on - Key needs to be included 
    ID: Key variable 
    binary_vars: list of categorical or binary variables for splitting
    continuous_vars: list of continuous variables for spltting 
    split_pct: subset percentage (integer from 0-1), even split = .05
    '''
    
    while True: 
        X1, X2, y1, y2 = train_test_split(Split_Criteria, ID, test_size= split_pct) 
        chi_vals = []  
        con_vals = []
        if binary_vars == None:
            for c in continuous_vars:
                # T Test 
                con_vals = stats.ttest_ind(X1[c], X2[c])
                con_vals = [con_vals.pvalue]
            p_vals = con_vals
            p_vals = np.array(p_vals)
            if all(i > .05 for i in p_vals) == True: 
                print("No Significant Differences in Splits")
                print(p_vals)
                break
                
        if continuous_vars == None:
            for n in binary_vars:
                tab = X1.groupby(n).count()
                tab = tab.iloc[:,0]
                tab2 = X2.groupby(n).count()
                tab2 = tab2.iloc[:,0]
                table = pd.concat([tab, tab2], axis=1)
                # Chi-square Test.
                c, p, dof, expected = chi2_contingency(table)
                chi_vals.append(p)
            p_vals = chi_vals
            p_vals = np.array(p_vals)
            if all(i > .05 for i in p_vals) == True: 
                print("No Significant Differences in Splits")
                print(p_vals)
                break
        
        else:
            for n in binary_vars:
                tab = X1.groupby(n).count()
                tab = tab.iloc[:,0]
                tab2 = X2.groupby(n).count()
                tab2 = tab2.iloc[:,0]
                table = pd.concat([tab, tab2], axis=1)
                # Chi-square test of independence.
                c, p, dof, expected = chi2_contingency(table)
                chi_vals.append(p)     
            for c in continuous_vars:
                con_vals = stats.ttest_ind(X1[c], X2[c])
                con_vals = [con_vals.pvalue]
            p_vals = chi_vals + con_vals
            p_vals = np.array(p_vals)
            if all(i > .05 for i in p_vals) == True: 
                print("No Significant Differences in Splits")
                print(p_vals)
                break
    
    Split_1 = pd.merge(y1, df, on='Key')
    Split_2 = pd.merge(y2, df, on='Key') 
    
    return Split_1, Split_2



#Example: 
#df = pd.read_csv('C:/Users/jacob.derosa/Desktop/Scripts/Full_CBCL_Splits/CBCL.csv')
#df = df.rename(columns={'Unnamed: 0': 'Key'})
#y = pd.DataFrame(df.Key)
#Split = df[['Key','ADHD_I','ADHD_C','ODD', 'ADHD_H','ADHD','ASD','ANX', 'NT','DEP','Other','LD','Age','Sex']] 
#binary_names = ['ADHD_I','ADHD_C','ODD', 'ADHD_H','ADHD','ASD','ANX', 'NT','DEP','Other','LD', 'Sex']
#continuous_names = ['Age']

### Run Chi_square Loop
#Split_1, Split_2 = chi_loop(df, Split, y, split_pct = .5, continuous_vars = continuous_names, binary_vars = binary_names)



