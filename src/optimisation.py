''''!!!TESTING AND OPTIMISING USING SMOTE, OPTUNA AND INFORMATION GAIN'''

import numpy as np
import pandas as pd

#mearest neightbours for SMOTE
from sklearn.neighbors import NearestNeighbors

'''
for SMOTE to work need to find the mminority class
@inputs: Y - output targets
@outputs: min_label - the minority label (0s or 1s)
    num_synthetic - the difference (number of synthetic data needed)
'''
def find_minority(Y):
    #find number of 0s or 1s
    counts = np.bincount(Y)
    min_label = np.argmin(counts)

    #find num of synthetic data needed
    num_synthetic = np.max(counts) - np.min(counts)
    return min_label, num_synthetic

'''
the SMOTE algorithm
@inputs: X_train - input features
    Y_train - output targets both after the test/train split
    k - number of nearest neightbours
@outputs: X - input features with additional data
'''
def apply_smote(X_train, Y_train, k=6):
    #save feature names
    feature_names = X_train.columns
    target_name = 'y'

    #convert to numpy arrays and find minority
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    min_label, num_synthetic = find_minority(Y_train)

    #only want minority class rows
    min_mask = (Y_train == min_label)
    X_min = X_train[min_mask]

    #create/fit nearest neighour model
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X_min)
    neighbours = nn.kneighbors(X_min, return_distance=False)

    #create empty numpy array for features and an array of the minority label for target
    n_samples, n_features = X_min.shape
    synthetic_X = np.zeros((num_synthetic, n_features))
    synthetic_Y = np.full(num_synthetic, min_label)

    #generate the synthetic array using random interpolation
    for i in range(num_synthetic):
        base_idx = np.random.randint(0, n_samples)
        base_point = X_min[base_idx]

        neighbor_idx = np.random.choice(neighbours[base_idx][1:])
        neighbor_point = X_min[neighbor_idx]
        
        #the difference between the actual point selected and its neighbour
        diff_vector = neighbor_point - base_point
        gap = np.random.uniform(0, 1)

        #randomly interpolate between the 2
        synthetic_X[i] = base_point + (gap * diff_vector)

    #add to the end of the synthetic training data 
    X_balanced = np.vstack((X_train, synthetic_X))
    Y_balanced = np.concatenate((Y_train, synthetic_Y))
    
    #reconvert to dataframe for the model
    X_balanced_df = pd.DataFrame(X_balanced, columns=feature_names)
    Y_balanced_series = pd.Series(Y_balanced, name=target_name)
    
    return X_balanced_df, Y_balanced_series




