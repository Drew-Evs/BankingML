''''!!!TESTING AND OPTIMISING USING SMOTE, OPTUNA AND INFORMATION GAIN'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#mearest neightbours for SMOTE
from sklearn.neighbors import NearestNeighbors

#calculating information gain
from sklearn.feature_selection import mutual_info_classif

#use for an upgraded SMOTE 
from imblearn.combine import SMOTEENN

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
def apply_smote(X_train, Y_train, k=7):
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


'''
want to study information gain of different features
allows selecting of the best ones
@input: X - features post SMOTE
    Y - targets post SMOTE
@outputsL info_gain - pddataframe of info gain
'''
def calc_info_gain(X, Y):
    #calculate mutual information score between features and target
    ig_scores = mutual_info_classif(X, Y, random_state=42)

    #create and sort dataframe
    info_gain = pd.DataFrame({
        'Feature': X.columns,
        'Info_Gain': ig_scores
    })
    info_gain = info_gain.sort_values(by='Info_Gain', ascending=False).reset_index(drop=True)
    return info_gain

#applying info gain
def apply_info_gain(X_prepared, Y_prepared, info_gain, gain_threshold=0):
    #information gain threshold (remove columns that are lower)
    cols = info_gain[info_gain['Info_Gain'] <= gain_threshold]['Feature'].tolist()
    X_prepared = X_prepared.drop(columns=cols, axis=1, errors='ignore')
    print(f"Dropped {len(cols)} useless features: {cols}")
    return X_prepared

#and to plot
def plot_information_gain(info_gain):
    #want top 20 
    plt.figure(figsize=(10, 8))
    top_features = info_gain.head(20)
    
    plt.barh(top_features['Feature'], top_features['Info_Gain'], color='skyblue')
    plt.gca().invert_yaxis()
    plt.xlabel('Information Gain Score')
    plt.title('Feature Importance based on Information Gain')
    plt.tight_layout()
    plt.show()

    #and also bottom 20 
    plt.figure(figsize=(10, 8))
    bottom_features = info_gain.tail(20)
    
    plt.barh(bottom_features['Feature'], bottom_features['Info_Gain'], color='skyblue')
    plt.gca().invert_yaxis()
    plt.xlabel('Information Gain Score')
    plt.title('Feature Importance based on Information Gain')
    plt.tight_layout()
    plt.show()

'''
SMOTE didnt work great by itself - need to enn to clean up
also operate only on the training data
'''
def apply_smote_enn(X_train, Y_train):
    #initate algorithm and fit/resample data
    smote_enn = SMOTEENN(random_state=42)
    X_resampled, Y_resampled = smote_enn.fit_resample(X_train, Y_train)
    
    #rebuild Pandas DataFrame
    X_balanced_df = pd.DataFrame(X_resampled, columns=X_train.columns)
    Y_balanced_series = pd.Series(Y_resampled, name='y')
    
    return X_balanced_df, Y_balanced_series

import optuna
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

#preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

#preprocesses the data as it would in the pipeline
def load_and_preprocess_globally():
    print("Loading data")
    X_raw, Y_raw = import_data()
    X, Y = prepare_data(X_raw, Y_raw)
    
    #split data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    print("Processing data")
    num_cols = X_train.select_dtypes(include='number').columns
    cat_cols = X_train.select_dtypes(include='object').columns
    encoder = ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ])

    X_train_enc = encoder.fit_transform(X_train)
    X_test_enc = encoder.transform(X_test)

    #SMOTE training data
    custom_smote = SMOTE(k_neighbors=7, random_state=42)
    smote_enn = SMOTEENN(smote=custom_smote, random_state=42)
    X_train_bal, Y_train_bal = smote_enn.fit_resample(X_train_enc, Y_train)

    #finally scale
    scaler = MinMaxScaler()
    X_train_final = scaler.fit_transform(X_train_bal)
    X_test_final = scaler.transform(X_test_enc)

    print(f"Preprocessing complete. Balanced Training Shape: {X_train_final.shape}")
    
    return X_train_final, X_test_final, Y_train_bal, Y_test

#for optimising xgboost
def objective_xgb(trial, X_train, X_test, Y_train, Y_test):
    #suggested params per trial
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'eval_metric': 'logloss',
        'random_state': 42,
        'n_jobs': -1
    }

    #initiate/fit model
    xgb_model = XGBClassifier(**param)
    xgb_model.fit(X_train, Y_train)

    #inference time and f1 score
    start_time = time.perf_counter()
    preds = xgb_model.predict(X_test)
    inference_time = time.perf_counter() - start_time
    f1 = f1_score(Y_test, preds)
    
    return f1, inference_time

#same for neural net
def objective_nn(trial, X_train, X_test, Y_train, Y_test):
    n_layers = trial.suggest_int('n_layers', 1, 2)
    layers = []
    for i in range(n_layers):
        layers.append(trial.suggest_int(f'n_units_l{i}', 16, 64))
    
    param = {
        'hidden_layer_sizes': tuple(layers),
        'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
        'solver': 'adam', 
        'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True),
        'max_iter': trial.suggest_int('max_iter', 100, 300),
        'early_stopping': True,
        'random_state': 42
    }

    nn_model = MLPClassifier(**param)
    nn_model.fit(X_train, Y_train)

    start_time = time.perf_counter()
    preds = nn_model.predict(X_test)
    inference_time = time.perf_counter() - start_time

    f1 = f1_score(Y_test, preds)
    
    return f1, inference_time

#execution loop
if __name__ == "__main__":
        
    # Custom imports
    from data_experiments import import_data
    from preprocessing_pipeline import prepare_data

    #do the split and preprocess
    X_train_final, X_test_final, Y_train_bal, Y_test = load_and_preprocess_globally()

    print("\n" + "="*50)
    print("Starting Lightning-Fast XGBoost Optimization...")
    print("Maximizing F1 Score | Minimizing Inference Time")
    print("="*50)
    
    #setup study 
    study_xgb = optuna.create_study(directions=['maximize', 'minimize'], study_name="XGB_Time_vs_F1")
    study_xgb.optimize(lambda trial: objective_xgb(trial, X_train_final, X_test_final, Y_train_bal, Y_test), n_trials=30)
    
    print("\n*** XGBoost Pareto Front (Best Trade-offs) ***")
    for i, trial in enumerate(study_xgb.best_trials):
        print(f"\nOption {i+1}: F1: {trial.values[0]:.4f} | Time: {trial.values[1]:.5f}s")
        print(f"Params: {trial.params}")

    print("\n" + "="*50)
    print("Starting Lightning-Fast Neural Network Optimization...")
    print("Maximizing F1 Score | Minimizing Inference Time")
    print("="*50)

    study_nn = optuna.create_study(directions=['maximize', 'minimize'], study_name="NN_Time_vs_F1")
    study_nn.optimize(lambda trial: objective_nn(trial, X_train_final, X_test_final, Y_train_bal, Y_test), n_trials=30)
    
    print("\n*** Neural Network Pareto Front (Best Trade-offs) ***")
    for i, trial in enumerate(study_nn.best_trials):
        print(f"\nOption {i+1}: F1: {trial.values[0]:.4f} | Time: {trial.values[1]:.5f}s")
        print(f"Params: {trial.params}")