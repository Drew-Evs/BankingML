from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

from data_experiments import import_data
import time

from sklearn.metrics import accuracy_score, log_loss, f1_score

import pandas as pd

import wandb 
import optuna

from sklearn.base import BaseEstimator, TransformerMixin

#calculating information gain
from sklearn.feature_selection import mutual_info_classif

from sklearn import set_config
set_config(transform_output="pandas")

from sklearn.feature_selection import f_classif
import numpy as np

class CustomImputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        #need to drop duration (cant know before call) - keep it in
        X = X.drop('duration', axis=1, errors='ignore')

        return X

#preprocesses the data as it would in the pipeline
def load_and_preprocess():
    print("Loading data")
    X, Y = import_data()

    le = LabelEncoder()
    Y = pd.Series(le.fit_transform(Y["y"]), index=Y.index)
    
    #split data into train and test
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    #then split train 70/30 into train/validate (not sure if this is the righ way)
    X_train, X_validate, Y_train, Y_validate = train_test_split(
        X_train, Y_train, test_size=0.3, random_state=42
    )

    print("Imputing Data")
    imputer = CustomImputer()
    X_train = imputer.fit_transform(X_train)
    X_validate = imputer.transform(X_validate)

    print("Processing data")
    num_cols = X_train.select_dtypes(include='number').columns
    cat_cols = X_train.select_dtypes(include='object').columns
    encoder_scalar = ColumnTransformer([
        ("num", MinMaxScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ])

    #do the encoding here (only transform the test)
    X_train_enc = encoder_scalar.fit_transform(X_train)
    X_validate_final = encoder_scalar.transform(X_validate)

    print(f"Preprocessing complete. Balanced Training Shape: {X_train_enc.shape}")
    
    return X_train_enc, X_validate_final, Y_train, Y_validate

#separate out smote to avoid having to constantly recall
'''important note for SMOTE runs after a one hot encoder could be issues with running between 0 and 1'''
'''going to run a test with SMOTENC'''
def apply_smote(knn, X_train, Y_train):
    #SMOTE training data - n_jobs should parallelise
    custom_smote = SMOTE(k_neighbors=knn, random_state=42)
    smote_enn = SMOTEENN(smote=custom_smote, random_state=42, n_jobs=-1)

    #apply smote to trianing data only
    X_train_final, Y_train_bal = smote_enn.fit_resample(X_train, Y_train)

    #need to return to pandas?
    if not isinstance(X_train_final, pd.DataFrame):
        X_train_final = pd.DataFrame(X_train_final, columns=X_train.columns)

    return X_train_final, Y_train_bal

#pass in the training data and the parameters for the model
#return the created model
def create_xgb_model(X_train, Y_train, params):
    model = XGBClassifier(**params)
    model.fit(X_train, Y_train)
    return model

#using an optuna trial (start with just 2 variables to log)
#either want to log full config or just the smote/info_gain threshold
def param_trial():
    #initiate runs
    run = wandb.init()

    #pull hyperparameters and link to model
    config = wandb.config
    params = {
        "max_depth": config.max_depth,
        "n_estimators": config.n_estimators,
        "learning_rate": config.learning_rate,
        "subsample": config.subsample,
        "colsample_bytree": config.colsample_bytree,
        "eval_metric": "logloss",
    }

    #create model
    xgb_model = create_xgb_model(X_train, Y_train, params)

    #get results
    f1, acc, loss, inference_time = return_results(xgb_model, X_validate, Y_validate)

    run.log({
        "Accuracy": acc, 
        "Loss": loss,
        "F1 Score": f1,
        "Time": inference_time,
        "Learning Rate": params['learning_rate'],
        "Num estimators": params['n_estimators'],
        "Depth": params['max_depth'],
        "Row Sampling": params['subsample'], 
        "Column Sampling": params['colsample_bytree'],
    })

#logic to sweep with a full configuration hyperoptimisation
def param_sweep():
    #preprocess data
    global X_train, X_validate, Y_train, Y_validate
    X_train, X_validate, Y_train, Y_validate = load_and_preprocess()

    #can run smote once
    X_train, Y_train = apply_smote(7, X_train, Y_train)

    #creating the wandb sweep
    print("XGBoost Optimisation")
    
    #initial params
    #adjust params with a sweep configuration
    sweep_config = {
        'method': 'bayes',  
        'metric': {
            'name': 'Accuracy',
            'goal': 'maximize'   
        },
        'parameters': {
            'n_estimators': {'min': 50, 'max': 750},
            'learning_rate': {'min': 0.01, 'max': 0.2}, 
            'max_depth': {'min': 2, 'max': 10},
            'subsample': {'min': 0.3, 'max': 1.0},
            'colsample_bytree': {'min': 0.3, 'max': 1.0}
        }
    }

    sweep_id = wandb.sweep(
        sweep=sweep_config, 
        project="BankingMLProject", 
        entity="heronic-technologies"
    )

    wandb.agent(sweep_id, function=param_trial, count=30)

#refitting infogainselection with f_classif (should run faster)
#has a larger range necessary to change sweep config
class InfoGainSelection(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=5.0): #prev 0.002
        self.threshold = threshold
        self.selected_features = None
    
    def fit(self, X, Y):
        #calculate info gain and filter out lower than threshold
        #ig_scores = mutual_info_classif(X, Y, random_state=42)

        #using f_classif - uses ANOVA F values (!!!RESEARCH THESE LATER!!!)
        f_scores, p_values = f_classif(X, Y)
        f_scores = np.nan_to_num(f_scores)

        #create and sort dataframe
        # info_gain = pd.DataFrame({
        #     'Feature': X.columns,
        #     'Info_Gain': ig_scores
        # })

        #ensures still runs if array instead of dataframe
        features = X.columns if hasattr(X, 'columns') else list(range(X.shape[1]))

        feature_scores = pd.DataFrame({
            'Feature': features,
            'Score': f_scores
        })

        feature_scores = feature_scores.sort_values(by='Score', ascending=False).reset_index(drop=True)

        self.selected_features = feature_scores[feature_scores['Score'] > self.threshold]['Feature'].tolist()

        #keep only best feature if threshold too high
        if len(self.selected_features) == 0:
            self.selected_features = [feature_scores.iloc[0]['Feature']]

        return self
    
    def transform(self, X):
        #actually filter out - works if dataframe or array
        if isinstance(X, pd.DataFrame):
            return X[self.selected_features]
        else:
            return X[:, self.selected_features]
    
#training the data with information gain and custom knns
def data_trial():
    run = wandb.init()
    config = wandb.config

    #then apply smote 
    X_train_temp, Y_train_temp = apply_smote(config.knn, X_train, Y_train)

    #then fit and transform using the info gain selection transformer
    info_gain_transformer = InfoGainSelection(threshold=config.gain_threshold)
    X_train_final = info_gain_transformer.fit_transform(X_train_temp, Y_train_temp)
    X_validate_final = info_gain_transformer.transform(X_validate)

    #and finally fit model to the training data
    xgb_model.fit(X_train_final, Y_train_temp)

    f1, acc, loss, inference_time = return_results(xgb_model, X_validate_final, Y_validate)

    run.log({
        "Accuracy": acc, 
        "Loss": loss,
        "F1 Score": f1,
        "Time": inference_time,
        "Smote K": config.knn,
        "Info Gain Threshold": config.gain_threshold,
    })

#logic to sweep different knns for SMOTE and info gain
def data_processing_sweep():
    #adjust params with a sweep configuration
    sweep_config = {
        'method': 'bayes',  
        'metric': {
            'name': 'Accuracy',
            'goal': 'maximize'   
        },
        'parameters': {
            'gain_threshold': {'min': 0.0, 'max': 50.0},
            'knn': {'min': 2, 'max': 12}, 
        }
    }
    
    #using static parameters for the model (selected from best perfroming)
    params = {
        "max_depth": 6,
        "n_estimators": 500,
        "learning_rate": 0.15,
        "subsample": 0.83,
        "colsample_bytree": 0.99,
        "eval_metric": "logloss",
    }

    #create model as global for this its the train and validate that change
    global xgb_model
    xgb_model = XGBClassifier(**params)

    #keep data global (can do due to seperation of smote)
    global X_train, X_validate, Y_train, Y_validate
    X_train, X_validate, Y_train, Y_validate = load_and_preprocess()

    sweep_id = wandb.sweep(
        sweep=sweep_config, 
        project="BankingMLProject", 
        entity="heronic-technologies"
    )

    wandb.agent(sweep_id, function=data_trial, count=17)

#refactored function to take in a model and x/y validate and return results
def return_results(xgb_model, X_validate, Y_validate):
    #predict and log values
    #inference time
    start_time = time.perf_counter()
    preds = xgb_model.predict(X_validate)
    inference_time = time.perf_counter() - start_time

    #probability for loss calculatoin
    probs = xgb_model.predict_proba(X_validate)
    
    #accuracy metrics
    f1 = f1_score(Y_validate, preds)
    acc = accuracy_score(Y_validate, preds)
    loss = log_loss(Y_validate, probs)

    return f1, acc, loss, inference_time

'''
want to allow different sweeps for different goals
usage: - python wandb_pipeline.py parameters - runs a hypereparameter optimisation
    - python wandb_pipeline.py data - alters the data processing pipeline
'''
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python wandb_pipeline.py <goal>")
        sys.exit(1)

    if sys.argv[1] == "parameters":
        param_sweep()
    elif sys.argv[1] == "data":
        data_processing_sweep()
    else:
        print("<goal> must be 'data' or 'parameter'")
        sys.exit(1)
