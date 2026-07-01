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
    Y_prepared = le.fit_transform(Y["y"])
    Y = pd.Series(Y_prepared)
    
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

    #SMOTE training data
    custom_smote = SMOTE(k_neighbors=7, random_state=42)
    smote_enn = SMOTEENN(smote=custom_smote, random_state=42)

    #apply smote to trianing data only
    X_train_final, Y_train_bal = smote_enn.fit_resample(X_train_enc, Y_train)
    print(f"Preprocessing complete. Balanced Training Shape: {X_train_final.shape}")
    
    return X_train_final, X_validate_final, Y_train_bal, Y_validate

#pass in the training data and the parameters for the model
#return the created model
def create_xgb_model(X_train, Y_train, params):
    model = XGBClassifier(**params)
    model.fit(X_train, Y_train)
    return model

#using an optuna trial (start with just 2 variables to log)
def train_sweep():
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

    #logging values
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

if __name__ == "__main__":
    #preprocess data
    global X_train, X_validate, Y_train, Y_validate
    X_train, X_validate, Y_train, Y_validate = load_and_preprocess()

    #creating the wandb sweep
    print("XGBoost Optimisation")
    
    #initial params
    #adjust params with a sweep configuration
    sweep_config = {
        'method': 'bayes',  
        'metric': {
            'name': 'F1 Score',
            'goal': 'maximize'   
        },
        'parameters': {
            'n_estimators': {'min': 50, 'max': 750},
            'learning_rate': {'value': 0.1}, 
            'max_depth': {'value': 6},
            'subsample': {'min': 0.3, 'max': 1.0},
            'colsample_bytree': {'min': 0.3, 'max': 1.0}
        }
    }

    sweep_id = wandb.sweep(
        sweep=sweep_config, 
        project="BankingMLProject", 
        entity="heronic-technologies"
    )

    wandb.agent(sweep_id, function=train_sweep, count=30)