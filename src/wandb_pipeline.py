from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

from data_experiments import import_data
from preprocessing_pipeline import prepare_data, CustomImputer
import time

from sklearn.metrics import accuracy_score, log_loss, f1_score

import pandas as pd

#preprocesses the data as it would in the pipeline
def load_and_preprocess():
    print("Loading data")
    X, Y = import_data()

    le = LabelEncoder()
    Y_prepared = le.fit_transform(Y["y"])
    Y = pd.Series(Y_prepared)
    
    #split data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    print("Imputing Data")
    imputer = CustomImputer()
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    print("Processing data")
    num_cols = X_train.select_dtypes(include='number').columns
    cat_cols = X_train.select_dtypes(include='object').columns
    encoder_scalar = ColumnTransformer([
        ("num", MinMaxScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ])

    #do the encoding here (only transform the test)
    X_train_enc = encoder_scalar.fit_transform(X_train)
    X_test_final = encoder_scalar.transform(X_test)

    #SMOTE training data
    custom_smote = SMOTE(k_neighbors=7, random_state=42)
    smote_enn = SMOTEENN(smote=custom_smote, random_state=42)

    #apply smote to trianing data only
    X_train_final, Y_train_bal = smote_enn.fit_resample(X_train_enc, Y_train)
    print(f"Preprocessing complete. Balanced Training Shape: {X_train_final.shape}")
    
    return X_train_final, X_test_final, Y_train_bal, Y_test

#pass in the training data and the parameters for the model
#return the created model
def create_xgb_model(X_train, Y_train, params):
    model = XGBClassifier(**params)
    model.fit(X_train, Y_train)
    return model

if __name__ == "__main__":
    import wandb 

    #preprocess data
    X_train, X_test, Y_train, Y_test = load_and_preprocess()

    #setup a wandb run
    run = wandb.init(
        entity="heronic-technologies",
        project="BankingMLProject",
        config={
            "model": "XGBoost",
            "dataset": "Portugese Telemarketing",
            "n_estimators": 275,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "eval_metric": "logloss",
        },
    )

    #define depth as custom x axis - want everything else to step with it
    run.define_metric("Depth")
    run.define_metric("Accuracy", step_metric="Depth")
    run.define_metric("Loss", step_metric="Depth")
    run.define_metric("F1 Score", step_metric="Depth")
    run.define_metric("Inference Time", step_metric="Depth")

    #want to experiment with different depths
    for depth in [4, 5, 6, 7, 8]:

        params = {
            "n_estimators": 275,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "max_depth": depth,
            "colsample_bytree": 0.7,
            "eval_metric": "logloss",
        }

        xgb_model = create_xgb_model(X_train, Y_train, params)

        #inference time
        start_time = time.perf_counter()
        preds = xgb_model.predict(X_test)
        inference_time = time.perf_counter() - start_time

        #probability for loss calculatoin
        probs = xgb_model.predict_proba(X_test)

        f1 = f1_score(Y_test, preds)
        
        # 3. Calculate metrics
        acc = accuracy_score(Y_test, preds)
        loss = log_loss(Y_test, probs)

        run.log({"Accuracy": acc, 
            "F1 Score": f1, 
            "Loss": loss,
            "Inference Time": inference_time,
            "Depth": depth 
        })

    run.finish()