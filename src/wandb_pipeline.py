from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

from data_experiments import import_data
from preprocessing_pipeline import prepare_data

from imblearn.pipeline import Pipeline

#preprocesses the data as it would in the pipeline
def load_and_preprocess():
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
    encoder_scalar = ColumnTransformer([
        ("num", MinMaxScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ])

    #SMOTE training data
    custom_smote = SMOTE(k_neighbors=7, random_state=42)
    smote_enn = SMOTEENN(smote=custom_smote, random_state=42)

    preprocessing_pipeline = Pipeline([
        ('encoder', encoder_scalar),
        ('smote_enn', smote_enn)
    ])

    #this should fit, transform and apply smote to just the training data
    X_train_final, Y_train_bal = preprocessing_pipeline.fit_resample(X_train, Y_train)
    #simply transform the test data
    X_test_final = preprocessing_pipeline.transform(X_test)

    print(f"Preprocessing complete. Balanced Training Shape: {X_train_final.shape}")
    
    return X_train_final, X_test_final, Y_train_bal, Y_test

#pass in the training data and the parameters for the model
#return the created model
def create_xgb_model(X_train, Y_train, params):
    model = XGBClassifier(**params)
    model.fit(X_train, Y_train)
    return model