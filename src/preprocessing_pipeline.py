'''
!!! THIS PAGE IS USED TO RUN THE PREPROCESSING PIPELINE FOR THE DATA
'''
import pandas as pd

#allows handling of category and numeric columns differently
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

#premade methods to scale numeric and encode categories
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler


#for more control over the data preprocessing 
from sklearn.base import BaseEstimator, TransformerMixin

#sklearn outputs in pandas
from sklearn import set_config
set_config(transform_output="pandas")

#allow inclusion of SMOTE in the pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE

from optimisation import apply_info_gain, calc_info_gain


'''
a custom cleaning/imputing class to control dropping of columns, removing of nan rows
    & imputing poutcome nans with other
'''
class CustomImputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        #need to drop duration (cant know before call) - keep it in
        #X = X.drop('duration', axis=1, errors='ignore')

        return X
    
'''
a custom transformer working with information gain
    allows placing of feature selection in the pipeline
'''
class InfoGainSelection(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.002):
        self.threshold = threshold
        self.selected_features = None
    
    def fit(self, X, Y):
        #calculate info gain and filter out lower than threshold
        info_gain = calc_info_gain(X, Y)
        self.selected_features = info_gain[info_gain['Info_Gain'] > self.threshold]['Feature'].tolist()
        return self
    
    def transform(self, X):
        #actually filter out
        return X[self.selected_features]

'''
build the updated pipeline
'''
def updated_pipeline(X, model, threshold=0.0):
    #clean duration column
    #keep duration in
    #imputer = CustomImputer()
    #X_clean = imputer.transform(X)

    num_cols = X.select_dtypes(include='number').columns
    cat_cols = X.select_dtypes(include='object').columns

    #preprocess with a one hot encoder
    preprocessor = ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ])

    #custom smote to increase number of neighbours
    custom_smote = SMOTE(k_neighbors=7, random_state=42)

    #the full pipeline as described in https://hrcak.srce.hr/file/452496
    full_pipeline = ImbPipeline([
        #("imputer", imputer),
        ("encode", preprocessor),                        
        ("smote_enn", SMOTEENN(smote=custom_smote, random_state=42)),        
        ("scaler", MinMaxScaler()),       
        ("classifier", model)   
    ])

    return full_pipeline

    
'''
build the pipeline for processing X
@inputs: X - the features
@output: full_pipeline - the pipeline used for the data
'''
def build_pipeline(X):
    #simulate clean X
    imputer = CustomImputer()
    X_clean = imputer.transform(X)

    #divide into nyumerical and categorical columns
    num_cols = X_clean.select_dtypes(include='number').columns
    cat_cols = X_clean.select_dtypes(include='object').columns

    #the 2 pipelines for each
    numeric_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    #combined into single preprocessor
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols)
    ])

    #run th imputer before preprocessor
    full_pipeline = Pipeline([
        ("imputer", imputer),
        ("preprocess", preprocessor)
    ])

    return full_pipeline

'''
process the data using the pipeline and a label encoder for use in ML
@inputs: X - the features
    Y - the targets
@outputs: X_prepared - prepared features
    Y_prepared - the prepared targets
'''
def prepare_data(X, Y):
    #droping the unkown job and education rows
    mask = X[['job', 'education']].notna().all(axis=1)
    X = X[mask]
    Y = Y[mask]

    #encoding y with a label encoder
    le = LabelEncoder()
    Y_prepared = le.fit_transform(Y["y"])
    Y_prepared = pd.Series(Y_prepared)
    
    #running the pipeline manually for testing
    imputer = CustomImputer()
    X_clean = imputer.transform(X)

    num_cols = X_clean.select_dtypes(include='number').columns
    cat_cols = X_clean.select_dtypes(include='object').columns

    preprocessor = ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ])
    
    X_encoded = preprocessor.fit_transform(X_clean)

    #applying info gain
    #selector = InfoGainSelection(threshold=0.0)
    #X_selected = selector.fit_transform(X_encoded, Y_prepared)

    #then run the scalar
    scaler = MinMaxScaler()
    X_final = scaler.fit_transform(X_encoded)

    print(f"Preprocessing Complete. Final Data Shape: {X_final.shape}")
    return X_final, Y_prepared


'''
found that evening the number of yes/no rows improves the flow of the data
@inputs: X - unfiltered input
    Y - unfiltered targets
@returns: X - filtered input
    Y - equal number of yes/no
'''
def equal_targets(X, Y):
    #work to even classes - stop just selecting no
    class_0 = Y[Y["y"] == "yes"]
    class_1 = Y[Y["y"] == "no"]

    #find smallest class and sample the same
    n = min(len(class_0), len(class_1))
    class_0 = class_0.sample(n, random_state=42)
    class_1 = class_1.sample(n, random_state=42)

    #recombine and find matching X rows
    Y = pd.concat([class_0, class_1]).sort_index()
    X = X.loc[Y.index]

    return X, Y

if __name__ == "__main__":
    from data_experiments import import_data
    X, Y = import_data()

    X, Y = prepare_data(X, Y)

    print(X.head())
    print(Y.head())