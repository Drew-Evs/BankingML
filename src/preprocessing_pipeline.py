'''
!!! THIS PAGE IS USED TO RUN THE PREPROCESSING PIPELINE FOR THE DATA
'''
import pandas as pd

#allows handling of category and numeric columns differently
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

#premade methods to scale numeric and encode categories
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

#for more control over the data preprocessing 
from sklearn.base import BaseEstimator, TransformerMixin

#sklearn outputs in pandas
from sklearn import set_config
set_config(transform_output="pandas")


'''
a custom cleaning/imputing class to control dropping of columns, removing of nan rows
    & imputing poutcome nans with other
'''
class CustomImputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        #logic from experiments
        X['poutcome'] = X['poutcome'].fillna('other')
        X = X.drop('contact', axis=1, errors='ignore')
        #experiment with dropping
        X = X.drop('duration', axis=1, errors='ignore')

        return X
    
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
    
    #get the pipeline and transform X
    pipeline = build_pipeline(X)
    X_prepared = pipeline.fit_transform(X)

    return X_prepared, Y_prepared

if __name__ == "__main__":
    from data_experiments import import_data
    X, Y = import_data()

    X, Y = prepare_data(X, Y)

    print(X.head())
    print(Y.head())