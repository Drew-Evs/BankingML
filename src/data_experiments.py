'''
!!! THIS PAGE IS USED TO TEST THE DATA AND FIND OUT WHAT PREPROCESSING NEEDS DOING
'''
import pandas as pd
import matplotlib.pyplot as plt

'''
importing the data from the repo 
@returns: X - the features & data
        Y - target data
'''
def import_data():
    from ucimlrepo import fetch_ucirepo 

    #fetch dataset
    bank_marketing = fetch_ucirepo(id=222) 
    
    #converts to pandas
    X = bank_marketing.data.features 
    Y = bank_marketing.data.targets 
    
    return X, Y

'''
Y is currently a yes or no need to convert to a value of 0 or 1
testing a label encoder to perform this function 
@input: unencoded Y with yes/no
@output: encoded Y with 0 or 1
'''
def label_encoder(Y):
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(Y["y"])
    y_encoded = pd.Series(y_encoded)
    return y_encoded

'''
3 numerical values - want to see how they correlate with the target
@inputs: X - the input features
        Y - the target features
@returns: a scatter matrix showing correlation between values
'''
def check_correlation(X, Y):
    #the specific columns we want
    cols = ['age', 'balance', 'duration']

    #combine into one dataframe
    df = X[cols].copy()
    df['target'] = Y

    print(df.corr(numeric_only=True))

'''
testing for scaling and imputer requirements 
if theres a large percent of NaNs (missing values) need an imputer 
if large range then need a scaler 
@inputs: X - input features
@outputs: range of each numerical feature 
        % NaNs for all features
'''
def check_pipeline_reqs(X):
    #select only number columns
    X_num = X.select_dtypes(include='number')
    
    #range test
    range_df = pd.DataFrame({
        "min": X_num.min(),
        "max": X_num.max(),
        "range": X_num.max() - X_num.min()
    })

    print(range_df.sort_values("range", ascending=False))

    #NaN test for all columns only print if missing some percent
    missing_pct = X.isna().mean() * 100
    print(missing_pct[missing_pct > 0])

'''
checking unique values in a specific column
@inputs: X - input features
        cat - the category to search
@returns: list of unique values
'''
def check_unique(X, cat):
    print(X[cat].unique())

'''
testing imputing/handling of Nan values - clean job/education
combine poutcome Nans with other & drop column contact type
@inputs: X - unimputed features
@outputs: imputed features
'''
def run_custom_imputer(X):
    X['poutcome'] = X['poutcome'].fillna('other')
    X = X.dropna(subset=['job', 'education'])
    X.drop('contact', axis=1, inplace=True)
    return X

'''
testing a standard scalar for the numeric values - run before imputer
@inputs: X - unscaled features
@outputs X_scaled - scaled features
'''
def run_scalar(X):
    #using standard scalar from sklearn - mean 0 standard deviation is 1
    from sklearn.preprocessing import StandardScaler

    #select numeric columns and copt to avoid modifying
    num_cols = X.select_dtypes(include='number').columns
    X_scaled = X.copy()

    #create and scale
    scaler = StandardScaler()
    X_scaled[num_cols] = scaler.fit_transform(X[num_cols])

    return X_scaled

'''
using a one hot encoder for the category input feautres
@inputs: unencoded X
@returns: encoded X
'''
def run_encoder(X):
    from sklearn.preprocessing import OneHotEncoder
    
    #create encoder
    encoder = OneHotEncoder()

    #select categories 
    cat = X.select_dtypes(include='object')
    X_cat_encoded = encoder.fit_transform(cat)

    #convert back to dataframe
    X_cat_encoded = pd.DataFrame(
        X_cat_encoded.toarray(),
        columns = encoder.get_feature_names_out(cat.columns)
    )  

    #recombine and return
    X_num = X.select_dtypes(exclude='object').reset_index(drop=True)
    X_cat_encoded = X_cat_encoded.reset_index(drop=True)
    X_encoded = pd.concat([X_num, X_cat_encoded], axis=1)

    return X_encoded

if __name__ == "__main__":
    X, Y = import_data()
    Y = label_encoder(Y)

    #testing what is required for pipeline
    check_pipeline_reqs(X)

    X = run_scalar(X)
    X = run_custom_imputer(X)
    X = run_encoder(X)
    print(X.head())

    #when rerunning now shouldnt need imputer or scalar anymore
    check_pipeline_reqs(X)