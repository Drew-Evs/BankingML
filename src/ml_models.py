from data_experiments import import_data
from preprocessing_pipeline import prepare_data
from optimisation import apply_smote, apply_smote_enn, apply_info_gain, calc_info_gain

from imblearn.combine import SMOTEENN

import pandas as pd
import matplotlib.pyplot as plt
import joblib
import time

#regression model imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#xgboost import
from xgboost import XGBClassifier

#neural net classifier
from sklearn.neural_network import MLPClassifier

#for results
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score

#for cross validation
from sklearn.model_selection import StratifiedKFold, cross_validate
from preprocessing_pipeline import updated_pipeline
import numpy as np

#make load paths relative to ml_models
import os
BASE_DIR = os.path.dirname(__file__)

'''
a class using scikitlearn logistic regression for prediction
imports/processes data and then trains the model
dedicated prediction method
'''
class LogisticModel():
    def __init__(self, name="logistic_model.pk1", data_path="data/preprocessed.pk1"):
        #relative paths
        self.data_path = os.path.join(BASE_DIR, data_path)
        self.model_path = os.path.join(BASE_DIR, "saved_models", name)

        #prepare data and split
        try:
            self.load_data(self.data_path)
        except Exception as e:
            self.data_setup(self.data_path)
        self.split_data()

        #try to load model if it exists if not train and save
        try:
            self.load_model(self.model_path)
        except Exception as e:
            self.train_model()
            self.save_model(self.model_path)


    def data_setup(self, name):
        #preprocess and split data
        X, Y = import_data()
        self.features, self.targets = prepare_data(X, Y)
        joblib.dump((self.features, self.targets), name)

    def load_data(self, name):
        self.features, self.targets = joblib.load(name)

    def save_model(self, name):
        joblib.dump(self.model, name)

    def load_model(self, name):
        self.model = joblib.load(name)

    def split_data(self):
        #spitting 80/20 for train test
        self.f_train, self.f_test, self.t_train, self.t_test = train_test_split(
            self.features,
            self.targets,
            test_size=0.2,
            random_state=42
        )

        #apply smote on the training data
        smote_enn = SMOTEENN(random_state=42)
        self.f_train, self.t_train = smote_enn.fit_resample(self.f_train, self.t_train)

    def train_model(self):
        #create a logistic regression for classification and then fit to data
        self.model = LogisticRegression()
        self.model.fit(self.f_train, self.t_train)

    def model_results(self):
        t_pred = self.model.predict(self.f_test)
        print(f"Accuracy : {accuracy_score(self.t_test, t_pred) * 100:.2f}%")
        print(f"Precision: {precision_score(self.t_test, t_pred) * 100:.2f}%")
        print(f"Recall   : {recall_score(self.t_test, t_pred) * 100:.2f}%")
        print(f"F1 Score : {f1_score(self.t_test, t_pred) * 100:.2f}%")
        print(confusion_matrix(self.t_test, t_pred))

    #using cross validation for accuracy
    def cross_validate(self, folds=5):
        #optimal param model
        logmodel = LogisticRegression()

        #use the updated pipeline which processes all the data and fits the model
        pipeline = updated_pipeline(self.features, logmodel)
        
        #create multiple splits and the scoring metrics before running
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']

        #run the splits in parallel
        cv_results = cross_validate(
            pipeline, 
            self.features, 
            self.targets, 
            cv=skf, 
            scoring=scoring_metrics,
            n_jobs=-1
        )
        
        #output results
        print("\n*** Average Results Across All Splits ***")
        print(f"Accuracy : {np.mean(cv_results['test_accuracy']) * 100:.2f}% (+/- {np.std(cv_results['test_accuracy']) * 100:.2f}%)")
        print(f"Precision: {np.mean(cv_results['test_precision']) * 100:.2f}% (+/- {np.std(cv_results['test_precision']) * 100:.2f}%)")
        print(f"Recall   : {np.mean(cv_results['test_recall']) * 100:.2f}% (+/- {np.std(cv_results['test_recall']) * 100:.2f}%)")
        print(f"F1 Score : {np.mean(cv_results['test_f1']) * 100:.2f}% (+/- {np.std(cv_results['test_f1']) * 100:.2f}%)")


'''
the literature recommends XGBoost
https://hrcak.srce.hr/clanak/452496
https://www.preprints.org/manuscript/202512.0447
'''
class XGBoostModel():
    def __init__(self, model_path="XGB_model.pk1", data_path="XGB_preprocessed.pk1"):
        #relative paths
        self.data_path = os.path.join(BASE_DIR, "data", data_path)
        self.model_path = os.path.join(BASE_DIR, "saved_models", model_path)

        #prepare data and split
        try:
            self.load_data(self.data_path)
        except Exception as e:
            self.data_setup(self.data_path)
        self.split_data()

        #try to load model if it exists if not train and save
        try:
            self.load_model(self.model_path)
        except Exception as e:
            self.train_model()
        self.save_model(self.model_path)

    def make_clean(self, model_path="logistic_model.pk1", data_path="data/preprocessed.pk1"):
        data_path = os.path.join(BASE_DIR, data_path)
        model_path = os.path.join(BASE_DIR, "saved_models", model_path)

        self.data_setup(self.data_path)
        self.split_data()

        self.train_model()
        self.save_model(self.model_path)

    def data_setup(self, name):
        #preprocess and split data
        X, Y = import_data()
        self.features, self.targets = prepare_data(X, Y)
        joblib.dump((self.features, self.targets), name)

    def load_data(self, name):
        self.features, self.targets = joblib.load(name)

    def save_model(self, name):
        joblib.dump(self.model, name)

    def load_model(self, name):
        self.model = joblib.load(name)

    def split_data(self):
        #spitting 80/20 for train test
        self.f_train, self.f_test, self.t_train, self.t_test = train_test_split(
            self.features,
            self.targets,
            test_size=0.2,
            random_state=42
        )

        #apply smote on the training data
        smote_enn = SMOTEENN(random_state=42)
        self.f_train, self.t_train = smote_enn.fit_resample(self.f_train, self.t_train)

    def train_model(self):
        #create a XGBclassiifier for classification and then fit to data 
        #potentially want to experiment with somte
        self.model = XGBClassifier(
            n_estimators=275,
            learning_rate=0.03,
            max_depth=7,
            subsample=0.78,
            colsample_bytree=0.68,
            eval_metric="logloss"
        )

        self.model.fit(self.f_train, self.t_train)
        
    def model_results(self):
        #align the correct features after info gain
        t_pred = self.model.predict(self.f_test)
        print(f"Accuracy : {accuracy_score(self.t_test, t_pred) * 100:.2f}%")
        print(f"Precision: {precision_score(self.t_test, t_pred) * 100:.2f}%")
        print(f"Recall   : {recall_score(self.t_test, t_pred) * 100:.2f}%")
        print(f"F1 Score : {f1_score(self.t_test, t_pred) * 100:.2f}%")
        print(confusion_matrix(self.t_test, t_pred))

    #using cross validation for accuracy - didnt work as wanted (ignore)
    def cross_validate(self, folds=5):
        #optimal param model
        xgb_model = XGBClassifier(
            n_estimators=275, 
            learning_rate=0.03,
            max_depth=7,
            subsample=0.78,
            colsample_bytree=0.,
            eval_metric="logloss"
        )

        #use the updated pipeline which processes all the data and fits the model
        pipeline = updated_pipeline(self.features, xgb_model)
        
        #create multiple splits and the scoring metrics before running
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']

        #run the splits in parallel
        cv_results = cross_validate(
            pipeline, 
            self.features, 
            self.targets, 
            cv=skf, 
            scoring=scoring_metrics,
            n_jobs=-1
        )
        
        #output results
        print("\n*** Average Results Across All Splits ***")
        print(f"Accuracy : {np.mean(cv_results['test_accuracy']) * 100:.2f}% (+/- {np.std(cv_results['test_accuracy']) * 100:.2f}%)")
        print(f"Precision: {np.mean(cv_results['test_precision']) * 100:.2f}% (+/- {np.std(cv_results['test_precision']) * 100:.2f}%)")
        print(f"Recall   : {np.mean(cv_results['test_recall']) * 100:.2f}% (+/- {np.std(cv_results['test_recall']) * 100:.2f}%)")
        print(f"F1 Score : {np.mean(cv_results['test_f1']) * 100:.2f}% (+/- {np.std(cv_results['test_f1']) * 100:.2f}%)")

'''
comparing with a NN model
literature tends to suggest that this is worse
https://www.sciencedirect.com/science/article/pii/S016792361400061X
'''
class NNModel():
    def __init__(self, model_path="NN_model.pk1", data_path="NN_preprocessed.pk1"):
        #relative paths
        self.data_path = os.path.join(BASE_DIR, "data", data_path)
        self.model_path = os.path.join(BASE_DIR, "saved_models", model_path)

        #prepare data and split
        try:
            self.load_data(self.data_path)
        except Exception as e:
            self.data_setup(self.data_path)
        self.split_data()

        #try to load model if it exists if not train and save
        try:
            self.load_model(self.model_path)
        except Exception as e:
            self.train_model()
            self.save_model(self.model_path)

    def make_clean(self, model_path="logistic_model.pk1", data_path="data/preprocessed.pk1"):
        data_path = os.path.join(BASE_DIR, data_path)
        model_path = os.path.join(BASE_DIR, "saved_models", model_path)

        self.data_setup(self.data_path)
        self.split_data()

        self.train_model()
        self.save_model(self.model_path)

    def data_setup(self, name):
        #preprocess and split data
        X, Y = import_data()
        self.features, self.targets = prepare_data(X, Y)
        joblib.dump((self.features, self.targets), name)

    def load_data(self, name):
        self.features, self.targets = joblib.load(name)

    def save_model(self, name):
        joblib.dump(self.model, name)

    def load_model(self, name):
        self.model = joblib.load(name)

    def split_data(self):
        #spitting 80/20 for train test
        self.f_train, self.f_test, self.t_train, self.t_test = train_test_split(
            self.features,
            self.targets,
            test_size=0.2,
            random_state=42
        )

        #apply smote on the training data
        smote_enn = SMOTEENN(random_state=42)
        self.f_train, self.t_train = smote_enn.fit_resample(self.f_train, self.t_train)

    def train_model(self):
        #create a NN classifier for classification
        self.model = MLPClassifier(
            hidden_layer_sizes=(40, 20),
            activation='tanh',           
            solver='adam',               
            learning_rate_init=0.001,
            max_iter=170,               
            early_stopping=True,     
            random_state=42
        )

        self.model.fit(self.f_train, self.t_train)

    def model_results(self):
        t_pred = self.model.predict(self.f_test)
        print(f"Accuracy : {accuracy_score(self.t_test, t_pred) * 100:.2f}%")
        print(f"Precision: {precision_score(self.t_test, t_pred) * 100:.2f}%")
        print(f"Recall   : {recall_score(self.t_test, t_pred) * 100:.2f}%")
        print(f"F1 Score : {f1_score(self.t_test, t_pred) * 100:.2f}%")
        print(confusion_matrix(self.t_test, t_pred))

    #using cross validation for accuracy
    def cross_validate(self, folds=5):
        #optimal param model
        nn_model = MLPClassifier(
            hidden_layer_sizes=(40, 20),
            activation='tanh',           
            solver='adam',               
            learning_rate_init=0.001,
            max_iter=170,               
            early_stopping=True,     
            random_state=42
        )

        #use the updated pipeline which processes all the data and fits the model
        pipeline = updated_pipeline(self.features, nn_model)
        
        #create multiple splits and the scoring metrics before running
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']

        #run the splits in parallel
        cv_results = cross_validate(
            pipeline, 
            self.features, 
            self.targets, 
            cv=skf, 
            scoring=scoring_metrics,
            n_jobs=-1
        )
        
        #output results
        print("\n*** Average Results Across All Splits ***")
        print(f"Accuracy : {np.mean(cv_results['test_accuracy']) * 100:.2f}% (+/- {np.std(cv_results['test_accuracy']) * 100:.2f}%)")
        print(f"Precision: {np.mean(cv_results['test_precision']) * 100:.2f}% (+/- {np.std(cv_results['test_precision']) * 100:.2f}%)")
        print(f"Recall   : {np.mean(cv_results['test_recall']) * 100:.2f}% (+/- {np.std(cv_results['test_recall']) * 100:.2f}%)")
        print(f"F1 Score : {np.mean(cv_results['test_f1']) * 100:.2f}% (+/- {np.std(cv_results['test_f1']) * 100:.2f}%)")


if __name__ == "__main__":
    print("-"*50)

    print("Logistic Model:")
    logistic_model = LogisticModel()
    #logistic_model.cross_validate()
    logistic_model.model_results()
    print("-"*50)

    print("XGBoost Model:")
    xg_model = XGBoostModel()
    #xg_model.cross_validate()
    xg_model.model_results()
    print("-"*50)

    print("Neural Net Model:")
    nn_model = NNModel()
    #nn_model.cross_validate()
    nn_model.model_results()
    print("-"*50)

