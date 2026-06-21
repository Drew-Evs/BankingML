from data_experiments import import_data
from preprocessing_pipeline import prepare_data

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
        start_training = time.perf_counter()
        try:
            self.load_data(self.data_path)
        except Exception as e:
            self.data_setup(self.data_path)
        self.split_data()
        end_training = time.perf_counter()

        start_loading = time.perf_counter()
        #try to load model if it exists if not train and save
        try:
            self.load_model(self.model_path)
        except Exception as e:
            self.train_model()
            self.save_model(self.model_path)
        end_loading = time.perf_counter()

        print(f'Preprocess time {end_training-start_training:.3f}s')
        print(f'Model time {end_loading-start_loading:.3f}s')

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
        start_training = time.perf_counter()
        try:
            self.load_data(self.data_path)
        except Exception as e:
            self.data_setup(self.data_path)
        self.split_data()
        end_training = time.perf_counter()

        start_loading = time.perf_counter()
        #try to load model if it exists if not train and save
        try:
            self.load_model(self.model_path)
        except Exception as e:
            self.train_model()
            self.save_model(self.model_path)
        end_loading = time.perf_counter()

        print(f'Preprocess time {end_training-start_training:.3f}s')
        print(f'Model time {end_loading-start_loading:.3f}s')

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

    def train_model(self):
        #create a XGBclassiifier for classification and then fit to data 
        #potentially want to experiment with somte
        self.model = XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="hist"
        )

        self.model.fit(self.f_train, self.t_train)

    def model_results(self):
        t_pred = self.model.predict(self.f_test)
        print(f"Accuracy : {accuracy_score(self.t_test, t_pred) * 100:.2f}%")
        print(f"Precision: {precision_score(self.t_test, t_pred) * 100:.2f}%")
        print(f"Recall   : {recall_score(self.t_test, t_pred) * 100:.2f}%")
        print(f"F1 Score : {f1_score(self.t_test, t_pred) * 100:.2f}%")
        print(confusion_matrix(self.t_test, t_pred))

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
        start_training = time.perf_counter()
        try:
            self.load_data(self.data_path)
        except Exception as e:
            self.data_setup(self.data_path)
        self.split_data()
        end_training = time.perf_counter()

        start_loading = time.perf_counter()
        #try to load model if it exists if not train and save
        try:
            self.load_model(self.model_path)
        except Exception as e:
            self.train_model()
            self.save_model(self.model_path)
        end_loading = time.perf_counter()

        print(f'Preprocess time {end_training-start_training:.3f}s')
        print(f'Model time {end_loading-start_loading:.3f}s')

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

    def train_model(self):
        #create a NN classifier for classification
        #allow early stopping
        #mess around with epochs and learning rate using optuna later
        self.model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',           
            solver='adam',               
            learning_rate_init=0.001,
            max_iter=300,               
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

if __name__ == "__main__":
    print("-"*50)

    print("Logistic Model\n")
    logistic_model = LogisticModel()
    logistic_model.model_results()
    print("-"*50)

    print("XGBoost Model\n")
    xg_model = XGBoostModel()
    xg_model.model_results()
    print("-"*50)

    print("Neural Net Model\n")
    nn_model = NNModel()
    nn_model.model_results()
    print("-"*50)

