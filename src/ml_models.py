from data_experiments import import_data
from preprocessing_pipeline import prepare_data

import pandas as pd
import matplotlib.pyplot as plt
import joblib
import time

#regression model imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#for results
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

'''
a class using scikitlearn logistic regression for prediction
imports/processes data and then trains the model
dedicated prediction method
'''
class LogisticModel():
    def __init__(self, name="logistic_model.pk1"):
        #prepare data and split

        start_training = time.perf_counter()
        X, Y = import_data()
        self.features, self.targets = prepare_data(X, Y)
        self.split_data()
        end_training = time.perf_counter()

        start_loading = time.perf_counter()
        #try to load model if it exists if not train and save
        try:
            self.load_model(name)
        except Exception as e:
            self.train_model()
            self.save_model(name)
        end_loading = time.perf_counter()

        print(f'Preprocess time {end_training-start_training:.3f}s')
        print(f'Model time {end_loading-start_loading:.3f}s')

    def save_model(self, name):
        joblib.dump(self.model, f'saved_models/{name}')

    def load_model(self, name):
        self.model = joblib.load(f'saved_models/{name}')

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
        print(classification_report(self.t_test, t_pred))
        print(confusion_matrix(self.t_test, t_pred))

if __name__ == "__main__":
    test_model = LogisticModel()
    test_model.model_results()

