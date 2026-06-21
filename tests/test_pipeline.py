#allows pulling functions from src 
import sys
import os
import unittest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data_experiments import import_data
from preprocessing_pipeline import prepare_data
from ml_models import LogisticModel, XGBoostModel, NNModel

#test that the import data function is working correctly
class TestImportData(unittest.TestCase):

    def test_runs(self):
        try:
            X, Y = import_data()
            self.assertIsNotNone(X)
            self.assertIsNotNone(Y)
        except Exception as e:
            self.fail(f"import_data() raised an exception: {e}")

#test that the data prep is working correctly
class TestDataPipeline(unittest.TestCase):

    def test_runs(self):
        try:
            X, Y = import_data()
            X, Y = prepare_data(X, Y)
            self.assertIsNotNone(X)
            self.assertIsNotNone(Y)
        except Exception as e:
            self.fail(f"prepare_data() raised an exception: {e}")

#test that the models works correctly
class TestXGBoost(unittest.TestCase):

    def test_runs(self):
        try:
            model = XGBoostModel()
            self.assertIsNotNone(model.model)
            model.model_results
        except Exception as e:
            self.fail(f"XGBoostModel() raised an exception: {e}")

class TestNNModel(unittest.TestCase):

    def test_runs(self):
        try:
            model = NNModel()
            self.assertIsNotNone(model.model)
            model.model_results
        except Exception as e:
            self.fail(f"NNModel() raised an exception: {e}")


if __name__ == "__main__":
    unittest.main()