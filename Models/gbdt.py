import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from Models.cart_learner import CARTLearner

"""
Gradient Boosted Decision Tree
"""

class GBDT:

    def __init__(self, trees=10, max_depth=10, learning_rate=0.1):
        self.trees = trees
        self.max_depth = max_depth
        self.gdbt_model = []
        self.gdbt_predictions = None      # To test training progress
        self.learning_rate = learning_rate

    def train(self, x, y):
        """
            Create a certain number of weak-models to do regressive gradient boosting
                @params:
                    x is a 2-D ndarray where columns are the X features and each row is a different training example
                    y is a 1-D ndarray with the matching labels/answers in the same order as the X training examples
        """

        self.gdbt_predictions = np.zeros_like(y) # GBDT model is initially empty.

        for _ in range(self.trees):

            # Error of the strong model
            error = self.gdbt_predictions - y

            # The weak model is a decision tree (CART right now and a specific max depth)
            weak_model = CARTLearner(max_depth=self.max_depth)
            weak_model.train(x, error)

            self.gdbt_model.append(weak_model)

            weak_predictions = weak_model.test(x)
            self.gdbt_predictions -= self.learning_rate * weak_predictions

    def test(self, x, y=None, return_errors=False):
        
        errors = []
        predictions = np.array([0 for _ in range(len(x))], dtype='float64')
        for weak_model in self.gdbt_model:
            weak_predictions = weak_model.test(x)
            predictions -= self.learning_rate * weak_predictions
            if y is not None:
                errors.append(mean_squared_error(y, predictions, squared=False))
        
        if return_errors:
            return predictions, errors

        return predictions

