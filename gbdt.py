import numpy as np
from cart_learner import CARTLearner

"""
Gradient Boosted Decision Tree
"""

class GBDT:

    def __init__(self, trees=10, max_depth=10):
        self.trees = trees
        self.max_depth = max_depth
        self.gdbt_model = []
        self.gdbt_predictions = None      # To test training progress

    def train(self, x, y):
        """
            Create a certain number of weak-models to do regressive gradient boosting
                @params:
                    x is a 2-D ndarray where columns are the X features and each row is a different training example
                    y is a 1-D ndarray with the matching labels/answers in the same order as the X training examples
        """

        self.gdbt_predictions = np.zeros_like(y) # GBDT model is initially empty.
        print(y[:5])
        print('---------------------------------')

        for _ in range(self.trees):

            # Error of the strong model
            error = self.gdbt_predictions - y

            # The weak model is a decision tree (CART right now wo/ pruning and a specific max depth)
            weak_model = CARTLearner(max_depth=self.max_depth)
            weak_model.train(x, error)

            self.gdbt_model.append(weak_model)

            weak_predictions = weak_model.test(x)   #[:,0]?

            print(self.gdbt_predictions[:5])
            print(weak_predictions[:5])
            self.gdbt_predictions -= weak_predictions
            print('---------------------------------')

    def test(self, x):

        """ FIX THIS """
        
        predictions = np.zeros_like(x[0, :])
        print(x.shape, len(x[0,:]), len(predictions))
        for weak_model in self.gdbt_model:
            predictions += weak_model.test(x)
        
        return predictions

