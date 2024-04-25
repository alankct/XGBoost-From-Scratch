import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from cart_learner import CARTLearner

"""
XGBoost Implementation
"""
class Node:
    def __init__(self, val=0, feature=None, left=None, right=None):
        # (Approximate Greedy Algorithm): feature is a conditional boolean (multiple thresholds rather than just one feature)?
        self.val = val
        self.feature = feature
        self.left = left
        self.right = right


class XGBoostTree:
    def __init__(self, leaf_size=1, max_depth=float('inf'), regularization=0, tree_pruning=0):
        self.leaf_size, self.max_depth = leaf_size, max_depth
        self.regularization, self.tree_pruning = regularization, tree_pruning
        
    def find_split_feature_and_value(self):
        """
        Calculates leaf similarity scores through different split thresholds
        Calculates gain to evaluate optimal split points
        (Approximate Greedy Algorithm): Next, we can improve on this^ by using quantiles instead
        """
        pass
    def train(self, x, y, depth=0, reset=True):
        """Last optimization: random subset of data/features"""
        if np.all(y == y[0]):
            # All y values are the same, returns a leaf with that value
            return Node(y[0])
        if np.all(x==x[0,:]) or len(y) <= self.leaf_size or depth >= self.max_depth:
            # If all X values are the same or if we reach a small leaf sub-set or if we go past max_depth: return a leaf
            return Node(np.mean(y))


        # Determine best x-value (feature) to split on
        feature_index = self.find_split_feature(x, y)

        # Split on the median value for the selected feature
        median = np.median(x[:, feature_index])
        less_than_median = x[:, feature_index] <= median
        left_x, right_x = x[less_than_median], x[~less_than_median]
        left_y, right_y = y[less_than_median], y[~less_than_median]

        if len(x) == max(len(left_x), len(right_x)):
            # The median did not split the arrays, so we return a leaf
            return Node(np.mean(y))

        # Create a decision node
        node = Node(median, feature=feature_index)
        if not self.root or reset:
            self.root = node

        # Build left & right sub-trees
        node.left = self.train(left_x, left_y, depth=depth+1, reset=False)
        node.right = self.train(right_x, right_y, depth=depth+1, reset=False)
        
        return node
    
    def test(self, x):
        pass


class XGBoost:

    def __init__(self, trees=10, max_depth=10, learning_rate=0.1, regularization=0, tree_pruning=0):
        self.trees = trees
        self.max_depth = max_depth
        self.gdbt_model = []
        self.gdbt_predictions = None
        self.learning_rate, self.regularization, self.tree_pruning = learning_rate, regularization, tree_pruning

    def train(self, x, y):
        """
            Create a certain number of (Weak) XGBoost Trees to do regressive gradient boosting
                @params:
                    x is a 2-D ndarray where columns are the X features and each row is a different training example
                    y is a 1-D ndarray with the matching labels/answers in the same order as the X training examples
        """

        # Stores all predictions so far (initially empty)
        self.gdbt_predictions = np.zeros_like(y)

        for _ in range(self.trees):

            # Error of the strong model
            error = self.gdbt_predictions - y

            # The weak model is a decision tree (weak XGBoost Tree)
            weak_model = XGBoostTree(max_depth=self.max_depth)
            weak_model.train(x, error)

            self.gdbt_model.append(weak_model)

            weak_predictions = weak_model.test(x)
            self.gdbt_predictions -= self.learning_rate * weak_predictions

    def test(self, x, y=None, return_errors=False):
        
        errors = []
        # Stores predictions so far (initially all 0)
        predictions = np.array([0 for _ in range(len(x))], dtype='float64')
        for weak_model in self.gdbt_model:
            weak_predictions = weak_model.test(x)
            predictions -= self.learning_rate * weak_predictions
            if y is not None:
                errors.append(mean_squared_error(y, predictions, squared=False))
        
        if return_errors:
            return predictions, errors

        return predictions

