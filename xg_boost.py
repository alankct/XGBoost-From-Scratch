import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from cart_learner import CARTLearner

"""
XGBoost Implementation

Notes:
    1. XGBoost only uses Approximate Greedy Algorithm, Parallel Learning, and Weighted Quantile Sketch
        when the training dataset is huge
    2. 
"""
class Node:
    def __init__(self, val=0, feature=None, left=None, right=None):
        # (Approximate Greedy Algorithm): feature is a conditional boolean (multiple thresholds rather than just one feature)?
        self.val = val
        self.feature = feature
        self.left = left
        self.right = right
        # self.consumed_nodes = consumed_nodes


class XGBoostTree:
    def __init__(self, leaf_size=1, max_depth=float('inf'), lamda=0, gamma=0, quantiles=4):
        self.leaf_size, self.max_depth = leaf_size, max_depth
        # Regularization (lamda) and tree pruning (gamma) params
        # When lamda > 0, it is easier to prune trees because the values for gain are smaller
        # When gamma is higher, it is easier to prune trees too
        self.lamda, self.gamma = lamda, gamma
        self.quantiles = quantiles
        self.root = None

    def find_split_feature_and_value(self, x, y):
        """
        Calculates leaf similarity scores through different split thresholds
        Calculates gain to evaluate optimal split points
        (Approximate Greedy Algorithm): We can improve on this^ by using quantiles instead
        """

        # Do we want multiple quantiles to do conditional splits?
        best_feature = None
        best_split = None
        best_gain = -1

        # df = pd.DataFrame(x)
        # print(df)

        base_ss = sum(y)**2 / (len(y) + self.lamda)

        for i, col in enumerate(x.T):
            _, bins = pd.qcut(col, self.quantiles, retbins=True, duplicates='drop')
            for cut in bins[1:-1]:
                left_mask= col[:] <= cut
                right_mask = ~left_mask
                left_ss = (np.sum(y[left_mask]) ** 2) / (len(y[left_mask]) + self.lamda)
                right_ss = (np.sum(y[right_mask]) ** 2) / (len(y[right_mask]) + self.lamda)
                gain = (left_ss + right_ss) - base_ss

                if gain >= best_gain:
                    best_gain, best_feature, best_split = gain, i, cut
        
        # print(best_gain, best_feature, best_split)

        if best_gain - self.gamma < 0:
            return -1, -1   # Tree pruning: we will not create this branch
        
        return best_feature, best_split

    def train(self, x, y, depth=0, reset=True):
        """Last optimization: random subset of data/features"""
        if np.all(y == y[0]) or np.all(x==x[0,:]) or len(y) <= self.leaf_size or depth >= self.max_depth:
            # If all y values are the same OR
            # If all X values are the same OR
            # If we reach a small leaf sub-set OR
            # If we go past max_depth of tree: return a leaf
            return Node(np.sum(y)/(len(y) + self.lamda))


        # Determine best x-value (feature) and its value to split on
        split_feat, split_val = self.find_split_feature_and_value(x, y)

        if split_feat == -1 and split_val == -1:        # Prune Tree
            return None

        # Split on the median value for the selected feature
        less_than_val = x[:, split_feat] <= split_val
        left_x, right_x = x[less_than_val], x[~less_than_val]
        left_y, right_y = y[less_than_val], y[~less_than_val]

        if len(x) == max(len(left_x), len(right_x)):
            # The median did not split the arrays, so we return a leaf
            return Node(np.sum(y)/(len(y) + self.lamda))

        # Create a decision node
        node = Node(split_val, feature=split_feat)
        if not self.root or reset:
            self.root = node

        # Build left & right sub-trees
        node.left = self.train(left_x, left_y, depth=depth+1, reset=False)
        node.right = self.train(right_x, right_y, depth=depth+1, reset=False)
        
        return node
    
    def test(self, x):
        """
            Returns predictions y (1-D ndarray of estimates) for each row of x (2-D ndarray of features)
        """

        predictions = []
        
        def dfs(curr, row):
            if curr.feature == None:
                # Found a leaf
                predictions.append(curr.val)
                return
            if row[curr.feature] <= curr.val:
                dfs(curr.left, row)
            else:
                dfs(curr.right, row)

        for row in x:
            dfs(self.root, row)

        return np.array(predictions)


class XGBoost:

    def __init__(self, trees=10, quantiles=4, max_depth=10, learning_rate=0.1, lamda=0, gamma=0):
        self.trees = trees
        self.max_depth = max_depth
        self.quantiles = quantiles
        self.gdbt_model = []
        self.gdbt_predictions = None
        self.learning_rate, self.lamda, self.gamma = learning_rate, lamda, gamma

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
            weak_model = XGBoostTree(leaf_size=1, max_depth=self.max_depth, lamda=self.lamda, gamma=self.gamma, quantiles=self.quantiles)
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



# x = np.array([[1, 11, 7], [3, 90, 3], [6, 99, 10], [10, 68, 3]])
# y = np.array([3, 9, 9, 5])
# find_split_feature_and_value(x, y)