import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from Models.cart_learner import CARTLearner

"""
XGBoost Implementation

Notes:
    1. XGBoost only uses Approximate Greedy Algorithm, Parallel Learning, and Weighted Quantile Sketch
        when the training dataset is huge
    2. 
"""
class Node:
    def __init__(self, val=0, feature=None, left=None, right=None, sparse_dir=None):
        # (Approximate Greedy Algorithm): feature is a conditional boolean (multiple thresholds rather than just one feature)?
        self.val = val
        self.feature = feature
        self.left = left
        self.right = right
        self.sparse_dir = sparse_dir
        # self.consumed_nodes = consumed_nodes
    def __str__(self):
        return f'Feature: {self.feature} Val: {self.val} Sparse Dir {self.sparse_dir != None}'


class XGBoostTree:
    def __init__(self, leaf_size=1, max_depth=float('inf'), lamda=0, gamma=0, quantiles=4, sparsity_aware=False):
        self.leaf_size, self.max_depth = leaf_size, max_depth
        # Regularization (lamda) and tree pruning (gamma) params
        # When lamda > 0, it is easier to prune trees because the values for gain are smaller
        # When gamma is higher, it is easier to prune trees too
        self.lamda, self.gamma = lamda, gamma
        self.quantiles = quantiles
        self.root = None
        self.sparsity_aware = sparsity_aware

    def find_split_feature_and_value(self, x, y, sparsity_aware=False):
        """
        Calculates leaf similarity scores through different split thresholds
        Calculates gain to evaluate optimal split points
        (Approximate Greedy Algorithm): We can improve on this^ by using quantiles instead
        """

        # Do we want multiple quantiles to do conditional splits?
        best_feature = None
        best_split = None
        best_gain = -1
        sparse_dir = -1

        # df = pd.DataFrame(x)
        # print(df)

        base_ss = sum(y)**2 / (len(y) + self.lamda)

        for i, col in enumerate(x.T):
            if sparsity_aware:
                missing_values = col[:] == 0
                col1 = col[~missing_values]
                # col2 = col[missing_values]      # Missing values x
                y1 = y[~missing_values]
                y2 = y[missing_values]          # Missing values y
                _, bins = pd.qcut(col1, self.quantiles, retbins=True, duplicates='drop')
            else:
                _, bins = pd.qcut(col, self.quantiles, retbins=True, duplicates='drop')
            for cut in bins[1:-1]:
                if sparsity_aware:
                    left_mask= col1[:] <= cut
                    right_mask = ~left_mask
                    gain1 = (np.sum(y1[left_mask]) + np.sum(y2))**2 / (len(y1[left_mask]) + len(y2) + self.lamda)
                    gain2 = (np.sum(y1[right_mask]) + np.sum(y2))**2 / (len(y1[right_mask])+ len(y2) + self.lamda)
                    if gain1 >= gain2:
                        gain = gain1
                        sparse_dir = 0  # 'Left'
                    else:
                        gain = gain2
                        sparse_dir = 1  # 'Right'
                else:
                    left_mask= col[:] <= cut
                    right_mask = ~left_mask
                    left_ss = (np.sum(y[left_mask]) ** 2) / (len(y[left_mask]) + self.lamda)
                    right_ss = (np.sum(y[right_mask]) ** 2) / (len(y[right_mask]) + self.lamda)
                    gain = (left_ss + right_ss) - base_ss

                if gain >= best_gain:
                    best_gain, best_feature, best_split = gain, i, cut
        
        # print(best_gain, best_feature, best_split)

        if best_gain - self.gamma < 0:
            return -1, -1, -1   # Tree pruning: we will not create this branch
        
        return best_feature, best_split, sparse_dir

    def train(self, x, y, depth=0, reset=True, sparsity_aware=False):
        """Last optimization: random subset of data/features"""
        if np.all(y == y[0]) or np.all(x==x[0,:]) or len(y) <= self.leaf_size or depth >= self.max_depth:
            # If all y values are the same OR
            # If all X values are the same OR
            # If we reach a small leaf sub-set OR
            # If we go past max_depth of tree: return a leaf
            return Node(np.sum(y)/(len(y) + self.lamda))


        # Determine best x-value (feature) and its value to split on
        split_feat, split_val, sparse_dir = self.find_split_feature_and_value(x, y, sparsity_aware=sparsity_aware)

        if split_feat == -1 and split_val == -1:        # Prune Tree
            return Node(np.sum(y)/(len(y) + self.lamda))

        # Split on the median value for the selected feature
        less_than_val = x[:, split_feat] <= split_val
        left_x, right_x = x[less_than_val], x[~less_than_val]
        left_y, right_y = y[less_than_val], y[~less_than_val]

        if len(x) == max(len(left_x), len(right_x)):
            # The median did not split the arrays, so we return a leaf
            return Node(np.sum(y)/(len(y) + self.lamda))

        # Create a decision node
        node = Node(split_val, feature=split_feat, sparse_dir=sparse_dir)
        if not self.root or reset:
            self.root = node

        # Build left & right sub-trees
        node.left = self.train(left_x, left_y, depth=depth+1, reset=False)
        node.right = self.train(right_x, right_y, depth=depth+1, reset=False)
        # if sparse_dir == 0:
        #     node.sparse_dir = node.left
        # elif sparse_dir == 1:
        #     node.sparse_dir = node.right
        
        return node
    
    def test(self, x):
        """
            Returns predictions y (1-D ndarray of estimates) for each row of x (2-D ndarray of features)
        """

        predictions = []
        
        def dfs(curr, row):
            # print(curr, row)
            if curr.feature == None:
                # Found a leaf
                predictions.append(curr.val)
                return
            if row[curr.feature] == 0 and curr.sparse_dir != None:
                if curr.sparse_dir == 0: dfs(curr.left, row)
                if curr.sparse_dir == 1: dfs(curr.right, row)
            if row[curr.feature] <= curr.val:
                dfs(curr.left, row)
            else:
                dfs(curr.right, row)

        for row in x:
            # print(self.root)
            dfs(self.root, row)

        return np.array(predictions)


class XGBoost:

    def __init__(self, trees=10, quantiles=4, max_depth=10, learning_rate=0.1, lamda=0, gamma=0, sparsity_aware=False):
        self.trees = trees
        self.max_depth = max_depth
        self.quantiles = quantiles
        self.gdbt_model = []
        self.gdbt_predictions = None
        self.learning_rate, self.lamda, self.gamma = learning_rate, lamda, gamma
        self.sparsity_aware = sparsity_aware

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
            weak_model = XGBoostTree(leaf_size=1, max_depth=self.max_depth, lamda=self.lamda, gamma=self.gamma, quantiles=self.quantiles, sparsity_aware=self.sparsity_aware)
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