import numpy as np
import pandas as pd

class Node:
    def __init__(self, val=0, feature=None, left=None, right=None):
        self.val = val
        self.feature = feature
        self.left = left
        self.right = right
    
    def __str__(self):
        return f'Value: {self.val}, Feature: {self.feature}, L: [{self.left is not None}], R: [{self.right is not None}]'

class CARTLearner:

    def __init__(self, leaf_size=1):
        # set up your object
        self.root = None
        self.leaf_size = leaf_size
    
    def find_split_feature(self, x, y):
        best_feature, best_correlation = None, -float("inf")

        # Transpose in order to have rows be the feautures (columns), and run corrcoef with y
        x = x.T
        for i in range(len(x)):
            if not np.all(x[i,:] == x[i,0]):
                correlation = np.abs(np.corrcoef(x[i], y)[0,-1])
                if correlation > best_correlation:
                    best_feature, best_correlation = i, correlation
        return best_feature

    def train(self, x, y):
        """
            Induce a decision tree based on this training data
                @params:
                    x is a 2-D ndarray where columns are the X features and each row is a different training example
                    y is a 1-D ndarray with the matching labels/answers in the same order as the X training examples
        """
        if np.all(y == y[0]):
            # All y values are the same, returns a leaf with that value
            return Node(y[0])
        if np.all(x==x[0,:]) or len(y) <= self.leaf_size:
            # np.all(x == x[0, 0]) or If it is not possible to split (all X values same), or if we reach a small sub-set, return a leaf
            return Node(np.mean(y))

        # Determine best x-value (feature) to split on
        feature_index = self.find_split_feature(x, y)

        # Split on the median value for the selected feature
        median = np.median(x[:, feature_index])
        less_than_median = x[:, feature_index] <= median
        left_x, right_x = x[less_than_median], x[~less_than_median]
        left_y, right_y = y[less_than_median], y[~less_than_median]

        if len(x) == len(left_x) or len(x) == len(right_x):
            # The median did not split the arrays, so we return a leaf
            return Node(np.mean(y))

        # Create a decision node
        node = Node(median, feature=feature_index)
        if not self.root:
            self.root = node

        # Build left & right sub-trees
        node.left = self.train(left_x, left_y)
        node.right = self.train(right_x, right_y)
        
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

        return predictions



