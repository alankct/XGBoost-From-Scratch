import numpy as np
import pandas as pd

"""
Simple model, same as before (I added a depth parameter for class and reset for train)
"""

class Node:
    def __init__(self, val=0, feature=None, left=None, right=None):
        self.val = val
        self.feature = feature
        self.left = left
        self.right = right

    def __str__(self):
        return f'Node val {self.val}, split feature: {self.feature}, right: {self.right is not None}, left {self.left is not None}'

class CARTLearner:

    def __init__(self, leaf_size=1, max_depth=float('inf')):
        self.root = None
        self.leaf_size = leaf_size
        self.max_depth = max_depth
    
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

    def train(self, x, y, depth=0, reset=True):
        """
            Induce a decision tree based on this training data
                @params:
                    x is a 2-D ndarray where columns are the X features and each row is a different training example
                    y is a 1-D ndarray with the matching labels/answers in the same order as the X training examples
        """
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
        """
            Returns predictions y (1-D ndarray of estimates) for each row of x (2-D ndarray of features)
        """

        predictions = []
        
        def dfs(curr, row):
            try:
                if curr.feature == None:
                    # Found a leaf
                    predictions.append(curr.val)
                    return
                if row[curr.feature] <= curr.val:
                    dfs(curr.left, row)
                else:
                    dfs(curr.right, row)
            except Exception as e:
                print(e)
                print(f'Predictions so far: {predictions}')
                print(f'Root: {self.root}')

        for row in x:
            dfs(self.root, row)

        return np.array(predictions)



