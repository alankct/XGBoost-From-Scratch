import numpy as np

class Node:
    def __init__(self, val=0, feature=None, left=None, right=None):
        self.val = val
        self.feature = feature
        self.left = left
        self.right = right

class PERTLearner:

    def __init__(self, leaf_size=1, max_depth=1000):
        self.root = None
        self.leaf_size = leaf_size
        self.max_depth = max_depth
    
    def train(self, x, y, depth=0, reset=False):
        """
            Induce a decision tree based on this training data
                @params:
                    x is a 2-D ndarray where columns are the X features and each row is a different training example
                    y is a 1-D ndarray with the matching labels/answers in the same order as the X training examples
        """

        # Base Cases: All y-values are the same or we reached leaf_size/max_depth, so we return a leaf
        if np.all(y == y[0]): return Node(y[0])
        if np.all((x==x[0,:])) or len(y) <= self.leaf_size or depth > self.max_depth: return Node(np.mean(y))

        for i in range(10):

            # Pick random data points that have different y-values
            i, j = 0, 0
            while y[i] == y[j]:
                i, j = np.random.randint(len(y)), np.random.randint(len(y))
            
            # Pick random split feature and split value
            split_feat_i = np.random.randint(len(x[0]))
            random_lerp = np.random.random() # 0 to 1
            split_val = (random_lerp * x[i, split_feat_i]) + ((1-random_lerp) * x[j, split_feat_i])
            
            # Split the data on the random feature/value to create left & right children
            mask = x[:, split_feat_i] <= split_val
            left_x, left_y = x[mask], y[mask]
            right_x, right_y = x[~mask], y[~mask]

            if len(left_y) != len(y) and len(right_y) != len(y):
                node = Node(split_val, split_feat_i)
                if not self.root:
                    self.root = node
                node.left = self.train(left_x, left_y, depth=depth+1, reset=False)
                node.right = self.train(right_x, right_y, depth=depth+1, reset=False)
                return node

        return Node(np.mean(y)) # Failed too many times, return a leaf
    
    def test(self, x):
        """
            Returns predictions y (1-D ndarray of estimates) for each row of x (2-D ndarray of features)
        """

        predictions = []

        def dfs(node, row):
            if node.feature == None:
                # Found a leaf
                predictions.append(node.val)
                return
            if row[node.feature] <= node.val:
                dfs(node.left, row)
            else:
                dfs(node.right, row)
              
        for row in x:
            dfs(self.root, row)
        
        return predictions