import numpy as np

class Node:
    def __init__(self, col, value=None, left = None, right= None):
        self.value = value
        self.right = right
        self.left = left
        self.col = col
    
    def setLChild(self, left):
        self.left = left
    
    def setRChild(self, right):
        self.right = right

class CARTLearner:

    def __init__(self, leaf_size=1):
        # set up your object
        self.leaf_size = leaf_size
        self.parent = None
        return
    
    def best_feature(self, x,y):
        bestCorr = -2**32
        feat = None
        for i in range(x.T.shape[0]):
            if not np.all(x.T[i] == x.T[i,0]):
                corr = np.corrcoef(x.T[i], y)[0,1]
                if (np.absolute(corr) > bestCorr):
                    bestCorr = np.absolute(corr)
                    feat = i
        return feat

        
    def build_tree(self, x, y):
        if np.all(y==y[0]):
            leaf = Node(col = None, value = y[0])
            return leaf
        elif np.all(x==x[0]) or (np.shape(x)[0] <= self.leaf_size):
            leaf = Node(col = None, value = np.mean(y))
            return leaf
        maxCorr = self.best_feature(x,y)
        median = np.median(x[:,maxCorr])
        lessEqSet = x[x[:,maxCorr] <= median]
        lessY = y[x[:,maxCorr] <= median]
        greatSet = x[x[:,maxCorr] > median]
        moreY = y[x[:,maxCorr] > median]
        if np.any(lessEqSet) and np.any(greatSet):
            rNode = self.build_tree(greatSet, moreY)
            lNode = self.build_tree(lessEqSet, lessY)
            parentNode = Node(maxCorr, value = median, left = lNode, right = rNode)
            return parentNode
        else:
            leaf = Node(col = None, value = np.mean(y))
            return leaf 
    
    def train(self, x, y):
        self.parent = self.build_tree(x,y)
        return

    def test(self, x):
        predictions = []
        for row in x:
            currNode = self.parent
            while(currNode.left != None):
                if row[currNode.col] <= currNode.value:
                    currNode = currNode.left
                else:
                    currNode = currNode.right
            predictions.append(currNode.value)
        return predictions
