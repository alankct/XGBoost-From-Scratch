from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np


"""
Can serve as reference
"""
class BootstrapLearner:

    def __init__(self, constituent=None, kwargs={}, bags=0):
        self.constituent = constituent
        self.kwargs = kwargs
        self.bags = bags
        self.all_models = []
    
    def train(self, x, y):
        
        for _ in range(self.bags):
            
            constituent = self.constituent(**self.kwargs)
            
            # Come up with a list of random data points (size len(y)) from 0->len(y)
            rand_indxs = [np.random.randint(len(y)) for _ in range(len(y))]
            rand_indxs = np.array(rand_indxs)
            x_train, y_train = x[rand_indxs, :], y[rand_indxs]

            constituent.train(x_train, y_train)
            self.all_models.append(constituent)
    
    def test(self, x):
        
        predictions = []
        for constituent in self.all_models:
            predictions.append(constituent.test(x))
        
        # Get the mean 1-d vector out of the 2-d predictions matrix
        predictions = np.mean(predictions, axis=0)

        return predictions