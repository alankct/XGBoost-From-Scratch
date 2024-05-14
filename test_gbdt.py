import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from Models.gbdt import GBDT
import pandas as pd
import matplotlib.pyplot as plt

"""
Testing function—can test the error for GBDT/XGBOOST like this
"""

if __name__ == '__main__':
    print(f'TESTING GBDT')
    for filename in ['winequality-red', 'winequality-white']:
        print('---------------------------------')
        data = np.genfromtxt(f"tree_data/{filename}.csv", delimiter=",")

        # Shuffle the rows and partition some data for testing.
        x_train, x_test, y_train, y_test = train_test_split(data[:,:-1], data[:,-1], test_size=0.4)


        for max_depth in [3,6,9]:
            #in_sample = pd.DataFrame(index=range(100,2000))
            #out_of_sample = pd.DataFrame(index=range(100,2000))
            for trees in [100]:
                
                # Construct our learner.
                lrn = GBDT(trees=trees, max_depth=max_depth, learning_rate=0.1)
                lrn.train(x_train, y_train)
                y_pred = lrn.test(x_test)

                # Test in-sample.
                y_pred, is_errors = lrn.test(x_train, y=y_train, return_errors=True)
                rmse_is = mean_squared_error(y_train, y_pred, squared=False)
                corr_is = np.corrcoef(y_train, y_pred)[0,1]
                #in_sample.iloc[trees, 0] = rmse_is
                # Test out-of-sample.
                y_pred, os_errors= lrn.test(x_test, y=y_test, return_errors=True)
                rmse_oos = mean_squared_error(y_test, y_pred, squared=False)
                corr_oos = np.corrcoef(y_test, y_pred)[0,1]
                #out_of_sample.iloc[trees,0] = rmse_oos

                plt.plot(is_errors, label='In-Sample')
                plt.plot(os_errors, label='Out-Of-Sample')
                plt.title(f'(In Sample vs Out of Sample) Errors for 100 models, with max_depth {max_depth}')
                plt.grid()
                plt.legend()
                plt.show()

                # Print summary.
                print(f'Testing {filename}.csv for 100 number of trees and {max_depth} Max Depth')
                print (f"In sample, RMSE: {rmse_is}, Corr: {corr_is}")
                print (f"Out of sample, RMSE: {rmse_oos}, Corr: {corr_oos}")
        break


