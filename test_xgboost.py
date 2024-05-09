import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xg_boost import XGBoost
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

"""
Testing OUR implementation of XGBOOST
"""

if __name__ == '__main__':
    print(f'TESTING XGBOOST')
    for filename in ['winequality-red', 'winequality-white']:
        print('---------------------------------')
        data = np.genfromtxt(f"tree_data/{filename}.csv", delimiter=",")

        # Shuffle the rows and partition some data for testing.
        x_train, x_test, y_train, y_test = train_test_split(data[:,:-1], data[:,-1], test_size=0.4)


        for max_depth in range(3, 7):
            # fig, ax = plt.subplots()
            for trees in range(20, 101, 20):
                
                # Construct our learner.
                lrn = XGBoost(trees=trees, max_depth=max_depth, learning_rate=0.1, lamda=0, gamma=0)
                lrn.train(x_train, y_train)
                y_pred = lrn.test(x_test)

                # Test in-sample.
                y_pred, is_errors = lrn.test(x_train, y=y_train, return_errors=True)
                rmse_is = mean_squared_error(y_train, y_pred, squared=False)
                corr_is = np.corrcoef(y_train, y_pred)[0,1]

                # Test out-of-sample.
                y_pred, os_errors= lrn.test(x_test, y=y_test, return_errors=True)
                rmse_oos = mean_squared_error(y_test, y_pred, squared=False)
                corr_oos = np.corrcoef(y_test, y_pred)[0,1]
                

                # ax.clear()  # clearing the axes
                # ax.scatter(x,y, s = y, c = 'b', alpha = 0.5)  # creating new scatter chart with updated data
                # fig.canvas.draw()  # forcing the artist to redraw itself

                plt.plot(is_errors, label='In-Sample')
                plt.plot(os_errors, label='Out-Of-Sample')
                plt.title(f'(In Sample vs Out of Sample) Errors for {trees} models, with max_depth {max_depth}')
                plt.grid()
                plt.legend()
                plt.pause(0.1)

                # Print summary.
                print(f'Testing {filename}.csv for 100 number of {trees} and {max_depth} Max Depth')
                print (f"In sample RMSE: {rmse_is} Corr: {corr_is}   ——   Out of sample RMSE: {rmse_oos} Corr: {corr_oos}")
            plt.show()
            print('------------')


"""
Problems:

    0. Fix None Tree Prediction
    1. Too costly of an operation (quantiles every time)
    2. Less trees? Less Max Depth? Play around with hyper-parameters
    3. Play around with the lamda/gamma values (does outputing a Null node work?)
"""