import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from Models.xg_boost import XGBoost
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

"""
Testing OUR implementation of XGBOOST
"""
WINDOW = 12
best_lr = {'ARE': 0.2, 'ARG': 0.0001, 'BGD': 0.01, 'BIH': 0.1, 'BRA': 0.001, 'BWA': 0.0001, 'CHN': 0.001, 'CIV': 0.01, 
           'COL': 0.0001, 'CZE': 0.0001, 'EGY': 0.2, 'EST': 0.0001, 'GHA': 0.001, 'HRV': 0.01, 'HUN': 0.3, 'IDN': 0.3, 
           'IND': 0.1, 'JAM': 0.01, 'JOR': 0.1, 'KAZ': 0.0001, 'KEN': 0.0001, 'LKA': 0.001, 'MAR': 0.01, 'MEX': 0.0001, 
           'NGA': 0.01, 'PAK': 0.0001, 'PER': 0.3, 'PHL': 0.0001, 'SEN': 0.01, 'SRB': 0.1, 'SVN': 0.3, 'THA': 0.2, 
           'TTO': 0.0001, 'TUN': 0.01, 'UKR': 0.3, 'VNM': 0.01, 'ZAF': 0.0001, 'ZWE': 0.01}


corrs = {}
corrs_oos = []

if __name__ == '__main__':
    print(f'TESTING XGBOOST')
    for filename in ['ARE', 'ARG', 'BGD', 'BIH', 'BRA', 'BWA', 'CHN', 'CIV', 'COL', 'CZE', 'EGY', 'EST', 'GHA', 'HRV', 'HUN'
                 , 'IDN', 'IND', 'JAM', 'JOR', 'KAZ', 'KEN', 'LKA', 'MAR', 'MEX', 'NGA', 'PAK', 'PER', 'PHL', 'SEN', 'SRB',
                 'SVN', 'THA', 'TTO', 'TUN', 'UKR', 'VNM', 'ZAF', 'ZWE']:               # ['winequality-red', 'winequality-white']
        print('---------------------------------')
        data = np.genfromtxt(f"Data/{filename}.csv", delimiter=",", usecols=np.arange(3, 18), skip_header=True)
        data[:,-1] = data[:,-1]/np.roll(data[:,-1], WINDOW) - 1

        # Shuffle the rows and partition some data for testing.
        x_train, x_test, y_train, y_test = train_test_split(data[WINDOW-1:,:-1], data[WINDOW-1:,-1], test_size=0.4, shuffle=False)

    
        for trees in [100]:
                
                # Construct our learner.
            lrn = XGBoost(trees=trees, max_depth=4, learning_rate=best_lr[filename], lamda=1, gamma=0, sparsity_aware=True)
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
            corrs[filename] = (corr_is, corr_oos)
            corrs_oos.append(abs(corr_oos))
                

            # ax.clear()  # clearing the axes
            # ax.scatter(x,y, s = y, c = 'b', alpha = 0.5)  # creating new scatter chart with updated data
            # fig.canvas.draw()  # forcing the artist to redraw itself

            # plt.plot(is_errors, label='In-Sample')
            # plt.plot(os_errors, label='Out-Of-Sample')
            # plt.title(f'(In Sample vs Out of Sample) Errors for {trees} models, with max_depth {max_depth}')
            # plt.grid()
            # plt.legend()
            # plt.show()
            # # plt.pause(0.1)

            # # Print summary.
            # print(f'Testing {filename}.csv for {trees} number of trees and {max_depth} Max Depth')
            # print (f"In sample RMSE: {rmse_is} Corr: {corr_is}   ——   Out of sample RMSE: {rmse_oos} Corr: {corr_oos}")
            # sys.exit(0)
        print('------------')
print(corrs)