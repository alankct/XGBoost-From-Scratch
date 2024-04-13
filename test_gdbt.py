import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from gbdt import GBDT

"""
Testing function—can test the error for GBDT/XGBOOST like this
"""

# Load the csv file and remove the header row and date column. '3_groups', 'Istanbul', 'ripple', 
for filename in ['winequality-red', 'winequality-white']:
    print('---------------------------------')
    data = np.genfromtxt(f"tree_data/{filename}.csv", delimiter=",")[1:,1:]

    # Shuffle the rows and partition some data for testing.
    x_train, x_test, y_train, y_test = train_test_split(data[:,:-1], data[:,-1], test_size=0.4)

    for trees in [1000]:
        # Construct our learner.
        lrn = GBDT(trees=trees, max_depth=6)
        lrn.train(x_train, y_train)
        y_pred = lrn.test(x_test)

        # Test in-sample.
        y_pred = lrn.test(x_train)
        rmse_is = mean_squared_error(y_train, y_pred, squared=False)
        corr_is = np.corrcoef(y_train, y_pred)[0,1]

        plt.plot(y_train, label='Ground Truth')
        plt.plot(y_pred, label='Pred')
        plt.legend()
        plt.title('In Sample Pred vs Ground Truth')
        plt.show()

        # Test out-of-sample.
        y_pred = lrn.test(x_test)
        rmse_oos = mean_squared_error(y_test, y_pred, squared=False)
        corr_oos = np.corrcoef(y_test, y_pred)[0,1]

        plt.plot(y_test, label='Ground Truth')
        plt.plot(y_pred, label='Pred')
        plt.title('In Sample Pred vs Ground Truth')
        plt.legend()
        plt.show()

        # Print summary.
        print(f'Testing {filename}.csv for {trees} number of trees')
        print (f"In sample, RMSE: {rmse_is}, Corr: {corr_is}")
        print (f"Out of sample, RMSE: {rmse_oos}, Corr: {corr_oos}")


