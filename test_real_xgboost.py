import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder


WINDOW = 6
# for filename in ['3_groups', 'ripple', 'winequality-red', 'winequality-white']:
#     print('---------------------------------')
#     data = np.genfromtxt(f"tree_data/{filename}.csv", delimiter=",")

#     # Shuffle the rows and partition some data for testing.
#     x_train, x_test, y_train, y_test = train_test_split(data[:,:-1], data[:,-1], test_size=0.4)
#     bst = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=.3)
#     bst.fit(x_train, y_train)
#     y_pred = bst.predict(x_train)
#     rmse_is = mean_squared_error(y_train, y_pred, squared=False)
#     corr_is = np.corrcoef(y_train, y_pred)[0,1]

#     y_pred = bst.predict(x_test)
#     rmse_oos = mean_squared_error(y_test, y_pred, squared=False)
#     corr_oos = np.corrcoef(y_test, y_pred)[0,1]

#     print(f'Testing {filename}.csv')
#     print (f"In sample, RMSE: {rmse_is}, Corr: {corr_is}")
#     print (f"Out of sample, RMSE: {rmse_oos}, Corr: {corr_oos}")



for filename in ['ARE', 'ARG', 'BGD', 'BIH', 'BRA', 'BWA', 'CHN', 'CIV', 'COL', 'CZE', 'EGY', 'EST', 'GHA', 'HRV', 'HUN'
                 , 'IDN', 'IND', 'JAM', 'JOR', 'KAZ', 'KEN', 'LKA', 'MAR', 'MEX', 'NGA', 'PAK', 'PER', 'PHL', 'SEN', 'SRB',
                 'SVN', 'THA', 'TTO', 'TUN', 'UKR', 'VNM', 'ZAF', 'ZWE']:
    print('---------------------------------')
    print(filename)
    data = np.genfromtxt(f"Data/{filename}.csv", delimiter=",", usecols=np.arange(3, 18), skip_header=True)
    data[:,-1] = data[:,-1]/np.roll(data[:,-1], WINDOW) - 1
    # data = data/np.roll(data, WINDOW) - 1
    x_train, x_test, y_train, y_test = train_test_split(data[WINDOW-1:,:-1], data[WINDOW-1:,-1], test_size=0.4, shuffle=False)
    # y_train = le.fit_transform(y_train)
    # y_test = le.fit_transform(y_test)
    for learning_rate in [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]:
        bst = XGBRegressor(n_estimators=1000, max_depth=4, learning_rate=learning_rate)
        bst.fit(x_train, y_train)
        y_pred = bst.predict(x_train)
        rmse_is = mean_squared_error(y_train, y_pred, squared=False)
        corr_is = np.corrcoef(y_train, y_pred)[0,1]

        y_pred = bst.predict(x_test)
        rmse_oos = mean_squared_error(y_test, y_pred, squared=False)
        corr_oos = np.corrcoef(y_test, y_pred)[0,1]

        print(f'Testing {filename}.csv for learning rate: {learning_rate}')
        print (f"In sample, RMSE: {rmse_is}, Corr: {corr_is}")
        print (f"Out of sample, RMSE: {rmse_oos}, Corr: {corr_oos}")
    