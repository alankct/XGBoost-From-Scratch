import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from matplotlib import pyplot as plt
from Models.xg_boost import XGBoost



WINDOW = 12


# for filename in ['3_groups', 'ripple', 'winequality-red', 'winequality-white']:
#     print('---------------------------------')
#     data = np.genfromtxt(f"tree_data/{filename}.csv", delimiter=",")

#     # Shuffle the rows and partition some data for testing.
#     x_train, x_test, y_train, y_test = train_test_split(data[:,:-1], data[:,-1], test_size=0.4)
#     bst = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=.1)
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


# #For window = 6
# best_lr = {'ARE': 0.2, 'ARG': 0.0001, 'BGD': 0.2, 'BIH': 0.2, 'BRA': 0.3, 'BWA': 0.2, 'CHN': 0.001, 'CIV': 0.001, 
#            'COL': 0.001, 'CZE': 0.001, 'EGY': 0.2, 'EST': 0.01, 'GHA': 0.3, 'HRV': 0.3, 'HUN': 0.1, 'IDN': 0.01, 
#            'IND': 0.01, 'JAM': 0.3, 'JOR': 0.0001, 'KAZ': 0.01, 'KEN': 0.0001, 'LKA': 0.001, 'MAR': 0.1, 'MEX': 0.1, 
#            'NGA': 0.001, 'PAK': 0.01, 'PER': 0.0001, 'PHL': 0.2, 'SEN': 0.3, 'SRB': 0.3, 'SVN': 0.001, 'THA': 0.0001, 
#            'TTO': 0.001, 'TUN': 0.3, 'UKR': 0.1, 'VNM': 0.01, 'ZAF': 0.001, 'ZWE': 0.001}

#Calculated best learning rates
best_lr = {'ARE': 0.2, 'ARG': 0.0001, 'BGD': 0.01, 'BIH': 0.1, 'BRA': 0.001, 'BWA': 0.0001, 'CHN': 0.001, 
           'COL': 0.0001, 'CZE': 0.0001, 'EGY': 0.2, 'EST': 0.0001, 'GHA': 0.001, 'HRV': 0.01, 'HUN': 0.3, 'IDN': 0.3, 
           'IND': 0.1, 'JAM': 0.01, 'JOR': 0.1, 'KAZ': 0.0001, 'KEN': 0.0001, 'LKA': 0.001, 'MAR': 0.01, 'MEX': 0.0001, 
           'NGA': 0.01, 'PAK': 0.0001, 'PER': 0.3, 'PHL': 0.0001, 'SEN': 0.01, 'SRB': 0.1, 'SVN': 0.3, 'THA': 0.2, 
           'TTO': 0.0001, 'TUN': 0.01, 'UKR': 0.3, 'VNM': 0.01, 'ZAF': 0.0001, 'ZWE': 0.01}

#List of countries
countries = ['ARE', 'ARG', 'BGD', 'BIH', 'BRA', 'BWA', 'CHN', 'COL', 'CZE', 'EGY', 'EST', 'GHA', 'HRV', 'HUN'
                 , 'IDN', 'IND', 'JAM', 'JOR', 'KAZ', 'KEN', 'LKA', 'MAR', 'MEX', 'NGA', 'PAK', 'PER', 'PHL', 'SEN', 'SRB',
                 'SVN', 'THA', 'TTO', 'TUN', 'UKR', 'VNM', 'ZAF', 'ZWE']

corrs = {
    "In Sample Correlation" : [],
    "Out Of Sample Correlation" : [],
    "In Sample Correlation Ours" : [],
    "Out Of Sample Correlation Ours" : []
}

oos_difference = []
is_difference = []
for filename in countries:
    print('---------------------------------')
    print(filename)
    
    #Cleaning data and calculating year to date returns
    data = np.genfromtxt(f"Data/{filename}.csv", delimiter=",", usecols=np.arange(3, 18), skip_header=True)
    data[:,-1] = data[:,-1]/np.roll(data[:,-1], WINDOW) - 1
    x_train, x_test, y_train, y_test = train_test_split(data[WINDOW-1:,:-1], data[WINDOW-1:,-1], test_size=0.4, shuffle=False)

    learning_rate = best_lr[filename]

    #Testing real XGBoost
    bst = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=learning_rate)
    bst.fit(x_train, y_train)
    y_pred = bst.predict(x_train)
    rmse_is = mean_squared_error(y_train, y_pred, squared=False)
    corr_is = np.corrcoef(y_train, y_pred)[0,1]

    y_pred = bst.predict(x_test)
    rmse_oos = mean_squared_error(y_test, y_pred, squared=False)
    corr_oos = np.corrcoef(y_test, y_pred)[0,1]

    print(f'Testing {filename}.csv for learning rate: {learning_rate} with XGBoost')
    print (f"In sample, RMSE: {rmse_is}, Corr: {corr_is}")
    print (f"Out of sample, RMSE: {rmse_oos}, Corr: {corr_oos}")

    corrs['In Sample Correlation'].append(abs(corr_is))
    corrs['Out Of Sample Correlation'].append(abs(corr_oos))

    #Testing Implemented XGBoost
    lrn = XGBoost(trees=100, max_depth=4, learning_rate=learning_rate, lamda=1, gamma=0, sparsity_aware=True)
    lrn.train(x_train, y_train)
    y_pred = lrn.test(x_test)

    # Test in-sample.
    y_pred, is_errors = lrn.test(x_train, y=y_train, return_errors=True)
    rmse_is_our = mean_squared_error(y_train, y_pred, squared=False)
    corr_is_our = np.corrcoef(y_train, y_pred)[0,1]

    # Test out-of-sample.
    y_pred, os_errors= lrn.test(x_test, y=y_test, return_errors=True)
    rmse_oos_our = mean_squared_error(y_test, y_pred, squared=False)
    corr_oos_our = np.corrcoef(y_test, y_pred)[0,1]

    print(f'Testing {filename}.csv for learning rate: {learning_rate} with Implementation')
    print (f"In sample, RMSE: {rmse_is_our}, Corr: {corr_is_our}")
    print (f"Out of sample, RMSE: {rmse_oos_our}, Corr: {corr_oos_our}")

    corrs['In Sample Correlation Ours'].append(abs(corr_is_our))
    if corr_oos_our != np.nan:
        corrs['Out Of Sample Correlation Ours'].append(abs(corr_oos_our))

    is_difference.append(abs(corr_is_our - corr_is))
    if corr_oos_our != np.nan:
        oos_difference.append(abs(corr_oos_our - corr_oos))
    corrs[filename] = (corr_is, corr_oos)
    
    # # For picking best LR
    # best_lr_oos = 0
    # best_corr_oos = 0
    # for learning_rate in [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]:
    #     bst = XGBRegressor(n_estimators=1000, max_depth=4, learning_rate=learning_rate)
    #     bst.fit(x_train, y_train)
    #     y_pred = bst.predict(x_train)
    #     rmse_is = mean_squared_error(y_train, y_pred, squared=False)
    #     corr_is = np.corrcoef(y_train, y_pred)[0,1]

    #     y_pred = bst.predict(x_test)
    #     rmse_oos = mean_squared_error(y_test, y_pred, squared=False)
    #     corr_oos = np.corrcoef(y_test, y_pred)[0,1]

    #     print(f'Testing {filename}.csv for learning rate: {learning_rate}')
    #     print (f"In sample, RMSE: {rmse_is}, Corr: {corr_is}")
    #     print (f"Out of sample, RMSE: {rmse_oos}, Corr: {corr_oos}")
    #     if abs(corr_oos) > abs(best_corr_oos):
    #         best_lr_oos = learning_rate
    #         best_corr_oos = corr_oos
    # best_lr[filename] = best_lr_oos

#For graphing 
# fig, ax = plt.subplots()

# ax.boxplot([corrs['In Sample Correlation'], corrs['In Sample Correlation Ours'], corrs['Out Of Sample Correlation'], corrs['Out Of Sample Correlation Ours']])
# ax.set_ylim(0,1)
# ax.set_title('Stock Return Prediction Correlations via XGBoost')
# ax.set_ylabel('Correlation')
# ax.set_xticklabels(['IS Native', 'IS Implementation', 'OOS Native', 'OOS Implementation'])
# # fig, ax = plt.subplots(layout='constrained')

# # for attribute, measurement in corrs.items():
# #     offset = width * multiplier
# #     rects = ax.bar(x + offset, measurement, width, label=attribute)
# #     ax.bar_label(rects, padding=3)
# #     multiplier += 1

# # ax.set_ylabel('Correlation')
# # ax.set_title('Country')
# # ax.set_xticks(x + width, countries)
# # ax.legend(loc='upper left', ncols=2)
# # ax.set_ylim(0, 1)

# plt.show()
# print(f'Max difference: IS : {np.max(is_difference)} OOS : {np.max(oos_difference)}')
# print(f'Mean difference: IS : {np.mean(is_difference)} OOS : {np.mean(oos_difference)}')