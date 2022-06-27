#!/usr/bin/env python

import os
import time
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, confusion_matrix, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from keras import backend as K
from math import sqrt
from keras import optimizers
from keras.models import Model
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, Flatten,TimeDistributed,Bidirectional, RepeatVector
from keras.layers.convolutional import Conv1D, MaxPooling1D
from sklearn.preprocessing import MinMaxScaler
from sklearn.inspection import permutation_importance


# # Path seeting
os.chdir('Your dir')

# # Read Data
data = pd.read_csv('file name')
# # print(df.head)
values = data.values

# scaled data
scaler = MinMaxScaler(feature_range=(0, 1))
values_scaled = scaler.fit_transform(values)
# test_scaled = scaler.fit_transform(test_data)

# # Data reshape: all (-t)
# def pre_data(values, n_time, n_feature, time_lag):
#     v = values.shape[0] - n_time - time_lag + 1
#     ozone = np.zeros([v, 1])
#     others = np.zeros([v, n_time * n_feature])
#     for i in range(v):
#         others[i, :] = values[i:i+n_time, :].reshape(n_time*n_feature, order='C')
#     for i in range(v):
#         ozone[i] = values[i+n_time+time_lag-1, 0]
#     X = np.hstack([others, ozone])
#     return X

# # Data reshape
def pre_data(values, n_time, n_feature, time_lag):
    v = values.shape[0] - n_time - time_lag
    X = np.zeros([v,n_time*n_feature+1])
    for i in range(v):
        for j in range(n_time):
            X[i,j*n_feature] = values[i+j,0]
            X[i,j*n_feature+1] = values[i+j,1]
            X[i,j*n_feature+2:j*n_feature+8] = values[i+j+time_lag,2:8]
        X[i,-1] = values[i+n_time+time_lag-1, 0]
    return X



# split into train and test sets
def split_data(X,n_time,n_feature,test_size):
    train = X[:-test_size, :]
    test = X[-test_size:,:]
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # # reshape input to be 3D [samples, timesteps, features]
    # train_X = train_X.reshape((-1, n_time, n_feature))
    # test_X = test_X.reshape((-1, n_time, n_feature))
    return train_X, train_y, test_X, test_y


def fit_model(train_X, train_y, test_X, test_y):
    # design network
    model = Sequential()
    # model.add(Conv1D(filters=100, kernel_size=3, activation='relu', input_shape=(train_X.shape[1], train_X.shape[2])))
    # model.add(Conv1D(filters=100, kernel_size=3, activation='relu', ))
    # # # model.add(Conv1D(filters=100, kernel_size=3, activation='relu', ))
    # # # model.add(MaxPooling1D(pool_size=2))
    # # # model.add(Flatten())
    # model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True, activation='relu'),)
    # # # model.add(Dropout(0.2))
    # model.add(LSTM(100,  activation='relu'), )
    # model.add(Dropout(0.2))
    # model.add(Bidirectional(LSTM(100, return_sequences=True,activation='relu'), input_shape=(train_X.shape[1], train_X.shape[2])),)
    # model.add(Bidirectional(LSTM(100, return_sequences=True,activation='relu'), ),)
    # model.add(Bidirectional(LSTM(100, return_sequences=True,activation='relu'), ),)
    # model.add(Bidirectional(LSTM(100, activation='relu'), ),)
    model.add(Dense(100,activation = 'relu',input_shape=(train_X.shape[1],)))
    # model.add(Dense(100,activation = 'relu'))
    # model.add(Dense(100,activation = 'relu'))
    model.add(Dense(100,activation = 'relu'))
    model.add(Dense(1))
    model.compile(loss=e_weight_mse, optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=50, batch_size=128,
                        validation_data=(test_X, test_y), verbose=2, shuffle=False)
    # make a prediction
    pred_y_train = model.predict(train_X)
    pred_y = model.predict(test_X)
    return history, pred_y_train, pred_y, model

# # Weighted loss function
def e_weight_mse(y_true, y_pred):
    return K.mean(K.exp(3  * y_true) * K.square(y_pred - y_true), axis=-1)

# transform regression into classification function
def  rg_2_cl (x, thre):
    y = copy.deepcopy(x)
    for i in range(y.shape[0]):
        if (y[i] < thre):
            y[i] = 0
        else:
            y[i] = 1
    return y


# set the parameter
n_time = 5
n_feature = 8
time_lag = 3
test_size = 365

# prepare the data
X = pre_data(values_scaled, n_time, n_feature, time_lag)  # The normalized data

# split the data
train_X, train_y, test_X, test_y = split_data(X, n_time, n_feature, test_size)

# # Model training
start = time.time()

# history, pred_y_train, pred_y, model = fit_model(train_X, train_y, test_X, test_y)
# print('NN finished')

# forest = RandomForestRegressor()
# model_rf = forest.fit(train_X,train_y)
# pred_y = model_rf.predict(test_X)
# print('RF FINISH')

end = time.time()
print('time:' + str(end - start))
# model.summary()

# # save the model
# model.save('multi_station_model.h5')
# model.save('regional_model_lstm.h5')

# # Plot Loss curve
# train_test_error
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='test')
# plt.title('Leanrning curve of MLP (with future meteo)')
# plt.legend()
# plt.show()

# # Plot Feature importance (All feature)

# # Bar plot
# feature_names = ['O(t-4)','NOx(t-4)','T(t-3)','SWR(t-3)','RAIN(t-3)','RH(t-3)','WS(t-3)','WD(t-3)',
# 'O(t-3)','NOx(t-3)','T(t-2)','SWR(t-2)','RAIN(t-2)','RH(t-2)','WS(t-2)','WD(t-2)',
# 'O(t-2)','NOx(t-2)','T(t-1)','SWR(t-1)','RAIN(t-1)','RH(t-1)','WS(t-1)','WD(t-1)',
# 'O(t-1)','NOx(t-1)','T(t)','SWR(t)','RAIN(t)','RH(t)','WS(t)','WD(t)',
# 'O(t)','NOx(t)','T(t+1)','SWR(t+1)','RAIN(t+1)','RH(t+1)','WS(t+1)','WD(t+1)',]

# result = permutation_importance(model, test_X, test_y,scoring='neg_root_mean_squared_error', n_repeats=10,random_state=0)
# forest_importances = pd.Series(result.importances_mean, index=feature_names)

# fig, ax = plt.subplots()
# forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
# ax.set_title("Feature importances using permutation on full model")
# ax.set_ylabel("Mean accuracy decrease")
# fig.tight_layout()
# plt.show()

# # Barh plot (Top 20)

# feature_names = ['O(t-4)','NOx(t-4)','T(t-3)','SWR(t-3)','RAIN(t-3)','RH(t-3)','WS(t-3)','WD(t-3)',
# 'O(t-3)','NOx(t-3)','T(t-2)','SWR(t-2)','RAIN(t-2)','RH(t-2)','WS(t-2)','WD(t-2)',
# 'O(t-2)','NOx(t-2)','T(t-1)','SWR(t-1)','RAIN(t-1)','RH(t-1)','WS(t-1)','WD(t-1)',
# 'O(t-1)','NOx(t-1)','T(t)','SWR(t)','RAIN(t)','RH(t)','WS(t)','WD(t)',
# 'O(t)','NOx(t)','T(t+1)','SWR(t+1)','RAIN(t+1)','RH(t+1)','WS(t+1)','WD(t+1)',]
# feature_names = np.r_[feature_names]

# result = permutation_importance(model, test_X, test_y,scoring='neg_root_mean_squared_error', n_repeats=10,random_state=0)
# sorted_idx = result.importances_mean.argsort()

# print(pd.Series(result.importances_mean, index=feature_names))

# y_ticks = np.arange(0, len(feature_names))
# fig, ax = plt.subplots()
# ax.barh(y_ticks[-20:], result.importances_mean[sorted_idx][-20:])
# ax.set_yticks(y_ticks[-20:])
# ax.set_yticklabels(feature_names[sorted_idx][-20:])
# # ax.set_title("Feature importances using Permutation Importances")
# fig.tight_layout()
# plt.show()
