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
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from sklearn.preprocessing import MinMaxScaler
from sklearn.inspection import permutation_importance

# # Read data
data = pd.read_csv('xxx.csv')
# # print(df.head)
values = data.values

# scaled data
scaler = MinMaxScaler(feature_range=(0, 1))
values_scaled = scaler.fit_transform(values)
# test_scaled = scaler.fit_transform(test_data)

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

def e_weight_mse(y_true, y_pred):
    return K.mean(K.exp(3  * y_true) * K.square(y_pred - y_true), axis=-1)

def fit_tl_model(reset_model, train_X, train_y, test_X, test_y):
    model = Sequential([
        reset_model,
        # LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True, activation='relu'),
        # # Dropout(0.2),
        # # LSTM(100,  return_sequences=True, activation='relu'),
        # # Dropout(0.2),
        # LSTM(100,  return_sequences=True, activation='relu'),
        # # Dropout(0.2),
        # LSTM(100,  return_sequences=True, activation='relu'),
        # # Dropout(0.2),
        # LSTM(100, activation='relu'),
        # Dropout(0.2),
        # Bidirectional(LSTM(100, return_sequences=True,activation='relu'), input_shape=(train_X.shape[1], train_X.shape[2])),
        # Bidirectional(LSTM(60, return_sequences=True,activation='relu'), ),
        # Bidirectional(LSTM(60, activation='relu'), ),
        # Dropout(0.2),
        Dense(100,activation = 'relu'),
        Dense(100,activation = 'relu'),
        Dense(100,activation = 'relu'),
        Dense(100,activation = 'relu'),
        Dense(1),
    ])
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss=e_weight_mse, optimizer=adam)
    history = model.fit(train_X, train_y, epochs=40, batch_size=256,
                        validation_data=(test_X, test_y), verbose=2, shuffle=False)
    pred_y = model.predict(test_X)
    pred_y_train = model.predict(train_X)
    return history, pred_y_train, pred_y, model

def load_reset_model(filename, layername):
    base_model = load_model(filename)
    base_model.summary()
    reset_model = Model(inputs=base_model.input, outputs=base_model.get_layer(layername).output)
    reset_model.summary()
    reset_model.trainable = False
    return reset_model

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
time_lag = 1
test_size = 365

# prepare the data
X = pre_data(values_scaled, n_time, n_feature, time_lag)  # The normalized data

# split the data
train_X, train_y, test_X, test_y = split_data(X, n_time, n_feature, test_size)

# Model training
start = time.time()

# 1-station model (CNN-LSTM/ S-BLSTM)
# history, pred_y_train, pred_y, model = fit_model(train_X, train_y, test_X, test_y)

# Original model
# model = load_model('multi_station_model_1.h5')
# model.summary()

# SPTL model
# reset_model = load_reset_model(model_name, 'dense_2') 
# history, pred_y_train, pred_y, model = fit_tl_model(reset_model, train_X, train_y, test_X, test_y)

# Random Forest
forest = RandomForestRegressor()
model_rf = forest.fit(train_X,train_y)
pred_y = model_rf.predict(test_X)
# print('RF FINISH')

end = time.time()
print('time:' + str(end - start))
# model.summary()

# # Plot Loss curve
# train_test_error
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='test')
# plt.title('Leanrning curve of MLP (with future meteo)')
# plt.legend()
# plt.show()
