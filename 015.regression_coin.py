# import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import pandas as pd
import pyupbit as pb
import matplotlib.pyplot as plt

ticker="KRW-BTG"
time='day'

df=pb.get_ohlcv(ticker,time)
df['ma7']=df.open.rolling(window=7).mean() #n일 이동평균선
df['std7']=df.open.rolling(window=7).std() #n일 편차
df['max']=df.open/df.high.rolling(window=7).max() # 최대비교값
df['min']=df.open/df.low.rolling(window=7).min() # 최소비교값
df=df.dropna()

# print(df)

def mean_norm(df_input):
    return df_input.apply(lambda x:(x-x.mean())/x.std(),axis=0)
df=mean_norm(df)

# print(df)

X_train_pre=df[['open','high','low','close','volume','value','ma7','std7','max','min']]
y=df['close'].values
z=df.index.values

X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(X_train_pre, y,z, test_size=0.2)

model=Sequential()
model.add(Dense(128,input_dim=X_train.shape[1],activation='relu'))
model.add(Dense(128,input_dim=X_train.shape[1],activation='relu'))
model.add(Dense(1))
# model.summary()

model.compile(optimizer ='adam', loss = 'mean_squared_error')
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=100)
checkpointer = ModelCheckpoint(filepath='h5/regression_coin.h5', monitor='val_loss', verbose=0, save_best_only=True)
history = model.fit(X_train, y_train, validation_split=0.25, epochs=3000, batch_size=32, callbacks=[early_stopping_callback,checkpointer])
# history = model.fit(X_train, y_train, validation_split=0.25, epochs=3000, batch_size=32, callbacks=[early_stopping_callback])

real_prices =[]
pred_prices = []
X_num = []

n_iter = 0
Y_prediction = model.predict(X_test).flatten()
c_mean=pb.get_ohlcv(ticker,time).close.mean()
c_std=pb.get_ohlcv(ticker,time).close.std()
for i in range(25):
    # time_line=pb.get_ohlcv(ticker,time,count=1,to=z_test[i]).close[0]
    # print(pb.get_ohlcv(ticker,time,count=1,to=str(z_test[i])[:9]+str(int(str(z_test[i])[9])+1)).close[0])
    # time_line=z_test[i]
    time_line=str(z_test[i])[:9]+str(int(str(z_test[i])[9])+1)
    real = y_test[i]*c_std+c_mean
    prediction = Y_prediction[i]*c_std+c_mean
    print("타임라인: {},실제가격: {:.3f}, 예상가격: {:.3f}".format(time_line,real, prediction))
    real_prices.append(real)
    pred_prices.append(prediction)
    n_iter = n_iter + 1
    X_num.append(n_iter)

plt.plot(X_num, pred_prices, label='predicted price')
plt.plot(X_num, real_prices, label='real price')
plt.legend()
plt.show()

