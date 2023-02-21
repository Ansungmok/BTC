import pyupbit as pb
import pandas as pd
import tensorflow as tf
import numpy as np
import datetime

ticker='KRW-OMG'
minute='minute30'
to=datetime.datetime.now()
# df=pb.get_ohlcv(ticker,minute)
# df=df[['close','open','high','low','volume']].reset_index(drop=True)
df=pd.concat([pb.get_ohlcv(ticker,'day'),pb.get_ohlcv(ticker,'day',to=to-datetime.timedelta(days=200))]).sort_index(ascending=True)
df=df[['close','open','high','low','volume']].reset_index(drop=True)

x_data=df[['open','high','low']]
x_list=[]
for i in range(len(x_data)):
   x_list.append(x_data.loc[i].tolist())
y_list=df.close.tolist()

a=tf.Variable(0.1)
b=tf.Variable(0.1)

def loss_function():
   return a*sum(x_list[0])/3+b

opt=tf.keras.optimizers.Adam(learning_rate=0.001)
for i in range(300):
   opt.minimize(loss_function,var_list=[a,b])
   print(a.numpy(),b.numpy())