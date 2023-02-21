import pyupbit as pb
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

ticker='KRW-BTG'
time='minute10'
interval=20
k=2

df=pb.get_ohlcv(ticker,time)
df=df[['open','high','low','close']]
df['ma%s' %interval]=df.open.rolling(window=interval).mean()
df['stddev']=df.open.rolling(window=interval).std()
df['upper'],df['under']=df['ma%s' %interval]+k*df['stddev'],df['ma%s' %interval]-k*df['stddev']
df['bb']=(df.open-df.under)/(df.upper-df.under)
df=df.dropna()
# print(df)

def mean_norm(df_input):
    return df_input.apply(lambda x:x/x.open,axis=1)

df['tomorrow']=(df.open-df.close).apply(lambda x:1 if x>0 else 0).shift(-1) # Y 데이터 셋
df=df.dropna()

test_y=df['tomorrow'].values

test_df=mean_norm(df[['open','high','low','close']])
test_df=pd.concat([test_df[['high','low','close']],df.bb],axis=1) # x 데이터 셋
# print(test_df)

X_train,X_test,y_train,y_test=train_test_split(test_df,test_y,test_size=0.1)
# X_train,X_test,y_train,y_test=train_test_split(df[['bb']],test_y,test_size=0.1)

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(32,input_dim=X_train.shape[1],activation='sigmoid'))
# model.add(tf.keras.layers.Dense(16,input_dim=test_df.shape[1],activation='sigmoid'))
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
# model.summary()

model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(X_train,y_train,epochs=200)

# print(X_train)
# print(y_train)

y_prediction=model.predict(X_test)
# print(pd.DataFrame(y_prediction))
# print(pd.DataFrame(y_test))
result=pd.concat([pd.DataFrame(y_prediction,columns=['pred']),pd.DataFrame(y_test,columns=['test'])],axis=1)
result['result']=result.apply(lambda x:True if abs(x.test-x.pred)<=0.5 else False,axis=1)
print(result)

# plt.subplot(2,1,1)




# bb와 현재가 고가 저가 종가를 가지고 다음날 종가가 상승마감인지, 하락마감인지 예측해 보자. (sigmoid)


