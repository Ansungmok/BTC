import numpy as np
import pyupbit as pb
import matplotlib.pyplot as plt
import tensorflow as tf

df=pb.get_ohlcv('KRW-ETC','day',count=200)

def mean_norm(df_input):
    return df_input.apply(lambda x:(x-df_input.open.mean())/df_input.open.std(),axis=0) #apply:엑셀 스프레드시트처럼 모든 변수에 함수 부여

normed_df=mean_norm(df)[['open','high','low','close']] #데이터 셋 정규화
normed_df['up']=1
normed_df['up'][normed_df.open>normed_df.close]=0
train_df=normed_df.sample(frac=0.9,random_state=0)
test_df=normed_df.drop(train_df.index)

train_x=np.array([[train_df.open[i],train_df.high[i],train_df.low[i]] for i in range(len(train_df))])
train_y=np.array([[train_df.up[i]] for i in range(len(train_df))])

test_x=np.array([[test_df.open[i],test_df.high[i],test_df.low[i]] for i in range(len(test_df))])
test_y=np.array([[test_df.up[i]] for i in range(len(test_df))])

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(8,input_dim=3,activation='sigmoid'))
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

# sgd=tf.keras.optimizers.SGD(lr=0.1)
model.compile(loss='mean_squared_error',optimizer='adam')

early_stop=tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

model.fit(train_x,train_y,epochs=200,callbacks=[early_stop])
model.evaluate(test_x,test_y)

plt.scatter(test_df.index,model.predict(test_x))
plt.scatter(test_df.index,test_df.up)
plt.show()

