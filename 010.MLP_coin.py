import numpy as np
import pyupbit as pb
import matplotlib.pyplot as plt

df=pb.get_ohlcv('KRW-ETC','day',count=200)

def mean_norm(df_input):
    return df_input.apply(lambda x:(x-df_input.open.mean())/df_input.open.std(),axis=0) #apply:엑셀 스프레드시트처럼 모든 변수에 함수 부여

normed_df=mean_norm(df)[['open','high','low','close']] #데이터 셋 정규화
normed_df['up']=1
normed_df['up'][normed_df.open>normed_df.close]=0

train_df=normed_df.sample(frac=0.9,random_state=0)
test_df=normed_df.drop(train_df.index)

def actf(x): #시그모이드 함수
    return 1/(1+np.exp(-x)) #np.exp : 밑이 e 인 지수함수

def actf_deriv(x): #시그모이드 함수의 미분값
    return x*(1-x)

x=np.array([[train_df.open[i],train_df.high[i],train_df.low[i],1] for i in range(len(train_df))])
y=np.array([[train_df.up[i]] for i in range(len(train_df))])

np.random.seed(0)
inputs=4 #입력층
hiddens=16 #은닉층
outputs=1 #출력층

weight0=2*np.random.random((inputs,hiddens))-1 #3by6
weight1=2*np.random.random((hiddens,outputs))-1


for i in range(20000):
    layer0=x
    # print(layer0)
    net1=np.dot(layer0,weight0)
    # print(net1)
    layer1=actf(net1)
    layer1[:,-1]=1
    net2=np.dot(layer1,weight1)
    layer2=actf(net2)

    layer2_error=layer2-y
    layer2_delta=layer2_error*actf_deriv(layer2) #미분하니깐 식이 확 차이남
    layer1_error=np.dot(layer2_delta,weight1.T)
    layer1_delta=layer1_error*actf_deriv(layer1)

    weight1+=-0.2*np.dot(layer1.T,layer2_delta)
    weight0+=-0.2*np.dot(layer0.T,layer1_delta)
    # print('weight1 :%s,weight2:%s'%(weight1,weight0))

x=np.array([[test_df.open[i],test_df.high[i],test_df.low[i],1] for i in range(len(test_df))])
y=np.array([[test_df.up[i]] for i in range(len(test_df))])
layer0=x
net1=np.dot(layer0,weight0)
layer1=actf(net1)
layer1[:,-1]=1
net2=np.dot(layer1,weight1)
layer2=actf(net2)

# print(layer2)
# print(test_df.up)


# plt.subplot(2,1,1)
# plt.scatter(test_df.index,layer2)
# plt.scatter(test_df.index,test_df.up)

# plt.subplot(2,1,2)
# plt.bar(test_df.index,abs(layer2.reshape(len(test_df.index))-np.array([[i] for i in test_df.up.tolist()]).reshape(len(test_df.index))))
# plt.show()

# print(layer2[0])
# print(len(layer2))

b=[]
for a in range(len(layer2)):
    if abs(layer2[a]-y[a])>0.5:
        b.append(a)
        
c=test_df.iloc[b,:]
print(c)


# plt.subplot(3,1,1)
# plt.scatter(train_df.index,layer2)
# plt.subplot(3,1,2)
# plt.scatter(train_df.index,train_df.up)
# plt.subplot(3,1,3)
# plt.bar(train_df.index,abs(layer2.reshape(80)-np.array([[i] for i in train_df.up.tolist()]).reshape(80)))
# plt.show()


# print(np.array([[i] for i in train_df.up.tolist()]).reshape(80))
