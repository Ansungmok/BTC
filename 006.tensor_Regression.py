import pandas as pd
import pyupbit as pb
import numpy as np

access_key='your_access_key'
secret_key="your_secret_key"

upbit=pb.Upbit(access_key,secret_key)

ticker='KRW-BTG'
dataset=pb.get_ohlcv(ticker,'minute15')

dataset_stats=dataset.transpose()
