import pandas as pd
import pyupbit as pb
import numpy as np

access_key='NUTCOjIcnh5q8uFFlmw415XItVxfLOmHz0OJGI9L'
secret_key="YRk9J3hHF5ho6PH690bxwByU4T5QYp1iXmyUmQ58"

upbit=pb.Upbit(access_key,secret_key)

ticker='KRW-BTG'
dataset=pb.get_ohlcv(ticker,'minute15')

dataset_stats=dataset.transpose()