import pyupbit as pb

access_key='rxUYjasRKNRuBwcuxv9ZBxfunCNBgkQI4Dnaj38U'
secret_key="xELGSbu1BcUXNfGRbxg6QLmgL5cFlnNLZx0qViBQ"

upbit=pb.Upbit(access_key,secret_key)

ticker_list=pb.get_tickers('KRW')
my_ticker_list=[]
goal_ticker_list=list(set(ticker_list)-set(my_ticker_list))

a=upbit.get_balances()
for i in range(len(a)):
    my_ticker_list.append(a[i].get('currency'))

# print(my_ticker_list)
print(goal_ticker_list)

ticker=goal_ticker_list[0]

def buy_all(ticker_list,k):
    for i in range(len(ticker_list)):
        upbit.buy_market_order(ticker_list[i],k)

krw=upbit.get_balance('KRW')
