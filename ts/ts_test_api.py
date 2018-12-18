import tushare as ts

print(ts.__version__)

ts.set_token('f4af65a5b4c12c6a753fa15b019a253c16ddf6ef48d9db5d17d8c0d3')

pro = ts.pro_api()

df = pro.daily(ts_code='000001.SZ')

print(df)

# data = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
#
# print(data)