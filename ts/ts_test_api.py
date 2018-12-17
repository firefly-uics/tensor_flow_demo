import tushare as ts

print(ts.__version__)


df = ts.get_hist_data('600848', start='2018-01', end='2018-02', ktype='60')

print(df)