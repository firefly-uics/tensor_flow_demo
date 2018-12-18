import tushare as ts
from sqlalchemy import create_engine


ts.set_token('f4af65a5b4c12c6a753fa15b019a253c16ddf6ef48d9db5d17d8c0d3')
pro = ts.pro_api()
engine = create_engine("mysql+pymysql://root:root@127.0.0.1:3306/ts", max_overflow=5)

def to_db(df, table_name):
    df.to_sql(table_name, engine,if_exists='append')

def to_stock_basic():
    data = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
    to_db(data, 'stock_basic')

def to_daily(code):
    df = pro.daily(ts_code=code, start_date='20180101')
    to_db(df, 'daily')

def to_hist_data(code):
    df = ts.get_hist_data(code, start='2018-{}-{}'.format('01', '01'), ktype='60')
    df.index.astype('str')
    to_db(df, 'hist_data_60_' + code)


if __name__ == '__main__':
    # to_daily('000001.SZ')
    # to_stock_basic();
    # df = ts.pro_bar(pro_api=pro, ts_code='002396.SZ', start_date='20180101', end_date='20181011', freq='D')
    # print(df)
    to_hist_data('002230')