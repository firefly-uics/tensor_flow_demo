from sqlalchemy import create_engine
import sqlalchemy as sa
import pandas as pd

engine = create_engine("mysql+pymysql://root:root@127.0.0.1:3306/ts", max_overflow=5)

daily_sql = sa.text('SELECT * FROM daily where ts_code =:code and trade_date > :start and trade_date < :end order by `index` desc')
daily_by_trade_date_sql = sa.text('SELECT * FROM daily where ts_code =:code and trade_date =:trade_date')
hist_data_sql_str = 'SELECT * FROM hist_data_60_%s where `date` > :start and `date` < :end order by `date` '
_def_start = '20180101'
_def_end = '20181230'


def get_daily(code, start=None, end=None):
    if None == start:
        start = _def_start
    if None == end:
        end = _def_end

    return pd.read_sql_query(daily_sql, engine, params={'code': code, 'start': start, 'end': end})

def get_daily_by_trade_date(code, trade_date):
    return pd.read_sql_query(daily_by_trade_date_sql, engine, params={'code': code, 'trade_date': trade_date})


def get_hist_data(code, start=None, end=None):
    if None == start:
        start = '2018-08'
    if None == end:
        end = '2018-09'

    if code.find('.') != -1:
        code = code.split('.')[0]

    return pd.read_sql_query(sa.text(hist_data_sql_str % code), engine, params={'start': start, 'end': end})

if __name__ == '__main__':
    # print(get_daily('002396.SZ'))
    print(get_hist_data('002396', '2018-08', '2018-09'))
