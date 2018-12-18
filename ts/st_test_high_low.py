import datetime
import logging

import tushare as ts
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
import tensorflow as tf

from ts.build_model import BuildModel

logging.basicConfig(level=logging.INFO)

column_names = ['open', 'high', 'low', 'close', 'change', 'vol']
code = '002396.SZ'
model_name = '_model.h5'
high_module_name = code +'_high'+ model_name
low_module_name = code +'_low'+ model_name

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)

# 目标价格
target_price = {'min_change': -0.02, 'max_change': 0.02}

change_fee = 0.0

high_model = keras.models.load_model(high_module_name)
low_model = keras.models.load_model(low_module_name)


def sell_fee(change_unit, price):
    s_fee = change_unit * price * 0.0001 + change_unit * price * 0.0003 + change_unit / 1000 * 1
    global change_fee
    change_fee = change_fee + s_fee
    return s_fee


def buy_fee(change_unit, price):
    b_fee = change_unit * price * 0.0001 + change_unit / 1000 * 1.1
    global change_fee
    change_fee = change_fee + b_fee
    return b_fee


def get_target_price(index):
    today = datetime.datetime.strptime(index, '%Y-%m-%d %H:%M:%S')
    yesterday = today - datetime.timedelta(days=1)

    yesterday_df = ts.get_hist_data(code, start=yesterday.strftime('%Y-%m-%d'), end=yesterday.strftime('%Y-%m-%d'))
    today_df = ts.get_hist_data(code, start=today.strftime('%Y-%m-%d'), end=today.strftime('%Y-%m-%d'))

    logging.debug("yesterday_df:%s", yesterday_df)
    if len(yesterday_df) == 0:
        return get_target_price(yesterday.strftime('%Y-%m-%d') + ' 00:00:01')

    stock_data = np.array(yesterday_df)
    columns = yesterday_df.columns.values.tolist()

    x_train_col = BuildModel.x_train_col_index(columns, column_names)
    x_train_col_l = x_train_col_index(columns, column_names)

    x_h = np.array(stock_data[:, x_train_col])
    x_l = np.array(stock_data[:, x_train_col_l])

    optimizer = tf.train.RMSPropOptimizer(0.001)

    if len(x_h) == 0:
        return 0, 0

    high_model.compile(loss='mse',
                       optimizer=optimizer,
                       metrics=['mae'])

    low_model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=['mae'])

    test_predictions_h = high_model.predict(x_h).flatten()[0] / 10000 * 1
    test_predictions_l = low_model.predict(x_l).flatten()[0] / 10000 * 1

    # today_df_h_l = today_df['high', 'low']

    logging.info("test_predictions_h:%f, test_predictions_l:%s, today_df:%s", test_predictions_h, test_predictions_l,
                 today_df)

    target_price['min_change'] = test_predictions_l
    target_price['max_change'] = test_predictions_h

    return test_predictions_l, test_predictions_h


def test_data(df):
    def print_account():
        logging.info(
            "account_init:%f,account_total:%f, account pre:%f, stock:%f, price:%f, target_price:%s, change_fee:%s",
            account_init,
            account_total,
            (account_total - account_init) / account_init * 100, T0, price, target_price, change_fee)

    # 账户总额
    account_init = 10000
    account_total = account_init
    change_unit = 100

    # 股票总额
    T0 = 0
    T1 = 0
    day = 0
    data_count = len(df)
    logging.info("data count:%s", data_count)
    old_index = ''

    for index, row in df.iterrows():
        day = day + 1
        price = float(row["open"] + row["price_change"])
        p_change = float(row["p_change"])
        date = index.split(' ')[0]

        if old_index != date:
            old_index = date
            T1 = T1 + T0
            T0 = 0
            test_predictions_l, test_predictions_h = get_target_price(index)

        if float(test_predictions_l) >= float(test_predictions_h):
            if day == data_count:
                logging.info("中间结束清空")
                account_total = account_total + T1 * price - sell_fee(T1, price)
                account_total = account_total + T0 * price - sell_fee(T0, price)
                T1 = 0
                print_account()
            continue

        if price < test_predictions_l:
            logging.debug("buy")
            if account_total > (change_unit * price):
                logging.info("buy success")
                account_total = account_total - change_unit * price - buy_fee(change_unit, price)
                T0 = T0 + change_unit
                print_account()

        if price > test_predictions_h:
            logging.debug("sell")
            if T1 > change_unit:
                logging.info("sell success")
                account_total = account_total + change_unit * price - sell_fee(change_unit, price)
                T1 = T1 - change_unit
                print_account()

        if day == data_count:
            logging.info("结束清空")
            account_total = account_total + T1 * price - sell_fee(T1, price)
            account_total = account_total + T0 * price - sell_fee(T0, price)
            T1 = 0
            print_account()

    return account_total, (account_total - account_init) / account_init * 100


if __name__ == "__main__":
    res = []
    for m in range(8, 9):
        month = ('%s' % m).rjust(2, '0')
        for d in [['01', '10'], ['10', '20'], ['20', '30']]:
            logging.info('start:%s, end:%s', '2018-{}-{}'.format(month, d[0]), '2018-{}-{}'.format(month, d[1]))
            df = ts.get_hist_data(code, start='2018-{}-{}'.format(month, d[0]), end='2018-{}-{}'.format(month, d[1]),
                                  ktype='60')
            df = df.sort_values(axis=0, ascending=True, by='date')
            account_total, account_pre = test_data(df)
            res.append(
                {'date': '2018-%s-%s' % (month, d[1]), 'account_total': account_total, 'account_pre': account_pre})

    logging.info("res:%s", res)
