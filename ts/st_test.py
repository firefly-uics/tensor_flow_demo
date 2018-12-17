import datetime
import logging

import tushare as ts
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
import tensorflow as tf

from ts.st_history_data import x_train_col_index, column_names, load_data, code

logging.basicConfig(level=logging.INFO)

#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)

df = ts.get_hist_data(code, start='2018-11-01', end='2018-11-30', ktype='60')

df = df.sort_values(axis=0, ascending=True, by='date')

# 目标价格
target_price = {'min_change': -0.02, 'max_change': 0.02}

# 账户总额
account_init = 10000
account_total = account_init
change_fee = 0.0
change_unit = 100

# 股票总额
stock_total = 0

data_count = len(df)
day = 0
T0 = 0
T1 = 0

logging.info("data count:%s", data_count)

new_model = keras.models.load_model(code + '_p_change_model.h5')
logging.info(new_model.summary())


def sell_fee(change_unit, price):
    s_fee = change_unit * price * 0.0001 + change_unit * price * 0.0003 + change_unit / 1000 * 1
    global change_fee
    change_fee = change_fee + s_fee
    return s_fee


def buy_fee(change_unit, price):
    b_fee = change_unit * price * 0.0001 + change_unit / 1000 * 1
    global change_fee
    change_fee = change_fee + b_fee
    return b_fee


def print_account():
    logging.info("account_init:%f,account_total:%f, account pre:%f, stock:%f, price:%f, target_price:%s, change_fee:%s",
                 account_init,
                 account_total,
                 (account_total - account_init) / account_init * 100, stock_total, price, target_price, change_fee)


def get_target_price(index):
    (train_data, train_labels), (test_data, test_labels) = load_data()

    today = datetime.datetime.strptime(index, '%Y-%m-%d %H:%M:%S')
    yesterday = today - datetime.timedelta(days=1)

    yesterday_df = ts.get_hist_data(code, start=yesterday.strftime('%Y-%m-%d'), end=yesterday.strftime('%Y-%m-%d'))
    today_df = ts.get_hist_data(code, start=today.strftime('%Y-%m-%d'), end=today.strftime('%Y-%m-%d'))

    logging.debug("yesterday_df:%s", yesterday_df)
    if len(yesterday_df) == 0:
        return get_target_price(yesterday.strftime('%Y-%m-%d') + ' 00:00:01')

    stock_data = np.array(yesterday_df)
    columns = yesterday_df.columns.values.tolist()

    x_train_col = x_train_col_index(columns, column_names)

    x = np.array(stock_data[:, x_train_col])

    optimizer = tf.train.RMSPropOptimizer(0.001)

    if len(x) == 0:
        return 0

    new_model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=['mae'])

    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    x = (x - mean) / std

    logging.debug("x:%s", x)

    test_predictions = new_model.predict(x).flatten()[0]
    today_p_change = today_df['p_change']
    logging.info("test_predictions:%f, today_df:%s, loss:%f", test_predictions, today_p_change, test_predictions-test_predictions)

    target_price['min_change'] = test_predictions * 0.01
    target_price['max_change'] = test_predictions * 0.09

    return test_predictions


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
        train_val = get_target_price(index)
        logging.debug("old_index:%s, target_price:%s, train_val:%s", old_index, target_price, train_val)

    if float(train_val) < 2 or float(train_val) > 8:
        if day == data_count:
            logging.info("清空")
            account_total = account_total + T1 * price - sell_fee(T1, price)
            T1 = 0
            print_account()
        continue
    if -1 > float(train_val) - p_change > 1:
        continue

    if p_change < target_price['min_change']:
        logging.debug("buy")
        print_account()
        if account_total > (change_unit * price):
            account_total = account_total - change_unit * price - buy_fee(change_unit, price)
            T0 = T0 + change_unit

    if p_change > target_price['max_change']:
        logging.debug("sell")
        print_account()
        if T1 > change_unit:
            account_total = account_total + change_unit * price - sell_fee(change_unit, price)
            T1 = T1 - change_unit

    if day == data_count:
        logging.info("清空")
        account_total = account_total + T1 * price - sell_fee(T1, price)
        T1 = 0
        print_account()
