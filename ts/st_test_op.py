import datetime
import logging

import tushare as ts
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
import tensorflow as tf

from ts.build_model import BuildModel
from ts.st_history_data import x_train_col_index, column_names, load_data, code

logging.basicConfig(level=logging.INFO)

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)

df = ts.get_hist_data(code, start='2018-06-01', end='2018-11-30', ktype='60')

df = df.sort_values(axis=0, ascending=True, by='date')

logging.debug('df:%s', df.head())

op_data = []
index = 0
cycle = 5
order_unit = 100


class op_row:
    buy_price = 0
    sell_price = 0
    profit = 0
    ma5 = 0
    cycles = 0

    def convert_to_dict(self):
        '''把Object对象转换成Dict对象'''
        dict = {}
        dict.update(self.__dict__)
        return dict

    def __init__(self, ma5, buy_price, sell_price, cycles, profit) -> None:
        super().__init__()
        self.buy_price = buy_price
        self.ma5 = ma5
        self.sell_price = sell_price
        self.profit = profit
        self.cycles = cycles

if __name__ == '__main__':
    for _, row in df.iterrows():
        for c in range(0, cycle):

            if index + cycle < len(df):
                buy_row = row
                sell_row = df.loc[df.index[index + c]]

                buy_price = buy_row.close
                sell_price = sell_row.close
                profit = (sell_price - buy_price) * order_unit / (buy_price * order_unit) * 100
                ma5 = buy_row.ma5
                cycles = c + 1

                logging.debug('ma5:%.2f, buy_price:%.2f, sell_price:%.2f, profit:%.2f, cycles:%d', ma5, buy_price,
                              sell_price, profit, cycles)

                op_data.append(op_row(ma5, buy_price, sell_price, profit, cycles).convert_to_dict())

        index += 1

    logging.debug('op_data:%s', op_data)

    column_names = ['ma5', 'buy_price', 'cycles']

    train_data = pd.DataFrame(op_data)

    logging.debug('train_data:%s', train_data)

    build_model = BuildModel(code + '_op', train_data, column_names, 'profit')

    build_model.create_model()
