import datetime
import logging

import tushare as ts
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
import tensorflow as tf

from ts.build_model import BuildModel
from ts.db_utils import get_daily_by_trade_date
from ts.simulation_history import SimulationHistory
from ts.st_history_data import x_train_col_index

class Change(SimulationHistory):
    model_cache = {}
    t1_predictions = None
    t0_predictions = 0
    t0_index = ''

    def is_sell(self, index, row):
        logging.debug('index: %s, date: %s', index, row['date'])
        today = datetime.datetime.strptime(row['date'], '%Y-%m-%d %H:%M:%S')

        df = get_daily_by_trade_date(self.get_code(), today.strftime('%Y%m%d'))

        change_predictions, true_predictions = self.predictions(df, ['open', 'high', 'low', 'close'], 'pct_chg',
                                                         self.get_code() + '_pct_chg_model.h5')

        logging.debug('change_predictions:%s, true_predictions:%s', change_predictions, true_predictions)

        if len(df) == 0:
            return False

        if self.t0_predictions == None:
            return False

        if self.t0_predictions <= 0:
            return False

        logging.debug('row[ma5] * (1+self.t0_predictions/100) :%s, ma5: %s, price:%s', row['ma5'] * (1+self.t0_predictions/100), row['ma5'], row['close'])

        return row['close'] > row['ma5'] * (1+self.t0_predictions/100)

    def is_buy(self, index, row):
        logging.debug('index: %s, date: %s', index, row['date'])
        today = datetime.datetime.strptime(row['date'], '%Y-%m-%d %H:%M:%S')

        df = get_daily_by_trade_date(self.get_code(), today.strftime('%Y%m%d'))

        change_predictions, true_predictions = self.predictions(df, ['open', 'high', 'low', 'close'], 'pct_chg',
                                                         self.get_code() + '_pct_chg_model.h5')

        self.t0_predictions = change_predictions

        logging.debug('change_predictions:%s, true_predictions:%s', change_predictions, true_predictions)

        if self.t0_index != index:
            self.t1_predictions = self.t0_predictions
            self.t0_index = index

        if len(df) == 0:
            return False

        if self.t0_predictions <= 0:
            return False

        logging.debug('row[ma5] * (1-change_predictions/100) :%s, ma5: %s, price:%s', row['ma5'] * (1-self.t0_predictions/100), row['ma5'], row['close'])

        return row['close'] < row['ma5'] * (1-self.t0_predictions/100)

    def predictions(self, df, column_names, label_name, module_name):
        columns = df.columns.values.tolist()
        stock_data = np.array(df)

        x_train_col = x_train_col_index(columns, column_names)
        y_train_col = x_train_col_index(columns, [label_name])[0]

        x = np.array(stock_data[:, x_train_col])
        y = np.array(stock_data[:, y_train_col])

        if len(x) == 0:
            return 0, 0

        model = self.model_cache.get(module_name)

        if model == None:
            model = keras.models.load_model(module_name)

            optimizer = tf.train.RMSPropOptimizer(0.001)

            model.compile(loss='mse',
                          optimizer=optimizer,
                          metrics=['mae'])

            self.model_cache[module_name] = model

        predictions = model.predict(x).flatten()[0]/10 + 1.5

        return predictions, y[0]
