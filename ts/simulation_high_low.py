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

class HighLow(SimulationHistory):
    model_cache = {}
    def is_sell(self, index, row):
        logging.debug('index: %s, date: %s', index, row['date'])
        today = datetime.datetime.strptime(row['date'], '%Y-%m-%d %H:%M:%S')

        df = get_daily_by_trade_date(self.get_code(), today.strftime('%Y%m%d'))

        h_predictions, true_predictions = self.predictions(df, ['open', 'low', 'close', 'change', 'vol'], 'high',
                                                         self.get_code() + '_high_model.h5')

        logging.debug('h_predictions:%s, true_predictions:%s', h_predictions, true_predictions)

        if len(df) == 0:
            return False

        if h_predictions < row['ma5']:
            return False

        if abs(row['ma5'] - h_predictions) > 3:
            return False

        logging.debug('h_predictions :%s, ma5: %s, price:%s', h_predictions, row['ma5'], row['close'])

        return row['close'] > h_predictions

    def is_buy(self, index, row):
        logging.debug('index: %s, date: %s', index, row['date'])
        today = datetime.datetime.strptime(row['date'], '%Y-%m-%d %H:%M:%S')

        df = get_daily_by_trade_date(self.get_code(), today.strftime('%Y%m%d'))

        l_predictions, true_predictions = self.predictions(df, ['open', 'high', 'close', 'change', 'vol'], 'low', self.get_code() + '_low_model.h5')

        logging.debug('l_predictions:%s, true_predictions:%s', l_predictions, true_predictions)

        if len(df) == 0:
            return False


        if l_predictions > row['ma5']:
            return False

        if abs(row['ma5'] - l_predictions) > 3:
            return False

        logging.debug('l_predictions :%s, ma5: %s, price:%s', l_predictions, row['ma5'], row['close'])

        return row['close'] < l_predictions

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

        predictions = model.predict(x).flatten()[0]/10000

        return predictions, y[0]
