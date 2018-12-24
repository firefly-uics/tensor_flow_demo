import logging

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from ts.simulation_history import SimulationHistory
from ts.st_history_data import x_train_col_index
from ts.st_test_op import op_row


class Op(SimulationHistory):
    model_cache = {}
    t1_predictions = None
    t0_predictions = 0
    t0_index = ''

    def is_sell(self, index, row):
        op_data = pd.DataFrame([op_row(row.ma5, row.close, 0, 2, 0).convert_to_dict()])

        logging.debug('op_data: %s', op_data)

        change_predictions, true_predictions = self.predictions(op_data, ['ma5', 'buy_price', 'cycles'],
                                                                'profit',
                                                                self.get_code() + '_op_model.h5')

        self.t0_predictions = change_predictions

        logging.debug('sell change_predictions:%s, true_predictions:%s', float(change_predictions), float(change_predictions) < 0.90)


        return float(change_predictions) < 0

    def is_buy(self, index, row):
        logging.debug('index: %s, date: %s', index, row['date'])

        op_data = pd.DataFrame([op_row(row.ma5, row.close, 0, 1, 0).convert_to_dict()])

        change_predictions, true_predictions = self.predictions(op_data, ['ma5', 'buy_price',  'cycles'], 'profit',
                                                         self.get_code() + '_op_model.h5')

        self.t0_predictions = change_predictions

        logging.debug('buy change_predictions:%s, true_predictions:%s', float(change_predictions), float(change_predictions) < 0.90)

        return float(change_predictions) > 0

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

        predictions = model.predict(x).flatten()[0]/10-0.85

        return predictions, y[0]
