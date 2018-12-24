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

class Order(SimulationHistory):
    def is_sell(self, index, row):
        return row['close'] > row['ma5'] * (1 + 0.02)

    def is_buy(self, index, row):
        if self.is_clean(index, row):
            return False

        return row['close'] < row['ma5'] * (1 - 0.02)

    def is_clean(self, index, row):
        if self.get_account().cycle >= 5:
            return True
        return (row['ma5'] - self.get_account().last_price)/row['ma5'] < -0.03



