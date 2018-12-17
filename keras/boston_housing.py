from __future__ import absolute_import, division, print_function

import logging
import ssl

import tensorflow as tf
from tensorflow import keras

import numpy as np

logging.basicConfig(level=logging.DEBUG)

ssl._create_default_https_context = ssl._create_unverified_context

boston_housing = keras.datasets.boston_housing

(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

print("train_labels.shape: {}".format(train_data.shape))

# Shuffle the training set 拆分样本
order = np.argsort(np.random.random(train_labels.shape))

logging.debug("order: %s", order)

train_data = train_data[order]
train_labels = train_labels[order]



#
print("Training set: {}".format(train_data.shape))  # 404 examples, 13 features
print("Testing set:  {}".format(test_data.shape))   # 102 examples, 13 features

import pandas as pd

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                'TAX', 'PTRATIO', 'B', 'LSTAT']

df = pd.DataFrame(train_data, columns=column_names)
print(df.head())


# 标准化特征
'''建议标准化使用不同比例和范围的特征。对于每个特征，用原值减去特征的均值，再除以标准偏差即可：'''

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

print(train_data[0])  # First training sample, normalized

# 创建模型

def build_model():
  model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu,
                       input_shape=(train_data.shape[1],)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])

  optimizer = tf.train.RMSPropOptimizer(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
  return model

model = build_model()

print(model.summary())

# 训练模型
# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 500

# Store training stats
history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[PrintDot()])

import matplotlib.pyplot as plt

def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [1000$]')
  plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
           label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
  plt.legend()
  plt.ylim([0, 5])

  plt.show()

plot_history(history)

model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[early_stop, PrintDot()])

plot_history(history)

[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)

print("Testing set Mean Abs Error: ${:7.2f}".format(mae * 1000))

# 总结
#
# 此笔记本介绍了几个处理回归问题的技巧。
#
# 均方误差 (MSE) 是用于回归问题的常见损失函数（与分类问题不同）。
# 同样，用于回归问题的评估指标也与分类问题不同。常见回归指标是平均绝对误差 (MAE)。
# 如果输入数据特征的值具有不同的范围，则应分别缩放每个特征。
# 如果训练数据不多，则选择隐藏层较少的小型网络，以避免出现过拟合。
# 早停法是防止出现过拟合的实用技术。