import logging

import tushare as ts
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
import tensorflow as tf

logging.basicConfig(level=logging.DEBUG)

column_names = ['open', 'low', 'close', 'volume']
code = '002396'
module_name = code + '_high_model.h5'

def x_train_col_index(all_columns, names):
    r = []
    for name in names:
        r.append(all_columns.index(name))
    return r


def load_data():
    '''
    获取测试数据
    :return:
    '''
    df = ts.get_hist_data(code)
    columns = df.columns.values.tolist()

    logging.debug('colums:%s', columns)

    stock_data = np.array(df)

    x_train_col = x_train_col_index(columns, column_names)
    y_train_col = x_train_col_index(columns, ['high'])[0]

    x = np.array(stock_data[:, x_train_col])
    y = np.array(stock_data[:, y_train_col])

    # 丢弃
    x = np.delete(x, -1, 0)
    y = np.delete(y, 0, 0)

    x = np.array(x)
    y = np.array(y)

    logging.debug("data set x: %s", len(x))
    logging.debug("data set y: %s", len(y))

    test_split = 0.2

    x_train = np.array(x[:int(len(x) * (1 - test_split))])
    y_train = np.array(y[:int(len(x) * (1 - test_split))])
    x_test = np.array(x[int(len(x) * (1 - test_split)):])
    y_test = np.array(y[int(len(x) * (1 - test_split)):])
    return (x_train, y_train), (x_test, y_test)


def fit():
    (train_data, train_labels), (test_data, test_labels) = load_data()

    logging.debug("train_data: %s", train_data)
    logging.debug("train_labels: %s", train_labels)
    logging.debug("train_data.shape: %s", train_data.shape)
    # Shuffle the training set 拆分样本
    order = np.argsort(np.random.random(train_labels.shape))

    logging.debug("order: %s", order)

    train_data = train_data[order]
    train_labels = train_labels[order]

    logging.debug("Training set: %s", train_data.shape)  # 404 examples, 13 features
    logging.debug("Testing set:  %s", test_data.shape)  # 102 examples, 13 features

    df = pd.DataFrame(train_data, columns=column_names)
    logging.debug("train_data: %s", df.head())

    # 标准化特征
    '''建议标准化使用不同比例和范围的特征。对于每个特征，用原值减去特征的均值，再除以标准偏差即可：'''

    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std

    logging.debug("train_data[0]: %s", train_data[0])  # First training sample, normalized

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

    logging.info("model:%s", model.summary())

    # 训练模型
    # Display training progress by printing a single dot for each completed epoch
    class PrintDot(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs):
            if epoch % 100 == 0: print('')
            print('.', end='')

    EPOCHS = 500

    #
    # The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)

    history = model.fit(train_data, train_labels, epochs=EPOCHS,
                        validation_split=0.02, verbose=0,
                        callbacks=[early_stop, PrintDot()])

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


    [loss, mae] = model.evaluate(test_data, test_labels, verbose=0)

    logging.info("Testing set Mean Abs Error: ${:7.2f} {}".format(mae, loss))

    x_test_data = np.array(test_data[0:1])
    y_test_labels = np.array(test_labels[0:1])

    results = model.evaluate(x_test_data, y_test_labels, verbose=0)
    logging.info("data:%s, label:%s,res:%s", x_test_data, y_test_labels, results)

    x_test_predictions = model.predict(x_test_data).flatten()
    logging.info("x_test_predictions data:%s, label:%s,res:%s", x_test_data, y_test_labels, x_test_predictions)

    test_predictions = model.predict(test_data).flatten()

    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values [1000$]')
    plt.ylabel('Predictions [1000$]')
    plt.axis('equal')
    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())
    _ = plt.plot([-100, 100], [-100, 100])

    plt.show()

    error = test_predictions - test_labels
    plt.hist(error, bins=50)
    plt.xlabel("Prediction Error [1000$]")
    _ = plt.ylabel("Count")

    plt.show()

    model.save(module_name)


if __name__ == "__main__":
    fit()

