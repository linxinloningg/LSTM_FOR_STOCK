import os
import numpy as np
import datetime as dt
from numpy import newaxis
from core.utils import Timer
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint


class Model:
    """用于构建和推断lstm模型的类"""

    def __init__(self):
        self.model = Sequential()

    def load_model(self, filepath):
        """
        导入模型
        :param filepath: .h5模型文件的路径
        :return:
        """
        print('[Model] Loading model from file %s' % filepath)
        self.model = load_model(filepath)

    def build_model(self, configs, input_timesteps, input_dim):
        """
        构建模型
        :param configs:配置文件
        :param input_timesteps:窗口数据的行数
        :param input_dim:窗口数据的列数
        :return:
        """
        timer = Timer()
        timer.start()

        for layer in configs['model']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None

            if layer['type'] == 'dense':
                self.model.add(Dense(neurons, activation=activation))
            if layer['type'] == 'lstm':
                self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'dropout':
                self.model.add(Dropout(dropout_rate))

        self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'])

        print('[Model] Model Compiled')
        timer.stop()

    def train(self, x, y, epochs, batch_size, validation_data, verbose, shuffle, validation_freq, save_dir):
        """
        训练模型
        :param x:输入的x值
        :param y:输入的y标签值
        :param epochs:每次梯度更新的样本数即批量大小
        :param batch_size:迭代次数
        :param validation_data:这个参数会覆盖 validation_split 即两个函数只能存在一个，
                                它的输入为元组 (x_val，y_val)，这作为验证数据。
        :param verbose: verbose = 0 为不在标准输出流输出日志信息，
                        verbose = 1 为输出进度条记录，
                        verbose = 2 为每个epoch输出一行记录
        :param shuffle:布尔值。是否在每轮迭代之前混洗数据
        :param validation_freq:使用验证集实施验证的频率。当等于1时代表每个epoch结束都验证一次
        :param save_dir:训练模型保存目录
        :return:
        """
        timer = Timer()
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))

        save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%Y%m%d-%H%M%S'), str(epochs)))

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=2),
            ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
        ]

        self.model.fit(
            x,
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            verbose=verbose,
            shuffle=shuffle,
            validation_freq=validation_freq,
            callbacks=callbacks
        )

        print('[Model] Training Completed. Model saved as %s' % save_dir)

        self.model.save(save_fname)

        timer.stop()

    def predict_point_by_point(self, data):
        # Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
        print('[Model] Predicting Point-by-Point...')
        predicted = self.model.predict(data)
        predicted = np.reshape(predicted, (predicted.size,))
        return predicted

    def predict_sequences_multiple(self, data, window_size, prediction_len):
        # Predict sequence of 50 steps before shifting prediction run forward by 50 steps
        print('[Model] Predicting Sequences Multiple...')
        prediction_seqs = []
        for i in range(int(len(data) / prediction_len)):
            curr_frame = data[i * prediction_len]
            predicted = []
            for j in range(prediction_len):
                predicted.append(self.model.predict(curr_frame[newaxis, :, :])[0, 0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
        return prediction_seqs

    # 最没用的预测
    def predict_sequence_full(self, data, window_size):
        # Shift the window by 1 new prediction each time, re-run predictions on new window
        print('[Model] Predicting Sequences Full...')
        curr_frame = data[0]
        predicted = []
        for i in range(len(data)):
            predicted.append(self.model.predict(curr_frame[newaxis, :, :])[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
        return predicted
