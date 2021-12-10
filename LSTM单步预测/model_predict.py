from core.model import Model
import numpy as np

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def plot_results(pointbypoint, fullseq, multiseq):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(fullseq, label='fullseq')
    ax = fig.add_subplot(111)
    ax.plot(multiseq, label='multiseq')
    plt.plot(pointbypoint, label='pointbypoint')
    plt.legend()
    plt.show()


class model_predict:
    def __init__(self, dataframe, cols):
        self.data_test = dataframe.get(cols).values[:]
        self.len_test = len(self.data_test)
        self.data_scale = dataframe.get(cols[0]).values[:]

    @staticmethod
    def normalise_windows(window_data, single_window=False):
        """归一化窗口的基值为零"""
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
                normalised_window.append(normalised_col)
            normalised_window = np.array(
                normalised_window).T  # 重塑和转置数组回到原来的多维格式
            normalised_data.append(normalised_window)
        return np.array(normalised_data)

    # 不成熟的函数（无法发挥作用）
    def reversal_normalis(self, predict_data):

        # 特征的归一化处理
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit_transform(self.data_scale.reshape(-1, 1))

        reversal_data = scaler.inverse_transform(predict_data.reshape(1, -1))

        return reversal_data

    def get_data(self, input_timesteps, normalise):
        """

        :param normalise:
        :param input_timesteps:
        :return:
        """

        data_windows = []
        for i in range(input_timesteps, self.len_test, input_timesteps):
            data_windows.append(self.data_test[i - input_timesteps:i])

        data_windows = np.array(data_windows).astype(float)
        data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows

        x = data_windows[:, :]
        y = data_windows[:, :, [0]]
        y = y.reshape(y.shape[0], y.shape[1])
        return x, y

    def forecast(self, model_file_path, input_timesteps, normalise):
        """

        :param normalise:
        :param input_timesteps:
        :param model_file_path: ../save_models/03122021-230019-e50.h5
        :return:
        """
        model = Model()
        model.load_model(model_file_path)

        x_test, y_test = self.get_data(input_timesteps, normalise)

        predictions_pointbypoint = model.predict_point_by_point(x_test)

        return predictions_pointbypoint, y_test
