import numpy as np


class DataLoader:
    """用于加载和转换lstm模型数据的类"""

    def __init__(self, dataframe, split, cols):
        """

        :param dataframe:传入dataframe类型的数据
        :param split: 训练集跟测试集分割比例
        :param cols: 用于训练的特征列表（将预测列表的第一个数据）
        """

        i_split = int(len(dataframe) * split)

        self.data_train = dataframe.get(cols).values[:i_split]
        self.data_test = dataframe.get(cols).values[i_split:]

        self.len_train = len(self.data_train)
        self.len_test = len(self.data_test)

        self.len_train_windows = None

    # 将测试数据构成一个数据窗口用于测试
    def get_test_data(self, seq_len, normalise):
        """
        创建x, y测试数据窗口
        警告:确保你有足够的内存来加载数据，否则加大训练集跟测试集分割比例。
        :param seq_len:序列长度
        :param normalise:是否归一化
        :return:
        """
        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i + seq_len])

        data_windows = np.array(data_windows).astype(float)
        data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows

        x = data_windows[:, :-1]
        y = data_windows[:, -1, [0]]
        return x, y

    # 将训练数据按序列长度分成多个训练数据窗口
    def get_train_data(self, seq_len, normalise):
        """
        创建x, y训练数据窗口
        警告:确保你有足够的内存来加载数据，否则减少训练集跟测试集分割比例。
        :param seq_len:序列长度
        :param normalise:是否归一化
        :return:
        """

        # 构建窗口数据
        def _next_window(i, seq_len, normalise):
            """
            从给定的索引位置i生成下一个数据窗口
            :param i:窗口序号
            :param seq_len: 序列长度
            :param normalise: 是否归一化
            :return:
            """
            window = self.data_train[i:i + seq_len]
            window = self.normalise_windows(window, single_window=True)[0] if normalise else window
            x = window[:-1]
            y = window[-1, [0]]
            return x, y

        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = _next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    @staticmethod
    def normalise_windows(window_data, single_window=False):
        """
        使用基值为零对窗口进行归一化
        :param window_data: 窗口数据
        :param single_window:数据是否为1维
        :return:
        """
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
