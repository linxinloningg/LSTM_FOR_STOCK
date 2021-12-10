from core.data_processor import DataLoader
import pandas as pd
import json
from core.model import Model
import matplotlib.pyplot as plt


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # 填充预测列表，将其在图表中移动到正确的开始位置
    for i, data in enumerate(predicted_data):
        padding = [None for _ in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()


if __name__ == "__main__":
    configs = json.load(open('config.json', 'r'))
    df = pd.read_csv('./data/sh600031.csv')  # 读取股票文件

    data = DataLoader(
        dataframe=df,
        split=configs['data']['train_test_split'],
        cols=configs['data']['columns']
    )
    x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        input_timesteps=configs['data']['input_timesteps'],
        normalise=configs['data']['normalise']
    )

    x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        input_timesteps=configs['data']['input_timesteps'],
        normalise=configs['data']['normalise']
    )

    model = Model()
    model.build_model(configs)

    # 内存中训练
    model.train(
        x,
        y,
        epochs=configs['training']['epochs'],
        batch_size=configs['training']['batch_size'],
        validation_data=(x_test, y_test),
        verbose=configs['training']['verbose'],
        shuffle=configs['training']['shuffle'],
        validation_freq=configs['training']['validation_freq'],
        save_dir=configs['model']['save_dir']
    )

    # 预测

    # 逐点预测
    predictions_pointbypoint = model.predict_point_by_point(x_test)

    # 画图
    plot_results(predictions_pointbypoint, y_test)
