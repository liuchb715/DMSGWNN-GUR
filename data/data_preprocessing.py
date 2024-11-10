# coding=utf-8
import numpy as np

def create_dataset_Multivariable(x_data, y_data, samplesLength: int, time_pass: int, slidingWindow_size: int ,
                                 time_pred_start=6*0, time_pred_end=6*10, average_size=6):
    arr_x, arr_y = [], []
    y_len = int((time_pred_end-time_pred_start)/average_size)
    for i in range(samplesLength):
        x = x_data[(i*slidingWindow_size): (i*slidingWindow_size + time_pass), :]
        y = y_data[(i*slidingWindow_size + time_pass + time_pred_start):(i*slidingWindow_size + time_pass + time_pred_end), :]

        y_average = np.zeros((y_len, y_data.shape[1]))
        for i in range(y_len):
            y_average[i, :] = np.mean(y[(i*average_size):((i+1)*average_size), :], axis=0)

        arr_x.append(x)
        arr_y.append(y_average)

    return np.array(arr_x), np.array(arr_y)