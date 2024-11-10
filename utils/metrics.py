# coding=utf-8
import numpy as np
from sklearn.metrics import mean_squared_error

def prediction_hit_rate(prediction, true_value, delta):
    data_minus = prediction - true_value
    tmp = np.where(abs(data_minus) <= delta, 1, 0)
    number = np.sum(tmp)

    return number / data_minus.shape[0]

def prediction_hit_rate_percentage_error(prediction, true_value, delta):
    data_minus = abs(prediction - true_value) / true_value
    tmp = np.where(abs(data_minus) <= delta, 1, 0)
    number = np.sum(tmp)

    return number / data_minus.shape[0]

def RMSE(prediction, ture_value):
    return np.sqrt(mean_squared_error(prediction, ture_value))

def SD(prediction, ture_value):
    # standard_deviation
    error = prediction - ture_value
    error_mean = error.mean()
    return np.sqrt(np.power(error - error_mean, 2).mean())