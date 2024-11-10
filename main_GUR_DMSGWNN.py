# coding=utf-8
import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import pairwise_distances
from sklearn import preprocessing
import joblib 
import argparse

import pandas as pd

from itertools import product
from tqdm import trange

import torch
import torch.nn as nn
import scipy.sparse as sparse

# Custom functions or classes
import models
from utils.utils import print_log, mkdir, EarlyStopping, get_logger
from utils.metrics import RMSE, SD, prediction_hit_rate, prediction_hit_rate_percentage_error
from data.data_preprocessing import create_dataset_Multivariable


parser = argparse.ArgumentParser(description='[DMSGWNN] Denoising Multiscale Spectral Graph Wavelet Neural Networks')
parser.add_argument('--seed', type=int, default=111, help='seed of random number generator')
parser.add_argument('--data_path', type=str, default='data_test.xlsx', help='data file') 
parser.add_argument('--correlation_matrix_path', type=str, default='data_GUR_correlation_matrix_test.csv', help='correlation_matrix data file')

parser.add_argument('--samplesLength', type=int, default=100, help='length of all samplkes')
parser.add_argument('--n_train', type=int, default=70, help='length of all samplkes')
parser.add_argument('--n_validation', type=int, default=90, help='length of all samplkes')

parser.add_argument('--time_pass', type=int, default=6 * 60 * 2, help='time span of process variable sampling')
parser.add_argument('--slidingWindow_size', type=int, default=6 * 1, help='size of sliding window for sampling')
parser.add_argument('--time_pred_start', type=int, default=6 * 0, help='starting time of predicted targets')
parser.add_argument('--time_pred_end', type=int, default=6 * 30, help='end time of predicted targets')

parser.add_argument('--batch_size', type=int, default=16, help='batch size for training')
parser.add_argument('--patience', type=int, default=100, help='level of patience with performance changes during the training process')
parser.add_argument('--n_epoch', type=int, default=3, help='number of training iterations')

parser.add_argument('--rsr_iter', type=int, default=100, help='iter of the RSR module')
parser.add_argument('--rsr_tolerance', type=float, default=1e-5, help='tolerance of the RSR module')

args = parser.parse_args()


def main():
    # -------------------------------#
    #    Parameters Setting
    # -------------------------------#
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch.set_default_dtype(torch.double)

    # obtaining the current timestamp
    timestamp_str = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    result_path = os.path.join(os.path.dirname(__file__), 'result')
    result_path_timestamp = os.path.join(result_path, "result" + timestamp_str)

    # creating the result directory, such as trainedModels, figures.
    mkdir(result_path_timestamp)
    mkdir(os.path.join(result_path_timestamp, 'trainedModels'))
    mkdir(os.path.join(result_path_timestamp, 'figures'))

    # Log setting
    logger = get_logger(os.path.join(result_path_timestamp, 'run.log'))
    print_func = logger.info
    print_func('Created the result file path: {}'.format(result_path_timestamp))

    # GPU setting
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print_func("GPU usage: {}".format(device))

    # -------------------------------#
    #    Loading data
    # -------------------------------#
    print_func("Loading data...\n")
    
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
    data_source_DIR = os.path.join(DATA_DIR, 'dataSource')
    data_path = os.path.join(data_source_DIR, args.data_path)  

    data_gas = pd.read_excel(data_path, sheet_name=0, header=0)
    features_raw = data_gas.iloc[:, 1:30].values
    gas_CO_raw = data_gas.iloc[:, 28].values.reshape(-1, 1)
    gas_CO2_raw = data_gas.iloc[:, 29].values.reshape(-1, 1)
    gas_UR_raw = data_gas.iloc[:, 30].values.reshape(-1, 1)

    data_path = os.path.join(data_source_DIR, args.correlation_matrix_path)   
    correlation_matrix = np.loadtxt(open(data_path, "rb"), delimiter=",", skiprows=1)
    correlation_matrix_MIC = correlation_matrix[:29, :]
    correlation_matrix_Pearson = np.abs(correlation_matrix[29:, :])

    #############    Data preprocessing    ###############
    # Normalization
    n_max = (args.samplesLength-1)*args.slidingWindow_size + args.time_pass + args.time_pred_end
    print_func("n_max:{}".format(n_max))

    X_timeserise = features_raw[:n_max, :]

    print_func("X_timeserise={}, gas_UR_raw={}".format(X_timeserise.shape, gas_UR_raw.shape))
    Y_timeserise = features_raw[:n_max, 27:X_timeserise.shape[1]]  # CO, CO2

    scaler = preprocessing.StandardScaler()  # 实例化
    X_timeserise_normalize = scaler.fit_transform(X_timeserise)  
    Y_timeserise_normalize = Y_timeserise / 100   # co、co2 /100

    X_data_normalize, Y_data_normalize = create_dataset_Multivariable(x_data=X_timeserise_normalize,
                                                                      y_data=Y_timeserise_normalize,
                                                                      samplesLength=args.samplesLength,
                                                                      time_pass=args.time_pass,
                                                                      slidingWindow_size=args.slidingWindow_size,
                                                                      time_pred_start=args.time_pred_start,
                                                                      time_pred_end=args.time_pred_end,
                                                                      average_size=6)

    X_data_normalize = X_data_normalize.transpose(0, 2, 1)  
    Y_data_normalize = Y_data_normalize.transpose(0, 2, 1)  

    # resampling Y
    Y_period = 6  # 1min
    Y_len = int(Y_data_normalize.shape[-1] / Y_period)
    Y_data_normalize_resampling = np.zeros((Y_data_normalize.shape[0], Y_data_normalize.shape[1], Y_len))
    for i in range(Y_len):
        # Y_data_normalize_resampling[:, :, i] = Y_data_normalize[:, :, i * Y_period:(i + 1) * Y_period].mean(axis=2)
        Y_data_normalize_resampling[:, :, i] = np.median(Y_data_normalize[:, :, i * Y_period:(i + 1) * Y_period], axis=2)
    Y_data_normalize = Y_data_normalize_resampling

    # shuffling dataset
    n_shuffle = args.n_validation
    idx_shuffle = np.random.permutation(n_shuffle)

    X_data_normalize_shuffle = np.concatenate((X_data_normalize[idx_shuffle, :, :],
                                               X_data_normalize[n_shuffle:, :, :]), axis=0)
    Y_data_normalize_shuffle = np.concatenate((Y_data_normalize[idx_shuffle, :, :],
                                               Y_data_normalize[n_shuffle:, :, :]), axis=0)

    X_train = X_data_normalize_shuffle[:args.n_train, :, :]
    X_validation = X_data_normalize_shuffle[args.n_train:args.n_validation, :, :]
    X_test = X_data_normalize_shuffle[args.n_validation:, :, :]

    Y_train = Y_data_normalize_shuffle[:args.n_train, :, :]
    Y_validation = Y_data_normalize_shuffle[args.n_train:args.n_validation, :, :]
    Y_test = Y_data_normalize_shuffle[args.n_validation:, :, :]

    X_train = torch.tensor(X_train).to(device)
    X_validation = torch.tensor(X_validation).to(device)
    X_test = torch.tensor(X_test).to(device)

    # -------------------------------#
    #       Training model
    # -------------------------------#
    print_func("Training model...\n")
    print_func("Number of input  samples: {}, {}, {}".format(X_train.shape, X_validation.shape, X_test.shape))
    print_func("Number of output samples: {}, {}, {}".format(Y_train.shape, Y_validation.shape, Y_test.shape))

    # ---------- gridSearchCV ---------#
    # parameters = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1], 'beta': [0.01, 0.1, 1, 10, 100],
    #               'scaling_tau': [1, 8, 16, 32, 64, 128],
    #               'num_wavelets': [1, 2, 4, 8, 16, 32],
    #               'approximation_order': [1, 3, 5, 8, 16, 32, 64], 'layer': [1, 3, 5, 8, 16, 32]}
    parameters = {'alpha': [0.01], 'beta': [0.1], 'scaling_tau': [128], 'num_wavelets': [4],
                  'approximation_order': [3], 'layer': [3]}
    param_values = [v for v in parameters.values()]

    # ---------- initial ---------#
    bestTrainedModel_path = os.path.join(result_path_timestamp, 'trainedModels')
    n_plt = 0

    for alpha, beta, scaling_tau, num_wavelets, approximation_order, layer in product(*param_values):
        str_parameters = "alpha{}_beta{}_scalingTau{}_numWavelets{}_approximationOrder{}_Layer{}".format(
                         alpha, beta, scaling_tau, num_wavelets, approximation_order, layer)
        print_func("\033[33m gridSearchCV: alpha: {}, beta: {}, scaling_tau: {}, num_wavelets: {}, approximation_order: {}, layer: {} \033[0m".format(
                alpha, beta, scaling_tau, num_wavelets, approximation_order, layer))

        # --------------------------------------------#
        # Graph laplacian and spectral graph wavelets
        # --------------------------------------------#
        print_func("Computing graph laplacian and spectral graph wavelets.\n")
        print_func("Graph_Laplacian.shape={}".format(correlation_matrix_MIC.shape))

        # ------- Computing Graph laplacian -----------#
        print_func("Computing graph laplacian.\n")

        # correlation_matrix_Pearson, correlation_matrix_MIC
        correlation_matrix = [correlation_matrix_Pearson, correlation_matrix_MIC]
        graph_laplacian = computing_graph_laplacian(correlation_matrix, mode='PM', threshold=0.1, sigma=0.1)

        graph_laplacian_density = len(graph_laplacian.nonzero()[0]) / (graph_laplacian.shape[0] * graph_laplacian.shape[1])
        graph_laplacian_density = str(round(100 * graph_laplacian_density, 2))
        print_func("Density of graph_laplacian: " + graph_laplacian_density + "%.")

        # ------- Computing spectral graph wavelets -----------#
        print_func("Computing spectral graph wavelets.\n")
        sparse_graph_laplacian = sparse.csc_matrix(graph_laplacian)
        sparse_wavelets = models.SpectralGraphWavelets(sparse_graph_laplacian, scaling_tau=scaling_tau, wavelets_tau=[0.01, 1],
                                                       num_wavelets=num_wavelets, approximation_order=approximation_order,
                                                       tolerance=1e-5, trace_func=print_func)
        sparse_wavelets.calculate_all_wavelets()
        wavelet_matrices = sparse_wavelets.wavelet_matrices[0]
        inverse_wavelet_matrices = sparse_wavelets.wavelet_matrices[1]
        wavelet_matrices_str = [wavelet_matrices.toarray(), inverse_wavelet_matrices.toarray()]

        # -------------------------------#
        # Regularized self-representation
        # -------------------------------#
        # ------------ Regularized self-representation ----------------#
        graph_laplacian = torch.tensor(graph_laplacian).to(device)
        X_train = torch.as_tensor(X_train).to(device)
        X_validation = torch.as_tensor(X_validation).to(device)

        # Seting parameters
        rsr_alpha = alpha
        rsr_beta = beta

        # X_train
        print_func("Regularized self-representation for X_train.")
        for i in range(X_train.shape[0]):
            P = Regularized_SelfRepresentation(X_train[i, :, :], graph_laplacian, iter=args.rsr_iter, alpha=rsr_alpha,
                                               beta=rsr_beta, tolerance=args.rsr_tolerance, device=device)
            X_train[i, :, :] = torch.mm(X_train[i, :, :], P)

        # X_validation
        print_func("Regularized self-representation for X_validation.")
        for i in range(X_validation.shape[0]):
            P = Regularized_SelfRepresentation(X_validation[i, :, :], graph_laplacian, iter=args.rsr_iter, alpha=rsr_alpha,
                                               beta=rsr_beta, tolerance=args.rsr_tolerance, device=device)
            X_validation[i, :, :] = torch.mm(X_validation[i, :, :], P)

        # -------------------------------#
        # Model training
        # -------------------------------#
        # building model
        model_DMSGWNN = models.GraphWavelet_RSR(input_shape=X_train.shape[1:],      # (29*360)
                                               hidden_dim_sgwnn=(256 * np.ones(layer, dtype=int)).tolist(),   # (128,128,128)->L:0.001970, CO:0.774272, CO2:0.469293
                                               output_shape=Y_train.shape[1:],     # (2, 10)
                                               wavelets_shape=wavelet_matrices_str[0].shape,    # (29*s, 29)
                                               dropout=0.5, device=device)
        # using GPU if self.device=cuda
        model_DMSGWNN = model_DMSGWNN.to(device)

        model_trainer = models.GraphWavelet_RSR_Trainer(model_DMSGWNN,
                                                            wavelet_matrices_str,
                                                            loss='mae',
                                                            lr=1e-4,
                                                            device=device)

        print_func(model_DMSGWNN)
        print_func("Model parameters: optimizer={}, loss={}".format(model_trainer.optimizer, model_trainer._loss))

        # initial
        train_loss_log = []
        validation_loss_log = []
        validation_loss_best = 100000
        best_epoch = 0

        bestTrainedModel_path = os.path.join(result_path_timestamp, 'trainedModels')
        early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=bestTrainedModel_path,
                                       name='Best_'+str_parameters+'.model', trace_func=print_func)

        for epoch in range(args.n_epoch):
            # training models
            train_loss = model_trainer.train_batch(X_train, Y_train, args.batch_size)
            train_loss_log.append(train_loss)

            # validating models
            validation_loss = model_trainer.validation_batch(X_validation, Y_validation)
            validation_loss_log.append(validation_loss)

            ###########Early Stopping ################
            early_stopping(validation_loss, model_trainer.model)

            if validation_loss_best >= validation_loss:
                validation_loss_best = validation_loss
                best_epoch = epoch

            print_func("\033[33m Epoch [{}/{}]: \t train_loss={:.6f}, validation_loss={:.6f},\033[0m"
                       "\t \033[31m best_epoch:{},\t validation_loss_best: {:.6f} \033[0m".format(
                       epoch + 1, args.n_epoch, train_loss, validation_loss, best_epoch, validation_loss_best))

            if early_stopping.early_stop:
                print("Early stopping")
                break

            if (epoch > 100 and validation_loss_best > 1000):
                print("Model divergence")
                break


        plt.figure('Training Loss {}'.format(n_plt))
        plt.plot(range(1, len(train_loss_log) + 1), train_loss_log, label='Training Loss')
        plt.plot(range(1, len(validation_loss_log) + 1), validation_loss_log, label='Validation Loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('TMSGWNN')
        plt.ylim(0, 0.2)  # consistent scale
        plt.xlim(0, len(train_loss_log) + 1)  # consistent scale
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(result_path_timestamp, 'figures/TMSGWNN_loss_' + str_parameters + '.png'), dpi=600, bbox_inches='tight')

        n_plt = n_plt + 1

    # -------------------------------#
    #         Testing model
    # -------------------------------#
    print_func("Testing model...\n")

    # 模型加载
    print_func("loading model")
    metric_predict_CO_time_list = []
    metric_predict_CO2_time_list = []
    metric_predict_GUR_time_list = []

    metric_predict_CO_parameter_list = []
    metric_predict_CO2_parameter_list = []
    metric_predict_GUR_parameter_list = []

    for alpha, beta, scaling_tau, num_wavelets, approximation_order, layer in product(*param_values):
        str_parameters = "alpha{}_beta{}_scalingTau{}_numWavelets{}_approximationOrder{}_Layer{}".format(
                         alpha, beta, scaling_tau, num_wavelets, approximation_order, layer)
        print_func("\033[33m gridSearchCV: alpha: {}, beta: {}, scaling_tau: {}, num_wavelets: {}, "
                   "approximation_order: {}, layer: {} \033[0m".format(alpha, beta, scaling_tau, num_wavelets, approximation_order, layer))

        # --------------------------------------------#
        # Graph laplacian and spectral graph wavelets
        # --------------------------------------------#
        print_func("Computing graph laplacian and spectral graph wavelets.\n")
        print_func("Graph_Laplacian.shape={}".format(correlation_matrix_MIC.shape))

        # ------- Computing Graph laplacian -----------#
        print_func("Computing graph laplacian.\n")

        # correlation_matrix_Pearson, correlation_matrix_MIC
        correlation_matrix = [correlation_matrix_Pearson, correlation_matrix_MIC]
        graph_laplacian = computing_graph_laplacian(correlation_matrix, mode='PM', threshold=0.1, sigma=0.1)

        graph_laplacian_density = len(graph_laplacian.nonzero()[0]) / (
                    graph_laplacian.shape[0] * graph_laplacian.shape[1])
        graph_laplacian_density = str(round(100 * graph_laplacian_density, 2))
        print_func("Density of graph_laplacian: " + graph_laplacian_density + "%.")

        # ------- Computing spectral graph wavelets -----------#
        print_func("Computing spectral graph wavelets.\n")
        sparse_graph_laplacian = sparse.csc_matrix(graph_laplacian)
        sparse_wavelets = models.SpectralGraphWavelets(sparse_graph_laplacian, scaling_tau=scaling_tau,
                                                       wavelets_tau=[0.01, 1],
                                                       num_wavelets=num_wavelets,
                                                       approximation_order=approximation_order,
                                                       tolerance=1e-5, trace_func=print_func)
        sparse_wavelets.calculate_all_wavelets()
        wavelet_matrices = sparse_wavelets.wavelet_matrices[0]
        inverse_wavelet_matrices = sparse_wavelets.wavelet_matrices[1]
        wavelet_matrices_str = [wavelet_matrices.toarray(), inverse_wavelet_matrices.toarray()]

        # -------------------------------#
        # Regularized self-representation
        # -------------------------------#
        # ------------ Regularized self-representation ----------------#
        graph_laplacian = torch.tensor(graph_laplacian).to(device)
        X_test = torch.as_tensor(X_test).to(device)

        # Seting parameters
        rsr_alpha = alpha
        rsr_beta = beta

        # X_test
        print_func("Regularized self-representation for X_test.")
        for i in range(X_test.shape[0]):
            P = Regularized_SelfRepresentation(X_test[i, :, :], graph_laplacian, iter=args.rsr_iter, alpha=rsr_alpha,
                                               beta=rsr_beta, tolerance=args.rsr_tolerance, device=device)
            X_test[i, :, :] = torch.mm(X_test[i, :, :], P)

        # -------------------------------#
        #         Model testing
        # -------------------------------#
        print_func("Model testing.\n")
        name_model = 'Best_' + str_parameters + '.model'
        bestTrainedModel_model_best = joblib.load(os.path.join(bestTrainedModel_path, name_model))

        y_predict = predict_GWNN(bestTrainedModel_model_best, X_test, Y_test, wavelet_matrices_str, device)
        Y_test_ture = Y_test

        Y_test_ture_CO = Y_test_ture[:, 0, :]
        Y_test_ture_CO2 = Y_test_ture[:, 1, :]
        Y_test_ture_GUR = Y_test_ture_CO2 / (Y_test_ture_CO + Y_test_ture_CO2)

        y_predict_CO = y_predict[:, 0, :]
        y_predict_CO2 = y_predict[:, 1, :]
        y_predict_GUR = y_predict_CO2 / (y_predict_CO + y_predict_CO2)

        time_step_pred = 1
        metric_predict_CO = metric(Y_test_ture_CO[:, time_step_pred - 1], y_predict_CO[:, time_step_pred - 1], print_func)
        metric_predict_CO_time_list.extend(metric_predict_CO)
        metric_predict_CO2 = metric(Y_test_ture_CO2[:, time_step_pred - 1], y_predict_CO2[:, time_step_pred - 1], print_func)
        metric_predict_CO2_time_list.extend(metric_predict_CO2)
        metric_predict_GUR = metric(Y_test_ture_GUR[:, time_step_pred - 1], y_predict_GUR[:, time_step_pred - 1], print_func)
        metric_predict_GUR_time_list.extend(metric_predict_GUR)

        time_step_pred = 2
        metric_predict_CO = metric(Y_test_ture_CO[:, time_step_pred - 1], y_predict_CO[:, time_step_pred - 1], print_func)
        metric_predict_CO_time_list.extend(metric_predict_CO)
        metric_predict_CO2 = metric(Y_test_ture_CO2[:, time_step_pred - 1], y_predict_CO2[:, time_step_pred - 1], print_func)
        metric_predict_CO2_time_list.extend(metric_predict_CO2)
        metric_predict_GUR = metric(Y_test_ture_GUR[:, time_step_pred - 1], y_predict_GUR[:, time_step_pred - 1], print_func)
        metric_predict_GUR_time_list.extend(metric_predict_GUR)

        time_step_pred = 4
        metric_predict_CO = metric(Y_test_ture_CO[:, time_step_pred - 1], y_predict_CO[:, time_step_pred - 1], print_func)
        metric_predict_CO_time_list.extend(metric_predict_CO)
        metric_predict_CO2 = metric(Y_test_ture_CO2[:, time_step_pred - 1], y_predict_CO2[:, time_step_pred - 1], print_func)
        metric_predict_CO2_time_list.extend(metric_predict_CO2)
        metric_predict_GUR = metric(Y_test_ture_GUR[:, time_step_pred - 1], y_predict_GUR[:, time_step_pred - 1], print_func)
        metric_predict_GUR_time_list.extend(metric_predict_GUR)

        time_step_pred = 5
        metric_predict_CO = metric(Y_test_ture_CO[:, time_step_pred - 1], y_predict_CO[:, time_step_pred - 1], print_func)
        metric_predict_CO_time_list.extend(metric_predict_CO)
        metric_predict_CO2 = metric(Y_test_ture_CO2[:, time_step_pred - 1], y_predict_CO2[:, time_step_pred - 1], print_func)
        metric_predict_CO2_time_list.extend(metric_predict_CO2)
        metric_predict_GUR = metric(Y_test_ture_GUR[:, time_step_pred - 1], y_predict_GUR[:, time_step_pred - 1], print_func)
        metric_predict_GUR_time_list.extend(metric_predict_GUR)

        # 1~30min average
        metric_predict_CO = metric_multi(Y_test_ture_CO, y_predict_CO, print_func)
        metric_predict_CO_time_list.extend(metric_predict_CO)
        metric_predict_CO2 = metric_multi(Y_test_ture_CO2, y_predict_CO2, print_func)
        metric_predict_CO2_time_list.extend(metric_predict_CO2)
        metric_predict_GUR = metric_multi(Y_test_ture_GUR, y_predict_GUR, print_func)
        metric_predict_GUR_time_list.extend(metric_predict_GUR)

        print_func("metric_predict_CO_time_list:\n{}".format(metric_predict_CO_time_list))
        print_func("metric_predict_CO2_time_list:\n{}".format(metric_predict_CO2_time_list))
        print_func("metric_predict_GUR_time_list:\n{}".format(metric_predict_GUR_time_list))

        metric_predict_CO_parameter_list.append(metric_predict_CO_time_list)
        metric_predict_CO2_parameter_list.append(metric_predict_CO2_time_list)
        metric_predict_GUR_parameter_list.append(metric_predict_GUR_time_list)

        metric_predict_CO_time_list = []
        metric_predict_CO2_time_list = []
        metric_predict_GUR_time_list = []

    metric_predict = np.concatenate((np.array(metric_predict_CO_parameter_list),
                                     np.array(metric_predict_CO2_parameter_list),
                                     np.array(metric_predict_GUR_parameter_list)), axis=0)
    metric_predict_df = pd.DataFrame(metric_predict)

    save_path_DIR = os.path.join(os.path.dirname(__file__), 'result/generalModel/resultData')
    save_path = os.path.join(save_path_DIR, "GURPrediction_DMSGWNN_parameters_Layer_result" + timestamp_str + ".csv")
    metric_predict_df.to_csv(save_path, index=False, header=False, encoding='utf-8')
    print_func("Result has saved!")


def metric(Y_test_ture, y_predict, print_func):
    metric_predict = [RMSE(Y_test_ture, y_predict), mean_absolute_error(Y_test_ture, y_predict),
                      SD(Y_test_ture, y_predict), r2_score(Y_test_ture, y_predict),
                      prediction_hit_rate_percentage_error(Y_test_ture, y_predict, 0.02),
                      prediction_hit_rate_percentage_error(Y_test_ture, y_predict, 0.01)]

    return np.array(metric_predict)

def metric_multi(Y_test_ture, y_predict, print_func):
    RMSE_average = []
    MAE_average = []
    SD_average = []
    R2_average = []
    HR2_average = []
    HR1_average = []

    for i in range(Y_test_ture.shape[1]):
        RMSE_average.append(RMSE(Y_test_ture[:, i], y_predict[:, i]))
        MAE_average.append(mean_absolute_error(Y_test_ture[:, i], y_predict[:, i]))
        SD_average.append(SD(Y_test_ture[:, i], y_predict[:, i]))
        R2_average.append(r2_score(Y_test_ture[:, i], y_predict[:, i]))
        HR2_average.append(prediction_hit_rate_percentage_error(Y_test_ture[:, i], y_predict[:, i], 0.02))
        HR1_average.append(prediction_hit_rate_percentage_error(Y_test_ture[:, i], y_predict[:, i], 0.01))

    RMSE_average = 1 / Y_test_ture.shape[1] * np.array(RMSE_average).sum()
    MAE_average = 1 / Y_test_ture.shape[1] * np.array(MAE_average).sum()
    SD_average = 1 / Y_test_ture.shape[1] * np.array(SD_average).sum()
    R2_average = 1 / Y_test_ture.shape[1] * np.array(R2_average).sum()
    HR2_average = 1 / Y_test_ture.shape[1] * np.array(HR2_average).sum()
    HR1_average = 1 / Y_test_ture.shape[1] * np.array(HR1_average).sum()

    metric_predict = [RMSE_average, MAE_average, SD_average, R2_average, HR2_average, HR1_average]

    return np.array(metric_predict)

def predict_GWNN(model_best, X_test, Y_test, wavelet_matrices_str, device):
    model_best.eval()

    wavelets_indices = torch.LongTensor(wavelet_matrices_str[0].nonzero()).to(device)
    wavelets_values = torch.tensor(wavelet_matrices_str[0][wavelet_matrices_str[0].nonzero()], dtype=torch.float64)
    wavelets_values = wavelets_values.view(-1).to(device)

    wavelets_inverse_indices = torch.LongTensor(wavelet_matrices_str[1].nonzero()).to(device)
    wavelets_inverse_values = torch.tensor(wavelet_matrices_str[1][wavelet_matrices_str[1].nonzero()], dtype=torch.float64)
    wavelets_inverse_values = wavelets_inverse_values.view(-1).to(device)

    with torch.no_grad():
        # Obtaining train dataset
        X_test_batch = X_test   # torch.tensor(X_test, dtype=torch.float64)
        Y_test_batch = torch.tensor(Y_test, dtype=torch.float64)

        # using GPU if self.device=cuda
        X_test_batch, Y_test_batch = X_test_batch.to(device), Y_test_batch.to(device)

        Y_test_batch_predict = model_best(wavelets_indices, wavelets_values,
                                          wavelets_inverse_indices, wavelets_inverse_values,
                                          X_test_batch)

        y_predict = Y_test_batch_predict.cpu().detach().numpy()
        return y_predict

def computing_graph_laplacian(correlation_matrix, mode='Pearson', threshold=0.2, sigma=1):
    '''
    computing graph_laplacian by a thresholded Gaussian kernel weighting function
    Parameters
    ----------
    correlation_matrix: similarity matrix
    threshold: threshold of elements of correlation_matrix
    sigma: pamameter of Gaussian kernel

    Returns
    -------
    graph_laplacian_normalized
    '''
    #-------------- Adjaceny matrix ----------------#
    correlation_matrix_Pearson = correlation_matrix[0]
    correlation_matrix_MIC = correlation_matrix[1]

    if mode == 'Pearson':
        graph_adj = correlation_matrix_Pearson
    elif mode == 'MIC':
        graph_adj = correlation_matrix_MIC
    elif mode == 'PM':
        graph_adj = 0.5 * (correlation_matrix_Pearson + correlation_matrix_MIC)

    graph_adj[graph_adj < threshold] = 0

    #--------------- Degree matrix -----------------#
    graph_degree = np.sum(graph_adj, axis=1)
    graph_degree_inv = np.power(graph_degree, -0.5)
    graph_degree_diag = np.diag(graph_degree)
    graph_degree_diag_inv = np.diag(graph_degree_inv)

    # ------------- Laplacian matrix ---------------#
    graph_laplacian = graph_degree_diag - graph_adj
    # Symmetric normalized Laplacian
    graph_laplacian_normalized = graph_degree_diag_inv.dot(graph_laplacian).dot(graph_degree_diag_inv)

    return graph_laplacian_normalized

def Regularized_SelfRepresentation(X, L, iter=50, alpha=0.1, beta=0.1, tolerance=1e-5, device='cpu'):
    X = X.double().to(device)
    L = L.double().to(device)
    U = torch.eye(X.shape[1]).double().to(device)

    XTX = torch.mm(X.T, X)
    XLX = torch.mm(torch.mm(X.T, L), X)

    for i in range(iter):
        A = XTX + alpha * XLX + beta * U
        P = torch.mm(A.inverse(), XTX)
        U = torch.diag_embed(1 / (2 * P.norm(2, 1)))

    P[torch.abs(P) < tolerance] = 0

    return P

def spectralGraphWavelets(L, scale=[0.5, 1]):
    if not isinstance(L, np.ndarray):
        L = np.array(L)

    eigenvalue, U = np.linalg.eig(L)
    eigenvalue_diag = np.diag(eigenvalue)
    filter = filter_heat(eigenvalue.max(), scale)
    n_filter = len(scale)
    dim_L = L.shape[0]
    wavelet_matrices = np.zeros((n_filter * dim_L, dim_L))
    for i in range(n_filter):
        wavelet_matrices[(i * dim_L):((i + 1) * dim_L), :] = np.matmul(np.matmul(U, filter[i](eigenvalue_diag)), U.T)

    inverse_wavelet_matrices = np.linalg.inv(np.matmul(wavelet_matrices.T, wavelet_matrices))
    inverse_wavelet_matrices = np.matmul(inverse_wavelet_matrices, wavelet_matrices.T)

    wavelet_matrices_str = [wavelet_matrices, inverse_wavelet_matrices]
    return wavelet_matrices_str

def filter_heat(graph_eigenvalue_max, tau):
    g = []
    for t in tau:
        g.append(lambda x, t=t: np.exp(-t * x / graph_eigenvalue_max))
    return g


if __name__ == "__main__":
    main()