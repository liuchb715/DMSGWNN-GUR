# coding=utf-8
import torch
import torch.optim
import torch.nn as nn
from tqdm import trange

from utils.utils import print_log

class GraphWavelet_RSR_Trainer(nn.Module):
    """
    Denoising Multi-scale Spectral Graph Wavelet Neural Networks for Gas Utilization Ratio Prediction in Blast Furnace.
    IEEE Transactions on Neural Networks and Learning Systems (TNNLS), 2024, DOI: 10.1109/TNNLS.2024.3453280.
    """
    def __init__(self, model, wavelet_matrices_str, loss='mse', lr=1e-3, device='cpu'):  # mse_with_regularizer
        super(GraphWavelet_RSR_Trainer, self).__init__()
        self.model = model
        self.lr = lr
        self._loss = loss
        self.device = device
        self.wavelet_matrices_str = wavelet_matrices_str

        self.wavelets_indices = torch.LongTensor(self.wavelet_matrices_str[0].nonzero()).to(self.device)
        self.wavelets_values = torch.tensor(self.wavelet_matrices_str[0][self.wavelet_matrices_str[0].nonzero()], dtype=torch.float64)
        self.wavelets_values = self.wavelets_values.view(-1).to(self.device)

        self.wavelets_inverse_indices = torch.LongTensor(self.wavelet_matrices_str[1].nonzero()).to(self.device)
        self.wavelets_inverse_values = torch.tensor(self.wavelet_matrices_str[1][self.wavelet_matrices_str[1].nonzero()], dtype=torch.float64)
        self.wavelets_inverse_values = self.wavelets_inverse_values.view(-1).to(self.device)

        # self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr, alpha=0.9)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.99), eps=1e-08, weight_decay=0.0005)

    def train_batch(self, X_train, Y_train, batch_size):
        print_log("\033[32m Start of batch training...")
        self.model.train()
        loss_train_total = 0.0

        # Number of train per batch
        if X_train.shape[0] % batch_size == 0:
            n_epochs = int(X_train.shape[0] / batch_size)
        else:
            n_epochs = int(X_train.shape[0] / batch_size) + 1

        iter = 0
        # for iter in range(0, X_train.shape[0], batch_size):
        self.epochs = trange(n_epochs, desc="Loss")
        for epoch in self.epochs:
            # Extracting batch train_data set
            if iter + batch_size > X_train.shape[0]:
                X_train_batch, Y_train_batch = X_train[iter:, :, :], Y_train[iter:, :, :]
            else:
                X_train_batch, Y_train_batch = X_train[iter:(iter + batch_size), :, :], Y_train[iter:(iter + batch_size), :, :]

            # training model
            # X_train_batch = torch.tensor(X_train_batch, dtype=torch.float64)
            Y_train_batch = torch.tensor(Y_train_batch, dtype=torch.float64)

            # using GPU if self.device=cuda
            X_train_batch, Y_train_batch = X_train_batch.to(self.device), Y_train_batch.to(self.device)

            self.optimizer.zero_grad()
            Y_train_batch_predict = self.model(self.wavelets_indices, self.wavelets_values,
                                               self.wavelets_inverse_indices, self.wavelets_inverse_values,
                                               X_train_batch)

            loss = self.loss(Y_train_batch, Y_train_batch_predict)
            loss.backward()
            self.optimizer.step()

            self.epochs.set_description("GWNN (Train Loss=%g)" % round(loss.item(), 4))
            iter = iter + batch_size

            loss_train_total = loss_train_total + loss.item()

        print_log("End of batch training...\033[0m")

        return loss_train_total

    def validation_batch(self, X_validation, Y_validation):
        self.model.eval()
        with torch.no_grad():
            # Obtaining train dataset
            X_validation_batch = X_validation #torch.tensor(X_validation, dtype=torch.float64)
            Y_validation_batch = torch.tensor(Y_validation, dtype=torch.float64)

            # using GPU if self.device=cuda
            X_validation_batch, Y_validation_batch = X_validation_batch.to(self.device), Y_validation_batch.to(self.device)

            Y_validation_batch_predict = self.model(self.wavelets_indices, self.wavelets_values,
                                                    self.wavelets_inverse_indices, self.wavelets_inverse_values,
                                                    X_validation_batch)

            validation_loss = self.loss(Y_validation_batch, Y_validation_batch_predict)

        return validation_loss.item()

    def predict(self, model_best, X_test, Y_test):
        model_best.eval()

        with torch.no_grad():
            # Obtaining train dataset
            X_test_batch = torch.tensor(X_test, dtype=torch.float64)
            Y_test_batch = torch.tensor(Y_test, dtype=torch.float64)

            # using GPU if self.device=cuda
            X_test_batch, Y_test_batch = X_test_batch.to(self.device), Y_test_batch.to(self.device)

            Y_test_batch_predict = model_best(self.wavelets_indices, self.wavelets_values,
                                              self.wavelets_inverse_indices, self.wavelets_inverse_values,
                                              X_test_batch)

            y_predict = Y_test_batch_predict.cpu().detach().numpy()
            return y_predict

    def loss(self, inputs, targets):
        if self._loss == 'mse':
            loss_func = torch.nn.MSELoss()
            return loss_func(inputs, targets)
        if self._loss == 'mae':
            loss_func = torch.nn.L1Loss()
            return loss_func(inputs, targets)
        if self._loss == 'huber':
            loss_func = torch.nn.SmoothL1Loss()
            return loss_func(inputs, targets)

        raise NameError('Loss not supported:', self._loss)