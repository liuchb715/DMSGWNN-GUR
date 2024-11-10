import torch
from torch_sparse import spspmm, spmm
import torch.nn.functional as F

class GraphWavelet_RSR(torch.nn.Module):
    """
    "Denoising Multi-scale Spectral Graph Wavelet Neural Networks for Gas Utilization Ratio Prediction in Blast Furnace.
    IEEE Transactions on Neural Networks and Learning Systems (TNNLS), 2024, DOI: 10.1109/TNNLS.2024.3453280."

    Graph Wavelet Neural Network class.
    :param input_shape: dimension of input.
    :param hidden_dim_sgwnn: dimension of hidden layer, such as hidden_dim = [32, 8].
    :param output_shape: dimension of output, such as output_dim=1.
    :param wavelets_shape: Shape of wavelets, such as wavelets_shape=[58, 29].
    :param device: Device used for training.
    """
    def __init__(self, input_shape, hidden_dim_sgwnn, output_shape, wavelets_shape, dropout=0.5, device='cpu'):
        super(GraphWavelet_RSR, self).__init__()
        self.input_variable = input_shape[0]
        self.input_dim = input_shape[1]

        self.hidden_dim_sgwnn = hidden_dim_sgwnn

        self.output_variable = output_shape[0]
        self.output_dim = output_shape[1]

        self.wavelets_shape = wavelets_shape
        self.dropout = dropout
        self.device = device

        self.dropout = dropout
        self.device = device

        # set graph wavelet convolution layer
        self.gwconvs = []
        input_dim_tmp = self.input_dim
        for hid_dim in self.hidden_dim_sgwnn:
            output_dim_tmp = hid_dim
            self.gwconvs.append(
                GraphWaveletLayer(input_dim_tmp, output_dim_tmp, self.wavelets_shape, self.dropout, self.device))
            input_dim_tmp = output_dim_tmp
        self.gwconvs = torch.nn.ModuleList(self.gwconvs)

        # output layer
        # self.regressor_out = torch.nn.Linear(self.input_variable * self.hidden_dim_sgwnn[-1], self.output_dim)
        self.regressor_CO = torch.nn.Linear(self.hidden_dim_sgwnn[-1], self.output_dim)
        self.regressor_CO2 = torch.nn.Linear(self.hidden_dim_sgwnn[-1], self.output_dim)

    def forward(self, wavelets_indices, wavelets_values, wavelets_inverse_indices,
                wavelets_inverse_values, x_features):
        """
        Forward propagation pass.
        :param graph_laplacian: graph_laplacian.
        :param wavelets: wavelet matrix.
        :param wavelets_inverse: Inverse wavelet matrix.
        :param x_features: Input feature matrix.
        :param predictions: Predicted node value vector, such as [n,2].
        """
        # ---------- Spectral Graph Wavelet Neural Network ------------#
        for i, gwconvs in enumerate(self.gwconvs):
            x_features = gwconvs(wavelets_indices, wavelets_values,
                                 wavelets_inverse_indices, wavelets_inverse_values, x_features)

        # --------------------- Linear Output -------------------------#
        x_features_CO = x_features[:, -2, :]   # CO
        x_features_CO2 = x_features[:, -1, :]  # CO2

        predictions_CO = self.regressor_CO(x_features_CO).unsqueeze(dim=1)   
        predictions_CO2 = self.regressor_CO2(x_features_CO2).unsqueeze(dim=1)   
        predictions = torch.cat((predictions_CO, predictions_CO2), dim=1)

        return predictions

class GraphWaveletLayer(torch.nn.Module):
    """
    Abstract Graph Wavelet Layer class.
    :param input_dim: dimension of input_dim.
    :param output_dim: dimension of output_dim.
    :param wavelets_shape: dimension of wavelets_shape.
    :param device: Device to train on.
    """
    def __init__(self, input_dim, output_dim, wavelets_shape, dropout, device):
        super(GraphWaveletLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.wavelets_row = wavelets_shape[0]
        self.wavelets_col = wavelets_shape[1]
        self.device = device
        self.dropout = dropout

        self.define_parameters()
        self.init_parameters()

    def define_parameters(self):
        """
        Defining diagonal filter matrix (Theta in the paper) and weight matrix.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.input_dim, self.output_dim))

        # sparse diagonal(F)
        self.diagonal_weight_indices = torch.LongTensor([[node for node in range(self.wavelets_row)],
                                                         [node for node in range(self.wavelets_row)]])
        self.diagonal_weight_indices = self.diagonal_weight_indices.to(self.device)
        self.diagonal_weight_filter = torch.nn.Parameter(torch.Tensor(self.wavelets_row, 1))

    def init_parameters(self):
        """
        Initializing the diagonal filter and the weight matrix.
        """
        torch.nn.init.uniform_(self.diagonal_weight_filter, 0.9, 1.1)
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, wavelets_indices, wavelets_values, wavelets_inverse_indices, wavelets_inverse_values, x_features):
        """
        Forward propagation pass.
        :param wavelets_indices: Sparse wavelet matrix index pairs.
        :param wavelets_values: Sparse wavelet matrix values.
        :param wavelets_inverse_indices: Inverse wavelet matrix index pairs.
        :param wavelets_inverse_values: Inverse wavelet matrix values.
        :param features: Feature matrix.
        :return localized_features: Filtered feature matrix extracted.
        """
        # (wavelets_col x wavelets_row) * (wavelets_row x wavelets_row) -> wavelets_col x wavelets_row
        rescaled_phi_indices, rescaled_phi_values = spspmm(wavelets_inverse_indices,
                                                           wavelets_inverse_values,
                                                           self.diagonal_weight_indices,
                                                           self.diagonal_weight_filter.view(-1),
                                                           self.wavelets_col,
                                                           self.wavelets_row,
                                                           self.wavelets_row)

        # (wavelets_col x wavelets_row) * (wavelets_row x wavelets_col) -> wavelets_col x wavelets_col
        phi_product_indices, phi_product_values = spspmm(rescaled_phi_indices,
                                                         rescaled_phi_values,
                                                         wavelets_indices,
                                                         wavelets_values,
                                                         self.wavelets_col,
                                                         self.wavelets_row,
                                                         self.wavelets_col)

        # x_features shape: N x variable_dim x input_dim
        x_features_SGWNN = torch.zeros((x_features.shape[0], x_features.shape[1], self.output_dim)).to(self.device)
        for i in range(x_features.shape[0]):
            # (variable_dim x input_dim) * (input_dim, output_dim) -> variable_dim x output_dim
            filtered_features = torch.mm(x_features[i, :, :], self.weight_matrix)
            localized_features = spmm(phi_product_indices,
                                      phi_product_values,
                                      self.wavelets_col,
                                      self.wavelets_col,
                                      filtered_features)
            x_features_SGWNN[i, :, :] = localized_features

        x_features_SGWNN = F.leaky_relu(x_features_SGWNN, negative_slope=0.2, inplace=True)
        x_features_SGWNN = F.dropout(x_features_SGWNN, self.dropout)

        return x_features_SGWNN