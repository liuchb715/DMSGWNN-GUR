import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

class SpectralGraphWavelets(object):
    """
    Object to sparsify the wavelet coefficients for a graph.
    """
    def __init__(self, graph_laplacian, scaling_tau=5, wavelets_tau=[0.02, 0.15], num_wavelets=3,
                 approximation_order=3, tolerance=0, trace_func=print):
        """
        :param graph_laplace: Laplace of graph object.
        :param scale: Kernel scale length parameter.
        :param approximation_order: Chebyshev polynomial order.
        :param tolerance: Tolerance for sparsification.
        """
        self.graph_laplacian = graph_laplacian
        self.num_nodes_graph = self.graph_laplacian.shape[0]
        self.lmax_graph_laplacian = self.graph_laplacian_eigenvalue_max(self.graph_laplacian)

        self.num_scales = num_wavelets + 1   
        self.scaling_tau = scaling_tau
        self.wavelets_tau = np.linspace(wavelets_tau[0], wavelets_tau[1], num_wavelets)

        self.approximation_order = approximation_order
        self.tolerance = tolerance
        self.trace_func = trace_func

        self.wavelet_matrices = []   

    def calculate_wavelet(self):
        """
        Creating sparse wavelets.
        :return remaining_waves: Sparse matrix of attenuated wavelets.
        """
        impulse = np.eye(self.num_nodes_graph, dtype=int)
        wavelet_coefficients = self.cheby_op(self.graph_laplacian, self.chebyshev, impulse)

        wavelet_coefficients[wavelet_coefficients < self.tolerance] = 0
        ind_1, ind_2 = wavelet_coefficients.nonzero()
        remaining_waves = sparse.csr_matrix((wavelet_coefficients[ind_1, ind_2], (ind_1, ind_2)),
                                            shape=(self.num_nodes_graph * self.num_scales, self.num_nodes_graph),
                                            dtype=np.float32)
        return remaining_waves

    def normalize_matrices(self):
        """
        Normalizing the wavelet and inverse wavelet matrices.
        """
        self.trace_func("Normalizing the sparsified wavelets.")
        for i, phi_matrix in enumerate(self.wavelet_matrices):
            self.wavelet_matrices[i] = normalize(self.wavelet_matrices[i], norm='l1', axis=1)

    def calculate_density(self):
        """
        Calculating the density of the sparsified wavelet matrices.
        """
        wavelet_density = len(self.wavelet_matrices[0].nonzero()[0]) / (self.num_nodes_graph*self.num_scales * self.num_nodes_graph)
        wavelet_density = str(round(100 * wavelet_density, 2))
        inverse_wavelet_density = len(self.wavelet_matrices[1].nonzero()[0]) / (self.num_nodes_graph*self.num_scales * self.num_nodes_graph)
        inverse_wavelet_density = str(round(100 * inverse_wavelet_density, 2))
        self.trace_func("Density of wavelets: "+wavelet_density+"%.")
        self.trace_func("Density of inverse wavelets: "+inverse_wavelet_density+"%.")

    def calculate_all_wavelets(self):
        """
        Graph wavelet coefficient calculation.
        """
        self.trace_func("Wavelet calculation and sparsification started.")
        self.trace_func("Max eigenvalue of graph laplacian:{}".format(self.lmax_graph_laplacian[0]))

        self.filter = self.filter(graph_eigenvalue_max=self.lmax_graph_laplacian, tau=self.wavelets_tau)

        #------------- Computing wavelets --------------#
        self.chebyshev = self.compute_cheby_coeff(self.filter, num_filter=self.num_scales,
                                                  graph_eigenvalue_max=self.lmax_graph_laplacian, m=self.approximation_order)

        sparsified_wavelets = self.calculate_wavelet()

        self.wavelet_matrices.append(sparsified_wavelets)

        # --------- Computing inverse wavelets ------------#
        sparsified_wavelets_inv = sparse.linalg.inv(sparsified_wavelets.T.dot(sparsified_wavelets)).dot(sparsified_wavelets.T)
        sparsified_wavelets_inv_numpy = sparsified_wavelets_inv.toarray()
        sparsified_wavelets_inv_numpy[sparsified_wavelets_inv_numpy < self.tolerance] = 0
        ind_1, ind_2 = sparsified_wavelets_inv_numpy.nonzero()
        remaining_wavelets_inv = sparse.csr_matrix((sparsified_wavelets_inv_numpy[ind_1, ind_2], (ind_1, ind_2)),
                                            shape=(self.num_nodes_graph, self.num_nodes_graph * self.num_scales),
                                            dtype=np.float32)

        self.wavelet_matrices.append(remaining_wavelets_inv)

        self.normalize_matrices()
        self.calculate_density()

    def graph_laplacian_eigenvalue_max(self, L):

        lmax = sparse.linalg.eigsh(L, k=1)
        lmax = lmax[0]
        lmax *= 1.01  

        return lmax

    def filter(self, graph_eigenvalue_max, tau):
        g = []

        # Low pass filtering, scaling filter
        g.append(lambda x: np.exp(-1 * self.scaling_tau * x / graph_eigenvalue_max))

        # Bass pass filtering, wavelet filter
        for t in tau:
            g.append(lambda x, t=t: np.exp(-1 * np.power((x / graph_eigenvalue_max - 0.5), 2) / (2 * t * t)))
        return g

    def filter_heat(self, graph_eigenvalue_max, tau):
        g = []
        for t in tau:
            g.append(lambda x, t=t: np.exp(-t * x / graph_eigenvalue_max))
        return g

    def compute_cheby_coeff(self, filter, num_filter=1, graph_eigenvalue_max=2, m=30, N=None, *args, **kwargs):
        if not N:
            N = m + 1

        a_arange = [0, graph_eigenvalue_max]

        a1 = (a_arange[1] - a_arange[0]) / 2
        a2 = (a_arange[1] + a_arange[0]) / 2
        c = np.zeros((num_filter, m + 1))

        tmpN = np.arange(N)
        num = np.cos(np.pi * (tmpN + 0.5) / N)
        for i in range(num_filter):
            for o in range(m + 1):
                c[i][o] = 2. / N * np.dot(filter[i](a1 * num + a2), np.cos(np.pi * o * (tmpN + 0.5) / N))
        return c

    def cheby_op(self, graph_laplacian, c, signal, **kwargs):
        r"""
        Chebyshev polynomial of graph Laplacian applied to vector.

        Parameters
        ----------
        G : Graph Laplacian
        c : ndarray or list of ndarrays
            Chebyshev coefficients for a Filter or a Filterbank
        signal : ndarray
            Signal to filter

        Returns
        -------
        r : ndarray
            Result of the filtering

        """
        # Handle if we do not have a list of filters but only a simple filter in cheby_coeff.
        if not isinstance(c, np.ndarray):
            c = np.array(c)

        c = np.atleast_2d(c)
        Nscales, M = c.shape

        num_nodes = graph_laplacian.shape[0]

        if M < 2:
            raise TypeError("The coefficients have an invalid shape")

        # thanks to that, we can also have 1d signal.
        try:
            Nv = np.shape(signal)[1]
            r = np.zeros((num_nodes * Nscales, Nv))
        except IndexError:
            r = np.zeros((num_nodes * Nscales))

        lmax = self.graph_laplacian_eigenvalue_max(graph_laplacian)
        a_arange = [0, lmax]

        a1 = float(a_arange[1] - a_arange[0]) / 2.
        a2 = float(a_arange[1] + a_arange[0]) / 2.

        twf_old = signal  # T0=1
        twf_cur = (graph_laplacian.dot(signal) - a2 * signal) / a1    # T1=y

        tmpN = np.arange(num_nodes, dtype=int)
        for i in range(Nscales):
            r[tmpN + num_nodes * i] = 0.5 * c[i, 0] * twf_old + c[i, 1] * twf_cur

        factor = 2 / a1 * (graph_laplacian - a2 * sparse.eye(num_nodes))
        for k in range(2, M):
            twf_new = factor.dot(twf_cur) - twf_old   # T2 = 2y*T1 - T0
            for i in range(Nscales):
                r[tmpN + num_nodes * i] += c[i, k] * twf_new

            twf_old = twf_cur
            twf_cur = twf_new



        return r 