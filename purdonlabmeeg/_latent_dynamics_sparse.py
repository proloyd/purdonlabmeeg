# Author: Proloy Das <pdas6@mgh.harvard.edu>
import numpy as np
from scipy import linalg
from ._spatial_map_utils import (update_sparse_spatial_hyperparameters,
                                 sparse_spatial_filter)
from ._spatial_map_utils import _spatial_extent

from ._latent_dynamics import Handler


class SparseHandler(Handler):
    def __init__(self, n_factors, n_channels, n_sources, n_samples, order):
        super().__init__(n_factors, n_channels, n_sources, n_samples, order)
        self.ks = None

    def initialize_sensor_map(self, ks, r, fws, sigma_ws=0.):
        super().initialize_sensor_map(None, None, r, fws, sigma_ws=0.)
        self.ks = np.asanyarray(ks).copy()

    def expectation_wrt_w(self, f, r):
        for sigma_w in self.sigma_ws:
            sigma_w[sigma_w < 0] = 0   # sigma_w is a diagonal matrix!
        super().expectation_wrt_w(f, r)

    def update_spatial_map(self, y, f, r, v=None, vinv=None):
        m = self.n_factors
        p = self.order
        x = self.xs[1][:, (p-1) * m:].T
        cov = self.sigma_xs[1][(p-1) * m:, (p-1) * m:]
        ll = sparse_spatial_filter(y, x, cov, f, r, self.ks,
                                   self.ws, self.sigma_ws, v, vinv)
        return ll

    def update_spatial_scale(self, f, r, sigma2=1.0, v=None, vinv=None):
        m = self.n_factors
        p = self.order
        x = self.xs[1][:, (p-1) * m:].T
        cov = self.sigma_xs[1][(p-1) * m:, (p-1) * m:]
        update_sparse_spatial_hyperparameters(x, cov, f, r, self.ks,
                                              self.ws, self.sigma_ws,
                                              sigma2, v, vinv)
        return self


class SparseExtentHandler(Handler):
    def __init__(self, n_factors, n_channels, n_sources, n_samples, order):
        super().__init__(n_factors, n_channels, n_sources, n_samples, order)
        self.ks = None

    def initialize_sensor_map(self, ks, r, fws, sigma_ws, f, conn, snr):
        print('initializing sensor maps')
        super().initialize_sensor_map(None, None, r, fws, sigma_ws=0.)
        whitener = np.sqrt(np.diag(r))
        whitened_gain = f / whitener[:, None]
        whitened_signals = fws.T / whitener[:, None]
        _whitened_gain_norm = np.sqrt((whitened_gain ** 2).sum(axis=0))
        _whitened_gain_norm[:] = 1.
        _whitened_gain = whitened_gain / _whitened_gain_norm[None, :]
        print('preparing gain matrix')
        u, s, vh = linalg.svd(_whitened_gain, full_matrices=False)
        print('initializing spatial maps')
        ks = [spatial_map_initialization(u, s, vh, _whitened_gain_norm,
                                         whitened_signal, conn, snr)
              for whitened_signal in whitened_signals.T]
        self.ks = ks

    def expectation_wrt_w(self, f, r):
        for sigma_w in self.sigma_ws:
            sigma_w[sigma_w < 0] = 0   # sigma_w is a diagonal matrix!
        super().expectation_wrt_w(f, r)

    def update_spatial_map(self, y, f, r, conn):
        m = self.n_factors
        p = self.order
        x = self.xs[1][:, (p-1) * m:].T
        cov = self.sigma_xs[1][(p-1) * m:, (p-1) * m:]
        ll = _spatial_extent.extent_spatial_filter(y, x, cov, f, r, self.ks,
                                                   self.ws, self.sigma_ws,
                                                   conn)
        return ll

    def update_spatial_scale(self, f, r, sigma2=1.0, conn=None):
        m = self.n_factors
        p = self.order
        x = self.xs[1][:, (p-1) * m:].T
        cov = self.sigma_xs[1][(p-1) * m:, (p-1) * m:]
        ks = _spatial_extent.update_sparse_spatial_hyperparameters(
                    x, cov, f, r, self.ks, self.ws, self.sigma_ws, sigma2, conn
                    )
        self.ks = ks
        return self


def spatial_map_initialization(u, s, vh, _whitened_gain_norm, whitened_signal,
                               conn, snr=10.):
    s_inv = 1 / s
    s_inv[s < s.max() * 1e-15] = 0.
    # s_inv[50:] = 0.0
    x = (vh.T * s_inv[None, :]).dot(u.T.dot(whitened_signal))
    x /= _whitened_gain_norm
    gamma1 = (conn.dot(x) ** 2) / 1000
    gamma1[:] = 0.001   # Trying with large values of gamma1, adaptive?
    gamma2 = x ** 2
    gamma2 += gamma2.max()
    return (gamma1, gamma2)
