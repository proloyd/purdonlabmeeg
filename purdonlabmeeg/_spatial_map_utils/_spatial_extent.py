# Author: Proloy Das <pdas6@mgh.harvard.edu>
import collections
import numpy as np
from scipy import linalg
from . import _sparse_map
from numba import jit, njit, float32, float64, int32


def prior_cov_inverse(gamma1, gamma2, conn):
    A = np.diag(gamma2).astype(np.float64)
    gamma1 = gamma1.astype(np.float64)
    _, indices, indptr = conn.data, conn.indices, conn.indptr
    return _prior_cov_inverse_opt(A, gamma1, indices, indptr)


@jit(["float64[:,::1](float64[:, ::1], float64[::1], int32[::1], int32[::1])"],
     nopython=True, cache=True)
def _prior_cov_inverse_opt(A, gamma1, indices, indptr):
    for k in (range(gamma1.shape[0])):
        i = indices[indptr[k]]
        j = indices[indptr[k] + 1]
#         v1 = conn[k].dot(A)
        v = A[i] - A[j]
#         temp1 = conn[k].dot(v1.T)
        temp = v[i] - v[j]
#         assert temp==temp1
        temp += gamma1[k]
        if np.abs(temp) > 1e-15:
            # print(f"{k}: {gamma1[k] + temp}")
            w = 1 / temp
            A -= w * np.outer(v, v)
    return A


def _gamma1_update(x, sigma_x, gamma1, conn):
    _, indices, indptr = conn.data, conn.indices, conn.indptr
    return _gamma1_update_opt(x, sigma_x, gamma1, indices, indptr)


@jit(["float64[::1](float64[::1], float64[:, ::1],  float64[::1],"
      "int32[::1], int32[::1])"], nopython=True, cache=True)
def _gamma1_update_opt(x, sigma_x, gamma1, indices, indptr):
    "inplace update"
    for k in (range(gamma1.shape[0])):
        i = indices[indptr[k]]
        j = indices[indptr[k] + 1]
        gamma1[k] = (x[i] - x[j]) ** 2
        # conn_p * sigma_x * conn_p.T = (sigma_x[i] - sigma_x[j][i]
        #                                - (sigma_x[i] - sigma_x[j][j]
        gamma1[k] += sigma_x[i, i] + sigma_x[j, j] - 2 * sigma_x[i, j]
    return gamma1


def prior_cov_inverse_sparse(gamma1, gamma2, conn):
    conn_gamma2 = conn.dot(np.diag(gamma2))
    temp = (np.diag(gamma1) + conn.dot(conn_gamma2.T))
    out = np.diag(gamma2) - conn_gamma2.T.dot(np.linalg.solve(temp,
                                                              conn_gamma2))
    return out


def _gamma1_update_sparse(x, sigma_x, gamma1, conn):
    gamma1 = conn.dot(x) ** 2
    temp = conn.multiply(conn.dot(sigma_x)).sum(axis=1)
    temp = np.squeeze(np.asarray(temp))
    assert gamma1.shape == temp.shape
    gamma1 += temp
    return gamma1


def _solve_for_qw(gamma1, gamma2, conn, zeta, xi2, sigma_n, gain):
    k = prior_cov_inverse_sparse(gamma1, gamma2, conn)
    return _sparse_map._solve_for_qw(k, zeta, xi2, sigma_n, gain)


def _solve_for_qw_chol(gamma1, gamma2, conn, zeta, xi2, sigma_n, gain):
    k = prior_cov_inverse_sparse(gamma1, gamma2, conn)
    return _sparse_map._solve_for_qw_chol(k, zeta, xi2, sigma_n, gain)


def extent_spatial_filter(y, x, cov, gain, sigma_n, ks, ws, sigma_ws,
                          conn=None):
    assert y.shape[1] == x.shape[1]
    n = x.shape[1]
    temp = x.dot(x.T) + n * cov
    fw = gain.dot(ws.T)
    total_ll = 0.
    for i in range(len(ws)):
        xi2 = temp[i, i]
        temp[i, i] = 0.0
        zeta = (y * x[i]).sum(axis=-1) - (fw * temp[:, i]).sum(axis=-1)
        try:
            this_w, this_sigma_w, ll = _solve_for_qw_chol(*ks[i], conn, zeta,
                                                          xi2, sigma_n, gain)
        except np.linalg.LinAlgError:
            this_w, this_sigma_w, ll = _solve_for_qw(*ks[i], conn, zeta, xi2,
                                                     sigma_n, gain)
        fw[:, i] = gain.dot(this_w)
        ws[i][:] = this_w
        sigma_ws[i][:] = this_sigma_w
        total_ll += ll
    return total_ll


def _update_hyperparameters(mu_w, gain, sigma_n, sigma_w, xi2, k, sigma2,
                            conn):
    gamma1, gamma2 = k
    # methos 1 (Easy)
    c = mu_w * mu_w
    gamma2[:] = c + np.diag(sigma_w)
    gamma1 = _gamma1_update_sparse(mu_w, sigma_w, gamma1, conn)
    return (gamma1, gamma2)


def update_sparse_spatial_hyperparameters(x, cov, gain, sigma_n, ks,
                                          ws, sigma_ws, sigma2=1., conn=None):
    n = x.shape[1]
    xi2s = (x * x).sum(axis=1)
    xi2s += n * np.diag(cov)
    ks = [_update_hyperparameters(mu_w, gain, sigma_n, sigma_w, xi2, ks[i],
                                  sigma2=sigma2, conn=conn)
          for i, (mu_w, sigma_w, xi2) in enumerate(zip(ws, sigma_ws, xi2s))]
    # for i, (mu_w, sigma_w, xi2) in enumerate(zip(ws, sigma_ws, xi2s)):
    #     _ = _update_hyperparameters(mu_w, gain, sigma_n, sigma_w, xi2, ks[i],
    #                                 sigma2=sigma2, conn=conn)
    return ks


def test_update_hyperparameters(var=None, seed=0, snr=10.):
    from matplotlib import pyplot as plt
    from numpy.random import default_rng
    from scipy import sparse

    rng = default_rng(seed=seed)

    m = 1000
    pos = (rng.uniform(-10, 10, size=m))
    sorted_idx = np.argsort(pos)
    lows = [-7, -3.25, 1.4, 6]
    highs = [-5, -2.75, 1.6, 8]
    spatial_maps = [1 / (1 + np.exp(-100*(pos-low))) -
                    1 / (1 + np.exp(-100*(pos-high)))
                    for low, high in zip(lows, highs)]
    spatial_maps = np.asanyarray(spatial_maps)
    spatial_maps = spatial_maps[1] - spatial_maps[2]
    fig, ax = plt.subplots(2)
    ax[0].plot(pos[sorted_idx], spatial_maps.T[sorted_idx])
    gain = rng.standard_normal((100, m))
    gain /= np.linalg.norm(gain, 'fro')
    y = gain.dot(spatial_maps.T)
    noise = rng.standard_normal(y.shape)
    sigma_n = y * y / snr
    y_noisy = y + np.sqrt(sigma_n) * noise
    ax[1].plot(y)
    ax[1].plot(y_noisy)

    nedges = len(sorted_idx) - 1
    nsources = m
    conn = sparse.coo_matrix((np.ones(nedges), (range(nedges), sorted_idx[:-1]))
                             , shape=(nedges, nsources))
    conn -= sparse.coo_matrix((np.ones(nedges), (range(nedges), sorted_idx[1:]))
                              , shape=(nedges, nsources))
    last = int(np.where(sorted_idx == m-1)[0])
    edges = conn.dot(spatial_maps)  # [np.concatenate((sorted_idx[:last],
    #                   sorted_idx[last:]))
    ax[0].plot(pos[sorted_idx][:-1], edges)
    fig.show()

    whitened_gain = gain / np.sqrt(sigma_n)[:, None]
    whitened_signal = y_noisy / np.sqrt(sigma_n)
    sigma_n = np.eye(sigma_n.shape[0])
    # lambda2 = (y_noisy.shape[0] / ((whitened_gain * whitened_gain).sum()))
    # lambda2 *= (snr if var is None else var)
    # gamma2 = lambda2 * np.ones(m)
    # gamma1 = 10**20 * lambda2 * nedges / nsources * np.ones(nedges)
    # k = (gamma1, gamma2)
    # x, sigma_x, ll = _sparse_map._solve_for_qw(np.diag(gamma2), whitened_signal
    #                                            , 1., sigma_n, whitened_gain)
    _whitened_gain_norm = np.sqrt((whitened_gain * whitened_gain).sum(axis=0))
    _whitened_gain_norm[:] = 1.
    _whitened_gain = whitened_gain / _whitened_gain_norm[None, :]
    u, s, vh = linalg.svd(_whitened_gain, full_matrices=False)
    s_inv = 1 / s
    x = (vh.T * s_inv[None, :]).dot(u.T.dot(whitened_signal))
    x /= _whitened_gain_norm
    ax[0].plot(pos[sorted_idx], gain.T.dot(y_noisy)[sorted_idx])
    ax[0].plot(pos[sorted_idx], x[sorted_idx])
    # x = gain.T.dot(y_noisy)
    # x = spatial_maps + var * rng.standard_normal(spatial_maps.shape)
    # gamma1 = (conn.dot(x) ** 2)
    # gamma1 = 1 / gamma1
    gamma1 = (conn.dot(spatial_maps) ** 2) + 0.01
    gamma1[:] = 0.001
    # gamma1[:] = 1. / 100
    gamma2 = x ** 2
    gamma2 += gamma2.max()
    # gamma2 += 1 / snr
    # gamma2 = x ** 2 + snr / 100
    k = (gamma1, gamma2)
    ipdb.set_trace()

    sigma2 = 1.0
    fig, ax = plt.subplots(3)
    ax[0].plot(spatial_maps[sorted_idx], label='Ground truth')
    plt.tight_layout()
    fig.show()
    for i in range(200):
        x, sigma_x, ll = _solve_for_qw(*k, conn, whitened_signal, 1., sigma_n,
                                       whitened_gain)
        ax[2].plot(np.abs(x))
        ax[2].plot(np.sqrt(x * x + np.diag(sigma_x)))
        # ipdb.set_trace()
        ax[0].plot(x[sorted_idx], label=f'iter-{i}')
        ax[0].legend()
        k = _update_hyperparameters(x, whitened_gain, sigma_n, sigma_x, 1., k,
                                    sigma2, conn)
        ax[1].plot(k[1], label=f'iter-{i}')
        ax[0].legend()
        fig.canvas.draw()
        fig.canvas.flush_events()
        # ipdb.set_trace()

    ci = np.diag(sigma_x).copy()
    ci[ci < 0] = 0.
    ci = np.sqrt(ci)
    fig, ax = plt.subplots()
    ax.plot(spatial_maps, label='Ground truth')
    ax.plot(x, label=f'iter-{i}')
    ax.plot(x + 2 * ci, label='upper')
    ax.plot(x - 2 * ci, label='lower')
    ax.legend()
    fig.show()
