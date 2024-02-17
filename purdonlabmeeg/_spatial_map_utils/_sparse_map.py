# Author: Proloy Das <pdas6@mgh.harvard.edu>
import collections
import numpy as np
from scipy import linalg


def _solve_for_qw(k, zeta, xi2, sigma_n, gain):
    gain_k_gainT = gain.dot(k).dot(gain.T)
    temp = gain_k_gainT + sigma_n / xi2
    e, v = linalg.eigh(temp)
    nz = e > 0
    e = e[nz]
    v = v[:, nz]
    temp = v.T.dot(np.atleast_2d(gain)).dot(k)
    temp /= np.sqrt(e[:, None])
    sigma_w = k - temp.T.dot(temp)

    # mu_w = linalg.solve(sigma_n, zeta)
    # mu_w = gain.T.dot(mu_w)
    # mu_w = sigma_w.dot(mu_w)

    temp = v.T.dot(zeta)
    temp /= e
    alpha = v.dot(temp)
    mu_w = k.dot(gain.T.dot(alpha)) / xi2
    ll = - (gain_k_gainT.dot(alpha) * alpha).sum() / 2
    return mu_w, sigma_w, ll


def _solve_for_qw_chol(k, zeta, xi2, sigma_n, gain):
    gain_k_gainT = gain.dot(k).dot(gain.T)
    temp = gain_k_gainT + sigma_n / xi2

    c, low = linalg.cho_factor(temp, lower=True)
    alpha = linalg.cho_solve((c, low), zeta)
    mu_w = k.dot(gain.T.dot(alpha)) / xi2
    v = linalg.solve_triangular(c, gain.dot(k), lower=low)
    sigma_w = k - v.T.dot(v)
    ll = - (gain_k_gainT.dot(alpha) * alpha).sum() / 2
    return mu_w, sigma_w, ll


def sparse_spatial_filter(y, x, cov, gain, sigma_n, ks, ws, sigma_ws,
                          v=None, vinv=None):
    assert y.shape[1] == x.shape[1]
    n = x.shape[1]
    temp = x.dot(x.T) + n * cov
    fw = gain.dot(ws.T)
    if v is None:
        v = np.eye(ws.shape[1])
        vinv = v
    gain = vinv.T.dot(gain.T).T
    total_ll = 0.
    for i in range(len(ws)):
        xi2 = temp[i, i]
        temp[i, i] = 0.0
        zeta = (y * x[i]).sum(axis=-1) - (fw * temp[:, i]).sum(axis=-1)
        try:
            this_w, this_sigma_w, ll = _solve_for_qw_chol(ks[i], zeta,
                                                          xi2, sigma_n, gain)
        except np.linalg.LinAlgError:
            this_w, this_sigma_w, ll = _solve_for_qw(ks[i], zeta, xi2,
                                                     sigma_n, gain)
        fw[:, i] = gain.dot(this_w)
        ws[i][:] = vinv.dot(this_w)
        sigma_ws[i][:] = vinv.dot(vinv.dot(this_sigma_w).T).T
        total_ll += ll
    return total_ll


def _update_hyperparameters(mu_w, gain, sigma_n, sigma_w, xi2, k, sigma2=1.):
    temp = gain.dot(k).dot(gain.T)
    temp += sigma_n / xi2
    try:
        l, low = linalg.cho_factor(temp, lower=True)
        temp = linalg.solve_triangular(l, gain, lower=low)
    except linalg.LinAlgError:
        e, v = linalg.eigh(temp)
        nz = e > 0
        e = e[nz]
        v = v[:, nz]
        temp = v.T.dot(np.atleast_2d(gain))
        temp /= np.sqrt(e[:, None])
    gamma = (temp * temp).sum(axis=0)
    c = mu_w * mu_w
    # method 1
    # gamma += 1. / sigma2
    # temp = np.sqrt(c * gamma + 4.)
    # k_ = (temp + 2) / gamma
    # # method 2
    # k_ = np.sqrt(c / gamma)
    # method 3
    k_ = c + np.diag(sigma_w)
    k.flat[::k.shape[0]+1] = k_
    return k


def update_sparse_spatial_hyperparameters(x, cov, gain, sigma_n, ks,
                                          ws, sigma_ws, sigma2=1., v=None,
                                          vinv=None):
    n = x.shape[1]
    xi2s = (x * x).sum(axis=1)
    xi2s += n * np.diag(cov)
    if v is None:
        v = np.eye(ws.shape[1])
        vinv = v
    ws = v.dot(ws.T).T
    gain = vinv.T.dot(gain.T).T
    for i, (mu_w, sigma_w, xi2) in enumerate(zip(ws, sigma_ws, xi2s)):
        sigma_w = v.dot(sigma_w).dot(v.T)
        _ = _update_hyperparameters(mu_w, gain, sigma_n, sigma_w, xi2, ks[i],
                                    sigma2=sigma2)
    return


def test_update_hyperparameters(sigma2=1.):
    from matplotlib import pyplot as plt
    snr = 10
    rng = np.random.default_rng(0)
    n = 300
    m = 7000
    gain = rng.standard_normal((n, m))
    gain /= linalg.norm(gain)
    noise = rng.standard_normal((n,))
    idx = rng.choice(m, 2)
    mu_w = np.zeros(m)
    mu_w[idx] = 1
    y = gain.dot(mu_w)
    sigma_n = y * y / snr
    y_noisy = y + np.sqrt(sigma_n) * noise
    whitened_gain = gain / np.sqrt(sigma_n)[:, None]
    whitened_signal = y_noisy / np.sqrt(sigma_n)
    lambda2 = (((whitened_signal * whitened_signal).sum() / y_noisy.shape[0] - 1)
               / (whitened_gain * whitened_gain).sum())
    k = lambda2 * np.eye(m)
    print(lambda2)
    ipdb.set_trace()

    fig, ax = plt.subplots(3)
    ax[0].plot(mu_w, label='Ground truth')
    plt.tight_layout()
    fig.show()
    for i in range(20):
        x, sigma_x = _solve_for_qw_chol(k, y_noisy, 1., sigma_n, gain)
        ax[2].plot(np.abs(x))
        ax[2].plot(np.sqrt(x * x + np.diag(sigma_x)))
        ipdb.set_trace()
        ax[0].plot(x, label=f'iter-{i}')
        ax[0].legend()
        k = _update_hyperparameters(x, gain, sigma_n, sigma_x * 0, 1., k, sigma2)
        ax[1].plot(np.diag(k), label=f'iter-{i}')
        ax[0].legend()
        fig.canvas.draw()
        fig.canvas.flush_events()
        # ipdb.set_trace()

    ci = np.diag(sigma_x).copy()
    ci[ci < 0] = 0.
    ci = np.sqrt(ci)
    fig, ax = plt.subplots()
    ax.plot(mu_w, label='Ground truth')
    ax.plot(x, label=f'iter-{i}')
    ax.plot(x + 2 * ci, label='upper')
    ax.plot(x - 2 * ci, label='lower')
    ax.legend()
    fig.show()
