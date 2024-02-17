import numpy as np
import matplotlib.pyplot as plt
from purdonlabmeeg._temporal_dynamics_utils._ar_utils import (lattice, __lattice, find_hidden_ar,
                            remove_aperiodic_signal)
from purdonlabmeeg._temporal_dynamics_utils.polyeig import polyeig
from purdonlabmeeg import BaseOscModel
import scipy.fft as fft
from scipy import linalg


def test_lattice():
    from numpy.random import default_rng
    from math import sqrt
    a = np.array([[0.118, -0.191], [0.847, 0.264]])
    b = np.array([[1, 2], [3, -4]])
    c = np.array([[0.811, -0.226], [0.477, 0.415]])
    d = np.array([[0.193, 0.689], [-0.320, -0.749]])
    rng = default_rng(seed=0)
    w = rng.standard_normal((2, 100)).T
    v = sqrt(0.1) * rng.standard_normal((2, 100)).T
    u = np.empty_like(w)
    z = np.empty_like(w)
    u[0] = rng.standard_normal(2)
    z[0] = rng.standard_normal(2)
    for k in range(1, 100):
        u[k] = c.dot(u[k-1]) + d.dot(w[k-1])
        z[k] = a.dot(z[k-1]) + b.dot(u[k-1])
    y = z + v
    x = np.vstack((y.T, u.T))

    results_ = lattice(1, x, 'VM',)
    results = __lattice(1, x, 'VM',)
    assert np.allclose(results[1][1], results_[1][1])

    results_ = lattice(1, x, 'NS',)
    results = __lattice(1, x, 'NS',)
    assert np.allclose(results[1][1], results_[1][1])


def scatter_with_errorbars(x, y, xerr, yerr, capsize=10.0, linestyle='None',
                           color='b', marker='o', ax=None):
    if ax is None:
        fig, ax = plt.subplots(1)
    ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt=color+marker, capsize=capsize)
    return ax


def test_find_ar():
    from numpy.random import default_rng
    from math import sqrt
    a = np.array([[0.118, -0.191], [0.847, 0.264]])
    b = np.array([[1, 2], [3, -4]])
    c = np.array([[0.811, -0.226], [0.477, 0.415]])
    d = np.array([[0.193, 0.689], [-0.320, -0.749]])
    rng = default_rng(seed=0)
    w = rng.standard_normal((2, 1000)).T
    v = sqrt(0.1) * rng.standard_normal((2, 1000)).T
    u = np.empty_like(w)
    z = np.empty_like(w)
    u[0] = rng.standard_normal(2)
    z[0] = rng.standard_normal(2)
    for k in range(1, 1000):
        u[k] = c.dot(u[k-1]) + d.dot(w[k-1])
        z[k] = a.dot(z[k-1]) + b.dot(u[k-1])
    y = z + v
    x = np.vstack((y.T, u.T))

    out = find_hidden_ar(1, x)
    v, e = polyeig(*out[1])

    p = 1
    out = find_hidden_ar(p, x)
    v, e = polyeig(*list(reversed(out[1])))
    # calulate theta and alpha from here
    real = np.isreal(e)
    complex = np.logical_not(real)
    e_real, v_real = e[real], v[:, real]
    e_com, v_com = e[complex], v[:, complex]
    e_com, v_com = e_com[::2], v_com[:, ::2]
    e = np.concatenate((e_real, e_com))
    alpha = np.absolute(e)
    freq = np.angle(e) / (2 * np.pi)

    sample_spectrum(x[0], freq)


def test_BaseOscModel(n=100):
    from numpy.random import default_rng
    from math import sqrt
    from purdonlabmeeg._spatial_map_utils._sensor_maps import (MixingMatrix,
                                                                    ClampedMixingMatrix)
    a = np.array([[0.118, -0.191], [0.847, 0.264]])
    b = np.array([[1, 2], [3, -4]])
    c = np.array([[0.811, -0.226], [0.477, 0.415]])
    d = np.array([[0.193, 0.689], [-0.320, -0.749]])
    rng = default_rng(seed=0)
    w = rng.standard_normal((2, n)).T
    u = np.empty_like(w)
    z = np.empty_like(w)
    u[0] = rng.standard_normal(2)
    z[0] = rng.standard_normal(2)
    for k in range(1, n):
        u[k] = c.dot(u[k-1]) + d.dot(w[k-1])
        z[k] = a.dot(z[k-1]) + b.dot(u[k-1])

    # v = sqrt(0.1) * rng.standard_normal((4, n))
    # x = np.vstack((z.T, u.T))
    # y = x + v

    v = sqrt(0.1) * rng.standard_normal((2, n)).T
    y = z + v
    y = np.vstack((y.T, u.T))

    oscmodel, C, r = BaseOscModel.initialize_from_data(4, y, 4, show_psd=True,
                                                       sfreq=100)
    print(C)
    mixing_mat_inst = ClampedMixingMatrix.from_mat(C, None)
    alpha = mixing_mat_inst.next_hyperparameter(True)
    r = r * np.eye(y.shape[0])
    fig, ax = plt.subplots()
    fe = -np.inf
    for jj in range(100):
        if jj%10 == 0: oscmodel.plot_osc_spectra(N=100, ax=ax)
        for ii in range(10):
            C, C_vars = mixing_mat_inst._get_vals()
            x_, s_, b, ll, _ = oscmodel.get_hidden_posterior(y, C, r, C_vars,
                                                             mixing_mat_inst=mixing_mat_inst)
            kld = mixing_mat_inst.update(y, x_, s_, alpha=alpha,
                                    lambdas=linalg.inv(r))
        print(f'====={jj} completed=====')
        new_fe = ll-kld
        if (new_fe - fe) / abs(fe) < 1e-3: break
        fe = new_fe
        oscmodel.update_params(x_, s_, b, update_sigma2=True)
        alpha = mixing_mat_inst.next_hyperparameter(True)
    print(C)
    
    fig.savefig('test_BaseOscModel(1).svg')

    fig, ax = plt.subplots(4)
    res = y - C.dot(x_.T)
    for ax_, this_x, this_res in zip(ax, y, res):
        ax_.plot(this_x)
        ax_.plot(this_res)
    fig.savefig('test_BaseOscModel(2).svg')

    fig, ax = plt.subplots(4)
    for ax_, this_x, this_res in zip(ax, x_.T, res):
        ax_.plot(this_x)
    fig.savefig('test_BaseOscModel(3).svg')

    fig, ax = plt.subplots(4)
    for i, ax_ in enumerate(ax):
        oscmodel.plot_indv_spectra(i, C, N=100, ax=ax_)
    y_hat = fft.fft(y)
    ff = fft.fftfreq(y.shape[1],  1)[:y.shape[1] // 2]
    for i, ax_ in enumerate(ax):
        ax_.plot(ff * 2 * np.pi,
                 10 * np.log10(np.abs(y_hat[i, :y.shape[1] // 2]) ** 2 /
                               y.shape[1]))
    fig.savefig('test_BaseOscModel(4).svg')


    fig, ax = plt.subplots()
    oscmodel.plot_osc_spectra(N=1000, ax=ax)
    fig.savefig('test_BaseOscModel(5).svg')


    y_hat = fft.fft(x_.T)
    ff = fft.fftfreq(y.shape[1],  1)[:y.shape[1] // 2]
    ax.semilogy(ff * 2 * np.pi,
                np.abs(y_hat[:, :y.shape[1] // 2].T) ** 2 / y.shape[1])
    return oscmodel


def test_BaseScalarOscModel(n=100):
    from numpy.random import default_rng
    from math import sqrt
    from purdonlabmeeg._spatial_map_utils._sensor_maps import ScalarMixingMatrix
    a = np.array([[0.118, -0.191], [0.847, 0.264]])
    b = np.array([[1, 2], [3, -4]])
    c = np.array([[0.811, -0.226], [0.477, 0.415]])
    d = np.array([[0.193, 0.689], [-0.320, -0.749]])
    rng = default_rng(seed=0)
    w = rng.standard_normal((2, n)).T
    u = np.empty_like(w)
    z = np.empty_like(w)
    u[0] = rng.standard_normal(2)
    z[0] = rng.standard_normal(2)
    for k in range(1, n):
        u[k] = c.dot(u[k-1]) + d.dot(w[k-1])
        z[k] = a.dot(z[k-1]) + b.dot(u[k-1])

    # v = sqrt(0.1) * rng.standard_normal((4, n))
    # x = np.vstack((z.T, u.T))
    # y = x + v

    v = sqrt(0.1) * rng.standard_normal((2, n)).T
    y = z + v
    y = np.vstack((y.T, u.T))
    y = y[:1]

    oscmodel, C, r = BaseOscModel.initialize_from_data(4, y, 4, show_psd=True,
                                                       sfreq=100)
    print(C)
    mixing_mat_inst = ScalarMixingMatrix.from_mat(C, None)
    alpha = mixing_mat_inst.next_hyperparameter(True)
    r = r * np.eye(y.shape[0])
    fig, ax = plt.subplots()
    fe = -np.inf
    for jj in range(100):
        if jj%10 == 0: oscmodel.plot_osc_spectra(N=100, ax=ax)
        for ii in range(10):
            C, C_vars = mixing_mat_inst._get_vals()
            x_, s_, b, ll, _ = oscmodel.get_hidden_posterior(y, C, r, C_vars,
                                                             mixing_mat_inst=mixing_mat_inst)
            kld = mixing_mat_inst.update(y, x_, s_, alpha=alpha,
                                    lambdas=linalg.inv(r))
        print(f'====={jj} completed=====')
        new_fe = ll-kld
        if (new_fe - fe) / abs(fe) < 1e-3: break
        fe = new_fe
        oscmodel.update_params(x_, s_, b, update_sigma2=True)
        alpha = mixing_mat_inst.next_hyperparameter(True)
    print(C)
    
    fig.savefig('test_BaseOscModel(1).svg')

    fig, ax = plt.subplots(y.shape[0])
    if y.shape[0] == 1: ax = [ax] 
    res = y - C.dot(x_.T)
    for ax_, this_x, this_res in zip(ax, y, res):
        ax_.plot(this_x)
        ax_.plot(this_res)
    fig.savefig('test_BaseOscModel(2).svg')

    fig, ax = plt.subplots(x_.shape[1])
    for ax_, this_x in zip(ax, x_.T):
        ax_.plot(this_x)
    fig.savefig('test_BaseOscModel(3).svg')

    fig, ax = plt.subplots()
    oscmodel.plot_indv_spectra(0, C, N=100, ax=ax)
    y_hat = fft.fft(y)
    ff = fft.fftfreq(y.shape[1],  1)[:y.shape[1] // 2]
    ax_.plot(ff * 2 * np.pi,
                10 * np.log10(np.abs(y_hat[0, :y.shape[1] // 2]) ** 2 /
                            y.shape[1]))
    fig.savefig('test_BaseOscModel(4).svg')


    fig, ax = plt.subplots()
    oscmodel.plot_osc_spectra(N=1000, ax=ax)
    fig.savefig('test_BaseOscModel(5).svg')


    y_hat = fft.fft(x_.T)
    ff = fft.fftfreq(y.shape[1],  1)[:y.shape[1] // 2]
    ax.semilogy(ff * 2 * np.pi,
                np.abs(y_hat[:, :y.shape[1] // 2].T) ** 2 / y.shape[1])
    return oscmodel


def test_complicated_matrix_inversion():
    from purdonlabmeeg._spatial_map_utils._sensor_maps import special_matrix_inversion
    from numpy.random import default_rng
    rng = default_rng(0)
    y = rng.normal(size=(3, 10))
    lambdas = y.dot(y.T)
    x = rng.normal(size=(5, 10))
    Cxx = x.dot(x.T)
    alpha = 2
    alpha_sqrt = alpha ** (1/2)
    n = 10
    rhs = rng.normal(size=(3, 5)) 

    A = np.kron(lambdas, Cxx)
    A = A + alpha * np.eye(*A.shape) / n
    b = linalg.inv(A).dot(np.ravel(rhs))
    u, mats, *rest = special_matrix_inversion(lambdas, Cxx, alpha_sqrt, n, rhs, False)
    assert np.allclose(u.dot(np.stack(mats)).ravel(), b)


if __name__ == '__main__':
    test_BaseOscModel()