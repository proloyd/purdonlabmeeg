# Author: Proloy Das <pdas6@mgh.harvard.edu>
import collections
import numpy as np
from scipy import linalg

from numba import jit, njit, float32, float64, uint


# @njit([float32[:](float32[:, :], float32[:]),
#       float64[:](float64[:, :], float64[:])],
#       cache=True)
@jit(["float32[::1](float32[:, ::1], float32[::1], float32[::1], float32)",
      "float64[::1](float64[:, ::1], float64[::1], float64[::1], float64)"],
     nopython=True, cache=True, nogil=True)
def _cg_solver(A, b, x, tol):
    """
    See the following website for cg:
    [link](https://scipy.github.io/old-wiki/pages/ConjugateGradientExample.html)
    [Ref: page 32](https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf)
    """
    nrow, ncol = A.shape

    # Initialization
    r = b - A.dot(x)
    r2 = (r * r).sum()
    p = r.copy()

    # Main conjugate gradient loop
    for k in range(nrow):
        Ap = A.dot(p)
        alpha = r2 / (p * Ap).sum()
        x += p * alpha
        r -= Ap * alpha
        # norm = np.linalg.norm(r, 2)
        # if norm < tol:
        #     break
        r2old = r2
        r2 = (r * r).sum()
        if r2 < nrow * tol:
            break
        beta = r2 / r2old
        # p = r + (p * beta)
        p *= beta
        p += r

    return x


def cg_solver(A, b, x=None, tol=1e-16):
    if A.dtype != b.dtype:
        A = A.astype(np.float64)
        b = b.astype(np.float64)

    if x is None:
        x = np.zeros(b.shape, b.dtype)

    return _cg_solver(A, b, x, tol)


def costf(theta, gpkernel, w, xi2, sigma_n, gain):
    """Returns ln|sigma_n + xi2 gain K gain.T| + ||w||_{kinv}"""
    # kinv, k, _ = gpkernel.inv_cov(theta=theta)
    k = gpkernel.cov(theta=theta)

    temp = gain.dot(k).dot(gain.T)
    temp *= xi2
    temp += sigma_n
    try:
        c = linalg.cholesky(temp)
        val = np.log(np.diag(c)). sum() * 2
    except linalg.LinAlgError:
        e = linalg.eigvalsh(temp)
        val = np.log(e).sum()

    # e, v = linalg.eigh(k)
    # # Set an relative cutoff for small singular values.
    # # We simply ignore the eigenvalues less than e.max() * 1e-15.
    # nz = e > e.max() * 0
    # e = e[nz]
    # v = v[:, nz]
    # temp = v.T.dot(w)
    # val += ((temp * temp) / (e)[:, None]).sum()
    # temp2, *residuals = np.linalg.lstsq(k, w, rcond=None)
    # val += (temp2 * w).sum()
    # y = kinv.dot(w)
    y = cg_solver(k, w)

    # temp = gain.dot(k).dot(gain.T)
    # temp += sigma_n / xi2
    #
    # c, low = linalg.cho_factor(temp, lower=True)
    # v = linalg.solve_triangular(c, gain.dot(k), lower=low)
    # y, *res = np.linalg.lstsq(k - v.T.dot(v), w, rcond=-1)
    # val = 2*np.log(np.diag(c)).sum() + (y * w).sum()
    val += (y * w).sum()

    out = np.asanyarray(val)
    out.shape = (1, )
    return out


def gradf(theta, gpkernel, w, xi2, sigma_n, gain):
    """Returns ln|sigma_n + xi2 gain K gain.T| + ||w||_{kinv}"""
    # kinv, k, _ = gpkernel.inv_cov(theta=theta)
    k = gpkernel.cov(theta=theta)

    gradks = gpkernel.diff_cov(theta=theta)
    if isinstance(gradks, collections.abc.Sequence):
        gradk = gradks[0]
    else:
        gradk = gradks

    if np.all(gradk):  # a large negative value of theta
        out = -1.0
        out = np.asanyarray(out)
        out.shape = (1, )
        return out

    temp = gain.dot(k).dot(gain.T)
    temp *= xi2
    temp = sigma_n + temp
    e, v = linalg.eigh(temp)
    nz = e > 0
    e = e[nz]
    v = v[:, nz]
    temp = v.T.dot(gain) / np.sqrt(e)[:, None]
    # grad = temp.T.dot(temp)
    out = (temp.dot(gradk) * temp).sum()
    out *= xi2

    # temp = kinv.dot(w)
    temp = cg_solver(k, w)
    out -= (temp * gradk.dot(temp)).sum()

    if np.isnan(temp).any():
        ipdb.set_trace()
    # grads = (grad * gradk).sum()
    # assert np.allclose(grads, out)
    # out = np.asanyarray(grads)
    out = np.asanyarray(out)
    out.shape = (1, )
    return out


def _solve_for_qw(theta, gpkernel, zeta, xi2, sigma_n, gain):
    k = gpkernel.cov(theta=theta)

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
    ll = (gain_k_gainT.dot(alpha) * alpha).sum() / 2
    return mu_w, sigma_w, ll


def _solve_for_qw_chol(theta, gpkernel, zeta, xi2, sigma_n, gain):
    k = gpkernel.cov(theta=theta)

    gain_k_gainT = gain.dot(k).dot(gain.T)
    temp = gain_k_gainT + sigma_n / xi2

    c, low = linalg.cho_factor(temp, lower=True)
    alpha = linalg.cho_solve((c, low), zeta)
    mu_w = k.dot(gain.T.dot(alpha)) / xi2
    v = linalg.solve_triangular(c, gain.dot(k), lower=low)
    sigma_w = k - v.T.dot(v)
    ll = - (gain_k_gainT.dot(alpha) * alpha).sum() / 2
    return mu_w, sigma_w, ll


def spatial_filter(y, x, cov, gain, sigma_n, thetas, gpkernel, ws, sigma_ws):
    assert y.shape[1] == x.shape[1]
    n = x.shape[1]
    temp = x.dot(x.T) + n * cov
    fw = gain.dot(ws.T)
    total_ll = 0
    for i in range(len(ws)):
        xi2 = temp[i, i]
        temp[i, i] = 0.0
        zeta = (y * x[i]).sum(axis=-1) - (fw * temp[:, i]).sum(axis=-1)
        try:
            this_w, this_sigma_w, ll = _solve_for_qw_chol(thetas[i], gpkernel,
                                                          zeta, xi2, sigma_n,
                                                          gain)
        except np.linalg.LinAlgError:
            this_w, this_sigma_w, ll = _solve_for_qw(thetas[i], gpkernel, zeta,
                                                     xi2, sigma_n, gain)
        ws[i][:] = this_w
        sigma_ws[i][:] = this_sigma_w
        fw[:, i] = gain.dot(this_w)
        total_ll += ll
    return total_ll


def _update_scale(mu_w, gain, gpkernel, sigma_n, xi2, theta_init=None,
                  tol=1e-4):
    from ._bb_utils import stab_bb
    if theta_init is None:
        theta_init = np.log(10)

    def Fx(x): return costf(x, gpkernel, mu_w, xi2, sigma_n, gain)
    def gradFx(x): return gradf(x, gpkernel, mu_w, xi2, sigma_n, gain)
    print('Starting theta optimization')
    theta_best, res = stab_bb(theta_init, costFn=Fx, gradFn=gradFx, tol=tol,
                              maxIt=2, verbose=True)
    print('Theta optimization Done')
    return theta_best


def update_spatial_scales(x, cov, gain, sigma_n, thetas, gpkernel, ws, sigma_ws,
                          tol=1e-4):
    n = x.shape[1]
    xi2s = (x * x).sum(axis=1)
    xi2s += n * np.diag(cov)
    for i, (mu_w, xi2) in enumerate(zip(ws, xi2s)):
        best_theta = _update_scale(mu_w, gain, gpkernel, sigma_n, xi2,
                                   thetas[i], tol)
        thetas[i] = best_theta
        print(f"best_theta for {i}th scale: {best_theta}")
    return


def _check_grad(gain, gpkernel, sigma_n, x, xi2, y_noisy, verbose=False):
    besttheta = np.log(10)
    w = _solve_for_qw(besttheta, gpkernel, y_noisy, xi2, sigma_n, gain)[0]
    def Fx(x): return costf(x, gpkernel, w, xi2, sigma_n, gain)
    def gradFx(x): return gradf(x, gpkernel, w, xi2, sigma_n, gain)
    rng = np.random.default_rng(1)
    x0 = rng.random() * 2.5
    return __check_grad(Fx, gradFx, x0, verbose=False)


def __check_grad(Fx, gradFx, x0, verbose=False):
    rng = np.random.default_rng(0)
    Fx0 = Fx(x0)
    gradFx0 = gradFx(x0)
    rel_errors = []
    for mul in reversed(np.logspace(-8, -2, 10)):
        delta = rng.random() * mul
        true_diff = Fx(x0 + delta) - Fx0
        estimated_diff = gradFx0 * delta
        rel_errors.append(np.abs((true_diff - estimated_diff) / true_diff))
        if verbose:
            print(
                f"{float(true_diff):+.8f} {float(estimated_diff):+.8f}" +
                f" {float(rel_errors[-1]):.8f}")
    return min(rel_errors) < 1e-4


def _demo_gpkernel_part(gain, gpkernel, sigma_n, xi2, y_noisy):

    besttheta = np.log(10)
    w = _solve_for_qw(besttheta, gpkernel, y_noisy, xi2, sigma_n, gain)[0]
    # from ._bb_utils import stab_bb
    # Fx = lambda x: costf(x, gpkernel, w, xi2, sigma_n, gain)
    # gradFx = lambda x: gradf(x, gpkernel, w, xi2, sigma_n, gain)
    # x0 = np.log(10)
    # besttheta, res = stab_bb(x0, costFn=Fx, gradFn=gradFx, tol=1e-4,)
    _update_scale(w, gain, gpkernel, sigma_n, xi2, theta_init=None, tol=1e-4)
    y_est = _solve_for_qw(besttheta, gpkernel, y_noisy, xi2, sigma_n, gain)
    return besttheta, y_est


def _simulate_random_data(gain=None, n=100, m=1000):
    rng = np.random.default_rng(100)
    x = rng.random(1000) * 100
    x = np.sort(x)
    x.shape = (m, 1)
    z = 1 / (1 + np.exp(-10 * (x - 49))) - 1 / (1 + np.exp(-10 * (x - 51)))
    z += 1 / (1 + np.exp(-1 * (x - 22))) - 1 / (1 + np.exp(-1 * (x - 28)))
    z += 1 / (1 + np.exp(-5 * (x - 70))) - 1 / (1 + np.exp(-5 * (x - 80)))
    if gain is None:
        gain = rng.standard_normal((n, m))
        gain /= np.linalg.norm(gain)
    y = gain.dot(z)
    sigma = 0.1 * (y.max() - y.min())
    y_noisy = y + sigma * rng.standard_normal(y.shape)

    sigma_n = sigma ** 2 * np.eye(y.shape[0])
    xi2 = 1
    return gain, sigma_n, x, xi2, y_noisy, z


def _demo_gpkernel_full(gain=None, n=100, m=1000):
    from ._kernels import (SquaredExponentailGPKernel, MaternGPKernel,
                           MaternGPKernel2, GammaExpGPKernel)
    import matplotlib.pyplot as plt

    gain, sigma_n, x, xi2, y_noisy, z = _simulate_random_data(gain, n, m)

    # Squared Gaussian
    segpkernel = SquaredExponentailGPKernel(X=x, fixed_params={'sigma_s':
                                                               np.sqrt(0.999)})
    besttheta_se, y_est_se = _demo_gpkernel_part(gain, segpkernel, sigma_n,
                                                 xi2, y_noisy)

    # Matern
    mgpkernel = MaternGPKernel(X=x, fixed_params={'sigma_s': np.sqrt(1)})
    besttheta_m, y_est_m = _demo_gpkernel_part(gain, mgpkernel, sigma_n,
                                               xi2, y_noisy)

    # Matern2
    mgpkernel2 = MaternGPKernel2(X=x, fixed_params={'sigma_s': np.sqrt(1)})
    besttheta_m2, y_est_m2 = _demo_gpkernel_part(gain, mgpkernel2, sigma_n,
                                                 xi2, y_noisy)

    # Gamma
    ggpkernel = GammaExpGPKernel(X=x, fixed_params={'gamma': 1.5,
                                                    'sigma_s': np.sqrt(1)})
    besttheta_g, y_est_g = _demo_gpkernel_part(gain, ggpkernel, sigma_n,
                                               xi2, y_noisy)

    fig, ax = plt.subplots(1)
    ax.plot(x, np.squeeze(z))
    ax.plot(x, np.squeeze(y_est_se[0]), label='Squared Exponential')
    ax.plot(x, np.squeeze(y_est_m[0]), label='Matern (eta=3/2)')
    ax.plot(x, np.squeeze(y_est_m2[0]), label='Matern (eta = 5/2)')
    ax.plot(x, np.squeeze(y_est_g[0]), label='Gamma (gamma=3/2)')
    ax.legend()
    fig.show()


def test_grad(verbose=False):
    from _kernels import (SquaredExponentailGPKernel, MaternGPKernel,
                          MaternGPKernel2, GammaExpGPKernel)
    gain, sigma_n, x, xi2, y_noisy, z = _simulate_random_data()

    # Squared Gaussian
    segpkernel = SquaredExponentailGPKernel(X=x, fixed_params={'sigma_s':
                                                               np.sqrt(0.999)})
    assert _check_grad(gain, segpkernel, sigma_n, x, xi2, y_noisy, verbose)

    # Matern
    mgpkernel = MaternGPKernel(X=x, fixed_params={'sigma_s': np.sqrt(1)})
    assert _check_grad(gain, mgpkernel, sigma_n, x, xi2, y_noisy, verbose)

    # Matern2
    mgpkernel2 = MaternGPKernel2(X=x, fixed_params={'sigma_s': np.sqrt(1)})
    assert _check_grad(gain, mgpkernel2, sigma_n, x, xi2, y_noisy, verbose)

    # Matern2
    ggpkernel = GammaExpGPKernel(X=x, fixed_params={'gamma': 1.5,
                                                    'sigma_s': np.sqrt(1)})
    assert _check_grad(gain, ggpkernel, sigma_n, x, xi2, y_noisy, verbose)


def test_handler3():
    from numpy.random import default_rng
    from math import sqrt
    import matplotlib.pyplot as plt
    from purdonlabmeeg import (SquaredExponentailGPKernel, MaternGPKernel,
                               MaternGPKernel2, GammaExpGPKernel, Handler)
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

    pos = (rng.uniform(-10, 10, size=100))
    sorted_idx = np.argsort(pos)
    lows = [-7, -4, 1, 6]
    highs = [-5, -2, 2, 8]
    spatial_maps = [1 / (1 + np.exp(-5*(pos-low))) -
                    1 / (1 + np.exp(-5*(pos-high)))
                    for low, high in zip(lows, highs)]
    spatial_maps = np.asanyarray(spatial_maps)
    fig, ax = plt.subplots()
    ax.plot(pos[sorted_idx], spatial_maps.T[sorted_idx, :])
    fig.show()
    gain = rng.standard_normal((10, 100))
    gain /= np.linalg.norm(gain, 'fro')
    sensing_matrix = gain.dot(spatial_maps.T)
    measurements = sensing_matrix.dot(x)
    noise = rng.standard_normal(measurements.shape)
    noisy_measurements = (measurements +
                          0.1 * measurements.std(axis=1)[:, None] * noise)

    r = np.diag(0.01 * measurements.var(axis=1))
    a_ = np.vstack((np.hstack((a, b)), np.hstack((np.zeros((2, 2)), c))))
    q_ = np.zeros((4, 4))
    q_[2:, 2:] = d.dot(d.T)
    print(a_)
    print(q_)
    gpkernel = GammaExpGPKernel(X=pos[:, None],
                                fixed_params={'gamma': 1.5,
                                              'sigma_s': np.sqrt(1)})
    # gpkernel = MaternGPKernel(X=pos[:, None],
    #                           fixed_params={'sigma_s': np.sqrt(1)})
    thetas = 0.1 * np.ones(4)
    cov = np.zeros((x.shape[0], x.shape[0]))

    handler = Handler(4, 10, 100, 100, 1)
    ws = np.zeros_like(handler.ws)
    sigma_ws = np.zeros_like(handler.sigma_ws)
    for _ in range(10):
        out = spatial_filter(noisy_measurements, x, cov, gain, r, thetas,
                             gpkernel, ws, sigma_ws)
        ax.plot(pos[sorted_idx], ws.T[sorted_idx, :])
        fig.canvas.draw()
        update_spatial_scales(x, cov, gain, r, thetas, gpkernel, ws,
                              sigma_ws, tol=1e-4)
    return out
