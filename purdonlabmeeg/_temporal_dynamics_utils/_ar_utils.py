import numpy as np
from scipy import linalg
from mne.utils import logger

from ._kalman_smoother import kalcvf


def lattice(p, x, method='NS'):
    """Computes multichannel autoregressive parameter matrix using either the
    Vieira-Morf algorithm (multi-channel generalization of the single-channel
    geometric lattice algorithm) or the Nuttall-Strand algorithm (multi-channel
    generalization of the single-channel Burg lattice algorithm).

    Parameters:
    p: int
        order of multichannel AR filter
    x: ndarray
        sample data array: x(channel #, sample #)
    method: str
        ’VM’ -- Vieira-Morf , ’NS’ -- Nuttall-Strand (Default)

    Returns:
    rho_f: ndarray
        forward linear prediction error/white noise covariance matrix
    a: list of ndarrays
        block vector of forward linear prediction/autoregressive matrix
        elements
    rho_b: ndarray
        backward linear prediction error/white noise covariance matrix
    b: list of ndarrays
        block vector of backward linear prediction/autoregressive matrix
        elements

    NOTE: 'VM'-method is more resilient to additive noise.
    """
    factors, n = x.shape
    # e_f = x.copy()
    # e_b = x.copy()
    # _e_f = np.empty_like(x)
    # _e_b = np.empty_like(x)
    # _rho_f = e_f[:, p+1:].dot(e_f[:, p+1:].T) / n
    # _rho_b = e_b[:, p:-1].dot(e_b[:, p:-1].T) / n
    # _rho_fb = e_f[:, p+1:].dot(e_b[:, p:-1].T) / n
    rho_f = x.dot(x.T) / n
    rho_b = rho_f.copy()
    _rho_f = x[:, p+1:].dot(x[:, p+1:].T) / n
    _rho_b = x[:, p:-1].dot(x[:, p:-1].T) / n
    _rho_fb = x[:, p+1:].dot(x[:, p:-1].T) / n
    a = [np.eye(factors)]
    b = [np.eye(factors)]
    return _lattice(p, a, b, rho_f, rho_b, _rho_f, _rho_b, _rho_fb, method)


def lattice_kalman_ss(p, x, cov, cross_cov, method='NS',):
    """Computes multichannel autoregressive parameter matrix using either the
    Vieira-Morf algorithm (multi-channel generalization of the single-channel
    geometric lattice algorithm) or the Nuttall-Strand algorithm (multi-channel
    generalization of the single-channel Burg lattice algorithm).

    Parameters:
    p: int
        order of multichannel AR filter
    x: ndarray
        sample data array: x(channel #, sample #)
    method: str
        ’VM’ -- Vieira-Morf , ’NS’ -- Nuttall-Strand (Default)
    cov: ndarray
        estimation error covariance matrix
    cross_cov: ndarray
        estimation error cross-covariance matrix
    Returns:
    rho_f: ndarray
        forward linear prediction error/white noise covariance matrix
    a: list of ndarrays
        block vector of forward linear prediction/autoregressive matrix
        elements
    rho_b: ndarray
        backward linear prediction error/white noise covariance matrix
    b: list of ndarrays
        block vector of backward linear prediction/autoregressive matrix
        elements

    NOTE: 'VM'-method is more resilient to additive noise.
    """
    factors, n = x.shape
    # e_f = x.copy()
    # e_b = x.copy()
    # _e_f = np.empty_like(x)
    # _e_b = np.empty_like(x)
    # _rho_f = e_f[:, p+1:].dot(e_f[:, p+1:].T) / n
    # _rho_b = e_b[:, p:-1].dot(e_b[:, p:-1].T) / n
    # _rho_fb = e_f[:, p+1:].dot(e_b[:, p:-1].T) / n
    rho_f = x.dot(x.T) / n + cov
    rho_b = rho_f.copy()
    _rho_f = x[:, p+1:].dot(x[:, p+1:].T) / n + cov * (n-p-1) / n
    _rho_b = x[:, p:-1].dot(x[:, p:-1].T) / n + cov * (n-p-1) / n
    _rho_fb = x[:, p+1:].dot(x[:, p:-1].T) / n + cross_cov * (n-p-2) / n
    a = [np.eye(factors)]
    b = [np.eye(factors)]
    return _lattice(p, a, b, rho_f, rho_b, _rho_f, _rho_b, _rho_fb, method)


def lattice_kalman(p, x, cov, cross_cov, method='NS',):
    """Computes multichannel autoregressive parameter matrix using either the
    Vieira-Morf algorithm (multi-channel generalization of the single-channel
    geometric lattice algorithm) or the Nuttall-Strand algorithm (multi-channel
    generalization of the single-channel Burg lattice algorithm).
    Parameters:
    p: int
        order of multichannel AR filter
    x: ndarray
        sample data array: x(channel #, sample #)
    method: str
        ’VM’ -- Vieira-Morf , ’NS’ -- Nuttall-Strand (Default)
    cov: ndarray
        estimation error covariance matrix
    cross_cov: ndarray
        estimation error cross-covariance matrix
    Returns:
    rho_f: ndarray
        forward linear prediction error/white noise covariance matrix
    a: list of ndarrays
        block vector of forward linear prediction/autoregressive matrix
        elements
    rho_b: ndarray
        backward linear prediction error/white noise covariance matrix
    b: list of ndarrays
        block vector of backward linear prediction/autoregressive matrix
        elements

    NOTE: 'VM'-method is more resilient to additive noise.
    """
    factors, n = x.shape
    # e_f = x.copy()
    # e_b = x.copy()
    # _e_f = np.empty_like(x)
    # _e_b = np.empty_like(x)
    # _rho_f = e_f[:, p+1:].dot(e_f[:, p+1:].T) / n
    # _rho_b = e_b[:, p:-1].dot(e_b[:, p:-1].T) / n
    # _rho_fb = e_f[:, p+1:].dot(e_b[:, p:-1].T) / n
    rho_f = x.dot(x.T) / n + cov.sum(axis=0) / n
    rho_b = rho_f.copy()
    _rho_f = x[:, p+1:].dot(x[:, p+1:].T) / n + cov[p+1:].sum(axis=0) / n
    _rho_b = x[:, p:-1].dot(x[:, p:-1].T) / n + cov[p:-1].sum(axis=0) / n
    _rho_fb = (x[:, p+1:].dot(x[:, p:-1].T) / n
               + cross_cov[p+1:].sum(axis=0) / n)
    a = [np.eye(factors)]
    b = [np.eye(factors)]
    return _lattice(p, a, b, rho_f, rho_b, _rho_f, _rho_b, _rho_fb, method)


def _lattice(p, a, b, rho_f, rho_b, _rho_f, _rho_b, _rho_fb, method='NS',):
    for i in range(1, p+1):
        if method == 'NS':
            rho_f_inv = stable_inverse(rho_f)
            rho_b_inv = stable_inverse(rho_b)
            ip_a = _rho_f.dot(rho_f_inv)
            ip_b = rho_b_inv.dot(_rho_b)
            ip_q = 2 * _rho_fb
            delta = linalg.solve_sylvester(ip_a, ip_b, ip_q)
            a_p = - delta.dot(rho_b_inv)
            b_p = - delta.T.dot(rho_f_inv)
        elif method == 'VM':
            # _rho_f_sqrt_inv = linalg.inv(linalg.cholesky(_rho_f, lower=True))
            _rho_f_sqrt_inv, _ = stable_sqrt_inverse(_rho_f)
            # _rho_b_sqrt_inv = linalg.inv(linalg.cholesky(_rho_b, lower=True))
            _rho_b_sqrt_inv, _ = stable_sqrt_inverse(_rho_b)
            lambdap = _rho_f_sqrt_inv.dot(_rho_fb.dot(_rho_b_sqrt_inv.T))
            # rho_f_sqrt = linalg.cholesky(rho_f, lower=True)
            # rho_b_sqrt = linalg.cholesky(rho_b, lower=True)
            # rho_f_sqrt_inv = linalg.inv(rho_f_sqrt)
            # rho_b_sqrt_inv = linalg.inv(rho_b_sqrt)
            rho_f_sqrt_inv, rho_f_sqrt = stable_sqrt_inverse(rho_f)
            rho_b_sqrt_inv, rho_b_sqrt = stable_sqrt_inverse(rho_b)
            a_p = - rho_f_sqrt.dot(lambdap.dot(rho_b_sqrt_inv))
            b_p = - rho_b_sqrt.dot(lambdap.dot(rho_f_sqrt_inv))
            delta = rho_f_sqrt.dot(lambdap.dot(rho_b_sqrt.T))

        # a, b updates
        a.append(np.zeros_like(a_p))  # replace 0 by zeros
        b.insert(0, np.zeros_like(b_p))  # replace 0 by zeros
        for a_i, b_i in zip(a, b):
            new_a_i = a_i + a_p.dot(b_i)
            new_b_i = b_i + b_p.dot(a_i)
            a_i[:] = new_a_i
            b_i[:] = new_b_i
        # rho updates
        rho_f = rho_f + a_p.dot(delta.T)
        rho_b = rho_b + b_p.dot(delta)
        # error updates
        # _e_f[:] = e_f[:]
        # _e_b[:] = e_b[:]
        # e_f[:, 1:] += a_p.dot(_e_b[:, :-1])
        # e_b[:, 1:] += b_p.dot(_e_f[:, :-1])
        a_p_rho_b = a_p.dot(_rho_b)
        b_p_rho_f = b_p.dot(_rho_f)
        _rho_fb_a_pT = _rho_fb.dot(a_p.T)
        b_p_rho_fb = b_p.dot(_rho_fb)
        __rho_f = _rho_f + a_p_rho_b.dot(a_p.T) + _rho_fb_a_pT + _rho_fb_a_pT.T
        __rho_b = _rho_b + b_p_rho_f.dot(b_p.T) + b_p_rho_fb.T + b_p_rho_fb
        __rho_fb = (_rho_fb + _rho_fb_a_pT.T.dot(b_p.T) + a_p_rho_b
                    + b_p_rho_f.T)

        _rho_f = __rho_f
        _rho_b = __rho_b
        _rho_fb = __rho_fb

    return rho_f, a, rho_b, b


def __lattice(p, x, method='NS', r=0):
    """Computes multichannel autoregressive parameter matrix using either the
    Vieira-Morf algorithm (multi-channel generalization of the single-channel
    geometric lattice algorithm) or the Nuttall-Strand algorithm (multi-channel
    generalization of the single-channel Burg lattice algorithm).

    Parameters:
    p: int
        order of multichannel AR filter
    x: ndarray
        sample data array: x(channel #, sample #)
    method: str
        ’VM’ -- Vieira-Morf , ’NS’ -- Nuttall-Strand (Default)

    Returns:
    rho_f: ndarray
        forward linear prediction error/white noise covariance matrix
    a: list of ndarrays
        block vector of forward linear prediction/autoregressive matrix
        elements
    rho_b: ndarray
        backward linear prediction error/white noise covariance matrix
    b: list of ndarrays
        block vector of backward linear prediction/autoregressive matrix
        elements

    NOTE: 'VM'-method is more resilient to additive noise.
    """
    factors, n = x.shape
    e_f = x.copy()
    e_b = x.copy()
    _e_f = np.empty_like(x)
    _e_b = np.empty_like(x)
    _rho_f = e_f[:, p+1:].dot(e_f[:, p+1:].T) / n
    _rho_b = e_b[:, p:-1].dot(e_b[:, p:-1].T) / n
    _rho_fb = e_f[:, p+1:].dot(e_b[:, p:-1].T) / n
    rho_f = x.dot(x.T) / n
    rho_b = rho_f.copy()
    a = [np.eye(factors)]
    b = [np.eye(factors)]
    for i in range(1, p+1):
        if method == 'NS':
            rho_f_inv = stable_inverse(rho_f)
            rho_b_inv = stable_inverse(rho_b)
            ip_a = _rho_f.dot(rho_f_inv)
            ip_b = rho_b_inv.dot(_rho_b)
            ip_q = 2 * _rho_fb
            delta = linalg.solve_sylvester(ip_a, ip_b, ip_q)
            a_p = - delta.dot(rho_b_inv)
            b_p = - delta.T.dot(rho_f_inv)
        elif method == 'VM':
            # _rho_f_sqrt_inv = linalg.inv(linalg.cholesky(_rho_f, lower=True))
            _rho_f_sqrt_inv, _ = stable_sqrt_inverse(_rho_f)
            # _rho_b_sqrt_inv = linalg.inv(linalg.cholesky(_rho_b, lower=True))
            _rho_b_sqrt_inv, _ = stable_sqrt_inverse(_rho_b)
            lambdap = _rho_f_sqrt_inv.dot(_rho_fb.dot(_rho_b_sqrt_inv.T))
            # rho_f_sqrt = linalg.cholesky(rho_f, lower=True)
            # rho_b_sqrt = linalg.cholesky(rho_b, lower=True)
            # rho_f_sqrt_inv = linalg.inv(rho_f_sqrt)
            # rho_b_sqrt_inv = linalg.inv(rho_b_sqrt)
            rho_f_sqrt_inv, rho_f_sqrt = stable_sqrt_inverse(rho_f)
            rho_b_sqrt_inv, rho_b_sqrt = stable_sqrt_inverse(rho_b)
            a_p = - rho_f_sqrt.dot(lambdap.dot(rho_b_sqrt_inv))
            b_p = - rho_b_sqrt.dot(lambdap.dot(rho_f_sqrt_inv))
            delta = rho_f_sqrt.dot(lambdap.dot(rho_b_sqrt.T))

        # a, b updates
        a.append(np.zeros_like(a_p))  # replace 0 by zeros
        b.insert(0, np.zeros_like(b_p))  # replace 0 by zeros
        for a_i, b_i in zip(a, b):
            new_a_i = a_i + a_p.dot(b_i)
            new_b_i = b_i + b_p.dot(a_i)
            a_i[:] = new_a_i
            b_i[:] = new_b_i
        # rho updates
        rho_f = rho_f + a_p.dot(delta.T)
        rho_b = rho_b + b_p.dot(delta)
        # error updates
        _e_f[:] = e_f[:]
        _e_b[:] = e_b[:]
        e_f[:, 1:] += a_p.dot(_e_b[:, :-1])
        e_b[:, 1:] += b_p.dot(_e_f[:, :-1])
        _rho_f = e_f[:, p+1:].dot(e_f[:, p+1:].T) / n
        _rho_b = e_b[:, p:-1].dot(e_b[:, p:-1].T) / n
        _rho_fb = e_f[:, p+1:].dot(e_b[:, p:-1].T) / n
    return rho_f, a, rho_b, b


def stable_inverse(a):
    from mne.utils.numerics import _safe_svd
    u, s, vh = _safe_svd(a)
    sinv = np.zeros_like(s)
    thresh = s.max() * 1e-8
    sinv[s > thresh] = 1 / s[s > thresh]
    return (vh.T * sinv[None, :]).dot(u.T)


def stable_sqrt_inverse(a):
    from scipy.linalg import eigh
    w, v = eigh(a)
    wsqrt = np.zeros_like(w)
    wsqrtinv = np.zeros_like(w)
    wsqrt[w>0] = np.sqrt(w[w>0])
    thresh = w.max() * 1e-16
    wsqrtinv[w>thresh] = 1 / wsqrt[w>thresh]
    return (v * wsqrtinv[None, :]).dot(v.T), (v * wsqrt[None, :]).dot(v.T)


def find_hidden_ar(p, x, method='NS', show_psd=False, sfreq=None, r0=None):
    from scipy import optimize
    factors, n = x.shape
    # x_ = np.vstack([x[:, p - i:n - i] for i in range(p+1)])
    # cov = x_.dot(x_.T) / n
    cov = x.dot(x.T) / n
    eigs = linalg.eigvalsh(cov)
    logger.debug(f'eigs:{eigs}')
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.plot(eigs[::-1])
    r_max = eigs.min()
    # r_max = eigs[-index]
    if r_max < 0.:
        raise ValueError(f'r_max: {r_max} is less than zero.')
    logger.debug(f'selected range (0 {r_max})')
    g = lambda r: - matsuda_g(r, p, x, method)
    if r0 is None:
        # res2 = optimize.minimize(g, x0=r_max/1.01, bounds=[(0, r_max)],)
        # r0 = res2.x
        stop = np.log10(r_max)
        rs = np.logspace(stop - 3, stop, endpoint=False, num=10)
        vals = list(map(g, rs))
        ii = np.argmax(vals)
        r0 = rs[ii]
        res2 = None
    else:
        res2 = None
    g_val = g(r0)
    logger.debug(f'rmax = {r_max}, r0 = {r0}, r_max>r0 = {r_max>r0}')

    _, a, rho_f = matsuda_g(r0, p, x, method, True)
    return r0, a, rho_f, g_val, res2


def matsuda_g(r, p, x, method='NS', return_params=False):
    if r < 0:
        logger.debug(f'r: {r} failed')
        return -1e15
    n_ch, n_time = x.shape
    rho_f = x.dot(x.T) / n_time
    rho_f.flat[::n_ch + 1] -= r
    rho_b = rho_f.copy()
    _rho_f = x[:, p+1:].dot(x[:, p+1:].T) / n_time
    _rho_f.flat[::n_ch + 1] -= r
    _rho_b = x[:, p:-1].dot(x[:, p:-1].T) / n_time
    _rho_b.flat[::n_ch + 1] -= r
    _rho_fb = x[:, p+1:].dot(x[:, p:-1].T) / n_time
    a = [np.eye(n_ch)]
    b = [np.eye(n_ch)]
    # rho_f, a, rho_b, b = _lattice(p, a, b, rho_f, rho_b, _rho_f, _rho_b,
    #                               _rho_fb, method)
    rho_f, a, _, _ = __lattice(p, x, method)
    logger.debug(f'{r}, rho_f: {np.diag(rho_f)}')
    a_ = np.zeros((n_ch * p, n_ch * p),
                  dtype=np.float64)
    a_.flat[n_ch * p * n_ch::n_ch * p + 1] = 1.
    for i in range(p):
        a_[:n_ch, i*n_ch:(i+1)*n_ch] = -a[i+1]
    f = np.hstack([np.eye(n_ch), np.zeros((n_ch, n_ch*(p-1)))])
    q = np.zeros_like(a_)
    rho_f = np.diag(np.diag(rho_f))
    q[:n_ch, :n_ch] = rho_f
    try:
        Nz = f.shape[-1]
        Ny = x.shape[0]
        var = np.zeros((Nz+Ny, Nz+Ny))
        var[:Nz, :Nz] = q
        var[Nz:, Nz:] = r * np.eye(n_ch)
        out = kalcvf(x, 0, np.zeros(Nz), a_, np.zeros(Ny), f, var,
                     None, None, use_numba=True)
        ll = out['ll']
        # *rest, ll, _ = sskf(x.T, a_, f, q, r * np.eye(factors), None,
        #                     None, None, initial_cond=0.0)
        logger.debug(f'{r}: {ll}')
    except linalg.LinAlgError as err:
        logger.debug(f'{r}: kalcvf failed')
        logger.debug(f'msg: {err}')
        logger.debug(f'q: {np.diag(rho_f)} r: {r}')
        ll = -1e15
    if return_params:
        return ll, a, rho_f
    else:
        return ll


def matsuda_g_new(r, p, x, method='NS', return_params=False):
    if np.any(r < 0):
        logger.debug(f'r: {r} failed')
        return -1e15
    # logger.debug(r)
    # TODO implement the actual Matsuda method.
    factors, n = x.shape
    x_ = np.vstack([x[:, p - i:n - i] for i in range(p+1)])
    cov = x_.dot(x_.T) / n
    C = cov[factors:, factors:] - r * np.eye(p*factors)
    c = cov[:factors, factors:]
    # try:
    #     a = linalg.solve(C, c.T)
    # except linalg.LinAlgError:
    #     print('Yoohoo')
    #     Cinv = linalg.pinvh(C)
    #     a = Cinv.dot(c.T)
    e, v = linalg.eigh(C)
    e_inv = np.zeros_like(e)
    e_inv[e > 0] = 1 / e[e > 0]
    a = (v * e_inv[None, :]).dot(v.T.dot(c.T))
    # import ipdb; ipdb.set_trace()
    # a = solve_with_norm_constarint(C, c.T)
    # import ipdb; ipdb.set_trace()
    # a[a > 1] = 0.99
    # a[a < -1] = -0.99
    rho_f = (cov[:factors, :factors] - a.T.dot(c.T) - c.dot(a)
             + a.T.dot(C).dot(a))
    assert np.allclose(rho_f, rho_f.T)
    a_ = np.zeros((factors * p, factors * p),
                  dtype=np.float64)
    a_.flat[factors * p * factors::factors * p + 1] = 1.
    a_[:factors] = a.T
    f = np.hstack([np.eye(factors), np.zeros((factors, factors*(p-1)))])
    q = np.zeros_like(a_)
    # rho_f = np.diag(np.diag(rho_f))
    q[:factors, :factors] = (rho_f + rho_f.T) / 2
    try:
        Nz = a.shape[0]
        Ny = x.shape[0]
        var = np.zeros((Nz+Ny, Nz+Ny))
        var[:Nz, :Nz] = q
        var[Nz:, Nz:] = r * np.eye(factors)
        out = kalcvf(x, 0, np.zeros(Nz), a_, np.zeros(Ny), f, var,
                     None, None, use_numba=True)
        ll = out['ll']
        # *rest, ll, _ = sskf(x.T, a_, f, q, r * np.eye(factors), None,
        #                     None, None, initial_cond=0.0)
    except Exception as e:
        # import ipdb; ipdb.set_trace()
        logger.debug(f'sskf failed: {e}')
        ll = -1e15
    if return_params:
        a_p = [np.eye(factors)]
        for i in range(p):
            a_p.append(-a[i*factors:(i+1)*factors].T)
        return -ll, a_p, rho_f
    else:
        return -ll

