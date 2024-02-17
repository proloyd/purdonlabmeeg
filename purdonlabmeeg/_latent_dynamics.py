import numpy as np
from scipy import linalg
from ._temporal_dynamics_utils import sskf, lattice_kalman_ss
from ._spatial_map_utils import spatial_filter, update_spatial_scales, cg_solver


class Handler:
    """Prepares (mostly memory allocation and type-casting) arrays for sskf()"""

    def __init__(self, n_factors, n_channels, n_sources, n_samples, order):
        self.a_ = np.zeros((n_factors * order, n_factors * order),
                           dtype=np.float64)
        self.a_.flat[n_factors * order * n_factors::n_factors * order + 1] = 1.
        self.a_upper = self.a_[:n_factors]

        self.q_ = np.zeros((n_factors * order, n_factors * order),
                           dtype=np.float64)
        self.q_upper = self.q_[:n_factors, :n_factors]

        self.fw_ = np.zeros((n_channels, n_factors * order), dtype=np.float64)
        self.fw_upper = self.fw_[:, :n_factors]

        self.c_ = np.zeros((n_factors * order, n_factors),
                           dtype=np.float64)
        self.c_upper = self.c_[:n_factors, :n_factors]

        self.delta_ = np.zeros((n_factors * order,), dtype=np.float64)
        self.delta_upper = self.delta_[:n_factors]
        self.fwT_rinv_fw = np.zeros((n_factors, n_factors), dtype=np.float64)

        self.ws = np.zeros((n_factors, n_sources), dtype=np.float64)
        self.sigma_ws = np.zeros((n_factors, n_sources, n_sources),
                                 dtype=np.float64)
        self.thetas = None
        self.gp_kernel = None

        _x = np.empty((n_samples, n_factors * order), dtype=np.float64)
        x_ = np.empty_like(_x)
        self.xs = (_x, x_)

        _sigma_x = np.empty((n_factors * order, n_factors * order),
                            dtype=np.float64)
        sigma_x_ = np.empty((n_factors * order, n_factors * order),
                            dtype=np.float64)
        sigma_x_hat = np.empty((n_factors * order, n_factors * order),
                               dtype=np.float64)
        self.sigma_xs = (_sigma_x, sigma_x_, sigma_x_hat)

        self._b = np.empty_like(_sigma_x)

        self.n_factors = n_factors
        self.n_channels = n_channels
        self.n_sources = n_sources
        self.n_samples = n_samples
        self.order = order

    def initialize_spatial_map(self, thetas, gp_kernel, ws, sigma_ws=0.0):
        self.gp_kernel = gp_kernel
        self.thetas = thetas
        self.ws[:] = ws
        self.sigma_ws[:] = sigma_ws

    # TODO: For source localizaion, fw_upper and delta_upper can be directly
    #       initialized without any call to expectation_wrt_w().
    def initialize_sensor_map(self, thetas, gp_kernel, r, fws, sigma_ws=0.0):
        self.gp_kernel = gp_kernel
        self.thetas = thetas
        self.fw_upper[:] = fws.T
        self.sigma_ws[:] = sigma_ws
        e, v = linalg.eigh(r)
        nz = e > 0
        e = e[nz]
        v = v[:, nz]
        temp_fw = v.T.dot(np.atleast_2d(fws.T))
        temp_fw /= np.sqrt(e[:, None])
        temp = temp_fw.T.dot(temp_fw)
        self.fwT_rinv_fw[:] = temp
        _c = linalg.cholesky(temp, lower=True)
        self.c_upper[:] = _c

    def _ravel_a(self):
        p, m, m_ = self.a_upper.shape
        assert m == m_
        return np.reshape(np.swapaxes(self.a_upper, 0, 1), (m, m * p))

    def _unravel_a(self):
        m, mp = self.a_upper.shape
        p = mp // m
        return np.swapaxes(np.reshape(self.a_upper, (m, p, m)), 0, 1)

    def expectation_wrt_w(self, f, r):
        assert self.ws.shape[0] == len(self.sigma_ws)
        e, v = linalg.eigh(r)
        nz = e > 0
        e = e[nz]
        v = v[:, nz]
        temp_f = v.T.dot(np.atleast_2d(f))
        temp_f /= np.sqrt(e[:, None])

        fw = temp_f.dot(self.ws.T)
        temp = fw.T.dot(fw)
        self.fwT_rinv_fw[:] = temp
        _delta = np.asanyarray([(temp_f.dot(sigma_w) * temp_f).sum()
                                for sigma_w in self.sigma_ws])
        temp.flat[::self.n_factors+1] += _delta
        _c = linalg.cholesky(temp, lower=True)
        self.c_upper[:] = _c
        self.delta_upper[:] = _delta
        self.fw_upper[:] = np.dot(f, self.ws.T)
        # import ipdb; ipdb.set_trace()
        return self

    def update_spatial_map(self, y, f, r):
        m = self.n_factors
        p = self.order
        x = self.xs[1][:, (p-1) * m:].T
        cov = self.sigma_xs[1][(p-1) * m:, (p-1) * m:]
        ll = spatial_filter(y, x, cov, f, r, self.thetas, self.gp_kernel,
                            self.ws, self.sigma_ws)
        return ll

    def update_spatial_scale(self, f, r, tol=1e-4):
        m = self.n_factors
        p = self.order
        x = self.xs[1][:, (p-1) * m:].T
        cov = self.sigma_xs[1][(p-1) * m:, (p-1) * m:]
        _ = update_spatial_scales(x, cov, f, r, self.thetas,
                                  self.gp_kernel, self.ws, self.sigma_ws,
                                  tol)
        return self

    def update_timecources(self, y, r, initial_cond=0):
        _, s, s_, b, s_hat, ll = sskf(y.T, self.a_, self.fw_, self.q_,
                                      r, self.c_, self.delta_, self.xs,
                                      initial_cond)
        self.sigma_xs[0][:] = s
        self.sigma_xs[1][:] = s_
        self.sigma_xs[2][:] = s_hat
        self._b[:] = b
        return ll

    def update_time_dynamics(self, method='VM'):
        p = self.order
        m = self.n_factors
        x = self.xs[1][:, :m].T
        cov = self.sigma_xs[1][(p-1) * m:, (p-1) * m:]
        cross_cov = self._b.dot(self.sigma_xs[1])[:m, :m].T
        rho_f, a, _, _ = lattice_kalman_ss(p, x, cov, cross_cov, method=method,)
        self.q_upper[:] = rho_f
        for i in range(self.order):
            self.a_upper[:, i*m:(i+1)*m] = -a[i+1]
        return self

    def _free_energy(self, y, f, r):
        p = self.order
        m = self.n_factors
        x = self.xs[1].T
        cov = self.sigma_xs[1]
        # cross_cov = self._b.dot(self.sigma_xs[1])
        cov_hat = self.sigma_xs[2][(p-1)*m:, (p-1)*m:]
        t = x.shape[1] - 1
        c0 = x[:m, 1:].dot(x[:m, 1:].T) + t * cov[(p-1)*m:, (p-1)*m:]
        # c1 = x[:, :-1].dot(x[:, :-1].T) + t * cov
        # c2 = x[:, :-1].dot(x[:m, 1:].T) + t * cross_cov[:, :m]
        # c3 = y.dot(x[:m].T)
        # d0 = self.fwT_rinv_fw
        # d1 = self.fw_upper
        # term_cnst = - t * self.n_sources * np.log(2*np.pi)

        # e, v = linalg.eigh(r)
        # e_sqrt = np.sqrt(e)
        # _c3 = v.T.dot(c3) / e_sqrt[:, None]
        # _d1 = v.T.dot(d1) / e_sqrt[:, None]
        # # _y = v.T.dot(y) / e_sqrt[:, None]
        # # term_0 = - np.sum(_y * _y)
        # term_1_2 = 2 * np.sum(_c3 * _d1)
        #
        # term_3 = - np.sum(d0 * c0)
        #
        # temp = self.a_upper.dot(c2)
        # temp2 = c0 - temp - temp.T + self.a_upper.dot(c1).dot(self.a_upper.T)
        # e, v = linalg.eigh(self.q_upper)
        # nz = e > 0
        # e = e[nz]
        # v = v[:, nz]
        # term_4 = - t * np.log(e).sum()
        # term_5 = - np.sum((v / e[None, :]).T * v.T.dot(temp2))

        ks = [self.gp_kernel.cov(theta=theta) for theta in self.thetas]
        # all = [self.gp_kernel.inv_cov(theta=theta) for theta in self.thetas]
        # kinvs, ks, _ = [i for i in zip(*all)]
        # term_1_4 = - np.sum([logdet_k_sigmawinv(k, f, xi2, r)
        #                      for k, xi2 in zip(ks, np.diag(c0))])
        # term_2a = - np.sum([(mu * cg_solver(k, mu)).sum()
        #                     for mu, k in zip(self.ws, ks)])
        term_1_2b_4 = - np.sum([trace_kinv_sigma(k, f, xi2, r)
                                for k, xi2 in zip(ks, np.diag(c0))])

        term_3 = t * (logdet(cov_hat) + m)

        term_5 = self.n_factors * self.n_sources
        free_energy = (term_1_2b_4 + term_3 + term_5)
        return free_energy / 2

    def free_energy(self, y, f, r):
        p = self.order
        m = self.n_factors
        x = self.xs[1].T
        cov = self.sigma_xs[1]
        cross_cov = self._b.dot(self.sigma_xs[1])
        cov_hat = self.sigma_xs[2][(p-1)*m:, (p-1)*m:]
        t = x.shape[1] - 1
        c0 = x[:m, 1:].dot(x[:m, 1:].T) + t * cov[(p-1)*m:, (p-1)*m:]
        c1 = x[:, :-1].dot(x[:, :-1].T) + t * cov
        c2 = x[:, :-1].dot(x[:m, 1:].T) + t * cross_cov[:, :m]
        c3 = y.dot(x[:m].T)
        d0 = self.fwT_rinv_fw
        d1 = self.fw_upper
        # term_cnst = - t * self.n_sources * np.log(2*np.pi)

        e, v = linalg.eigh(r)
        e_sqrt = np.sqrt(e)
        _c3 = v.T.dot(c3) / e_sqrt[:, None]
        _d1 = v.T.dot(d1) / e_sqrt[:, None]
        # _y = v.T.dot(y) / e_sqrt[:, None]
        # term_0 = - np.sum(_y * _y)
        term_1_2 = 2 * np.sum(_c3 * _d1)

        term_3 = - np.sum(d0 * c0)

        temp = self.a_upper.dot(c2)
        temp2 = c0 - temp - temp.T + self.a_upper.dot(c1).dot(self.a_upper.T)
        e, v = linalg.eigh(self.q_upper)
        nz = e > 0
        e = e[nz]
        v = v[:, nz]
        term_4 = - t * np.log(e).sum()
        term_5 = - np.sum((v / e[None, :]).T * v.T.dot(temp2))

        out = np.seterr(invalid='raise')
        ks = [self.gp_kernel.cov(theta=theta) for theta in self.thetas]
        # all = [self.gp_kernel.inv_cov(theta=theta) for theta in self.thetas]
        # kinvs, ks, _ = [i for i in zip(*all)]
        term_6_11 = - np.sum([logdet_k_sigmawinv(k, f, xi2, r)
                             for k, xi2 in zip(ks, np.diag(c0))])
        np.seterr(**out)
        term_7 = - np.sum([(mu * cg_solver(k, mu)).sum()
                           for mu, k in zip(self.ws, ks)])
        # term_7 = - np.sum([(mu * kinv.dot(mu)).sum()
        #                    for mu, kinv in zip(self.ws, kinvs)])
        # term_8 = - np.sum([(kinv * sigma_w).sum()
        #                    for sigma_w, kinv in zip(self.sigma_ws, kinvs)])
        term_8 = 0.

        term_9_10 = t * (logdet(cov_hat) + self.n_factors)

        # term_12 = self.n_factors * self.n_sources
        term_12 = 0.
        free_energy = (term_1_2 + term_3 + term_4 + term_5
                       + term_6_11 + term_7 + term_8 + term_9_10 + term_12)
        # ipdb.set_trace()
        return free_energy


def trace_kinv_sigma(k, gain, xi2, sigma_n):
    e, v = linalg.eigh(sigma_n)
    nz = e > 0
    e = e[nz]
    v = v[:, nz]
    gain = v.T.dot(np.atleast_2d(gain))
    gain /= np.sqrt(e[:, None])
    temp = gain.dot(k).dot(gain.T)
    e = linalg.eigvalsh(temp)
    e[e < 0.0] = 0.0
    e = (1 + xi2 * e)
    return np.sum(1 / e + np.log(e)) + gain.shape[1] - gain.shape[0]


def logdet_k_sigmawinv(k, gain, xi2, sigma_n):
    temp = gain.dot(k).dot(gain.T)
    temp *= xi2
    temp += sigma_n
    return logdet(temp)


def logdet(a):
    try:
        c = linalg.cholesky(a)
        val = np.log(np.diag(c)).sum() * 2
    except linalg.LinAlgError:
        e = linalg.eigvalsh(a)
        e = e[e > 0]
        val = np.log(e).sum()
    return val


def test_handler():
    from numpy.random import default_rng
    rng = default_rng(seed=0)
    w = rng.standard_normal(1000)
    v = rng.standard_normal((2, 1000))
    u = np.empty_like(w)
    z = np.empty_like(w)
    u[0] = rng.standard_normal()
    z[0] = rng.standard_normal()
    for k in range(1, 1000):
        u[k] = 0.7 * u[k-1] + w[k-1]
        z[k] = 0.8 * z[k-1] + u[k-1]
    x = np.vstack((u, z))
    f = np.asanyarray([[1, 1], [1, -1]]) / np.sqrt(2)
    y = f.dot(x)
    y_noisy = y + y.std(axis=1)[:, None] * 0.1 * v

    a = np.asanyarray([[0.7, 0], [1, 0.8]])
    q = np.asanyarray([[1, 0], [0, 0]])
    r = np.diag(0.01 * x.var(axis=1))

    handler = Handler(2, 2, 2, 1000, 1)
    handler.ws.flat[::3] = 1
    handler.expectation_wrt_w(f, r)
    # handler.a_upper[:] = a
    # handler.q_upper[:] = q
    handler.a_upper[:] = 0
    handler.q_upper[:] = np.eye(2)
    print('VM')
    lls_vm = []
    initial_cond = np.zeros(2)
    for j in range(99):
        ll = handler.update_timecources(y_noisy, r, initial_cond)
        handler.update_time_dynamics('VM')
        lls_vm.append(ll)
        initial_cond[:] = handler.xs[1][0]
        print(f'{linalg.norm(handler.a_ - a)}, {linalg.norm(handler.q_ - q)}' +
              f' {ll}')
        if len(lls_vm) > 2 and lls_vm[-1] < lls_vm[-2]:
            break

    handler = Handler(2, 2, 2, 1000, 1)
    handler.ws.flat[::3] = 1
    handler.expectation_wrt_w(f, r)
    # handler.a_upper[:] = a
    # handler.q_upper[:] = q
    handler.a_upper[:] = 0
    handler.q_upper[:] = np.eye(2)
    print('NS')
    lls_ns = []
    initial_cond = np.zeros(2)
    for j in range(99):
        ll = handler.update_timecources(y_noisy, r, initial_cond)
        handler.update_time_dynamics('NS')
        # print(handler.q_)
        lls_ns.append(ll)
        print(f'{linalg.norm(handler.a_ - a)}, {linalg.norm(handler.q_ - q)}' +
              f' {ll}')
        initial_cond[:] = handler.xs[1][0]
        if len(lls_ns) > 2 and lls_ns[-1] < lls_ns[-2]:
            break

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1)
    ax.plot(lls_vm, label='VM')
    ax.plot(lls_ns, label='NS')
    ax.legend()
    fig.show()


def test_handler2():
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

    r = 0.01 * np.eye(4)
    a_ = np.vstack((np.hstack((a, b)), np.hstack((np.zeros((2, 2)), c))))
    q_ = np.zeros((4, 4))
    q_[2:, 2:] = d.dot(d.T)

    handler = Handler(4, 4, 4, 100, 1)
    handler.ws.flat[::5] = 1
    handler.expectation_wrt_w(np.eye(4), r)
    # handler.a_upper[:] = a_
    # handler.q_upper[:] = q_
    handler.a_upper[:] = np.zeros_like(a_)
    handler.q_upper[:] = np.eye(4)
    print('VM')
    lls_vm = []
    initial_cond = np.zeros(4)
    for j in range(100):
        ll = handler.update_timecources(x, r, initial_cond)
        handler.update_time_dynamics('VM')
        print(f'{linalg.norm(handler.a_ - a_)}, {linalg.norm(handler.q_ - q_)}'
              + f' {ll}')
        lls_vm.append(ll)
        initial_cond[:] = handler.xs[1][0]
        if len(lls_vm) > 2 and lls_vm[-1] < lls_vm[-2]:
            break

    handler = Handler(4, 4, 4, 100, 1)
    handler.ws.flat[::5] = 1
    handler.expectation_wrt_w(np.eye(4), r)
    # handler.a_upper[:] = a_
    # handler.q_upper[:] = q_
    handler.a_upper[:] = np.zeros_like(a_)
    handler.q_upper[:] = np.eye(4)
    print('NS')
    lls_ns = []
    initial_cond = np.zeros(4)
    for j in range(100):
        ll = handler.update_timecources(x, r, initial_cond)
        handler.update_time_dynamics('NS')
        print(f'{linalg.norm(handler.a_ - a_)}, {linalg.norm(handler.q_ - q_)},'
              + f' {ll}')
        lls_ns.append(ll)
        initial_cond[:] = handler.xs[1][0]
        if len(lls_ns) > 2 and lls_ns[-1] < lls_ns[-2]:
            break

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1)
    ax.plot(lls_vm, label='VM')
    ax.plot(lls_ns, label='NS')
    ax.legend()
    fig.show()


def test_handler3():
    from numpy.random import default_rng
    from math import sqrt
    import matplotlib.pyplot as plt
    from purdonlabmeeg import (SquaredExponentailGPKernel, MaternGPKernel,
                               MaternGPKernel2, GammaExpGPKernel)
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
    f = gain.copy()
    # gpkernel = GammaExpGPKernel(X=pos[:, None],
    #                             fixed_params={'gamma': 1.5,
    #                                           'sigma_s': np.sqrt(1)})
    gpkernel = SquaredExponentailGPKernel(X=pos[:, None],
                                          fixed_params={'sigma_s': np.sqrt(0.99)})
    thetas = 0.1 * np.ones(4)

    # n_factors, n_channels, n_sources, n_samples, order
    handler = Handler(4, 10, 100, 100, 1)
    handler.initialize_spatial_map(thetas, gpkernel, spatial_maps)
    # handler.xs[1][:] = x.T
    handler.ws[:] = spatial_maps
    handler.expectation_wrt_w(f, r)
    # handler.a_upper[:] = a_
    # handler.q_upper[:] = q_
    handler.a_upper[:] = np.zeros_like(a_)
    handler.q_upper[:] = np.eye(4)
    print('scale_update')
    lls_vm = []
    initial_cond = np.zeros(4)
    for j in range(20):
        ll = handler.update_timecources(noisy_measurements, r, initial_cond)
        handler.update_spatial_map(noisy_measurements, f, r)
        handler.update_time_dynamics('VM')
        # ipdb.set_trace()
        handler.update_spatial_scale(f, r, tol=1e-4)
        handler.expectation_wrt_w(f, r)
        print(f'{linalg.norm(handler.a_ - a_) / linalg.norm(handler.a_) },'
              + f' {linalg.norm(handler.q_ - q_) / linalg.norm(handler.q_)}'
              + f' {ll}')
        lls_vm.append(ll)
        initial_cond[:] = handler.xs[1][0]
        # print(handler.a_)
        # print(handler.q_)
        # if len(lls_vm) > 2 and lls_vm[-1] < lls_vm[-2]:
        #     break

    fig, ax = plt.subplots()
    ax.plot(pos[sorted_idx], spatial_maps.T[sorted_idx, :], label='true')
    ax.plot(pos[sorted_idx], handler.ws.T[sorted_idx, :], label='estimated')
    ax.legend()
    fig.show()

    fig, ax = plt.subplots()
    ax.plot(x.T, label='true')
    ax.plot(handler.xs[1], label='estimated')
    ax.legend()
    fig.show()

    print(handler.thetas)
