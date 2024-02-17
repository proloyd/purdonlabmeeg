# author: Proloy Das <pdas6@mgh.harvard.edu> 
import numpy as np
from scipy import linalg
from functools import cached_property


class MixingMatrix:
    def __init__(self, shape):
        if shape[0] == 1:
            raise ValueError(f"for shape={shape} use ScalarMixingMatrix.")
        self.shape = shape
        self._mean = np.empty(shape)
        self._var = None
        self._u = np.eye(shape[0])
        
    @cached_property
    def _df(self) :
        return np.prod(self.shape)
    
    def _do_arrange(self, idx):
        self._mean = self._mean[:, idx]
        if self._var is not None:
            self._var = self._var[:, idx, :][:, :, idx]
    
    def __getitem__(self, key):
        obj = type(self).__new__(self.__class__)
        _mean = self._mean[:, key].copy()
        _var = self._var if self._var is None else self._var[:, key][:, :, key].copy()
        obj.__dict__.update(
            dict(
                shape=_mean.shape,
                _mean=_mean,
                _var=_var,
                _u=self._u.copy(),
            )
        )
        return obj

    @classmethod
    def from_mat(cls, mat, var=None):
        inst = cls(mat.shape)
        inst._mean[:] = mat
        inst._var = var
        return inst

    @staticmethod
    def _compute_summary_stats(ys, xs, sigma_xs):
        Cxxs = []
        Cyxs = []
        ns = []
        for y, x, sigma_x in zip(ys, xs, sigma_xs):
            n = x.shape[0]
            if sigma_x.ndim == 3:
                cov = sigma_x.mean(axis=0)
            elif sigma_x.ndim == 2:
                cov = sigma_x
            Cxxs.append(x.T.dot(x) / n + cov)
            Cyxs.append(y.dot(x) / n)
            ns.append(n)
        Cxxs, Cyxs = [np.stack(C, axis=-1) for C in (Cxxs, Cyxs)] 
        ns = np.array(ns)
        n = np.sum(ns)
        Cxx, Cyx = [np.sum(C * ns, axis=-1) / n for C in (Cxxs, Cyxs)]
        # Cxx = np.sum(Cxxs * ns, axis=-1) / n
        # Cyx = np.sum(Cyxs * ns, axis=-1) / n
        assert np.allclose(Cyx, Cyxs.mean(axis=-1))
        assert np.allclose(Cxx, Cxxs.mean(axis=-1))
        if np.any(np.isinf(Cxx)) or np.any(np.isnan(Cxx)):
            raise RuntimeError(
                f'Computation diverged: Cxx has nan or inf elements')
        return Cxx, Cyx, n

    def update(self, y, x, sigma_x, alpha=None, lambdas=None):
        n_chan = y[0].shape[0]
        n_osc2 = x[0].shape[1]
        if alpha is not None and lambdas is None:
            raise ValueError('lambdas must not be None,' +
                             f' when alpha is not {alpha}')
        elif isinstance(lambdas, float) or isinstance(lambdas, int):
            lambdas = lambdas * np.ones(n_chan)
        if lambdas.shape[0] != n_chan:
            raise ValueError(f'lambdas shape:{lambdas.size} does not match ' +
                             f'y dimension {n_chan}')

        if isinstance(alpha, np.ndarray):
            if alpha.size != n_osc2:
                raise ValueError(f'alpha shape:{alpha.size} does not match ' +
                                 f'x dimension {n_osc2}')
            alpha_as_array = True
        else:
            alpha_as_array = False

        Cxx, Cyx, n = self._compute_summary_stats(y, x, sigma_x,)

        if alpha is None:
            mixing_mat, *rest = linalg.lstsq(Cxx, Cyx.T)
            proj = np.eye(n_chan)
            if lambdas is not None:
                lambdas = np.squeeze(lambdas)
                lambdas_ = lambdas if lambdas.ndim == 2 else np.diag(lambdas)
                _, s, vh = linalg.svd(lambdas_)
                zeros = s < (s.max() * 1e-15)
                vh = vh[zeros]
                proj -= vh.T.dot(vh)
            self._mean[:] = proj.dot(mixing_mat.T)
            return 0.
        else:
            mixing_mat = []
            mixing_matrix_var = []
            alpha_sqrt = alpha ** (1/2)
            alpha = 1.

            lambdas = np.squeeze(lambdas)
            lambdas_ = lambdas if lambdas.ndim == 2 else np.diag(lambdas)
            rhs = lambdas_.dot(Cyx)
            un, mixing_mat, mixing_matrix_var, lambdas, s = \
                _special_matrix_inversion_ba(
                    lambdas_, Cxx, alpha_sqrt, n, rhs, alpha_as_array)

            # Cxx = _apply_balancing(Cxx, alpha_sqrt, alpha_as_array)
            # alpha = 1.
            # if np.isinf(alpha_sqrt).any() or np.isnan(alpha_sqrt).any():
            #     raise RuntimeError(f'Computation diverged: alpha (prior) has nan or inf elements')
            # u, s, vh = linalg.svd(Cxx)
            # s[s < 0.] = 0.
            # lambdas = np.squeeze(lambdas)
            # if lambdas.ndim == 2:
            #     un, lambdas, vnh = linalg.svd(lambdas)
            #     lambdas[lambdas < 0.] = 0.
            #     Cyx = un.T.dot(Cyx)
            #     self._u[:] = un
            # else:
            #     self._u[:] = np.eye(u.shape[0])
            # for i, (Cyxi, lambdai) in enumerate(zip(Cyx, lambdas)):
            #     s_inv = 1 / (s + alpha/(lambdai * n))
            #     Cxxi_inv = u.dot(u.T * s_inv[:, None])
            #     Cxxi_inv = _apply_balancing(Cxxi_inv, alpha_sqrt,
            #                                 alpha_as_array)
            #     mixing_mat.append(Cyxi.dot(Cxxi_inv))
            #     mixing_matrix_var.append(Cxxi_inv / (lambdai * n))
            #     if np.any(np.diag(mixing_matrix_var[-1]) < 0):
            #         raise ValueError('Computation diverged: diag of variance is negative')

            mixing_mat = np.vstack(mixing_mat)
            self._mean[:] = mixing_mat
            if self._var is None:
                self._var = np.stack(mixing_matrix_var) / n
            else:
                self._var[:] = np.stack(mixing_matrix_var) / n
            self._u[:] = un

            # KL Divergence computation
            logdet = np.log(lambdas * s[:, None] * n + alpha).sum()
            # trace = (1 / (lambdas * n * s[:, None] + alpha)).sum()
            # mixing_mat = self._u.dot(mixing_mat)
            # mixing_mat[0, 1::2] = 0.0
            # if alpha_as_array:
            #     temp = mixing_mat * alpha_sqrt[None, :]
            # else:
            #     temp = mixing_mat * alpha_sqrt
            # kld = ((temp * temp).sum() + logdet +
            #        trace - mixing_mat.size) / 2
            kld = (logdet - self._df) / 2
            return kld

    def next_hyperparameter(self, scalar_alpha):
        mixing_mat, mixing_matrix_var = self._get_vals()
        if scalar_alpha:
            alpha = (mixing_mat * mixing_mat).sum()
            if mixing_matrix_var is not None:
                for mixing_matrix_var_i in mixing_matrix_var:
                    alpha += np.trace(mixing_matrix_var_i)
            if alpha < 0:
                raise ValueError(f'alpha: {alpha} is negative, Line 928')
            return mixing_mat.size / alpha
        else:
            d = mixing_mat.shape[0]
            mixing_mat = mixing_mat.copy()
            mixing_mat.shape = (d, -1, 2)
            alpha = (mixing_mat * mixing_mat).sum(axis=-1).sum(axis=0)
            del mixing_mat
            if mixing_matrix_var is not None:
                for mixing_matrix_var_i in mixing_matrix_var:
                    var = np.diag(mixing_matrix_var_i)
                    var.shape = (-1, 2)
                    alpha += var.sum(axis=-1)
            return np.repeat(2 * d / alpha, 2)

    def ctrc(self, r):
        if r.ndim == 1:
            r = np.diag(r)
        if r.ndim != 2:
            raise ValueError('input must be a 1d or 2d np.array')
        mixing_mat, _ = self._get_vals()
        val = mixing_mat.T.dot(r).dot(mixing_mat)
        if self._var is not None:
            temp = (r.dot(self._u) * self._u).sum(axis=1)
            delta = (np.asanyarray(self._var) *
                     temp[:, None, None]).sum(axis=0)
            val = val + delta
        return val

    def crct(self, r):
        if r.ndim != 2:
            raise ValueError('input must be a 2d array')
        e, v = linalg.eigh(r)
        e[e < 0] = 0
        mixing_mat, _ = self._get_vals()
        mv = mixing_mat.dot(v)
        val = (mv * e[None, :]).dot(mv.T)
        if self._var is not None:
            temp = [(r * var_i).sum() for var_i in self._var]
            val = val + self._u.T.dot(self._u * np.asanyarray(temp)[:, None])
        return val

    def _get_vals(self,):
        un = self._u
        mixing_mat = self._mean
        mixing_matrix_var = self._var
        mixing_mat = un.dot(mixing_mat)
        mixing_mat[0, 1::2] = 0.
        if mixing_matrix_var is not None:
            mixing_matrix_var_ = np.stack(mixing_matrix_var)
            mixing_matrix_var = [np.sum(mixing_matrix_var_ *
                                        this_u[:, None, None] ** 2, axis=0)
                                 for this_u in un]
        return mixing_mat, mixing_matrix_var


class ClampedMixingMatrix(MixingMatrix):
    def __init__(self, shape):
        super().__init__(shape)
        self._mean[0] = np.tile(np.array((1, 0)), self.shape[-1] // 2)

    @cached_property
    def _df(self) :
        return np.prod(self.shape) - self.shape[-1]

    def update(self, y, x, sigma_x, alpha=None, lambdas=None):
        n_chan = y[0].shape[0]
        n_osc2 = x[0].shape[1]
        if alpha is not None and lambdas is None:
            raise ValueError('lambdas must not be None,' +
                             f' when alpha is not {alpha}')
        elif isinstance(lambdas, float) or isinstance(lambdas, int):
            lambdas = lambdas * np.ones(n_chan)
        if lambdas.shape[0] != n_chan:
            raise ValueError(f'lambdas shape:{lambdas.size} does not match ' +
                             f'y dimension {n_chan}')

        if isinstance(alpha, np.ndarray):
            if alpha.size != n_osc2:
                raise ValueError(f'alpha shape:{alpha.size} does not match ' +
                                 f'x dimension {n_osc2}')
            alpha_as_array = True
        else:
            alpha_as_array = False

        Cxx, Cyx, n = self._compute_summary_stats(y, x, sigma_x)

        lambdas = np.squeeze(lambdas)
        if lambdas.ndim == 1:
            lambdas = np.diag(lambdas)
        a = lambdas.dot(Cyx)  # shall be aceesed row-wise

        # first row of C matrix
        # c1 = np.tile(np.array((1, 0)), self.shape[-1] // 2)
        c1 = self._mean[0]
        _rhs = a[1:] - np.outer(lambdas[1:, :1], Cxx.dot(c1))

        if alpha is None:
            # new_Cyx = linalg.inv(lambdas[1:][:, 1:]).dot(_rhs)
            u, s, vh = linalg.svd(lambdas)
            w = np.zeros_like(s)
            w[s > s.max() * 1e-15] = 1 / w
            new_Cyx = vh.T.dot((u.T.dot(_rhs) * w[:, None]))
            mixing_mat, *rest = linalg.lstsq(Cxx, new_Cyx.T)
            self._mean[:, 1:] = mixing_mat.T
            return 0.

        alpha_sqrt = alpha ** (1/2)
        alpha = 1.

        _un, mats, mat_vars, lls, s = \
            _special_matrix_inversion_ba(lambdas[1:][:, 1:],  # leave the first row, and coulmn
                                     Cxx, alpha_sqrt, n, _rhs, alpha_as_array)

        # mixing_mat = [c1]
        # mixing_mat.extend(mats)
        # mixing_matrix_var.extend(mat_vars)

        if self._var is None:
            self._mean[:1] = c1
            self._u[0, 0] = 1.
            mixing_matrix_var = [np.zeros_like(Cxx)]
            mixing_matrix_var.extend(mat_vars)
            self._var = np.stack(mixing_matrix_var) / n
        else:
            self._var[1:] = np.stack(mat_vars) / n
        self._mean[1:] = np.stack(mats)
        self._u[1:, 1:] = _un

        # KL Divergence computation
        logdet = np.log(lls * s[:, None] * n + alpha).sum()
        # trace = (1 / (lls * n * s[:, None] + alpha)).sum()
        # mixing_mat = self._u.dot(mixing_mat)[1:]
        # if alpha_as_array:
        #     temp = mixing_mat * alpha_sqrt[None, :]
        # else:
        #     temp = mixing_mat * alpha_sqrt
        # kld = ((temp * temp).sum() + logdet +
        #        trace - mixing_mat.size) / 2
        kld = logdet / 2 # - self._df / 2
        return kld

    def next_hyperparameter(self, scalar_alpha):
        mixing_mat, mixing_matrix_var = self._get_vals()
        mixing_mat = mixing_mat[1:]
        if scalar_alpha:
            alpha = (mixing_mat * mixing_mat).sum()
            if mixing_matrix_var is not None:
                for mixing_matrix_var_i in mixing_matrix_var:
                    alpha += np.trace(mixing_matrix_var_i)
            if alpha < 0:
                raise ValueError(f'alpha: {alpha} is negative, Line 928')
            return mixing_mat.size / alpha
        else:
            d = mixing_mat.shape[0]
            mixing_mat = mixing_mat.copy()
            mixing_mat.shape = (d, -1, 2)
            alpha = (mixing_mat * mixing_mat).sum(axis=-1).sum(axis=0)
            del mixing_mat
            if mixing_matrix_var is not None:
                for mixing_matrix_var_i in mixing_matrix_var:
                    var = np.diag(mixing_matrix_var_i)
                    var.shape = (-1, 2)
                    alpha += var.sum(axis=-1)
            return np.repeat(2 * d / alpha, 2)


class ScalarMixingMatrix(ClampedMixingMatrix):
    def __init__(self, shape):
        if shape[0] != 1:
            raise ValueError(
                f"for shape={shape} use MixingMatrix or ClampedMixingMatrix.")
        self.shape = shape
        self._mean = np.tile(np.array([[1, 0]]), self.shape[-1] // 2)
        self._var = None
        self._u = np.eye(shape[0])

    def update(self, y, x, sigma_x, alpha=None, lambdas=None):
        # No update
        return 0

    def next_hyperparameter(self, scalar_alpha):
        # Remains None always
        return None


def _special_matrix_inversion(lambdas, Cxx, alpha_sqrt, n, rhs, alpha_as_array):
    """Solves  the following probelm:
        (np.kron(lambdas, Cxx) + alpha * I / n) x = rhs.ravel() 
    arising in mixing matrix update.
    """
    u1, s1, v1h = linalg.svd(lambdas)
    s1[s1 < s1.max() * 1e-15] = 0.

    rhs = u1.T.dot(rhs)

    Cxx = _apply_balancing(Cxx, alpha_sqrt, alpha_as_array)
    alpha = 1.
    u2, s2, v2h = linalg.svd(Cxx)
    s2[s2 < s2.max() * 1e-15] = 0.

    mats = []
    mat_vars = []
    for Cyxi, lambdai in zip(rhs, s1):
        s_inv = 1 / (s2 * lambdai + alpha / n)
        Cxxi_inv = u2.dot(u2.T * s_inv[:, None])
        Cxxi_inv = _apply_balancing(Cxxi_inv, alpha_sqrt,
                                    alpha_as_array)
        mats.append(Cxxi_inv.dot(Cyxi))
        mat_vars.append(Cxxi_inv)
        if np.any(np.diag(mat_vars[-1]) < 0):
            raise ValueError(
                'Computation likely diverged: negative definite Hessian')
    return u1, mats, mat_vars, s1, s2

# ("float64[:,:](float64[:,:], float[:], bool)")
def _apply_balancing(Cxx, alpha_sqrt, alpha_as_array):
    if alpha_as_array:
        Cxx = Cxx / alpha_sqrt[:, None]
        Cxx = Cxx / alpha_sqrt[None, :]
    else:
        Cxx = Cxx / alpha_sqrt
        Cxx = Cxx / alpha_sqrt
    return Cxx


from numba import njit, prange, guvectorize
@guvectorize(["void(float64[:,:], float64[:], float64[:,:])"],
             "(m,m),(n)->(m,m)", cache=True)
def f(Cxx, alpha_sqrt, out):
    m = Cxx.shape[0]
    for i in range(m):
        for j in range(m): 
            out[i, j] = Cxx[i, j] / (alpha_sqrt[i] * alpha_sqrt[j])


@njit("float64[:,:](float64[:,:], float64[:], boolean)", parallel=False,
nogil=True, fastmath=True, cache=True)
def _apply_balancing_ba(Cxx, alpha_sqrt, alpha_as_array):
    if alpha_as_array:
        out = Cxx / alpha_sqrt
        out = out.T / alpha_sqrt
        # out[:, :] = Cxx
        # for i in range(Cxx.shape[0]):
        #     for j in range(Cxx.shape[0]): 
        #         out[i, j] = Cxx[i, j] / (alpha_sqrt[i] * alpha_sqrt[j])
        # f(Cxx, alpha_sqrt, out)
    else:
        out = Cxx / (alpha_sqrt[0] ** 2)
    return out
    

@njit("float64[:](float64[:], float64[:, :], boolean, int64, float64[:, :], float64[:, :, :], float64[:], float64[:,:], int64)",
nogil=True, fastmath=True, cache=True)
def _matrix_inverseion_loop_opt(alpha_sqrt, rhs, alpha_as_array, ydim, mats, mat_vars, s1, Cxx, n):
    u2, s2, _ = np.linalg.svd(Cxx)
    for i in range(s2.shape[0]):
        if s2[i] < s2.max() * 1e-15:
            s2[i] = 0.
    for i in prange(ydim):
        (Cyxi, lambdai) = rhs[i], s1[i]
        s_inv = 1 / (s2 * lambdai + 1 / n)
        Cxxi_inv = np.dot(u2 * s_inv, u2.T)
        Cxxi_inv = _apply_balancing_ba(Cxxi_inv, alpha_sqrt,
                                    alpha_as_array)
        mats[i, :] = np.dot(Cxxi_inv, Cyxi)
        mat_vars[i, :, :] = Cxxi_inv
    return s2


def _special_matrix_inversion_ba(lambdas, Cxx, alpha_sqrt, n, rhs, alpha_as_array):
    """Solves  the following probelm:
        (np.kron(lambdas, Cxx) + alpha * I / n) x = rhs.ravel() 
    arising in mixing matrix update.
    """
    ydim, xdim = lambdas.shape[0], Cxx.shape[0]
    mats = np.zeros((ydim, xdim), np.float64, order='C')
    mat_vars = np.zeros((ydim, xdim, xdim), np.float64)
    
    u1, s1, _ = linalg.svd(lambdas)
    s1[s1 < s1.max() * 1e-15] = 0.

    rhs = np.dot(u1.T, rhs)
    if not alpha_as_array: alpha_sqrt = np.array([alpha_sqrt])

    Cxx = _apply_balancing_ba(Cxx, alpha_sqrt, alpha_as_array)

    s2 = _matrix_inverseion_loop_opt(alpha_sqrt, rhs, alpha_as_array, ydim, mats, mat_vars, s1, Cxx, n)

    return u1, mats, mat_vars, s1, s2


def update_noise_var(y, mixing_mat, mixing_matrix_var, x, sigma_x,
                          mixing_mat_inst=None, use_inv_gamma_prior='auto',
                          prior=None):
    if mixing_mat_inst is not None:
        mixing_mat, mixing_matrix_var = mixing_mat_inst._get_vals()

    if use_inv_gamma_prior == 'auto':
        use_inv_gamma_prior = True if mixing_mat_inst is not None else False

    n, Cxx, _lambdas = _compute_summary_stats_for_noise_var(
        y, x, sigma_x, mixing_mat)

    if mixing_mat_inst is None:
        temp = mixing_mat.dot(Cxx).dot(mixing_mat.T)
        temp[np.diag_indices_from(temp)] =\
            [(Cxx * mixing_matrix_var_i).sum()
             for mixing_matrix_var_i in mixing_matrix_var]
    else:
        temp = mixing_mat_inst.crct(Cxx)

    lambdas = _lambdas + temp
    df = 0  # prior degrees of freedom

    if use_inv_gamma_prior:
        # use Inverse-Wishart prior, this is such that lambdas = E[X^{-1}]^{-1}
        n_chan = y[0].shape[0]
        if prior is None:
            data = 1e-8 * np.eye(n_chan)
            df = 1
        else:
            data = prior['data']
            df = prior['df']
        lambdas = (lambdas * n + data * df) / (n + df + n_chan)
        # lambdas = 2 * y.dot(y.T) / n - ((y.dot((y - pred).T) / n
        #                                 - pred.dot(y.T) / n +
        #                                  mixing_mat_inst.crct(Cxx)))
        # lambdas = (lambdas * n + n * 1e-10*np.eye(n_chan)) / (n + n_chan)
        # Try diagonal loading
        # lambdas = 0.01 * np.diag(np.diag(lambdas)) + 0.09 * lambdas
    _, logdet = np.linalg.slogdet(lambdas * (n + df + n_chan))

    return lambdas, - logdet * (n + df) / 2


def _compute_summary_stats_for_noise_var(ys, xs, sigma_xs, mixing_mat):
    ns = []
    Cxxs = []
    _lambdas = []
    for y, x, sigma_x in zip(ys, xs, sigma_xs):
        n = y.shape[-1]

        pred = mixing_mat.dot(x.T)
        res = y - pred
        _lambdas.append(y.dot(res.T) / n - pred.dot(y.T) / n)

        if sigma_x.ndim == 3:
            sigma_x = sigma_x.mean(axis=0)
        Cxxs.append(sigma_x + x.T.dot(x) / n)

        ns.append(n)
    Cxxs = np.stack(Cxxs, axis=-1)
    _lambdas = np.stack(_lambdas, axis=-1)
    ns = np.array(ns)
    n = np.sum(ns)
    Cxx = np.sum(Cxxs * ns, axis=-1) / n
    lambdas = np.sum(_lambdas * ns, axis=-1) / n
    return n, Cxx, lambdas


def test_special_matrix_inversion():
    """(np.kron(lambdas, Cxx) + alpha * I / n) x = rhs.ravel() """
    from numpy.random import default_rng
    from codetiming import Timer
    n = 100
    ydim = 3
    xdim = 4
    lambdas = np.eye(ydim)
    rng = default_rng(seed=143)
    a = rng.standard_normal((xdim, n))
    Cxx = a.dot(a.T) / n
    alpha = rng.standard_normal(xdim) ** 2
    rhs = rng.standard_normal((ydim, xdim))
    x1 = np.linalg.solve(np.kron(lambdas, Cxx) + np.kron(np.eye(ydim), np.diag(alpha)) / n, rhs.ravel())

    alpha_sqrt = np.sqrt(alpha)
    u1, mat, *rest = _special_matrix_inversion(lambdas, Cxx, alpha_sqrt, n, rhs, True)
    x2 = u1.dot(mat).ravel()

    alpha_sqrt = np.sqrt(alpha)
    u1_, mat_, *rest_ = _special_matrix_inversion_ba(lambdas, Cxx, alpha_sqrt, n, rhs, True)
    x3 = u1_.dot(mat_).ravel()

    assert np.allclose(x1, x2)
    assert np.allclose(x2, x3)
    
    import ipdb; ipdb.set_trace()

    alpha_sqrt = np.sqrt(alpha)
    u1_, mat_, *rest_ = _special_matrix_inversion_ba(lambdas, Cxx, 0.5, n, rhs, False)
    x3 = u1_.dot(mat_).ravel()


def benchmark_special_matrix_inversion():
    """(np.kron(lambdas, Cxx) + alpha * I / n) x = rhs.ravel() """
    from numpy.random import default_rng
    from codetiming import Timer
    n = 1000
    ydim = 120
    xdim = 140
    lambdas = np.eye(ydim)
    rng = default_rng(seed=143)
    a = rng.standard_normal((xdim, n))
    Cxx = a.dot(a.T) / n
    alpha = rng.standard_normal(xdim) ** 2
    rhs = rng.standard_normal((ydim, xdim))
    # x1 = np.linalg.solve(np.kron(lambdas, Cxx) + np.kron(np.eye(ydim), np.diag(alpha)) / n, rhs.ravel())

    alpha_sqrt = np.sqrt(alpha)
    u1_, mat_, *rest_ = _special_matrix_inversion_ba(lambdas, Cxx, alpha_sqrt, n, rhs, True)

    alpha_sqrt = np.sqrt(alpha)
    with Timer() as _:
        u1, mat, *rest = _special_matrix_inversion(lambdas, Cxx, alpha_sqrt, n, rhs, True)
        x2 = u1.dot(mat).ravel()

    alpha_sqrt = np.sqrt(alpha)
    with Timer() as t:
        u1_, mat_, *rest_ = _special_matrix_inversion_ba(lambdas, Cxx, alpha_sqrt, n, rhs, True)
        x3 = u1_.dot(mat_).ravel()

    assert np.allclose(x2, x3)
    
    with Timer() as t:
        u1_, mat_, *rest_ = _special_matrix_inversion(lambdas, Cxx, 0.5, n, rhs, False)
        x3 = u1_.dot(mat_).ravel()

    with Timer() as t:
        u1_, mat_, *rest_ = _special_matrix_inversion_ba(lambdas, Cxx, 0.5, n, rhs, False)
        x3 = u1_.dot(mat_).ravel()