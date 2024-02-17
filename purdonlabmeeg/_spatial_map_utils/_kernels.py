# Author: Proloy Das <pdas6@mgh.harvard.edu>
# General Note: using a sparse matrix for distance is probably a bad idea,
# since this sort of thresholding to covariance matrices, i.e.
#         k[i, j] = 0 if dist[i, j]  > min_dist
# forces the covariance matrix to lose its positive-definite property
# (RED FLAG!).
import numpy as np
from scipy import linalg, sparse
from scipy.spatial.distance import pdist, is_valid_y, squareform
from math import exp


def _fasteigendecomposition(k, lh, rh):
    assert lh + rh == k.shape[0]
    eu, vu = linalg.eigh(k[:lh, :lh])
    el, vl = linalg.eigh(k[lh:, lh:])
    v = linalg.block_diag(vu, vl)
    e = np.hstack((eu, el))
    return e, v


class _GPKernel:
    __func_name__ = None
    _csrargs = None

    def __init__(self, X=None, metric='euclidean', dist=None,
                 fixed_params=None, splits=None):
        if X is None and dist is None:
            raise ValueError('Both X and dist cannot be None')
        elif dist is None:
            self._metric = metric
            self.ndim = X.shape[0]
            self._dist = pdist(X, metric)
        elif X is None:
            if sparse.issparse(dist):
                self._metric = 'Unknown'
                self.ndim = dist.shape[0]
                dist = dist.tocsr()
                dist, indices, inptr = dist.data, dist.indices, dist.indptr
                self._csrargs = (indices, inptr)
            self._dist = dist.copy()
        self._fixed_params = fixed_params
        self._splits = splits

    def __repr__(self, ):
        return (f'GPKernel(func={self.__func_name__}) on {self.ndim} points' +
                f' with {self._fixed_params} fixed params)')

    def __func__(self, params):
        raise NotImplementedError

    def __diff_func__(self, params):
        raise NotImplementedError

    def __inv_func__(self, params):
        raise NotImplementedError

    def cov(self, **params):
        all_params = self._fixed_params.copy()
        all_params.update(params)
        return self.__func__(**all_params)

    def diff_cov(self, **params):
        all_params = self._fixed_params.copy()
        all_params.update(params)
        return self.__diff_func__(**all_params)

    def inv_cov(self, **params):
        all_params = self._fixed_params.copy()
        all_params.update(params)
        return self.__inv_func__(**all_params)

    def _tosquare(self, mat):
        if self._csrargs is None:
            return squareform(mat) if is_valid_y(mat) else mat
        else:
            mat = sparse.csr_matrix((mat, *self._csrargs), (self.ndim,
                                                            self.ndim))
            return mat.toarray()

    @ staticmethod
    def _add_eye(mat):
        if sparse.issparse(mat):
            mat += sparse.eye(mat.shape[0])
        else:
            mat.flat[::mat.shape[0]+1] = 1.0
        return mat


class SquaredExponentailGPKernel(_GPKernel):
    __func_name__ = 'Squared Error'

    def __func__(self, theta, sigma_s):
        theta = np.squeeze(theta)
        k = np.zeros_like(self._dist)
        try:
            tau = exp(2*theta)
            if tau > 1e-15:
                temp = self._dist ** 2 / tau
                k = np.exp(- temp / 2)
                k *= sigma_s ** 2
        except OverflowError:
            pass
        k = self._tosquare(k)
        # c.flat[::self.ndim+1] += 1 - sigma_s**2
        k = self._add_eye(k)
        return k

    def __inv_func__(self, theta, sigma_s):
        theta = np.squeeze(theta)
        k = np.zeros_like(self._dist)
        try:
            tau = exp(2*theta)
            if tau > 1e-15:
                temp = self._dist ** 2 / tau
                k = np.exp(- temp / 2)
        except OverflowError:
            pass
        k = self._tosquare(k)
        # c.flat[::self.ndim+1] += 1 - sigma_s**2
        k = self._add_eye(k)
        k *= sigma_s ** 2
        if self._splits is None:
            e, v = linalg.eigh(k)
        else:
            # Do th left and right hemi separately
            e, v = _fasteigendecomposition(k, *self._splits)
        e[e < 0] = 0.
        sigma_n2 = 1 - sigma_s ** 2
        e += sigma_n2
        det = np.log(e).sum()
        kinv = (v / e).dot(v.T)
        k = self._add_eye(k)  # make sure the diagonals of k are 1.0
        return kinv, k, det

    def __diff_func__(self, theta, sigma_s):
        theta = np.squeeze(theta)
        diff1 = np.zeros_like(self._dist)
        try:
            tau = exp(2 * theta)
            if tau > 1e-15:
                temp = self._dist ** 2 / np.exp(2 * theta)
                diff1 = temp * np.exp(- temp / 2)
                diff1 *= sigma_s**2
        except OverflowError:
            pass
        diff2 = np.zeros_like(diff1)
        return [self._tosquare(diff) for diff in (diff1, diff2)]


class MaternGPKernel(_GPKernel):
    __func_name__ = 'Matern (eta=3/2)'

    def __func__(self, theta, sigma_s):
        if np.exp(theta) < 1e-15:
            k = np.zeros_like(self._dist)
        else:
            temp = self._dist / np.exp(theta)
            temp *= np.sqrt(3)
            k = (1 + temp) * np.exp(-temp)
            k *= sigma_s**2
        k = self._tosquare(k)
        k = self._add_eye(k)
        return k

    def __diff_func__(self, theta, sigma_s):
        if np.exp(theta) < 1e-15:
            diff_k = np.zeros_like(self._dist)
        else:
            temp = self._dist / np.exp(theta)
            temp *= np.sqrt(3)
            diff_k = temp * temp
            diff_k *= np.exp(-temp)
            diff_k *= sigma_s**2
        diff_k = self._tosquare(diff_k)
        return diff_k


class MaternGPKernel2(_GPKernel):
    __func_name__ = 'Matern (eta=5/2)'

    def __func__(self, theta, sigma_s):
        if np.exp(theta) < 1e-15:
            k = np.zeros_like(self._dist)
        else:
            temp = self._dist / np.exp(theta)
            temp *= np.sqrt(5)
            k = (1 + temp + temp * temp / 3) * np.exp(-temp)
            k *= sigma_s**2
        k = self._tosquare(k)
        k = self._add_eye(k)
        return k

    def __diff_func__(self, theta, sigma_s):
        if np.exp(theta) < 1e-15:
            diff_k = np.zeros_like(self._dist)
        else:
            temp = self._dist / np.exp(theta)
            temp *= np.sqrt(5)
            diff_k = temp * temp
            diff_k *= (temp + 1) / 3
            diff_k *= np.exp(-temp)
            diff_k *= sigma_s**2
        diff_k = self._tosquare(diff_k)
        return diff_k


class GammaExpGPKernel(_GPKernel):
    __func_name__ = 'gamma Exponential'

    def __func__(self, theta, sigma_s, gamma):
        if np.exp(2*theta) < 1e-15:
            k = np.zeros_like(self._dist)
        else:
            temp = self._dist / np.exp(theta)
            temp **= gamma
            k = np.exp(-temp)
            k *= sigma_s**2
        k = self._tosquare(k)
        k = self._add_eye(k)
        return k

    def __diff_func__(self, theta, sigma_s, gamma):
        if np.exp(theta) < 1e-15:
            diff_k = np.zeros_like(self._dist)
        else:
            temp = self._dist / np.exp(theta)
            temp **= gamma
            diff_k = np.exp(-temp) * temp
            diff_k *= gamma
            diff_k *= sigma_s**2
        diff_k = self._tosquare(diff_k)
        return diff_k


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from _bb_utils import stab_bb
    x = np.random.rand(1000) * 100
    x = np.sort(x)
    # x = np.arange(100)
    x.shape = (1000, 1)
    dist = pdist(x, 'euclidean')
    dist[dist > 90] = 0.0
    dist = squareform(dist)
    dist = sparse.csr_matrix(dist)
    y = 1 / (1 + np.exp(-10 * (x - 49))) - 1 / (1 + np.exp(-10 * (x - 51)))
    sigma = 0.1
    y_noisy = y + sigma * np.random.randn(*y.shape)

    theta = 5
    sigma = 0.10

    def f(theta, gpkernel):
        k = gpkernel.cov(theta=theta)
        if sparse.base.issparse(k):
            k = k.toarray()
        e, v = np.linalg.eigh(k)
        # e[e<0] = 0
        temp = v.T.dot(y_noisy)
        val = np.log(e + sigma ** 2).sum()
        # import ipdb; ipdb.set_trace()
        val += ((temp * temp) / (e + sigma ** 2)[:, None]).sum()
        out = val / 2
        out = np.asanyarray(out)
        out.shape = (1, )
        return out

    def df(theta, gpkernel):
        k = gpkernel.cov(theta=theta)
        dk = gpkernel.diff_cov(theta=theta)
        if sparse.base.issparse(k):
            k = k.toarray()
        if sparse.base.issparse(dk):
            dk = dk.toarray()
        e, v = np.linalg.eigh(k)
        if np.any(e < 0):
            import ipdb
            ipdb.set_trace()
        kinv = (v / (e + sigma ** 2)[None, :]).dot(v.T)
        k.flat[::e.shape[0]+1] += sigma ** 2
        try:
            assert np.allclose(kinv.dot(k), np.eye(k.shape[0]))
        except AssertionError:
            import ipdb
            ipdb.set_trace()
        temp = kinv.dot(y_noisy)
        diffk = kinv - temp.dot(temp.T)
        out = (diffk * dk).sum() / 2
        out = np.asanyarray(out)
        out.shape = (1, )
        return out

    def check_grad(x0, gpkernel):
        def Fx(x): return f(x, gpkernel)
        def gradFx(x): return df(x, gpkernel)
        for delta in reversed(np.logspace(-13, -2, 10)):
            true_diff = Fx(x0+delta) - Fx(x0)
            estimated_diff = gradFx(x0) * delta
            print(true_diff, estimated_diff, (true_diff - estimated_diff)/delta)

    mgpkernel = MaternGPKernel(X=None, dist=dist, fixed_params={'sigma_s':
                                                                np.sqrt(1)})
    fx = [f(i, mgpkernel) for i in np.arange(np.log(5), np.log(100), 0.4)]
    print(fx)
    x0 = np.arange(np.log(5), np.log(100), 0.4)[np.nanargmin(np.asarray(fx))]
    besttheta, res_m = stab_bb(x0=x0, costFn=lambda x: f(x, mgpkernel),
                               gradFn=lambda x: df(x, mgpkernel), tol=1e-4)
    print(f'Best theta: {np.exp(besttheta)}')
    k = mgpkernel.cov(theta=besttheta)
    if sparse.base.issparse(k):
        k = k.toarray()
    temp = np.linalg.solve(k + sigma**2 * np.eye(k.shape[0]), y_noisy)
    y_est_m = k.dot(temp)

    mgpkernel2 = MaternGPKernel2(X=x, fixed_params={'sigma_s': np.sqrt(1)})
    fx = [f(i, mgpkernel2) for i in np.arange(np.log(5), np.log(100), 0.4)]
    x0 = np.arange(np.log(5), np.log(100), 0.4)[np.asarray(fx).argmin()]
    besttheta, res_m = stab_bb(x0=x0, costFn=lambda x: f(x, mgpkernel2),
                               gradFn=lambda x: df(x, mgpkernel2), tol=1e-4)
    print(f'Best theta: {np.exp(besttheta)}')
    k = mgpkernel2.cov(theta=besttheta)
    if sparse.base.issparse(k):
        k = k.toarray()
    temp = np.linalg.solve(k + sigma**2 * np.eye(k.shape[0]), y_noisy)
    y_est_m2 = k.dot(temp)

    segpkernel = SquaredExponentailGPKernel(X=x, fixed_params={'sigma_s':
                                                               np.sqrt(0.999)})
    fx = [f(i, segpkernel) for i in np.arange(np.log(5), np.log(100), 0.4)]
    x0 = np.arange(np.log(5), np.log(100), 0.4)[np.asarray(fx).argmin()]
    besttheta, res_se = stab_bb(x0=x0, costFn=lambda x: f(x, segpkernel),
                                gradFn=lambda x: df(x, segpkernel), tol=1e-4,)
    print(f'Best theta: {np.exp(besttheta)}')
    k = segpkernel.cov(theta=besttheta)
    if sparse.base.issparse(k):
        k = k.toarray()
    temp = np.linalg.solve(k + sigma**2 * np.eye(k.shape[0]), y_noisy)
    y_est_se = k.dot(temp)

    gegpkernel = GammaExpGPKernel(X=x, fixed_params={'gamma': 1.5,
                                                     'sigma_s': np.sqrt(1)})
    fx = [f(i, gegpkernel) for i in np.arange(np.log(5), np.log(100), 0.4)]
    x0 = np.arange(np.log(5), np.log(100), 0.4)[np.asarray(fx).argmin()]
    besttheta, res_se = stab_bb(x0=x0, costFn=lambda x: f(x, gegpkernel),
                                gradFn=lambda x: df(x, gegpkernel), tol=1e-4,)
    print(f'Best theta: {np.exp(besttheta)}')
    k = gegpkernel.cov(theta=besttheta)
    if sparse.base.issparse(k):
        k = k.toarray()
    temp = np.linalg.solve(k + sigma**2 * np.eye(k.shape[0]), y_noisy)
    y_est_ge = k.dot(temp)

    fig, ax = plt.subplots(1)
    ax.scatter(x, y_noisy)
    ax.plot(x, y, label='True')
    ax.plot(x, y_est_m, color='red', label='Matern GP prior')
    ax.plot(x, y_est_m2, color='cyan', label='Matern GP prior-2')
    ax.plot(x, y_est_se, color='green', label='Squared Exponential GP prior')
    ax.plot(x, y_est_ge, color='magenta', label='Gamma-Exponential GP prior')
    ax.legend()
    fig.show()
