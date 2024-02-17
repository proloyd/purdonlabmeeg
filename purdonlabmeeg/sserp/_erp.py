# Author: Proloy Das <pdas6@mgh.harvard.edu>
import numpy as np
from scipy import linalg, stats, signal
import collections.abc
from ..viz import _get_cycler


class ImpulseResponse:
    _mean = None
    _cov = None
    _a = None
    n_stims = None
    offsets = None
    scalings = None

    def __init__(self, lag_max, lag_min=0, dtype=np.float64):
        if not isinstance(lag_max, int):
            raise ValueError('max_lag needs to be int,' +
                             f' received {type(lag_max)}')
        self.lag_max = lag_max
        self.lag_min = lag_min
        self.dtype = dtype

    def _setup_rcorr(self, stims):
        if isinstance(stims, np.ndarray) and stims.ndim > 1:
            stims = [stim for stim in stims]
        elif not isinstance(stims, collections.abc.Sequence):
            stims = [stims]
        self.n_stims = len(stims)
        nparams = self.lag_max - self.lag_min
        stims_lagged = []
        scalings = []
        offsets = []
        for stim in stims:
            _stim = np.zeros(self.lag_max + len(stim) - self.lag_min, dtype=stim.dtype)
            offsets.append(stim.mean())
            offsets[-1] = 0.    # no offset
            stim = np.squeeze(stim)
            stim = stim[:-self.lag_min] if self.lag_min > 0 else stim 
            _stim[self.lag_max:self.lag_max+len(stim)] = stim - offsets[-1]
            scalings.append(_stim.std())
            scalings[-1] = 1.   # no normalization
            _stim /= scalings[-1]
            temp = [_stim[nparams-k:-k] if k > 0 else _stim[nparams:]
                    for k in range(nparams)]
            stim_lagged = np.vstack(temp)
            stims_lagged.append(stim_lagged)
        stim_lagged = np.vstack(stims_lagged)
        rcorr = stim_lagged.dot(stim_lagged.T)
        self.offsets = np.array(offsets)
        self.scalings = np.array(scalings)
        del temp, _stim,
        nparams =  nparams * self.n_stims
        self._mean = np.zeros(nparams, self.dtype)
        self._cov = np.zeros((nparams, nparams), self.dtype)
        self._a = np.eye(nparams)
        return stim_lagged, rcorr

    def _format_a(self, a):
        nparams = self.lag_max - self.lag_min
        nparams_ =  nparams * self.n_stims
        self._a.flat[nparams_::nparams_ + 1] = - a
        self._a.flat[(nparams_ + 1)*nparams-1::(nparams_ + 1)*nparams] = 0

    def update(self, stim_lagged, rcorr, sigma2, y, a, tau):
        self._format_a(a)
        gamma = 1 / (tau + 1e-15)  # add a small number for numerical stability
        gamma[gamma < 0] = 0
        a_mat = np.sqrt(gamma[:, None]) * self._a
        cov_inv = rcorr / sigma2 + a_mat.T.dot(a_mat)
        e, v = linalg.eigh(cov_inv)
        e_inv = np.zeros_like(e)
        e_inv[e > 0] = 1 / e[e > 0]
        cov = np.dot(v, e_inv[:, None] * v.T, out=self._cov)
        import pdb; pdb.set_trace()
        b = np.sum(stim_lagged * y, axis=-1) / sigma2
        mu = np.dot(cov, np.squeeze(b), out=self._mean)
        # import ipdb; ipdb.set_trace()
        e = tau * e
        kld = np.log(e[e > 0.]).sum() + (a_mat.dot(mu) * gamma).sum()
        return - kld / 2

    def get_a_next(self, tau):
        cov = np.diagonal(self._cov[:-1])
        cross_cov = np.diagonal(self._cov, offset=1)
        num = (self._mean[1:] * self._mean[:-1] + cross_cov) / tau[1:]
        den = (self._mean[:-1] * self._mean[:-1] + cov) / tau[1:]
        return min(max(num.sum() / den.sum(), -1.), 1.)

    def get_tau_next(self, a):
        self._format_a(a)
        cov = self._cov
        # cov = self._a.dot(self._cov).dot(self._a.T)
        return self._a.dot(self._mean) ** 2 + np.diagonal(cov)

    def h(self,):
        if self._mean is None:
            raise ValueError('Did you fit ERP against data?')
        return (np.reshape(self._mean, (self.n_stims, self.lag_max - self.lag_min)) /
                self.scalings[:, None])

    def h_ci(self,):
        if self._mean is None:
            raise ValueError('Did you fit ERP against data?')
        ci = np.sqrt(np.reshape(np.diagonal(self._cov),
                                (self.n_stims, self.lag_max - self.lag_min)))
        ci /= self.scalings[:, None]
        return ci

    def plot(self, confidence=None, sfreq=1.0, ax=None, alpha=.5):
        import matplotlib.pyplot as plt
        if ax is None:
            ax = plt.gca()
        colors = _get_cycler(self.n_stims)
        ax.set_prop_cycle(colors)
        h = self.h()
        lines = ax.plot(np.arange(self.lag_min, self.lag_max) / sfreq, h.T, linewidth=0.5)
        if not isinstance(lines, collections.abc.Sequence):
            lines = [lines]
        if confidence is not None:
            ci = stats.t.ppf((1 + confidence) / 2., 100000) * self.h_ci()
            [ax.fill_between(np.arange(self.lag_min, self.lag_max) / sfreq,
                             lb, ub,
                             color=line.get_color(),
                             alpha=alpha)
             for lb, ub, line in zip(h - ci, h + ci, lines)]
            if len(lines) > 1:
                ax.legend(lines, [f'stim {i}' for i in range(len(lines))])
        return lines

    def __setstate__(self, state):
        self.__dict__.update(state)
        try: 
            getattr(self, 'lag_min')
        except AttributeError:
            self.lag_min = 0
        
        try:
            getattr(self, 'lag_max')
        except AttributeError:
            self.lag_max = self.max_lag


class KernelImpulseResponse(ImpulseResponse):
    _roll_extent = None
    _window = None
    _kernel = None

    def _setup_rcorr(self, stims, window=None):
        if window is None:
            window = ('gaussian', 5)
        stim_lagged, rcorr = super()._setup_rcorr(stims)
        self._a = None
        window_len = self.lag_max - self.lag_min
        window = signal.windows.get_window(window, window_len)
        roll_extent =  window_len // 2 - 1
        self._window = window
        self._roll_extent = roll_extent
        self._kernel = np.vstack([np.roll(window, ii) for ii in range(-roll_extent, roll_extent+1)]).T
        if len(self.scalings) > 1:
            from scipy.sparse import block_diag 
            self._kernel = block_diag([self._kernel] * len(stims))
        stim_lagged = self._kernel.T.dot(stim_lagged)
        rcorr = self._kernel.T @ rcorr @ self._kernel
        nparams = stim_lagged.shape[0]
        self._mean = np.zeros(nparams, self.dtype)
        self._cov = np.zeros((nparams, nparams), self.dtype)
        return stim_lagged, rcorr

    def update(self, stim_lagged, rcorr, sigma2, y, a, tau):
        tau_sqrt = np.sqrt(tau)
        stim_lagged = stim_lagged * tau_sqrt[:, None]
        tau_sqrt[np.isnan(tau_sqrt)] = 0.
        nonzero = (stim_lagged ** 2).sum(axis=1) > 1e-15
        # stim_lagged = stim_lagged[:, nonzero]
        temp = stim_lagged.dot(stim_lagged.T)
        temp.flat[::temp.shape[0]+1] += np.float64(sigma2)
        temp2 = stim_lagged.T @ np.linalg.pinv(temp) @ stim_lagged
        temp2 *= -1
        temp2.flat[::temp2.shape[0]+1] += 1
        temp2 /= np.float64(sigma2)
        self._mean[:] = np.squeeze(tau_sqrt[:, None] * stim_lagged.dot(temp2.dot(y.T)))
        temp3 = stim_lagged.dot(temp2).dot(stim_lagged.T)
        temp3 *= -1
        temp3.flat[::temp3.shape[0]+1] += 1
        self._cov[:] = tau_sqrt[:, None] * (temp3 * tau_sqrt[None, :])
        sign, logdet = np.linalg.slogdet(temp3)
        return  np.nansum(np.log(tau_sqrt[nonzero])) + (logdet +  np.nansum(self._mean / tau_sqrt) + nonzero.sum()) / 2

    def get_tau_next(self, a=0):
        cov = self._cov
        return self._mean ** 2 + np.diagonal(cov)
    
    def h(self,):
        if self._mean is None:
            raise ValueError('Did you fit ERP against data?')
        return (np.reshape(self._kernel.dot(self._mean), 
                           (self.n_stims, self.lag_max - self.lag_min)
                           ) /
                            self.scalings[:, None])
    
    def h_ci(self,):
        if self._mean is None:
            raise ValueError('Did you fit ERP against data?')
        ci = np.sqrt(np.reshape(np.diagonal(self._kernel @ self._cov @ self._kernel.T),
                                (self.n_stims, self.lag_max - self.lag_min)))
        ci /= self.scalings[:, None]
        return ci


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    stim = np.zeros(1000)
    stim[::100] = 1
    fir = ImpulseResponse(50)
    stim_lagged, rcorr = fir._setup_rcorr(stim)
    print(f'stim_lagged = \n{stim_lagged}')
    print(f'rcorr =\n{rcorr}')
    times = np.arange(50)/100
    h = (5 * np.sin(30 * times) *
            np.exp(- (times - 0.15) ** 2 / 0.01))
    y = h.dot(stim_lagged)
    y = y + 1 * np.random.randn(*y.shape)
    y = y[None, :]

    fir = ImpulseResponse(100)
    stim_lagged, rcorr = fir._setup_rcorr(stim)
    tau = 0.35*np.ones(100)
    for i in range(10):
        fir.update(stim_lagged, rcorr, 1, y, 1, tau)
        tau = fir.get_tau_next(1)

    fig, axes = plt.subplots(2, sharex=True)
    [ax.plot(x) for x, ax in zip((h, fir._mean), axes)]

    fir = KernelImpulseResponse(100)
    stim_lagged, rcorr = fir._setup_rcorr(stim)
    tau = 1*np.ones(99)
    for i in range(10):
        fir.update(stim_lagged, rcorr, 1, y, 1, tau)
        tau = fir.get_tau_next()

    fig, axes = plt.subplots(2, sharex=True)
    [ax.plot(x) for x, ax in zip((h, fir._kernel @ fir._mean), axes)]

    
