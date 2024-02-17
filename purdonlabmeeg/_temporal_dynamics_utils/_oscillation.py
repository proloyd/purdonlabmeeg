import imp
import math
from warnings import warn
import collections

import numpy as np
from mne import get_config
from mne.utils import logger
from scipy import linalg
from scipy.special import iv

from ..viz import _get_cycler
from ._ar_utils import find_hidden_ar
from ._kalman_smoother import kalman_smoother
from ._temporal_filtering import stableblockfiltering
from .polyeig import polyeig


def tr(a):
    return a[0, 0] + a[1, 1]


def rt(a):
    return a[1, 0] - a[0, 1]


def sample_spectrum(y, fmin, fmax, sfreq):
    from mne.time_frequency import psd_array_multitaper
    pbf, freq = psd_array_multitaper(y, sfreq, fmin=fmin, fmax=fmax,
                                     bandwidth=1.0, adaptive=True,
                                     low_bias=True, normalization='length',
                                     n_jobs=4)
    return pbf, freq


def ar_spectrum(ar_params, f):
    a, freq = ar_params
    cosfreq = math.cos(2 * math.pi * freq)
    _b = (1 + a ** 2) / (a * cosfreq)
    b = - (_b + math.sqrt(_b ** 2 - 4)) / 2
    sigma2 = - a * cosfreq / b
    temp = np.absolute(1 - 2 * a * cosfreq * np.exp(-1j * 2 * math.pi * f)
                       + (a ** 2) * np.exp(-1j * 4 * math.pi * f)) ** 2
    temp2 = np.absolute(1 + b * np.exp(-1j * 2 * math.pi * f)) ** 2
    return sigma2 * temp2 / temp


class BaseOscModel(object):
    a = None
    q = None

    def __init__(self, alpha=[], freq=[], sigma2=[], a_var=None, freq_var=None, populate=True):
        assert len(alpha) == len(freq)
        assert len(alpha) == len(sigma2)
        self.alpha = np.asanyarray(alpha)
        self.freq = np.asanyarray(freq)
        self.sigma2 = np.asanyarray(sigma2)
        self.n = len(self.alpha)
        self.cosfreq = np.cos(2 * math.pi * self.freq)
        self.sinfreq = np.sin(2 * math.pi * self.freq)
        self.__malloc__()
        if populate:
            self.__populate__()
        self.a_var = np.zeros_like(self.alpha) if a_var is None else a_var
        self.freq_var = (np.zeros_like(self.freq) if freq_var is None
                         else freq_var)
        self._alphas = [self.alpha.copy()]
        self._freqs = [self.freq.copy()]
        self._sigma2s = [self.sigma2.copy()]
        self._update_osc_names()

    def _update_osc_names(self,):
        from itertools import product
        self._osc_names = ['OSC%03d%s' % ii for ii in
                           product(range(self.n), ['x', 'y'])]

    def _expand_indices(self, idx):
        assert max(idx) < self.n
        idx = np.repeat(idx, 2)
        idx = idx * 2
        idx[1::2] += 1
        return idx

    def __add__(self, other):
        alpha = np.hstack(self.alpha, other.alpha)
        freq = np.hstack(self.freq, other.freq)
        sigma2 = np.hstack(self.sigma2, other.sigma2)
        return BaseOscModel(alpha, freq, sigma2)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __len__(self):
        return self.n
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            key = np.arange(start, stop, step)
        if isinstance(key, int):
            key = np.array([key])
        try:
            iter(key)
        except TypeError:
            raise TypeError(f'Invalid argument type: {type(key)}')
        alpha = self.alpha[key].copy()
        freq = self.freq[key].copy()
        sigma2 = self.sigma2[key].copy()
        obj = type(self).__new__(self.__class__)
        obj.__init__(alpha=alpha, freq=freq, sigma2=sigma2)
        return obj

    def __malloc__(self, ):
        self.a = np.zeros((2 * self.n, 2 * self.n), np.float64)
        self.q = np.zeros((2 * self.n, 2 * self.n), np.float64)

    def __populate__(self, ):
        cosfreq = self.alpha * self.cosfreq
        sinfreq = self.alpha * self.sinfreq
        self.a.flat[0::4*self.n+2] = cosfreq
        self.a.flat[1::4*self.n+2] = - sinfreq
        self.a.flat[2*self.n::4*self.n+2] = sinfreq
        self.a.flat[2*self.n+1::4*self.n+2] = cosfreq
        self.q.flat[0::4*self.n+2] = self.sigma2
        self.q.flat[2*self.n+1::4*self.n+2] = self.sigma2

    def _do_arange(self, order):
        self.alpha[:] = self.alpha[order]
        self.freq[:] = self.freq[order]
        self.sigma2[:] = self.sigma2[order]
        self.cosfreq[:] = self.cosfreq[order]
        self.sinfreq[:] = self.sinfreq[order]
        self.__populate__()

    @staticmethod
    def _compute_atomic_summary_stats(x, sigma_x, b):
        if x.ndim == 2:
            n = x.shape[0]
            if sigma_x.ndim == 2:
                cov = sigma_x
                cross_cov = b.dot(sigma_x)
                M1 = x[1:].T.dot(x[1:]) / (n-1) + cov  # C
                M2 = x[1:].T.dot(x[:-1]) / (n-1) + cross_cov  # B
                M3 = x[:-1].T.dot(x[:-1]) / (n-1) + cov  # A
            elif sigma_x.ndim == 3:
                M1 = (x[1:].T.dot(x[1:]) + sigma_x[1:].sum(axis=0)) / (n-1)  # C
                M2 = (x[1:].T.dot(x[:-1]) + b[:-1].sum(axis=0)) / (n-1)  # B
                M3 = (x[:-1].T.dot(x[:-1]) +
                      sigma_x[:-1].sum(axis=0)) / (n-1)  # A
        return M1, M2, M3, n

    def _compute_summary_stats(self, x, sigma_x, b):
        Ms = [self._compute_atomic_summary_stats(this_x, this_sigma_x, this_b) for
              this_x, this_sigma_x, this_b in zip(x, sigma_x, b)]
        M1s, M2s, M3s, ns = [np.stack(M, axis=-1) for M in zip(*Ms)]
        ns = np.squeeze(ns)
        n = np.sum(ns - 1)
        M1, M2, M3 = [np.sum(M * (ns-1), axis=-1) / n for M in (M1s, M2s, M3s)]
        return M1, M2, M3, n

    def update_params(self, x, sigma_x, cross_sigma_x, update_sigma2=False):
        """Update the oscillator parameters (M-step)"""
        M1, M2, M3, n = self._compute_summary_stats(x, sigma_x, cross_sigma_x)

        for j in range(self.n):
            idx = 2 * j
            M1j = M1[idx:idx+2, idx:idx+2]
            M2j = M2[idx:idx+2, idx:idx+2]
            M3j = M3[idx:idx+2, idx:idx+2]

            trM2j = tr(M2j)
            rtM2j = rt(M2j)
            trM3j = tr(M3j)
            trM1j = tr(M1j)

            temp = trM2j ** 2 + rtM2j ** 2
            sqrt_temp = math.sqrt(temp)
            self.freq[j] = math.atan2(rtM2j, trM2j) / (2 * math.pi)
            self.sinfreq[j] = rtM2j / sqrt_temp
            self.cosfreq[j] = trM2j / sqrt_temp
            # self.alpha[j] = max(sqrt_temp / trM3j, 0.01)
            self.alpha[j] = sqrt_temp / trM3j

            # if math.fabs(self.freq[j]) < 5e-5:
            #     self.freq[j] = math.copysign(5e-5, self.freq[j])
            #     self.sinfreq[j] = math.sin(2*math.pi*self.freq[j])
            #     self.cosfreq[j] = math.cos(2*math.pi*self.freq[j])
            #     temp_alpha = (trM2j * self.sinfreq[j]
            #                   + rtM2j * self.cosfreq[j]) / trM3j
            #     self.alpha[j] = max(temp_alpha, 0.01)
            # Take care of he unstable oscillators
            # if self.alpha[j] > 1.0:
            #     self.alpha[j] = 1. - 1e-3
            
            # Update state variance
            if update_sigma2:
                # self.sigma2[j] = (trM1j - temp / trM3j) / 2
                # self.sigma2[j] = (trM1j - self.alpha[j] ** 2 * trM3j) / 2
                temp_sigma2 = (trM1j + (self.alpha[j] ** 2) * trM3j -
                               2 * self.alpha[j] * (trM2j * self.cosfreq[j]
                               + rtM2j * self.sinfreq[j])) / 2
                self.sigma2[j] = np.abs(temp_sigma2)
            self.a_var[j] = self.sigma2[j] / trM3j
            kappa = temp / (self.sigma2[j] * trM3j)
            if kappa < 500:
                self.freq_var[j] = ((1 - (iv(1, kappa) / iv(0, kappa))) /
                                    (2 * np.pi) * 2)
            else:
                self.freq_var[j] = (1 / kappa / (2 * np.pi) * 2)
            if np.isnan(self.freq_var[j]):
                print(kappa)

        self.__populate__()
        self._alphas.append(self.alpha.copy())
        self._freqs.append(self.freq.copy())
        self._sigma2s.append(self.sigma2.copy())

    def _get_sorted_idx(self,):
        """Sorts the components according to 'magnitude'.

        See Neumair and Schneider (2001) for the definition of magnitude.
        Note than sorting is done in ascending order.
        """
        excitation = self.sigma2 / (1 - self.alpha ** 2)
        return np.argsort(excitation)

    def get_hidden_posterior(self, y, mixing_mat, r, mixing_matrix_var=None,
                             initial_cond=0.0, mixing_mat_inst=None, rinv=None, state_input=None,  proj=None, ncomp=0):
        # TODO: take care of the initial condition
        from ._temporal_filtering import get_noise_cov_inverse
        a = self.a
        q = self.q
        if mixing_mat_inst is not None:
            mixing_mat, mixing_matrix_var = mixing_mat_inst._get_vals()
        if rinv is None:
            rinv, _ = get_noise_cov_inverse(r, proj, ncomp,)
        outs = [self._get_hidden_posterior(this_y, mixing_mat, mixing_matrix_var, mixing_mat_inst,
                                           state_input, a, q, r, rinv, initial_cond) for this_y in y]
        x, phi_inv, X_minus_1, ll_ = list(zip(*outs))
        return x, phi_inv, X_minus_1, sum(ll_), initial_cond

    @staticmethod
    def _get_hidden_posterior(y, mixing_mat, mixing_matrix_var, mixing_mat_inst,
                              state_input, a, q, r, rinv, initial_cond):
        if y.ndim == 1:
            y = y[None, :]
        if mixing_matrix_var is None:
            out = kalman_smoother(a, q, np.zeros(a.shape[0]), q.copy(), mixing_mat, r, y, state_input,
                                  ss=False)
            if out['ss']:
                for elem in ['vsm', 'cvsm']:
                    out[elem] = np.repeat(out[elem], y.shape[-1],
                                          axis=-1)
                out['ss'] = False
            x, phi_inv, X_minus_1 = map(lambda a: np.moveaxis(a, -1, 0),
                                        (out['sm'], out['vsm'], out['cvsm']))
            # X_minus_1 = out['cvsm'][..., 1:]
            ll_ = out['ll'].item()
            
            debug = False
            if debug:
                from somata.exact_inference import djkalman
                check = djkalman(a, q, np.zeros((a.shape[0], 1)), q.copy(), mixing_mat, r, y, None, skip_interp=True)
                if not np.allclose(check[0][..., 1:], out['sm']):
                    import pdb; pdb.set_trace()
                if not np.allclose(check[1][..., 1:], out['vsm']):
                    import pdb; pdb.set_trace()
                if not np.allclose(check[2][..., 2:], out['cvsm'][..., :-1]):
                    import pdb; pdb.set_trace()
        else:
            if state_input is not None:
                raise NotImplementedError('state_input is not supported')
            try:
                x, phi_inv, X_minus_1, ll_ = stableblockfiltering(y.T, a, None,
                                                                q, r,
                                                                None,
                                                                None,
                                                                initial_cond,
                                                                mixing_mat_inst=mixing_mat_inst,
                                                                rinv=rinv)
            except np.linalg.LinAlgError as err:
                raise RuntimeError(str(err))
        return x, phi_inv, X_minus_1, ll_

    def get_hidden_prior(self, y, mixing_mat, r, mixing_matrix_var=None):
        # TODO: take care of the epochs
        # call kalman filter with formatted Data
        a = self.a
        q = self.q
        if mixing_matrix_var is None:
            delta = None
            chol = None
        else:
            warn('mixing_matrix_var is ignored for stand-alone predcition.')
        out = kalman_smoother(a, q, None, None, mixing_mat, r, y, ss=True)
        if out['ss']:
            for elem in ['vpred', ]:
                out[elem] = np.repeat(out[elem], y.shape[-1],
                                      axis=-1)
            out['ss'] = False
        _x, _s = map(lambda a: np.moveaxis(a, -1, 0),
                     (out['pred'], out['vpred'],))
        ll = out['ll']
        return _x, _s, ll

    @classmethod
    def initialize_from_data_old(cls, *args, show_psd=False, sfreq=None,
                             n_comp=None, show=False, proj=None, proj_ncomp=0,
                             exclude_zero_freq=True):
        """Initialization follows from the following paper

        Matsuda, T., & Komaki, F. (2017). Multivariate Time Series Decomposition
        into Oscillation Components. Neural Computation, 29, 2055-2075.
        https://doi.org/:10.1162/NECO_a_00981
        """
        if len(args) == 2:
            ar_order, x = args
            n_osc = None
        elif len(args) == 3:
            ar_order, x, n_osc = args
        if isinstance(x, collections.abc.Sequence):
            x = np.concatenate(x, axis=-1)
        if x.ndim == 3:
            x = np.concatenate(list(x), axis=-1)
        if x.shape[0] > 1:
            if False:   # We are not using shared_component_analysis anymore.
                loading_mat, y, is_significant_component, _ = shared_component_analysis(x)
                logger.debug(is_significant_component.all())
                logger.debug(is_significant_component.any())
                if get_config('MNE_LOGGING_LEVEL') == 'debug':
                    import matplotlib.pyplot as plt
                    fig, axes = plt.subplots(50, figsize=(4, len(y[:50])*1))
                    for y_, ax in zip(y[:50], axes):
                        ax.plot(y_)
                    fig.show()
            else:
                x_norm = np.sqrt((x * x).sum(axis=-1, keepdims=True))
                loading_mat = np.diag(np.squeeze(x_norm))
                y = x.copy() / x_norm
                is_significant_component = np.ones(x.shape[0], np.bool_)

            # automatic choice for n_comp
            if n_comp is None:
                out = np.nonzero(~is_significant_component)[0]
                if len(out) == 0 or out[0] == 0:
                    n_comp = x.shape[0]
                else:
                    n_comp = out[0]
            y_ = y[:n_comp]
            loading_mat_ = loading_mat[:, :n_comp]
        else:
            x_norm = np.sqrt((x * x).sum())
            y_ = x.copy() / x_norm
            n_comp = 1
            loading_mat = np.array([[1., ], ]) * x_norm
            loading_mat_ = loading_mat[:, :n_comp]
        
        ar_order = max(ar_order, 2 * int(np.ceil(n_osc / y_.shape[0]))) if n_osc is not None else ar_order
        # out = find_hidden_ar(p, y_, show_psd=show_psd, sfreq=sfreq)
        # outs = [find_hidden_ar(i, y_, show_psd=show_psd, sfreq=sfreq) + (i,)
        #         for i in range(1, p)]
        # outs.sort(key=itemgetter(-3))
        # p = outs[0][-1]
        logger.debug(ar_order)
        out = list(find_hidden_ar(ar_order, y_, show_psd=show_psd, sfreq=sfreq))
        # # Fit each Common component seperately and concatenate them
        # outs = [find_hidden_ar(ar_order, y_[i:i+1], show_psd=show_psd, sfreq=sfreq)
        #         for i in range(len(y_))]
        # new_out = []
        # for each in zip(*outs):
        #     if isinstance(each[0], np.ndarray):
        #         new_out.append(np.diag(np.squeeze(np.stack(each))))
        #     elif isinstance(each[0], list):
        #         new_out.append([np.diag(np.squeeze(np.stack(each_nested)))
        #                         for each_nested in zip(*each)])
        #     else:
        #         new_out.append(each)
        # out = new_out

        v, e = polyeig(*list(reversed(out[1])))
        v *= loading_mat_[0, 0]
        y_ *= loading_mat_[0, 0]
        loading_mat_ /= loading_mat_[0, 0]
        v = v / v[0] # Rescaling to make mixing matrix first row [1, 0]
        V = np.vstack([v / (e ** i) for i in range(ar_order)])
        Vinv = linalg.pinv(V)
        n = y_.shape[-1]
        w = Vinv.dot(np.vstack([y_[:, ar_order-ii:n-ii] for ii in range(ar_order)]))
        eps2 = (np.abs(w[:, 1:] - e[:, None] * w[:, :-1]) ** 2).mean(axis=-1)
        Vinv = Vinv[:, :out[2].shape[0]]
        excitation = (Vinv.dot(out[2]) * Vinv.conj()).sum(axis=1)
        excitation /= (1 - np.abs(e) ** 2)

        # calulate theta and alpha
        v = loading_mat_.dot(v)
        if proj_ncomp > 0:
            v = proj.dot(v)
        
        real = np.isreal(e)
        complex = np.logical_not(real)
        e_real, v_real, exc_real = e[real], v[:, real], excitation[real]
        e_com, v_com, exc_com = e[complex], v[:, complex], excitation[complex]
        e_com, v_com, exc_com = e_com[::2], v_com[:, ::2], exc_com[::2]

        excs = []
        sigma2s = []
        for val, chunk in zip(('real', 'complex'), (real, complex)):
            v_, w_ = v[:, chunk], w[chunk]
            eps2_ = eps2[chunk] 
            jj = 1 if val == 'real' else 2
            excs.append(np.array([np.linalg.norm(v_[:, jj*ii:jj*(ii+1)].dot(w_[jj*ii:jj*(ii+1)]))
                         for ii in range(np.sum(chunk) // jj)]))
            sigma2s.append(np.array([eps2_[jj*ii:jj*(ii+1)].mean()
                                    for ii in range(np.sum(chunk) // jj)]))
        exc_real, exc_com = excs        
        sigma2_real, sigma2_com = sigma2s

        if exclude_zero_freq:
            e = e_com
            v = v_com
            excitation = exc_com.real
            sigma2 = sigma2_com
        else:
            e = np.hstack((e_real, e_com))
            v = np.hstack((v_real, v_com))
            excitation = np.hstack([exc_real, exc_com]).real
            sigma2 = np.hstack((sigma2_real, sigma2_com))

        # excitation *= linalg.norm(v, axis=0) ** 2
        # idx = np.argsort(excitation / (1 - np.abs(e) ** 2))[::-1]
        idx = np.argsort(excitation)[::-1]

        if n_osc is not None and n_osc < len(idx):
            idx = idx[:n_osc]
        e = e[idx]
        v = v[:, idx]
        excitation = excitation[idx]
        sigma2 = sigma2[idx]
        alpha = np.absolute(e)
        freq = np.angle(e) / (2 * math.pi)

        # if x.ndim == 1:
        #     b, fs = sample_spectrum(x[None, :], 0.001, sfreq / 2.5, sfreq)
        # else:
        #     b, fs = sample_spectrum(x[:1], 0.001, sfreq / 2.5, sfreq)
        # a = np.vstack([ar_spectrum(arg, fs) for arg in zip(alpha, freq)]).T
        # try:
        #     sigma2 = linalg.solve(a, b[0])
        # except (linalg.LinAlgError, ValueError):
        #     sigma2, *rest = linalg.lstsq(a, b[0])
        # sigma2[sigma2 <= 0] = 1e-4 * sigma2[sigma2 > 0].min()
        # # print(f"Theo: {excitation / linalg.norm(v, axis=0) ** 2}")
        # # print(f"emp: {sigma2}")
        # sigma2 = excitation / linalg.norm(v, axis=0) ** 2

        c = np.empty((x.shape[0], 2 * len(alpha)))
        c[:, :2*v.shape[1]:2] = v.real
        c[:, 1:2*v.shape[1]:2] = - v.imag
        assert np.allclose(c[0, :2], np.array([1, 0])) 
        out[0] = np.diag(out[0]) if out[0].ndim == 2 else out[0]
        # Initialize noise covariance by a
        r0 = _observation_noise_by_hugo(y_, sfreq) * y.shape[1]
        logger.debug(r0)
        if r0.ndim < 1:
            loading_mat[:, :n_comp] *= (r0) ** 0.5
        else:
            loading_mat[:, :n_comp] *= (r0[None, :]) ** 0.5
        r = np.diag((loading_mat.dot(loading_mat.T) / y.shape[1]))
        logger.debug(f'observation noise: {r}')
        logger.debug('selected frequencies:\n')
        [logger.debug(f'({f}: {a}, {q})\n') for f, a, q in zip(freq, alpha, sigma2)]
        return cls(alpha, freq, sigma2), c, r

    @classmethod
    def initialize_from_data_(cls, *args, show_psd=False, sfreq=None, 
                        exclude_zero_freq=True, **kwargs):
        """Initialization follows from the following paper

        Matsuda, T., & Komaki, F. (2017). Multivariate Time Series Decomposition
        into Oscillation Components. Neural Computation, 29, 2055-2075.
        https://doi.org/:10.1162/NECO_a_00981
        """
        if len(args) == 2:
            ar_order, x = args
            n_osc = None
        elif len(args) == 3:
            ar_order, x, n_osc = args
        if isinstance(x, collections.abc.Sequence):
            x = np.concatenate(x, axis=-1)
        if x.ndim == 3:
            x = np.concatenate(list(x), axis=-1)

        if x.shape[0] > 1:
            x_norm = np.sqrt((x * x).sum(axis=-1, keepdims=True))
            loading_mat_ = np.diag(np.squeeze(x_norm))
            y_ = x.copy() / x_norm
        else:
            x_norm = np.sqrt((x * x).sum())
            y_ = x.copy() / x_norm
            loading_mat_ = np.array([[1., ], ]) * x_norm

        ar_order = max(ar_order, 2 * int(np.ceil(n_osc / x.shape[0]))) if n_osc is not None else ar_order
        logger.debug(ar_order)
        
        from statsmodels.tsa.api import VAR
        model = VAR(y_.T)
        result = model.fit(maxlags=ar_order, ic='aic')
        coeffs, var = result.coefs, result.resid_corr
        coeffs = [loading_mat_ @ -Ak @ linalg.inv(loading_mat_).T for Ak in coeffs]
        coeffs.insert(0, np.eye(y_.shape[0]))
        var = loading_mat_ @ var @ loading_mat_.T

        v, e = polyeig(*list(reversed(coeffs)))
        v = v / v[0] # Rescaling to make mixing matrix first row [1, 0]
        V = np.vstack([v / (e ** i) for i in range(ar_order)])
        Vinv = linalg.pinv(V)
        excitation = np.diag(Vinv[:, :var.shape[0]] @ var @ Vinv[:, :var.shape[0]].T) 
        n = x.shape[-1]
        w = Vinv.dot(np.vstack([x[:, ar_order-ii:n-ii] for ii in range(ar_order)]))
        eps2 = (np.abs(w[:, 1:] - e[:, None] * w[:, :-1]) ** 2).mean(axis=-1)   # time-average

        # calulate theta and alpha
        real = np.isreal(e)
        complex = np.logical_not(real)
        e_real, e_com,  = e[real], e[complex][::2]
        v_real, v_com = v[:, real], v[:, complex][:, ::2]

        excs = []
        orig_sigma2s = []
        sigma2s = []
        for chunk, jj in zip((real, complex), (1, 2)):
            v_, w_ = v[:, chunk], w[chunk]
            eps2_ = eps2[chunk]
            excs.append(np.array([np.linalg.norm(v_[:, jj*ii:jj*(ii+1)].dot(w_[jj*ii:jj*(ii+1)]))
                         for ii in range(np.sum(chunk) // jj)]))
            orig_sigma2s.append(np.array([excitation[jj*ii:jj*(ii+1)].mean()
                                    for ii in range(np.sum(chunk) // jj)]))
            sigma2s.append(np.array([eps2_[jj*ii:jj*(ii+1)].mean()
                                    for ii in range(np.sum(chunk) // jj)]))
        exc_real, exc_com = excs
        orig_sigma2_real, orig_sigma2_com = orig_sigma2s
        sigma2_real, sigma2_com = sigma2s

        if exclude_zero_freq:
            e = e_com
            v = v_com
            excitation = exc_com.real
            sigma2 = sigma2_com
        else:
            e = np.hstack((e_real, e_com))
            v = np.hstack((v_real, v_com))
            excitation = np.hstack([exc_real, exc_com]).real
            sigma2 = np.hstack((sigma2_real, sigma2_com))

        idx = np.argsort(excitation)[::-1]
        logger.debug(f'Polyeig discovered normalized frequencies:\n{sfreq * np.angle(e[idx]) / (2 * math.pi)}\n'
                     'sorted decreasing oder in contribution.')

        if n_osc is not None and n_osc < len(idx):
            idx = idx[:n_osc]
        e = e[idx]
        v = v[:, idx]
        excitation = excitation[idx]
        sigma2 = sigma2[idx]
        alpha = np.absolute(e)
        freq = np.angle(e) / (2 * math.pi)

        c = np.empty((x.shape[0], 2 * len(alpha)))
        c[:, :2*v.shape[1]:2] = v.real
        c[:, 1:2*v.shape[1]:2] = - v.imag
        assert np.allclose(c[0, :2], np.array([1, 0]))
        # out[0] = np.diag(out[0]) if out[0].ndim == 2 else out[0]
        # Initialize noise covariance by a
        r = _observation_noise_by_hugo(x, sfreq) #* x.shape[1]
        logger.debug(f'observation noise: {r}')
        logger.debug('selected frequencies:\n')
        [logger.debug(f'({f}: {a}, {q})\n') for f, a, q in zip(freq, alpha, sigma2)]
        return cls(alpha, freq, sigma2), c, r

    @classmethod
    def initialize_from_data(cls, *args, show_psd=False, sfreq=None, **kwargs):
        if sfreq is None:
            raise ValueError(f'sfreq cannot be None')
        from statsmodels.tsa.api import VAR
        from statsmodels.tsa.ar_model import AutoReg
        if len(args) == 2:
            ar_order, x = args
            n_osc = None
        elif len(args) == 3:
            ar_order, x, n_osc = args
        if isinstance(x, collections.abc.Sequence):
            x = np.concatenate(x, axis=-1)
        if x.ndim == 3:
            x = np.concatenate(list(x), axis=-1)

        ar_order = max(ar_order, 2 * int(np.ceil(n_osc / x.shape[0]))) \
                                        if n_osc is not None else ar_order
        logger.debug(ar_order)
        if x.shape[0] > 2:
            model = VAR(x.T)
            result = model.fit(maxlags=ar_order, ic='aic')
            coeffs = list(-result.coefs)
            coeffs.insert(0, np.eye(x.shape[0]))
            v, e = polyeig(*list(reversed(coeffs)))
        else:
            result = AutoReg(x[0], lags=ar_order, trend='n').fit()
            A = np.zeros((ar_order, ar_order))
            A[0] = result.params
            A[1:, :-1] = np.eye(ar_order-1)
            e, v = np.linalg.eig(A)
            # e = 1 / result.roots
            # v = np.ones((1, e.shape[-1]), dtype=np.complex64)
            vv = np.linalg.inv(v)[:, :1]
            cov = (v[0][:, None] * (vv.dot(vv.T.conj())) * v[0].conj()) * result.fpe
            v = v[:1]

        pos_freq = e.imag > 0
        # retain only positive frequencies.
        e = e[pos_freq]
        v = v[:, pos_freq]
        cfreq = np.angle(e) / (2*np.pi)
        alpha = np.abs(e)
        sel = np.logical_and(alpha > 0.2, cfreq < 5/12)  # 5/12 is completely
                                                         # arbitrary
        cfreq = cfreq[sel]
        alpha = alpha[sel]
        v = v[:, sel]

        v /= v[0]
        psd, fbins = sample_spectrum(x[0:1], 0.1, sfreq/2-1., sfreq)
        mpsd = [ar_spectrum(arg, fbins/sfreq) for arg in zip(alpha, cfreq)]
        mpsd = 2 * np.vstack(mpsd)
        if x.shape[0] > 2:
            # sigma2 = np.linalg.solve(mpsd.dot(mpsd.T) + np.eye(mpsd.shape[0]),
            #                                 psd.dot(mpsd.T)[0])
            # sigma2 = np.linalg.lstsq(mpsd.T, psd.T)
            from scipy import optimize
            sigma2 = optimize.nnls(mpsd.T, psd[0])[0]
        else:
            sigma2 = np.abs(np.diag(cov)[pos_freq][sel])
        pow = sigma2 * mpsd.sum(axis=1) * (np.abs(v) ** 2).sum(axis=0)
        sorted_pow = np.argsort(pow)[::-1]
        sel_idx = sorted_pow[:min(n_osc, sum(pow > 0))]
        v = v[:, sel_idx]
        sigma2 = sigma2[sel_idx]
        cfreq = cfreq[sel_idx]
        alpha = alpha[sel_idx]

        c = np.empty((x.shape[0], 2 * len(alpha)))
        c[:, :2*v.shape[1]:2] = v.real
        c[:, 1:2*v.shape[1]:2] = - v.imag
        assert np.allclose(c[0, :2], np.array([1, 0]))
        # out[0] = np.diag(out[0]) if out[0].ndim == 2 else out[0]
        # Initialize noise covariance by a
        r = _observation_noise_by_hugo(x, sfreq)  #* x.shape[1]
        # r = _observation_noise_by_hugo_new(x, sfreq)  #* x.shape[1]
        logger.debug(f'observation noise: {r}')
        logger.debug('selected frequencies:\n')
        [logger.debug(f'({f}: {a}, {q})\n') for
                f, a, q in zip(cfreq, alpha, sigma2)]
        return cls(alpha, cfreq, sigma2), c, r

    def plot_osc_spectra(self, N=100, ax=None, fs=None, show_freq_var=False):
        import matplotlib.pyplot as plt
        if fs is None:
            fs = 2 * math.pi
        if ax is None:
            ax = plt.gca()
        f_bins = np.linspace(0, 1/2, N, endpoint=False)
        alpha, freq = self.alpha, self.freq
        colors = _get_cycler(len(self))
        ax.set_prop_cycle(colors)
        a = np.vstack([ar_spectrum(arg, f_bins) for arg in zip(alpha, freq)]).T
        lines = ax.semilogy(f_bins * fs, a * self.sigma2[None, :])
        ax.legend(lines, [f'osc-{i}' for i in range(len(self))])
        if show_freq_var and self.freq_var is not None:
            for freq, freq_var, color in zip(self.freq, self.freq_var, colors):
                freq = np.abs(freq)
                ax.axvline(freq * fs, color=color['color'], lw=0.5)
                if np.isinf(freq_var) or np.isnan(freq_var):
                    continue
                std = np.sqrt(freq_var)
                ax.axvspan((freq-std) * fs, (freq+std) * fs,
                           facecolor=color['color'], alpha=.1)
        return ax

    def plot_indv_spectra(self, j, C, N=20, ax=None, fs=None, select=None):
        import matplotlib.pyplot as plt
        if fs is None:
            fs = 2 * math.pi
        if ax is None:
            ax = plt.gca()
        f_bins = np.linspace(0, 1/2, N, endpoint=False)
        alpha, freq = self.alpha, self.freq
        a = np.vstack([ar_spectrum(arg, f_bins) for arg in zip(alpha, freq)]).T
        a *= self.sigma2[None, :]
        c = C[j] ** 2
        c = (np.reshape(c, (-1, 2)).sum(axis=-1))
        pow = (a * c[None, :])
        if select is None:
            idx = np.arange(pow.shape[1])
        else:
            idx = np.argsort(pow.sum(axis=0))[::-1]
            idx = idx[:select]
        ax.plot(f_bins * fs, 10 * np.log10(pow[:, idx]))
        return ax


def _observation_noise_by_hugo(x, sfreq, sample_average=None):
    from mne.time_frequency import psd_array_multitaper
    psds, freqs = psd_array_multitaper(x, sfreq,
                                       bandwidth=2.,
                                       normalization='length',
                                       n_jobs=4)
    if sample_average is None:
        # Average estimates of last 10Hz
        sample_average = np.where(freqs > freqs[-1] - 10.0)[0][0]
    observation_noise = psds[..., sample_average:].mean(axis=-1) / 2
    return observation_noise


def _observation_noise_by_hugo_new(x, sfreq, sample_average=None):
    # from mne.time_frequency import psd_array_multitaper
    # psds, freqs = psd_array_multitaper(x, sfreq,
    #                                    bandwidth=2.,
    #                                    normalization='length',
    #                                    n_jobs=4)
    if sample_average is None:
        # Average estimates of last 10Hz
        sample_average = sfreq / 2 - 10.0
    from mne.filter import filter_data
    y = filter_data(x, sfreq, sample_average, None)
    observation_noise = y.var()
    return observation_noise

