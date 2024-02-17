import math
import itertools
import collections
import numpy as np

from scipy import linalg, stats
from numba import njit

from ._erp import ImpulseResponse, KernelImpulseResponse
from .._temporal_dynamics_utils import BaseOscModel
from .._spatial_map_utils import sensor_map
from .._temporal_dynamics_utils._oscillation import _observation_noise_by_hugo, sample_spectrum, ar_spectrum
from .._temporal_dynamics_utils._kalman_smoother import (kalcvf, kalcvs)


class ssERP:
    def __init__(self, inst, x, fir, tau, alpha, epochs, r, lls):
        self.oscs = inst
        self._osc_timecourse = x
        self.fir = fir
        self._alpha = alpha
        self._tau = tau
        self._epochs = epochs
        self._noise_var = r
        self._lls = lls
        self.ll = lls[-1]

    def viz_erp(self, ax, sfreq, confidence=None, color='k'):
        if not isinstance(ax, (collections.abc.Sequence, np.ndarray)):
            lines = self.fir.plot(confidence=confidence, sfreq=sfreq, ax=ax)
            ax = [ax]
        else:
            if not isinstance(color, (collections.abc.Sequence, np.ndarray)):
                color = [color] * self.fir.n_stims
            if len(ax) != self.fir.n_stims:
                raise ValueError(f"Could not find {self.fir.n_stims} axis")
            if self.fir.n_stims % len(color) != 0:
                raise ValueError(f"Could not find {self.fir.n_stims} colors")
            h = self.fir.h()
            color = itertools.cycle(color)
            lines = [this_ax.plot(np.arange(self.fir.max_lag) / sfreq, this_h,
                                  color=this_color, linewidth=0.5)
                     for this_h, this_ax, this_color in zip(h, ax, color)]
            if confidence is not None:
                ci = (stats.t.ppf((1 + confidence) / 2., 100000)
                      * self.fir.h_ci())
                [this_ax.fill_between(np.arange(self.fir.max_lag) / sfreq,
                                      lb, ub, color=this_color,
                                      alpha=0.2)
                 for this_ax, lb, ub, line, this_color in zip(ax, h - ci,
                                                              h + ci, lines,
                                                              color)]
        for this_ax in ax:
            this_ax.set_frame_on(False)
            this_ax.grid(True, linestyle='-.', linewidth=0.5)
            this_ax.tick_params(labelleft=True, left=False, bottom=False,
                                labelbottom=True)
            this_ax.set_xlabel('Time (in s)')
            this_ax.set_ylabel('\u03bcV')
        return lines

    def viz_epoch_means(self, axes, sfreq, confidence, color='k'):
        if not isinstance(color, (collections.abc.Sequence, np.ndarray)):
            color = [color] * self.fir.n_stims
        if len(axes) != self.fir.n_stims:
            raise ValueError(f"Could not find {self.fir.n_stims} axis")
        if self.fir.n_stims % len(color) != 0:
            raise ValueError(f"Could not find {self.fir.n_stims} colors")
        color = itertools.cycle(color)
        taxis = np.arange(0, self.fir.max_lag) / sfreq
        for ii, (es, ax, cl) in enumerate(zip(self._epochs, axes, color)):
            classical_erp, lci, uci = confidence_interval(es, axis=0,
                                                          confidence=confidence)
            ax.plot(taxis, classical_erp.T, color=cl, linewidth=0.5)
            ax.fill_between(taxis,
                            np.squeeze(lci),
                            np.squeeze(uci),
                            color=cl,
                            alpha=0.2)

        for i, ax in enumerate(axes):
            ax.set_frame_on(False)
            ax.grid(True, linestyle='-.', linewidth=0.5)
            ax.tick_params(labelleft=True, left=False, bottom=False,
                           labelbottom=True)
            ax.set_xlabel('Time (in s)')
            ax.set_ylabel('\u03bcV')

    def viz_osc(self, ax, sfreq, data):
        from mne import time_frequency
        self.oscs.plot_osc_spectra(N=500, ax=ax, fs=sfreq)
        kwargs = dict(bandwidth=1.0, adaptive=False, low_bias=True,
                      normalization='length')
        if data is not None:
            x = data-data.mean(axis=-1)
            psd, freq = time_frequency.psd_array_multitaper(x,
                                                            sfreq,
                                                            **kwargs)
            ax.semilogy(freq, psd.T, 'k')
        pbg, freq = time_frequency.psd_array_multitaper(self._osc_timecourse[::2],
                                                        sfreq,
                                                        **kwargs)
        ax.semilogy(freq, pbg.T, 'g')

        ax.set_xlabel('Frequency (in Hz)')
        ax.set_ylabel('PSD')
        ax.set_ylim([1e-4, 1e5])
        ax.set_xlim([0, sfreq/2])
        ax.set_frame_on(False)
        ax.grid(True, linestyle='-.', linewidth=0.5)
        ax.tick_params(labelleft=True, left=False, bottom=False,
                       labelbottom=True)


class BackGroundOscModel(BaseOscModel):
    def get_hidden_posterior(self, y, mixing_mat, r, mixing_matrix_var=None,
            initial_cond=0, mixing_mat_inst=None, state_input=None):
        phi = self.a
        q = self.q
        var = np.eye((q.shape[0] + 1))
        var[:q.shape[0], :q.shape[0]] = q
        var[q.shape[0]:, q.shape[0]:] = r
        out_kf = kalcvf(y, 0, np.zeros(2*self.n), phi, np.zeros(1), mixing_mat, var, return_pred=True,
                        return_filt=False, return_L=True, return_Dinve=True,
                        return_Dinv=True, return_K=False)
        out_sm = kalcvs(y, np.zeros(2*self.n), phi, np.zeros(1), mixing_mat, var,
                        out_kf['pred'],
                        out_kf['vpred'],
                        Dinvs=out_kf['Dinv'],
                        Dinves=out_kf['Dinve'],
                        Ls=out_kf['L'])
        args = map(lambda a: np.moveaxis(a, -1, 0), (out_sm['sm'], out_sm['vsm'], out_sm['cvsm']))
        res = tuple(args) + (out_kf['ll'], 0)
        return res

    def get_hidden_prior(self, y, mixing_mat, r, mixing_matrix_var=None):
        phi = self.a
        q = self.q
        var = np.eye((q.shape[0] + 1))
        var[:q.shape[0], :q.shape[0]] = q
        var[q.shape[0]:, q.shape[0]:] = r
        out_kf = kalcvf(y, 0, np.zeros(2*self.n), phi, np.zeros(1), mixing_mat, var, return_pred=True,
                        return_filt=False, return_L=True, return_Dinve=True,
                        return_Dinv=True, return_K=True)
        return out_kf

    @classmethod
    def initialize_from_data(cls, y, n_osc=2, max_order=13, sfreq=100,):
        from statsmodels.tsa.ar_model import ar_select_order
        sel = ar_select_order(y.T, max_order,)
        res = sel.model.fit()
        sel_roots = (res.roots.imag > 0)      # choose only the positive frequency.
        cfreq = np.angle(res.roots[sel_roots]) / (2*np.pi)
        alpha = np.abs(res.roots[sel_roots])
        cfreq, alpha = tuple(map(lambda x: x[:n_osc], (cfreq, alpha)))

        psd, fbins = sample_spectrum(y, 1, sfreq/2-5., sfreq)
        mpsd = [ar_spectrum(arg, fbins / sfreq) for arg in zip(alpha, cfreq)]

        mpsd = 2 * np.vstack(mpsd)
        sigma2 = np.linalg.solve(mpsd.dot(mpsd.T), psd.dot(mpsd.T)[0])
        r = res.sigma2
        c = np.zeros((1, 2 * len(cfreq)))
        c[:, ::2] = 1.
        return cls(alpha, cfreq, sigma2), c, r 


def get_erp_equation_terms(out_kf, stim_lagged, mixing_mat):
    N, T = stim_lagged.shape
    X = np.ascontiguousarray(stim_lagged.T)
    HM = np.zeros((N, T))
    hess = np.zeros((N, N))
    b = np.zeros((N))
    K = out_kf['K']
    Dinv = out_kf['Dinv']
    Dinve = out_kf['Dinve']
    L = out_kf['L']
    M_k = np.zeros((K.shape[0], N))
    # [print(arg.dtype, arg.shape) for arg in (X, HM, K, Dinv, Dinve, L, M_k, hess, b)]
    get_erp_equation_terms_opt(T, X, mixing_mat, HM, K, Dinv, Dinve, L, M_k, hess, b)
    return hess, b, HM

@njit('void(int64, float64[:,:], float64[:,:], float64[:,:], float64[:,:,:], float64[:,:,:], float64[:,:],'
      'float64[:,:,:], float64[:,:], float64[:,:], float64[:])')
def get_erp_equation_terms_opt(T, X, H, HM, K, Dinv, Dinve, L, M_k, hess, b):
    for k in range(1, T):
        M_k = np.dot(L[:, :, k-1], M_k) + np.dot(K[:, :, k-1], X[k-1:k])
        v = H.dot(M_k) - X[k:k+1]
        HM[:, k] = v
        hess += v.T.dot(Dinv[:, :, k].dot(v))
        b += v.T.dot(Dinve[:, k])
    return


class ForeGroundImpulseResponse(ImpulseResponse):
    def update(self, a, tau, hess, b):
        self._format_a(a)
        gamma = 1 / (tau + 1e-15)  # add a small number for numerical stability
        gamma[gamma < 0] = 0
        a_mat = np.sqrt(gamma[:, None]) * self._a
        # erp = np.linalg.solve(hess + a_mat.T.dot(a_mat), -b)
        cov_inv = hess + a_mat.T.dot(a_mat)
        e, v = linalg.eigh(cov_inv)
        e_inv = np.zeros_like(e)
        e_inv[e > 0] = 1 / e[e > 0]
        cov = np.dot(v, e_inv[:, None] * v.T, out=self._cov)
        mu = np.dot(cov, np.squeeze(-b), out=self._mean)

        kld = 0.5 * (np.log(gamma).sum() + np.log(e_inv[e > 0.]).sum()\
                     - (a_mat * a_mat.dot(cov + mu.dot(mu.T))).sum()) 
        # kld = np.log(e[e > 0.]).sum() + (a_mat.dot(mu) * gamma).sum()
        return kld

    def get_tau_next(self, a, tau):
        self._format_a(a)
        gamma = 1 / (tau + 1e-15)  # add a small number for numerical stability
        gamma[gamma < 0] = 0
        a_mat = np.sqrt(gamma[:, None]) * self._a
        cov = self._cov
        mu = self._mean
        h = (a_mat * a_mat.dot(cov)).sum(axis=0)
        g = (self._a * self._a.dot(cov + mu.dot(mu.T))).sum(axis=0)
        return np.sqrt(g / h)


def fit_sserp(eeg, stims, sfreq, t_max, t_min, a='auto', n_osc=2, ar_order=13,
              use_sparsifying_prior=False, max_iter=100):
    """utility function to fit sserp

    See test_sserp for example usage:
    ```
    sserp = fit_sserp(eeg, [stim], 100, .50, .0, a=1, n_osc=2, ar_order=13,
                    use_sparsifying_prior=False, max_iter=100)
    ```
    where `eeg.shape = (1, 1000), stim.shape = (1000,)`.
    Parameters:
    ----------
    eeg: single channel eeg data  
        np.ndarray of shape (1, T) (raw) or np.ndarray of shape (n_trials, 1, T) (epochs)
    stims: stimulus
        Sequences of np.ndarray (T,) (raw) or np.ndarray (n_trials, T) (epochs)
    sfreq: sampling frequency
        scalar / float
    t_max: end time of ERP (in secs)
        scalar / float
    lag_min: start time of ERP (< 0 usually)
        scalar / float
    a: controls smoothness of ERP
        scalar / float (0 < a <= 1)
    n_osc: number of backgorund oscillators to be fitted
        int
    ar_order: number of ar lags to consider for initialization
        int
    Output:
    ------
    sserp: ssERP object containing all fitted values.
    """
    if eeg.ndim == 3: # epochs data structure
        assert stims[0].shape[0] == eeg.shape[0], f'stims epochs(={stims[0].shape[0]}) does not match with eeg epochs(={eeg.shape[0]})' 
        eeg = np.hstack(eeg)
        stims = [np.hstack(stim) for stim in stims]

    lag_max = int(math.ceil(t_max * sfreq))
    lag_min = int(math.floor(t_min * sfreq))
    nparams = lag_max - lag_min
    if a == 'auto':
        a = 1.
        update_a = True
    else:
        update_a = False

    denoised_fir = ForeGroundImpulseResponse(lag_max, lag_min)
    stim_lagged, _ = denoised_fir._setup_rcorr(stims)
    erp = np.linalg.pinv(stim_lagged.dot(stim_lagged.T)
                         + 1*np.eye(nparams * len(stims))).dot(stim_lagged.dot(eeg.T))
    osc, H, r_init = BackGroundOscModel.initialize_from_data(eeg - erp.T.dot(stim_lagged),
                                                             n_osc, ar_order)
    r_init = _observation_noise_by_hugo(eeg, sfreq)
    r = r_init.copy() / 10
    tau = 1 * np.ones(nparams * len(stims))
    lls = np.nan * np.empty(max_iter)
    for j in range(max_iter):
        out = osc.get_hidden_prior(eeg, H, r,)
        # osc.plot_osc_spectra()

        hess, b, HM = get_erp_equation_terms(out, stim_lagged, H)
        kld = denoised_fir.update(a, tau, hess, b)
        tau = denoised_fir.get_tau_next(a, tau)
        if use_sparsifying_prior:
            pass
        else:
            for i in range(denoised_fir.n_stims):
                tau[i*nparams + 1:(i+1)*nparams] = \
                        tau[i*nparams + 1:(i+1)*nparams].mean()
        tau[:] = tau.mean()
        a = denoised_fir.get_a_next(tau) if update_a else a

        err = eeg - H.dot(out['pred'][...,:-1]) + denoised_fir._mean.dot(HM)
        r = r * (err * err * out['Dinv']).mean(axis=-1)

        err = eeg - denoised_fir._mean[None, :].dot(stim_lagged)
        out_ = osc.get_hidden_posterior(err, H, r,)
        osc.update_params([out_[0]], [out_[1]], [out_[2]])
        # psd, fbins = sample_spectrum(out_[0].T, 2, 40, 100)
        # plt.semilogy(fbins/100 * 2 * np.pi, psd.T[:,::2])
        
        err = (err - H.dot(out_[0].T))
        # r = (err * err + np.moveaxis(H @ out_[1] @ H.T, 0, -1)).mean(axis=-1)
        # r = r + (stim_lagged * denoised_fir._cov.dot(stim_lagged)).mean(axis=-1).sum()
        
        lls[j] = out_[3] + kld
        if np.abs(lls[j] - lls[j-1]) / np.abs(lls[j-1]) < 1e-8:
            break

    # evoked = fir._mean.dot(stim_lagged)

    return ssERP(osc, out_[0], denoised_fir, tau, a, None, r, lls)


def fit_old_sserp(eeg, stims, sfreq, t_max, t_min, a='auto', n_osc=2,
                  ar_order=13, use_sparsifying_prior=False,
                  max_iter=100, initial_guess=None):
    out = _fit_erp(eeg, sfreq, stims, int(t_max*sfreq), int(t_min*sfreq),
                   a, n_osc, ar_order, initial_guess, use_sparsifying_prior)

    return ssERP(out[0], out[2], out[1], out[4][0], out[4][1], out[5],
                 out[6][1], out[6][0])    


def fit_erp(noisy_data, sfreq, stims, lag_max, lag_min=0, a='auto', n_osc=2, ar_order=8,
            initial_guess=None, use_sparsifying_prior=False, window=False, window_params=None):
    if window != False:
        if window_params is not None:
            window = (window, window_params*sfreq)
        out = _fit_kernel_erp(noisy_data, sfreq, stims, int(lag_max*sfreq), int(lag_min*sfreq),
                        a, n_osc, ar_order, initial_guess, use_sparsifying_prior=True,
                        window=window)        
    else:
        out = _fit_erp(noisy_data, sfreq, stims, int(lag_max*sfreq), int(lag_min*sfreq),
                a, n_osc, ar_order, initial_guess, use_sparsifying_prior)

    return ssERP(out[0], out[2], out[1], out[4][0], out[4][1], out[5],
                 out[6][1], out[6][0])


def _fit_erp(noisy_data, sfreq, stims, lag_max, lag_min, a='auto', n_osc=2, ar_order=8,
             initial_guess=None, use_sparsifying_prior=False, **kwargs):
    """utility function

    Parameters
    ----------
    noisy_data: ndarray (1, T)  [Needs to be in mV, not in V.]
    sfreq: scalar
    stims: Sequences of ndarray (1, T) | np.ndarray (K, T)
    lag_max: end index of ERP
    lag_min: start index of erp (< 0 usually)
    a: scalar [0, 1] proportional to smoothness of ERP
    n_osc: int number of oscillators
    ar_order: int number of ar lags to consider for initialization
    initial_guess: (alpha, freq, sigma2) | None

    NOTE: n_osc and ar_order are ignored when initial_guess is not None.
    """
    nparams = lag_max - lag_min
    y = noisy_data.copy()
    all_epochs = []
    for events in stims:
        epochs = []
        for idx in np.nonzero(events)[0]:
            if idx+lag_max < noisy_data.shape[-1]:
                epochs.append(noisy_data[..., idx+lag_min:idx+lag_max])
        epochs = np.stack(epochs)
        all_epochs.append(epochs)

    fir = ImpulseResponse(lag_max, lag_min)
    stim_lagged, rcorr = fir._setup_rcorr(stims)
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.matshow(stim_lagged[:, :2000])
    # for events, c in zip(stims, ['g', 'y']):
    #     for idx in np.nonzero(events)[0]:
    #         if idx < 2000:
    #             ax.axvline(idx, color='c')
    # fig.show()
    # for i, epochs in enumerate(all_epochs):
    #     fir._mean[i*fir.max_lag: (i+1)*fir.max_lag] = (epochs.mean(axis=0) *
    #                                                    fir.scalings[i])
    temp = np.linalg.solve(rcorr, (stim_lagged * y).sum(axis=-1))
    fir._mean[:] = np.squeeze(temp)
    y1 = y - fir._mean.dot(stim_lagged)
    if initial_guess is None:
        inst, C, r_orig = BaseOscModel.initialize_from_data(ar_order,
                                                            y1 - y1.mean(),
                                                            n_osc,
                                                            show_psd=True,
                                                            sfreq=sfreq,)
    else:
        inst = BaseOscModel(*initial_guess)
        n_osc = len(initial_guess[0])
        C = (np.arange(2 * n_osc) % 2)[::-1][None, :]
        r_orig = _observation_noise_by_hugo(y1 - y1.mean(axis=-1),
                                            sfreq, sample_average=None)
    # r_orig *= 4

    mixing_mat_inst = sensor_map.ScalarMixingMatrix.from_mat(C, None)
    alpha = None
    if a == 'auto':
        a = 1.
        update_a = True
    else:
        update_a = False

    lls = []
    tau = (np.diff(temp) ** 2).mean() * np.ones(nparams * len(stims))
    r = r_orig * np.eye(1)
    y2 = [y.copy()]
    ll = 0
    print(r_orig)
    # for j in range(400):
    #     ll = fir.update(stim_lagged, rcorr, r, y2, a, tau)
    #     y1 = y - fir._mean.dot(stim_lagged)
    #     x_, s_, b, ll_, initial_cond = inst.get_hidden_posterior(
    #                                       y1, C, r,
    #                                       None, None,
    #                                       mixing_mat_inst=None)
    #     # x_[:] = 0
    #     y2 = y - C.dot(x_.T)
    #     # ll += fir.update(stim_lagged, rcorr, r, y2, a, tau)
    #     ll += ll_
    #     r = update_noise_var(y, C, x_, s_, stim_lagged, rcorr, fir)
    #     r = (9 * r + r_orig) / 10    
    for j in range(100):
        y1 = [this_y - fir._mean.dot(stim_lagged) for this_y in y]
        # y1 = y.copy()
        for kk in range(5):
            if kk != -1:
                x_, s_, b, ll_, _ = inst.get_hidden_posterior(
                                                y1, C, r,
                                                None, None,
                                                mixing_mat_inst=mixing_mat_inst)
                y2 = [this_y - C.dot(this_x.T) for this_y, this_x in zip(y, x_)]
            else:
                # _x, _s, ll_,  = inst.get_hidden_prior(y1[0][None, :], C, r)
                # y2 = [this_y - C.dot(this_x.T)[..., 1:-1] for this_y, this_x in zip(y, [_x])]
                pass

            ll = fir.update(stim_lagged, rcorr, r, y2, a, tau)
            ll += ll_
            # r = update_noise_var(y, C, x_, s_, stim_lagged, rcorr, fir)
            r = update_noise_var((y,), C, x_, s_, (stim_lagged,), rcorr, fir)
            # r = (9 * r + r_orig) / 10
            y1 = [this_y - fir._mean.dot(stim_lagged) for this_y in y]
        debug = False
        if debug:
            import matplotlib.pyplot as plt
            from mne.time_frequency import psd_array_multitaper
            xpsds, xfreqs = psd_array_multitaper(x_[0].T, sfreq,
                                               bandwidth=2.,
                                               normalization='length',
                                               n_jobs=4)
            ypsds, yfreqs = psd_array_multitaper(y[0], sfreq,
                                               bandwidth=2.,
                                               normalization='length',
                                               n_jobs=4)
            fig, ax = plt.subplots()
            ax.semilogy(xfreqs, xpsds.T)
            ax.semilogy(yfreqs, ypsds)
            # ax.plot(C.dot(x_[0].T).T, alpha = 0.2)
            # ax.plot(fir._mean.dot(stim_lagged))
            # import pdb; pdb.set_trace()

        lls.append(ll)
        print(ll)
        if j > 1 and np.abs((lls[-1] - lls[-2]) / lls[-2]) < 1e-8:
            print('Convergence reached!')
            break

        inst.update_params(x_, s_, b, update_sigma2=True)
        # r = update_noise_var((y,), C, x_, s_, (stim_lagged,), rcorr, fir)
        # r = (9 * r + r_orig) / 10

        # print('freq')
        # print(inst.freq * sfreq)
        # print('alpha')
        # print(inst.alpha)
        # print('sigma2')
        # print(inst.sigma2)
        a = fir.get_a_next(tau) if update_a else a
        tau = fir.get_tau_next(a)
        if use_sparsifying_prior:
            pass
        else:
            for i in range(fir.n_stims):
                tau[i*nparams + 1:(i+1)*nparams] = \
                        tau[i*nparams + 1:(i+1)*nparams].mean()
    y2 = fir._mean.dot(stim_lagged)

    return inst, fir, x_[0].T, y2, (tau, a), all_epochs, (lls, r), sfreq


def _fit_kernel_erp(noisy_data, sfreq, stims, lag_max, lag_min, a='auto', n_osc=2, ar_order=8,
             initial_guess=None, use_sparsifying_prior=True, **kwargs):
    """utility function

    Parameters
    ----------
    noisy_data: ndarray (1, T)  [Needs to be in mV, not in V.]
    sfreq: scalar
    stims: Sequences of ndarray (1, T) | np.ndarray (K, T)
    lag_max: end index of ERP
    lag_min: start index of erp (< 0 usually)
    a: scalar [0, 1] proportional to smoothness of ERP
    n_osc: int number of oscillators
    ar_order: int number of ar lags to consider for initialization
    initial_guess: (alpha, freq, sigma2) | None

    NOTE: n_osc and ar_order are ignored when initial_guess is not None.
    """
    nparams = lag_max - lag_min
    y = noisy_data.copy()
    all_epochs = []
    for events in stims:
        epochs = []
        for idx in np.nonzero(events)[0]:
            if idx+lag_max < noisy_data.shape[-1]:
                epochs.append(noisy_data[..., idx+lag_min:idx+lag_max])
        epochs = np.stack(epochs)
        all_epochs.append(epochs)

    window = kwargs.get("window", None)
    fir = KernelImpulseResponse(lag_max, lag_min)
    stim_lagged, rcorr = fir._setup_rcorr(stims, window=window)
    temp = np.linalg.solve(rcorr, (stim_lagged * y).sum(axis=-1))
    fir._mean[:] = np.squeeze(temp)
    y1 = y - fir._mean.dot(stim_lagged)
    import pdb; pdb.set_trace()
    if initial_guess is None:
        inst, C, r_orig = BaseOscModel.initialize_from_data(ar_order,
                                                            y1 - y1.mean(),
                                                            n_osc,
                                                            show_psd=True,
                                                            sfreq=sfreq,)
    else:
        inst = BaseOscModel(*initial_guess)
        n_osc = len(initial_guess[0])
        C = (np.arange(2 * n_osc) % 2)[::-1][None, :]
        r_orig = _observation_noise_by_hugo(y1 - y1.mean(axis=-1),
                                            sfreq, sample_average=None)
    # r_orig *= 4

    mixing_mat_inst = sensor_map.ScalarMixingMatrix.from_mat(C, None)
    alpha = None
    if a == 'auto':
        a = 1.
        update_a = True
    else:
        update_a = False

    lls = []
    tau = np.ones_like(temp)
    r = r_orig * np.eye(1)
    y2 = [y.copy()]
    ll = 0
    print(r_orig)   
    for j in range(100):
        # import pdb; pdb.set_trace()
        y1 = [this_y - fir._mean.dot(stim_lagged) for this_y in y]
        # y1 = y.copy()
        for kk in range(5):
            if kk != -1:
                x_, s_, b, ll_, _ = inst.get_hidden_posterior(
                                                y1, C, r,
                                                None, None,
                                                mixing_mat_inst=mixing_mat_inst)
                y2 = [this_y - C.dot(this_x.T) for this_y, this_x in zip(y, x_)]
            else:
                # _x, _s, ll_,  = inst.get_hidden_prior(y1[0][None, :], C, r)
                # y2 = [this_y - C.dot(this_x.T)[..., 1:-1] for this_y, this_x in zip(y, [_x])]
                pass

            ll = fir.update(stim_lagged, rcorr, r, y2[0], a, tau)
            ll += ll_
            # r = update_noise_var(y, C, x_, s_, stim_lagged, rcorr, fir)
            r = update_noise_var((y,), C, x_, s_, (stim_lagged,), rcorr, fir)
            # r = (9 * r + r_orig) / 10
            y1 = [this_y - fir._mean.dot(stim_lagged) for this_y in y]
        debug = False
        if debug:
            import matplotlib.pyplot as plt
            from mne.time_frequency import psd_array_multitaper
            xpsds, xfreqs = psd_array_multitaper(x_[0].T, sfreq,
                                               bandwidth=2.,
                                               normalization='length',
                                               n_jobs=4)
            ypsds, yfreqs = psd_array_multitaper(y[0], sfreq,
                                               bandwidth=2.,
                                               normalization='length',
                                               n_jobs=4)
            fig, ax = plt.subplots()
            ax.semilogy(xfreqs, xpsds.T)
            ax.semilogy(yfreqs, ypsds)
            # ax.plot(C.dot(x_[0].T).T, alpha = 0.2)
            # ax.plot(fir._mean.dot(stim_lagged))
            # import pdb; pdb.set_trace()

        lls.append(ll)
        print(ll)
        if j > 1 and np.abs((lls[-1] - lls[-2]) / lls[-2]) < 1e-8:
            print('Convergence reached!')
            break

        inst.update_params(x_, s_, b, update_sigma2=True)
        # r = update_noise_var((y,), C, x_, s_, (stim_lagged,), rcorr, fir)
        # r = (9 * r + r_orig) / 10

        # print('freq')
        # print(inst.freq * sfreq)
        # print('alpha')
        # print(inst.alpha)
        # print('sigma2')
        # print(inst.sigma2)
        a = fir.get_a_next(tau) if update_a else a
        tau = fir.get_tau_next(a)
        if use_sparsifying_prior:
            pass
        else:
            for i in range(fir.n_stims):
                tau[i*nparams + 1:(i+1)*nparams] = \
                        tau[i*nparams + 1:(i+1)*nparams].mean()
    y2 = fir._mean.dot(stim_lagged)

    return inst, fir, x_[0].T, y2, (tau, a), all_epochs, (lls, r), sfreq


def _fit_state_erps(noisy_data, sfreq, stims, n_lags, a='auto', n_osc=2, ar_order=8,
             initial_guess=None, use_sparsifying_prior=False, **kwargs):
    """utility function

    noisy_data: ndarray (1, T)
    sfreq: scalar
    stims: Sequences of ndarray (1, T) | np.ndarray (K, T)
    n_lags: length of ERP
    a: scalar [0, 1] proportional to smoothness of ERP
    n_osc: int number of oscillators
    ar_order: int number of ar lags to consider for initialization
    initial_guess: (alpha, freq, sigma2) | None

    NOTE: n_osc and ar_order are ignored when initial_guess is not None.
    """
    y = noisy_data.copy()

    # ERP initialization
    all_epochs = []
    for events in stims:
        epochs = []
        for idx in np.nonzero(events)[0]:
            if idx+n_lags < y.shape[-1]:
                epochs.append(noisy_data[:, idx:idx+n_lags])
        epochs = np.stack(epochs)
        all_epochs.append(epochs)
    fir = ImpulseResponse(n_lags)
    stim_lagged, rcorr = fir._setup_rcorr(stims)
    temp = np.linalg.solve(rcorr, (stim_lagged * y).sum(axis=-1))
    fir._mean[:] = np.squeeze(temp)
    tau = (np.diff(temp) ** 2).mean() * np.ones(n_lags * len(stims))
    if kwargs.get('debug', False):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(sum(epochs).T / len(epochs), label='average')
        ax.plot(temp, label='rev-corr')

    # Osc initialization
    y1 = y - fir._mean.dot(stim_lagged)
    if initial_guess is None:
        inst, C, r_orig = BaseOscModel.initialize_from_data(ar_order,
                                                            y1 - y1.mean(),
                                                            n_osc,
                                                            show_psd=True,
                                                            sfreq=sfreq,)
    else:
        inst = BaseOscModel(*initial_guess)
        n_osc = len(initial_guess[0])
        C = (np.arange(2 * n_osc) % 2)[::-1][None, :]
        r_orig = _observation_noise_by_hugo(y1 - y1.mean(axis=-1),
                                            sfreq, sample_average=None)
    r = r_orig.copy() * np.eye(1)

    fir.update(stim_lagged, rcorr, r, y, 1, tau)
    if kwargs.get('debug', False):
        ax.plot(fir._mean, label='post-mean')
        ax.legend()
        fig.savefig('debgug-1.png')

    # State ERP initialization
    x_, s_, b, ll_, _ = inst.get_hidden_posterior(y1, C, r, None, None,
                                                  mixing_mat_inst=None)
    state_firs = [ImpulseResponse(n_lags) for _ in range(2*n_osc)]
    state_fir_utils = [_fir._setup_rcorr(stims) for _fir in state_firs]
    diff_x = np.zeros_like(x_)
    diff_x[1:] = x_[1:] - x_[:-1].dot(inst.a.T)
    diff_x[0] = x_[0]
    [_fir.update(_stim_lagged, _rcorr, r, x, a, tau)
     for _fir, x, (_stim_lagged, _rcorr) in zip(state_firs, diff_x.T, state_fir_utils)]
    state_taus = np.vstack([_fir.get_tau_next(a) for _fir in state_firs])
    for nn in range(n_osc):
        for i in range(state_firs[0].n_stims):
            state_taus[2*nn: 2*(nn+1), i*state_firs[0].max_lag + 1:(i+1)*state_firs[0].max_lag] = \
                    state_taus[2*nn: 2*(nn+1), i*state_firs[0].max_lag + 1:(i+1)*state_firs[0].max_lag].mean()

    if a == 'auto':
        a = 1.
        update_a = True
    else:
        update_a = False

    lls = []
    ll = 0
    state_input = np.vstack([_fir._mean.dot(_stim_lagged) for _fir, (_stim_lagged, _)
                                in zip(state_firs, state_fir_utils)])
    for j in range(40):
        print(f"---{j}---")
        y2 = y.copy()
        y1 = y.copy()
        for kk in range(2):
            state_input = None
            for _ in range(10):
                x_, s_, b, ll2, _ = inst.get_hidden_posterior(y1, C, r, None, None,
                                                            mixing_mat_inst=None,
                                                            state_input=state_input)

                diff_x[1:] = x_[1:] - x_[:-1].dot(inst.a.T)
                diff_x[0] = x_[0]
                if j == 0 and kk == 0:
                    print(diff_x.shape)
                ll3 = [_fir.update(_stim_lagged, _rcorr, r, x, a, _tau)
                    for _fir, x, (_stim_lagged, _rcorr), _tau
                    in zip(state_firs, diff_x.T, state_fir_utils, state_taus)]
                state_input = np.vstack([_fir._mean.dot(_stim_lagged) for _fir, (_stim_lagged, _)
                                        in zip(state_firs, state_fir_utils)]) * 0
                print(ll2 + sum(ll3))

            y2 = y - C.dot(x_.T)
            ll1 = fir.update(stim_lagged, rcorr, r, y2, a, tau)
            y1 = y - fir._mean.dot(stim_lagged)
        print('--------')

        ll = ll1 + ll2 + sum(ll3)
        lls.append(ll)
        if j > 1 and np.abs((lls[-1] - lls[-2]) / lls[-2]) < 1e-6:
            print('Convergence reached!')
            break

        xx_ = x_ - state_input.T
        inst.update_params(xx_, s_, b, update_sigma2=True)
        if np.isnan(inst.a).any() or np.isinf(inst.a).any():
            raise ValueError('check inst.a')
        a = fir.get_a_next(tau) if update_a else a

        tau = fir.get_tau_next(a)
        if use_sparsifying_prior:
            pass
        else:
            for i in range(fir.n_stims):
                tau[i*fir.max_lag + 1:(i+1)*fir.max_lag] = \
                        tau[i*fir.max_lag + 1:(i+1)*fir.max_lag].mean()

        state_taus = np.vstack([_fir.get_tau_next(a) for _fir in state_firs])
        for nn in range(n_osc):
            for i in range(state_firs[0].n_stims):
                state_taus[2*nn: 2*(nn+1), i*state_firs[0].max_lag + 1:(i+1)*state_firs[0].max_lag] = \
                        state_taus[2*nn: 2*(nn+1), i*state_firs[0].max_lag + 1:(i+1)*state_firs[0].max_lag].mean()

        r = update_noise_var((y,), C, x_, s_, (stim_lagged,), rcorr, fir)
        r = (9 * r + r_orig) / 10

    y2 = fir._mean.dot(stim_lagged)

    return inst, (fir, state_firs), x_.T, y2, ((tau, state_taus), a), all_epochs, (lls, r), sfreq


def update_noise_var(y, mixing_mat, x, sigma_x, stim_lagged, rcorr, fir):
    # TODO: with the new ScalarMixingMatrix.
    lambdas = []
    for this_y, this_x, this_sigma_x, this_stim_lagged in zip(y, x, sigma_x, stim_lagged):
        n = this_y.shape[-1]
        res = this_y - fir._mean.dot(this_stim_lagged)
        res = res - mixing_mat.dot(this_x.T)
        lambdas.append((res * res).sum(axis=1) / n)

        if this_sigma_x.ndim == 3:
            this_sigma_x = this_sigma_x.mean(axis=0)
        e, v = linalg.eigh(this_sigma_x)
        e[e < 0] = 0
        mv = mixing_mat.dot(v)
        temp = ((mv * e[None, :]) * mv).sum(axis=1)
        lambdas[-1] = lambdas[-1] + temp
        temp = (np.sum(rcorr * fir._cov) / n)
        lambdas[-1] = lambdas[-1] + temp
    lambdas = sum(lambdas)
    # n = y.shape[-1]
    # res = [this_y - fir._mean.dot(stim_lagged) for this_y in y]
    # res = [this_res - mixing_mat.dot(this_x.T) for this_res, this_x in zip(res, x)]
    # lambdas = (res * res).sum(axis=1) / n

    # if sigma_x.ndim == 3:
    #     sigma_x = sigma_x.mean(axis=0)
    # e, v = linalg.eigh(sigma_x)
    # e[e < 0] = 0
    # mv = mixing_mat.dot(v)
    # temp = ((mv * e[None, :]) * mv).sum(axis=1)
    # lambdas = lambdas + temp
    # temp = (np.sum(rcorr * fir._cov) / n)
    # lambdas = lambdas + temp
    return np.diag(lambdas)


def confidence_interval(data, axis=-1, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a, axis=axis), stats.sem(a, axis=axis)
    # se /= np.sqrt(data.shape[axis])
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def test_sserp():
    import scipy.sparse as sparse
    from numpy.random import default_rng
    import matplotlib.pyplot as plt

    T = 2000
    rng = default_rng(seed=100)
    w = rng.standard_normal((T, 4))
    v = rng.standard_normal((1, T))
    u = np.empty_like(w)
    u[0] = rng.standard_normal(4)
    Phi1 = np.array([[np.cos(np.pi/50), np.sin(np.pi/50)], [-np.sin(np.pi/50), np.cos(np.pi/50)]])
    Phi2 = np.array([[np.cos(np.pi/6.25), np.sin(np.pi/6.25)], [-np.sin(np.pi/6.25), np.cos(np.pi/6.25)]])
    Phi = sparse.block_diag([0.99 * Phi1, 0.96 * Phi2]).toarray()
    for k in range(1, T):
        u[k] = np.dot(Phi, u[k-1]) + w[k-1]
    # x_noisy = x + x.std(axis=1)[:, None] * 0.1 * v
    f = np.array([[1.0, 0, 1.0, 0]], dtype=float)
    y = f.dot(u.T)

    y_noisy = y + y.std(axis=-1)[:, None] * v * 0.5

    stim = np.zeros(T)
    stim[50::100] = 1
    fir = ImpulseResponse(50)
    stim_lagged, rcorr = fir._setup_rcorr(stim)
    times = np.arange(50)/100
    h = (5 * np.sin(30 * times) *
            np.exp(- (times - 0.15) ** 2 / 0.01))
    evoked = h.dot(stim_lagged)
    evoked = evoked[None, :]
    eeg = y_noisy + evoked

    sserp = fit_sserp(eeg, [stim], 100, .50, .0, a=1, n_osc=2, ar_order=13,
                      use_sparsifying_prior=False, max_iter=100)
    # sserp = fit_erp(eeg, 100, [stim], .50, lag_min=0, a=1, n_osc=2,
    #                 ar_order=8)
    fir = ImpulseResponse(50)
    stim_lagged, rcorr = fir._setup_rcorr(stim)
    tau = 1*np.ones(50)
    for i in range(100):
        fir.update(stim_lagged, rcorr, 1, eeg, 1, tau)
        tau = fir.get_tau_next(1)
        tau[:] = tau.mean()

    fig, axes = plt.subplots(2, figsize=(7, 6))
    axes[0].plot(sserp._lls)
    sserp.viz_erp(ax=axes[1], sfreq=100, confidence=0.95)
    axes[1].plot(np.arange(50)/100, h.T, color='k', label='ground truth')
    axes[1].plot(np.arange(50)/100, fir._mean, label='Tikonov-erp', color='r')
    axes[1].legend(loc='upper right')
    plt.show()
