from time import time
from warnings import warn
import collections
import warnings

import numpy as np
from mne.epochs import BaseEpochs
from mne.io.base import BaseRaw
from mne.io.pick import _picks_to_idx, pick_info, pick_types, _picks_by_type
from mne.preprocessing.ica import _check_start_stop
from mne.utils.numerics import _PCA
from mne.utils import _validate_type, logger
from scipy import linalg
from copy import deepcopy

from .viz import (plot_oca_components, plot_oca_sources,
            plot_oca_properties, plot_oca_coh_spectra)

from ._temporal_dynamics_utils import (BaseOscModel, get_noise_cov_inverse)
from ._spatial_map_utils import sensor_map


def _fit_oca(y, ar_order, n_osc, max_iter=1000, initial_guess=None,
             sfreq=None, scalar_alpha=False, update_sigma2=True, tol=1e-6,
             noise_cov_prior=None, proj=None, ncomp=0):
    """Helper function to fit OCA to ndarrays"""
    if initial_guess is None:
        inst, C, r = BaseOscModel.initialize_from_data(ar_order, y, n_osc,
                                                       show_psd=False,
                                                       sfreq=sfreq,
                                                       proj=proj, proj_ncomp=ncomp)
        C_vars = None
        if C.shape[0] > 1:
            mixing_mat_inst = sensor_map.ClampedMixingMatrix.from_mat(C, C_vars)
            if scalar_alpha:
                alpha = C.size / (C * C).sum()
            else:
                _C = C.copy()
                _C.shape = (C.shape[0], -1, 2)
                alpha = (_C * _C).sum(axis=-1).sum(axis=0)
                alpha = np.repeat((2 * C.shape[0]) / alpha, 2)
        else:
            mixing_mat_inst = sensor_map.ScalarMixingMatrix.from_mat(C, None)
            alpha = None
        logger.debug(list(zip(inst.alpha, inst.freq)))
    else:
        inst, mixing_mat_inst, r, alpha = initial_guess
    
    initial_cond = 0.0
    if r.ndim < 2:
        r = r * np.eye(C.shape[0])
    lls = []
    dkls = []
    wlls = []
    improvements = []
    max_improvement = 1e-6
    neg_counter = 0
    if noise_cov_prior is None:
        noise_cov_prior = dict(data=r.copy(),
                            df = np.sum([this_y.shape[-1] for this_y in y]))
    # noise_cov_prior = None
    try:
        for j in range(max_iter):
            for _ in range(20):
                rinv, _ = get_noise_cov_inverse(r, proj, ncomp,)
                x_, s_, b, ll, initial_cond = inst.get_hidden_posterior(
                    y, None, r, None, initial_cond,
                    mixing_mat_inst=mixing_mat_inst, rinv=rinv,
                    proj=proj, ncomp=ncomp,)

                kld = mixing_mat_inst.update(y, x_, s_, alpha=alpha,
                                                lambdas=rinv)

                r, wll = sensor_map.update_noise_var(y, None, None, x_, s_, mixing_mat_inst,
                                                prior=noise_cov_prior)
            lls.append(ll)
            dkls.append(kld)
            wlls.append(wll)

            logger.debug(list(zip(inst.freq * sfreq, inst.alpha, inst.sigma2)))
            logger.debug(f'{lls[-1]} \t {dkls[-1]} \t {wlls[-1]} \t {lls[-1] - dkls[-1]+ wlls[-1]}')
            if j > 1:
                ll_diff = lls[-1] - lls[-2] + wlls[-1] - wlls[-2]
                dkl_diff = dkls[-1] - dkls[-2]
                improvements.append(ll_diff - dkl_diff)
                max_improvement = max(max_improvement, improvements[-1])
                if improvements[-1] < 0:
                    neg_counter += 1
                r_r = (np.abs(improvements[-1]) /
                    (max(np.abs(ll_diff), np.abs(dkl_diff)) + 1e-6))
                r_n = abs(improvements[-1]) / max_improvement
                if r_r < tol or r_n < tol:
                    # dkls.append(np.nan)
                    break
            
            inst.update_params(x_, s_, b, update_sigma2=update_sigma2)
            alpha = mixing_mat_inst.next_hyperparameter(scalar_alpha)

            # Stability check
            if inst.sigma2.max() > 1e15:
                raise RuntimeError(f'Computation diverged, i.e. possible overflow in sigma2.')
    except Exception as e:
        raise e
    return (inst, mixing_mat_inst, r, alpha, np.array(lls) + np.array(wlls),
            np.array(dkls))


def _exp_var_ncomp(var, n):
    "Copied from MNE"
    cvar = np.asarray(var, dtype=np.float64)
    cvar = cvar.cumsum()
    cvar /= cvar[-1]
    # We allow 1., which would give us N+1
    n = min((cvar <= n).sum() + 1, len(cvar))
    return n, cvar[n - 1]


def make_projector(info, projs=None, ch_names=None):
    """Create the projection operator
    Author: Proloy Das <pd640@mgh.harvard.edu>
    
    Input:
    -----
    info: mne.Info
    projs : list | None (default)
        List of projection vectors.
    ch_names: list of str | None (default)
        List of channels to include in the projection matrix.
    Output:
    ------
    """
    from mne.io.pick import pick_info
    from mne.io.proj import make_projector
    from mne.utils import logger

    projs = info['projs'] if projs is None else projs
    ch_names = info['ch_names'] if ch_names is None else ch_names
    if info['ch_names'] != ch_names:
        info = pick_info(info, [info['ch_names'].index(c) for c in ch_names])
    assert info['ch_names'] == ch_names

    proj, ncomp, _ = make_projector(projs, ch_names)
    if ncomp > 0:
        logger.info('    Created an SSP operator (subspace dimension = %d)'
                    % ncomp)
    return proj, ncomp


class BaseOCA:
    def __init__(self, n_oscillations, n_pca_components=None, pca_whiten=True,*, noise_cov=None,
                fit_params=None, max_iter='auto', allow_ref_meg=False,
                verbose=None):
        _validate_type(n_pca_components, (float, 'int-like', None))
        
        self.noise_cov = noise_cov

        for (kind, val) in [('n_oscillations', n_oscillations)]:
            if isinstance(val, float) and not 0 < val < 1:
                raise ValueError('Selecting OCA components by explained '
                                 'variance needs values between 0.0 and 1.0 '
                                 f'(exclusive), got {kind}={val}')
            if isinstance(val, int) and val == 1:
                warn(f'Selecting one component with {kind}={val}: use '
                    'n_component=1.0 for using all the pca components')
        self.current_fit = 'unfitted'
        self._max_pca_components = None
        self.n_pca_components = n_pca_components
        self.pca_whiten = pca_whiten
        self.ch_names = None

        if fit_params is None:
            fit_params = {}
        fit_params = deepcopy(fit_params)  # avoid side effects
        _validate_type(max_iter, (str, 'int-like'), 'max_iter')
        if isinstance(max_iter, str):
            if max_iter != 'auto':
                raise ValueError("max_iter can only be 'auto' when str")
            max_iter = 500
        fit_params.setdefault('max_iter', max_iter)
        self.max_iter = max_iter
        fit_params.setdefault('n_oscillations', n_oscillations)   # fit_params['n_oscillations'] priority             
        self.fit_params = fit_params

        self.info = None
        self.labels_ = dict()

    def fit(self, inst, picks=None, start=None, stop=None, decim=None,
            reject=None, flat=None, tstep=2.0, reject_by_annotation=True,
            verbose=None):
        """Run the OCA decomposition on raw data.
        Parameters
        ----------
        inst : instance of Raw or Epochs
            The data to be decomposed.
        %(picks_good_data_noref)s
            This selection remains throughout the initialized OCA solution.
        start, stop : int | float | None
            First and last sample to include. If float, data will be
            interpreted as time in seconds. If ``None``, data will be used from
            the first sample and to the last sample, respectively.
            .. note:: These parameters only have an effect if ``inst`` is
                      `~mne.io.Raw` data.
        decim : int | None
            Increment for selecting only each n-th sampling point. If ``None``,
            all samples  between ``start`` and ``stop`` (inclusive) are used.
        reject, flat : dict | None
            Rejection parameters based on peak-to-peak amplitude (PTP)
            in the continuous data. Signal periods exceeding the thresholds
            in ``reject`` or less than the thresholds in ``flat`` will be
            removed before fitting the OCA.
            .. note:: These parameters only have an effect if ``inst`` is
                      `~mne.io.Raw` data. For `~mne.Epochs`, perform PTP
                      rejection via :meth:`~mne.Epochs.drop_bad`.
            Valid keys are all channel types present in the data. Values must
            be integers or floats.
            If ``None``, no PTP-based rejection will be performed. Example::
                reject = dict(
                    grad=4000e-13, # T / m (gradiometers)
                    mag=4e-12, # T (magnetometers)
                    eeg=40e-6, # V (EEG channels)
                    eog=250e-6 # V (EOG channels)
                )
                flat = None  # no rejection based on flatness
        tstep : float
            Length of data chunks for artifact rejection in seconds.
            .. note:: This parameter only has an effect if ``inst`` is
                      `~mne.io.Raw` data.
        %(reject_by_annotation_raw)s
            .. versionadded:: 0.14.0
        %(verbose)s
        Returns
        -------
        self : instance of OCA
            Returns the modified instance.
        """
        _validate_type(inst, (BaseRaw, BaseEpochs), 'inst', 'Raw or Epochs')

        if np.isclose(inst.info['highpass'], 0.):
            warn('The data has not been high-pass filtered. For good OCA '
                 'performance, it should be high-pass filtered (e.g., with a '
                 '1.0 Hz lower bound) before fitting OCA.')

        if isinstance(inst, BaseEpochs) and inst.baseline is not None:
            warn('The epochs you passed to OCA.fit() were baseline-corrected. '
                 'However, we suggest to fit OCA only on data that has been '
                 'high-pass filtered, but NOT baseline-corrected.')

        if not isinstance(inst, BaseRaw):
            ignored_params = [
                param_name for param_name, param_val in zip(
                    ('start', 'stop', 'reject', 'flat'),
                    (start, stop, reject, flat)
                )
                if param_val is not None
            ]
            if ignored_params:
                warn(f'The following parameters passed to OCA.fit() will be '
                     f'ignored, as they only affect raw data (and it appears '
                     f'you passed epochs): {", ".join(ignored_params)}')

        picks = _picks_to_idx(inst.info, picks, allow_empty=False,)

        # Actually start fitting
        t_start = time()
        if self.current_fit != 'unfitted':
            self._reset()

        logger.info('Fitting OCA to data using %i channels '
                    '(please be patient, this may take a while)' % len(picks))

        # filter out all the channels the raw wouldn't have initialized
        self.info = pick_info(inst.info, picks)

        if self.info['comps']:
            with self.info._unlock():
                self.info['comps'] = []
        self.ch_names = self.info['ch_names']
 
        if isinstance(inst, BaseRaw):
            self._fit_raw(inst, picks, start, stop, decim, reject, flat,
                          tstep, reject_by_annotation, verbose)
        else:
            assert isinstance(inst, BaseEpochs)
            self._fit_epochs(inst, picks, decim, verbose)

        # sort OCA components by explained variance
        var = oca_explained_variance(self, inst)
        var_ord = var.argsort()[::-1]
        self._sort_components(var_ord)
        t_stop = time()
        logger.info("Fitting OCA took {:.1f}s.".format(t_stop - t_start))
        return self

    def _reset(self):
        """Aux method."""
        for key in ('pre_whitener_', 'post_colorer_',
                    'pca_whiten', 'n_pca_components_', 'n_samples_', 'pca_components_',
                    'pca_explained_variance_',
                    'pca_mean_', 'n_iter_',):
            if hasattr(self, key):
                delattr(self, key)

    def _fit_raw(self, raw, picks, start, stop, decim, reject, flat, tstep,
                 reject_by_annotation, verbose):
        """Aux method."""
        start, stop = _check_start_stop(raw, start, stop)

        reject_by_annotation = 'omit' if reject_by_annotation else None
        # this will be a copy
        data = raw.get_data(picks, start, stop, reject_by_annotation)

        # this will be a view
        if decim is not None:
            data = data[:, ::decim]

        # this will make a copy
        if (reject is not None) or (flat is not None):
            from mne.utils.numerics import _reject_data_segments
            self.reject_ = reject
            data, self.drop_inds_ = _reject_data_segments(data, reject, flat,
                                                          decim, self.info,
                                                          tstep)

        self._fit([data], 'raw')

        return self

    def _fit_epochs(self, epochs, picks, decim, verbose):
        """Aux method."""
        if epochs.events.size == 0:
            raise RuntimeError('Tried to fit OCA with epochs, but none were '
                               'found: epochs.events is "{}".'
                               .format(epochs.events))

        # this should be a copy (picks a list of int)
        data = epochs.get_data()[:, picks]
        # this will be a view
        if decim is not None:
            data = data[:, :, ::decim]

        self._fit(data, 'epochs')

        return self

    def _compute_pre_whitener(self, data):
        """Aux function."""
        from mne.cov import prepare_noise_cov, compute_whitener
        data = self._do_proj(data)

        info = self.info
        if self.noise_cov is None:
            # use standardization as whitener
            # Scale (z-score) the data by channel type
            pre_whitener = np.empty([len(data), 1])
            post_colorer = np.empty([len(data), 1])
            for _, picks_ in _picks_by_type(info, ref_meg=False, exclude=[]):
                pre_whitener[picks_] = np.std(data[picks_])
                post_colorer[picks_] = 1 / np.std(data[picks_]) 
        else:
            noise_cov = prepare_noise_cov(self.noise_cov, info,)
            pre_whitener, _, post_colorer = compute_whitener(
                noise_cov, info, picks=info.ch_names,
                pca=True, return_colorer=True, verbose=None)
            assert data.shape[0] == pre_whitener.shape[1]
        self.pre_whitener_ = pre_whitener
        self.post_colorer_ = post_colorer

    def _do_proj(self, data):
        from mne.io.proj import make_projector
        if self.info is not None and self.info['projs']:
            proj, nproj, _ = make_projector(
                [p for p in self.info['projs'] if p['active']],
                self.info['ch_names'], include_active=True)
            if nproj:
                logger.info(
                    f'    Applying projection operator with {nproj} vectors'
                    )
                if self.noise_cov is None:  # otherwise it's in pre_whitener_
                    data = proj @ data
        return data

    def _pre_whiten(self, data):
        data = self._do_proj(data)
        if self.noise_cov is None:
            data /= self.pre_whitener_
        else:
            data = self.pre_whitener_ @ data
        return data

    def _post_color(self, data):
        "Note that _post_color cannot undo the projections"
        if self.noise_cov is None:
            data /= self.post_colorer_
        else:
            data = self.post_colorer_ @ data
        return data

    def _compute_pca(self, data):
        pca = _PCA(n_components=None, whiten=self.pca_whiten)
        data = pca.fit_transform(data.T)
        use_ev = pca.explained_variance_ratio_
        
        # If user passed a float, select the PCA components explaining the
        # given cumulative variance. This information will later be used to
        # only submit the corresponding parts of the data to OCA.
        if isinstance(self.n_pca_components, float):
            self.n_pca_components_, ev = _exp_var_ncomp(use_ev, self.n_pca_components)
            if self.n_pca_components_ == 1:
                raise RuntimeError(
                    'One PCA component captures most of the '
                    f'explained variance ({100 * ev}%), your threshold '
                    'results in 1 component. You should select '
                    'a higher value.')
            msg = 'Selecting by explained variance'
        elif self.n_pca_components is None:
            # Selecting by non-zero PCA components
            self.n_pca_components_ = _exp_var_ncomp(use_ev, 0.999999)[0]
            msg = 'Selecting by non-zero PCA components'
        else:
            import operator
            msg = 'Selecting by number'
            self.n_pca_components_ = operator.index(self.n_pca_components)
        assert isinstance(self.n_pca_components_, (int, np.int_))
        logger.info('%s: %s components' % (msg, self.n_pca_components_))

        # the things to store for PCA
        self.pca_whiten_ = pca.whiten
        self.pca_mean_ = pca.mean_
        self.pca_components_ = pca.components_
        self.pca_explained_variance_ = pca.explained_variance_
        del pca
        return data.T

    def _fit(self, data, fit_type):
        raise NotImplementedError

    def _get_sorting_order(self, data=None):
        if data is None:
            return self._oscillators_._get_sorted_idx()
        else:
            source_powers = self._osc_explained_variance(data)
        return np.argsort(source_powers)[::-1]

    def _osc_explained_variance(self, data):
        sources = self._transform(data)
        sources = np.hstack(sources)
        source_powers = np.sum(sources ** 2, axis=-1)
        mixing_mat_ = self.get_components()
        source_powers *= (mixing_mat_ ** 2).sum(axis=0)
        return np.reshape(source_powers, (-1, 2)).sum(axis=-1)
    
    def _update_oca_names(self):
        """Update OCA names when n_components_ is set."""
        self._osc_names = self._oscillators_._osc_names

    def _get_spatial_filters(self,):
        from mne.cov import prepare_noise_cov, compute_whitener
        from mne.fixes import _safe_svd
        noise_cov = self.get_fitted_noise_cov()
        info = self.info
        noise_cov = prepare_noise_cov(noise_cov, info,)
        pre_whitener, _ = compute_whitener(
            noise_cov, info, picks=info.ch_names,
            pca=False, return_colorer=False, verbose=None)
        mm = self.get_components()
        wmm = pre_whitener @ mm
        blocks = [np.inner(wmm.T[2*i:2*(i+1)], wmm.T[2*i:2*(i+1)]) for i in range(self.n_oscillations)]
        blocks = linalg.block_diag(*blocks)
        import ipdb; ipdb.set_trace()
        # u, s, vh = _safe_svd(wmm, full_matrices=False)
        # sinv = 1 / s
        # sinv[np.isnan(sinv)] = 0.
        # spatial_filters = (vh.T * sinv[None, :]) @ u.T @ pre_whitener
        spatial_filters = linalg.inv(blocks) @ wmm.T @ pre_whitener
        return spatial_filters

    def _apply_spatial_filter(self, data):
        filters = self._get_spatial_filters()
        with DataManager(data) as f:
            f.apply(filters.dot)
        sources = f.data
        return np.stack(sources)

    def _spatial_filter_raw(self, raw, start, stop, reject_by_annotation=False):
        """Transform raw data."""
        if not hasattr(self, '_mixing_matrix_inst_'):
            raise RuntimeError('No fit available. Please fit OCA.')
        start, stop = _check_start_stop(raw, start, stop)
        picks = self._get_picks(raw)
        reject = 'omit' if reject_by_annotation else None
        data = [raw.get_data(picks, start, stop, reject)]
        return self._apply_spatial_filter(data)[0]

    def _spatial_filter_epochs(self, epochs):
        """Aux method."""
        if not hasattr(self, '_mixing_matrix_inst_'):
            raise RuntimeError('No fit available. Please fit OCA.')
        picks = self._get_picks(epochs)
        data = epochs.get_data()[:, picks]
        sources = self._apply_spatial_filter(data)
        return sources

    def get_network_activities(self, inst, start=None, stop=None):
        """Estimate sources given the unmixing matrix.
        This method will return the sources in the container format passed.
        Typical usecases:
        1. pass Raw object to use `raw.plot <mne.io.Raw.plot>` for OCA sources
        2. pass Epochs object to compute trial-based statistics in OCA space
        3. pass Evoked object to investigate time-locking in OCA space
        Parameters
        ----------
        inst : instance of Raw, Epochs or Evoked
            Object to compute sources from and to represent sources in.
        add_channels : None | list of str
            Additional channels  to be added. Useful to e.g. compare sources
            with some reference. Defaults to None.
        start : int | float | None
            First sample to include. If float, data will be interpreted as
            time in seconds. If None, the entire data will be used.
        stop : int | float | None
            Last sample to not include. If float, data will be interpreted as
            time in seconds. If None, the entire data will be used.
        Returns
        -------
        sources : instance of Raw, Epochs or Evoked
            The OCA sources time series.
        """
        if isinstance(inst, BaseRaw):
            sources = self._sources_as_raw(inst, start, stop, method='spatial_filter')
        elif isinstance(inst, BaseEpochs):
            sources = self._sources_as_epochs(inst, method='spatial_filter')
        else:
            raise ValueError('Data input must be of Raw, or Epochs'
                             'type')
        return sources

    def _transform(self, data):
        """Compute sources from data (operates inplace)."""
        with DataManager(data) as f:
            # data = self._pre_whiten(data)
            f.apply(self._pre_whiten)
            if self.pca_mean_ is not None:
                f.data -= self.pca_mean_[:, None]  # inplace operations are fine

            # Apply unmixing
            pca_components_ = self.pca_components_[:self.n_pca_components_]
            if self.pca_whiten_:
                norms = self.pca_explained_variance_[:self.n_pca_components_]
                norms = np.sqrt(norms)[:, np.newaxis]
                norms[norms == 0] = 1.
                pca_components_ = pca_components_ / norms
            # pca_data = np.dot(pca_components_, data)
            f.apply(lambda x: np.dot(pca_components_, x))
        pca_data = f.data
        # Apply OCA
        r = self._noise_var_fit_
        sources, *rest = self._oscillators_.get_hidden_posterior(pca_data, None, r, None, 0.0, self._mixing_matrix_inst_)
        sources = np.stack([sources_.T for sources_ in sources], axis=0)
        return sources

    def _transform_raw(self, raw, start, stop, reject_by_annotation=False):
        """Transform raw data."""
        if not hasattr(self, '_mixing_matrix_inst_'):
            raise RuntimeError('No fit available. Please fit OCA.')
        start, stop = _check_start_stop(raw, start, stop)
        picks = self._get_picks(raw)
        reject = 'omit' if reject_by_annotation else None
        data = [raw.get_data(picks, start, stop, reject)]
        return self._transform(data)[0]

    def _transform_epochs(self, epochs):
        """Aux method."""
        if not hasattr(self, '_mixing_matrix_inst_'):
            raise RuntimeError('No fit available. Please fit OCA.')
        picks = self._get_picks(epochs)
        data = epochs.get_data()[:, picks]
        sources = self._transform(data)
        return sources

    def _get_picks(self, inst):
        """Pick logic for _transform method."""
        picks = _picks_to_idx(
            inst.info, self.ch_names, exclude=[], allow_empty=True)
        if len(picks) != len(self.ch_names):
            if isinstance(inst, BaseRaw):
                kind, do = 'Raw', "doesn't"
            elif isinstance(inst, BaseEpochs):
                kind, do = 'Epochs', "don't"
            else:
                raise ValueError('Data input must be of Raw, Epochs or Evoked '
                                 'type')
            raise RuntimeError("%s %s match fitted data: %i channels "
                               "fitted but %i channels supplied. \nPlease "
                               "provide %s compatible with oca.ch_names"
                               % (kind, do, len(self.ch_names), len(picks),
                                  kind))
        return picks

    def get_sources(self, inst, start=None, stop=None):
        """Estimate sources given the unmixing matrix.
        This method will return the sources in the container format passed.
        Typical usecases:
        1. pass Raw object to use `raw.plot <mne.io.Raw.plot>` for OCA sources
        2. pass Epochs object to compute trial-based statistics in OCA space
        3. pass Evoked object to investigate time-locking in OCA space
        Parameters
        ----------
        inst : instance of Raw, Epochs or Evoked
            Object to compute sources from and to represent sources in.
        add_channels : None | list of str
            Additional channels  to be added. Useful to e.g. compare sources
            with some reference. Defaults to None.
        start : int | float | None
            First sample to include. If float, data will be interpreted as
            time in seconds. If None, the entire data will be used.
        stop : int | float | None
            Last sample to not include. If float, data will be interpreted as
            time in seconds. If None, the entire data will be used.
        Returns
        -------
        sources : instance of Raw, Epochs or Evoked
            The OCA sources time series.
        """
        if isinstance(inst, BaseRaw):
            sources = self._sources_as_raw(inst, start, stop, method='transform')
        elif isinstance(inst, BaseEpochs):
            sources = self._sources_as_epochs(inst, method='transform')
        else:
            raise ValueError('Data input must be of Raw, or Epochs'
                             'type')
        return sources

    def _sources_as_raw(self, raw, start, stop, method='transform'):
        """Aux method."""
        # merge copied instance and picked data with sources
        start, stop = _check_start_stop(raw, start, stop)
        # data_ = self._transform_raw(raw, start=start, stop=stop)
        method = getattr(self, f'_{method}_raw')
        data_ = method(raw, start, stop)
        assert data_.shape[-1] == stop - start

        preloaded = raw.preload
        if raw.preload:
            # get data and temporarily delete
            data = raw._data
            raw.preload = False
            del raw._data
        # copy and crop here so that things like annotations are adjusted
        try:
            out = raw.copy().crop(
                start / raw.info['sfreq'],
                (stop - 1) / raw.info['sfreq'])
        finally:
            # put the data back (always)
            if preloaded:
                raw.preload = True
                raw._data = data

        # populate copied raw.
        out._data = data_
        out._first_samps = [out.first_samp]
        out._last_samps = [out.last_samp]
        out._filenames = [None]
        out.preload = True
        out._projector = None
        self._export_info(out.info, raw)

        return out
    
    def _sources_as_epochs(self, epochs, method='transform'):
        """Aux method."""
        out = epochs.copy()
        # sources = self._transform_epochs(epochs)
        method = getattr(self, f'_{method}_epochs')
        sources = method(epochs)
        out._data = sources
        self._export_info(out.info, epochs)
        out.preload = True
        out._raw = None
        out._projector = None

        return out

    def _export_info(self, info, container):
        """Aux method."""
        from mne.io.constants import FIFF
        # set channel names and info
        ch_names = []
        ch_info = []
        for ii, name in enumerate(self._osc_names):
            ch_names.append(name)
            ch_info.append(dict(
                ch_name=name, cal=1, logno=ii + 1,
                coil_type=FIFF.FIFFV_COIL_NONE,
                kind=FIFF.FIFFV_MISC_CH,
                coord_frame=FIFF.FIFFV_COORD_UNKNOWN,
                unit=FIFF.FIFF_UNIT_NONE,
                loc=np.zeros(12, dtype='f4'),
                range=1.0, scanno=ii + 1, unit_mul=0))

        with info._unlock(update_redundant=True, check_after=True):
            info['chs'] = ch_info
            info['projs'] = []  # make sure projections are removed.
            info['bads'] = []  # make sure bad channels are removed.

    def _transform_to_sensorspace(self,):
        pca_components_ = self.pca_components_
        if self.pca_whiten_:
            norms = self.pca_explained_variance_
            norms = np.sqrt(norms)
            norms[norms == 0] = 1.
            pca_components_ = pca_components_ * norms[:, np.newaxis]
        pca_data = self._post_color(pca_components_.T)
        return pca_data

    def get_frequencies(self):
        return np.abs(self._oscillators_.freq) * self.info['sfreq']

    def get_components(self):
        """Get OCA topomap for components as numpy arrays.
        Returns
        -------
        components : array, shape (n_channels, n_components)
            The OCA components (maps).
        """
        pca_data = self._transform_to_sensorspace()
        # pca_components_ = np.sqrt(self.pca_explained_variance_[:, np.newaxis]) * self.pca_components_
        # pca_data = self._post_color(pca_components_.T)
        mixing_matrix_, _ = self._mixing_matrix_inst_._get_vals()
        return np.dot(pca_data[:, :self.n_pca_components_], mixing_matrix_)

    def get_fitted_noise_cov(self):
        """Get residual noise covariance.
        Returns
        -------
        mne.Covariance : shape (n_channels, n_channels)
            The fitted noise covariance (Cov).
        """
        from scipy import sparse
        from mne import Covariance
        pca_data = self._transform_to_sensorspace()

        lower_diag = np.ones(len(self.pca_explained_variance_) - self.n_pca_components_) \
                    if self.pca_whiten else self.pca_explained_variance_[self.n_pca_components_:]

        noise_cov = sparse.block_diag((self._noise_var_fit_,
                                    np.diag(lower_diag))).toarray()
        n_samples_ = getattr(self, 'n_samples_', None)
        if n_samples_ is not None:
            nfree = n_samples_ - self.n_oscillations * (3 + len(self.info['ch_names']))
        noise_cov = pca_data @ noise_cov @ pca_data.T
        return Covariance(noise_cov, self.info.ch_names, 
                          bads=[], projs=[], nfree=nfree, 
                          eig=None, eigvec=None, method='custom',
                          loglik=None, verbose=None)

    def apply(self, inst, include=None, exclude=None, n_pca_components=None,
              start=None, stop=None, verbose=None):
        """Remove selected components from the signal.
        Given the unmixing matrix, transform the data,
        zero out all excluded components, and inverse-transform the data.
        This procedure will reconstruct M/EEG signals from which
        the dynamics described by the excluded components is subtracted.
        Parameters
        ----------
        inst : instance of Raw, Epochs or Evoked
            The data to be processed (i.e., cleaned). It will be modified
            in-place.
        include : array_like of int
            The indices referring to columns in the ummixing matrix. The
            components to be kept. If ``None`` (default), all components
            will be included (minus those defined in ``oca.exclude``
            and the ``exclude`` parameter, see below).
        exclude : array_like of int
            The indices referring to columns in the ummixing matrix. The
            components to be zeroed out. If ``None`` (default) or an
            empty list, only components from ``oca.exclude`` will be
            excluded. Else, the union of ``exclude`` and ``oca.exclude``
            will be excluded.
        %(n_pca_components_apply)s
        start : int | float | None
            First sample to include. If float, data will be interpreted as
            time in seconds. If None, data will be used from the first sample.
        stop : int | float | None
            Last sample to not include. If float, data will be interpreted as
            time in seconds. If None, data will be used to the last sample.
        %(verbose)s
        Returns
        -------
        out : instance of Raw, Epochs or Evoked
            The processed data.
        Notes
        -----
        .. note:: Applying OCA may introduce a DC shift. If you pass
                  baseline-corrected `~mne.Epochs` or `~mne.Evoked` data,
                  the baseline period of the cleaned data may not be of
                  zero mean anymore. If you require baseline-corrected
                  data, apply baseline correction again after cleaning
                  via OCA. A warning will be emitted to remind you of this
                  fact if you pass baseline-corrected data.
        """
        kwargs = dict(include=include, exclude=exclude,
                      n_pca_components=n_pca_components)
        if isinstance(inst, BaseRaw):
            kind, meth = 'Raw', self._apply_raw
            kwargs.update(raw=inst, start=start, stop=stop)
        elif isinstance(inst, BaseEpochs):
            kind, meth = 'Epochs', self._apply_epochs
            kwargs.update(epochs=inst)
        else:  # isinstance(inst, Evoked):
            raise ValueError(f'{type(inst)} not supported.')

        if isinstance(inst, (BaseEpochs)):
            if getattr(inst, 'baseline', None) is not None:
                warn('The data you passed to OCA.apply() was '
                     'baseline-corrected. Please note that OCA can introduce '
                     'DC shifts, therefore you may wish to consider '
                     'baseline-correcting the cleaned data again.')

        logger.info(f'Applying OCA to {kind} instance')
        return meth(**kwargs)

    def _apply_raw(self, raw, include, exclude, n_pca_components, start, stop):
        """Aux method."""
        if not raw.preload:
            raise RuntimeError(
                "By default, MNE does not load data into main memory to "
                "conserve resources. oca.apply requires raw data to be"
                'loaded. Use preload=True (or string) in the constructor or '
                'raw.load_data().' )

        start, stop = _check_start_stop(raw, start, stop)

        picks = pick_types(raw.info, meg=False, include=self.ch_names,
                           exclude='bads', ref_meg=False)

        data = raw[picks, start:stop][0]
        data = self._pick_sources(data, include, exclude, n_pca_components)

        raw = raw.copy()
        raw[picks, start:stop] = data[0]
        return raw

    def _apply_epochs(self, epochs, include, exclude, n_pca_components):
        """Aux method."""
        if not epochs.preload:
            raise RuntimeError(
                "By default, MNE does not load data into main memory to "
                "conserve resources. oca.apply requires epochs data to be"
                'loaded. Use preload=True (or string) in the constructor or '
                'epochs.load_data().' )

        picks = pick_types(epochs.info, meg=False, ref_meg=False,
                           include=self.ch_names,
                           exclude='bads')

        # special case where epochs come picked but fit was 'unpicked'.
        if len(picks) != len(self.ch_names):
            raise RuntimeError('Epochs don\'t match fitted data: %i channels '
                               'fitted but %i channels supplied. \nPlease '
                               'provide Epochs compatible with '
                               'oca.ch_names' % (len(self.ch_names),
                                                 len(picks)))

        data = epochs.get_data(picks)
        data = self._pick_sources(data, include, exclude, n_pca_components)

        # restore epochs, channels, tsl order
        epochs = epochs.copy()
        epochs._data[:, picks] = np.stack(data, axis=0)
        epochs.preload = True
        return epochs

    def _pick_sources(self, data, include, exclude, n_pca_components):
        """Aux function."""
        sources = self._transform(data)
        mixing, _ = self._mixing_matrix_inst_._get_vals()

        sel_keep = np.arange(self.n_oscillations)
        if include not in (None, []):
            sel_keep = np.unique(include)
        elif exclude not in (None, []):
            sel_keep = np.setdiff1d(np.arange(self.n_oscillations), exclude)

        n_zero = self.n_oscillations - len(sel_keep)
        logger.info(f'    Zeroing out {n_zero} OCA component' + 
                    's' if n_zero else '')

        sel_keep = self._oscillators_._expand_indices(sel_keep)
        data = mixing[:, sel_keep] @ sources[:, sel_keep] 

        logger.info(f'    Projecting back using {self.n_pca_components_} '
                    f'PCA components')
        pca_components_ = self.pca_components_
        if self.pca_whiten_:
            norms = self.pca_explained_variance_
            norms = np.sqrt(norms)
            norms[norms == 0] = 1. 
            pca_components_ = pca_components_ * norms[:, np.newaxis]

        with DataManager(data) as f:
            # data = pca_components_[:self.n_pca_components_].T @ data
            f.apply(lambda x: pca_components_[:self.n_pca_components_].T @ x)

            if self.pca_mean_ is not None:
                f.data += self.pca_mean_[:, None]

            # restore scaling
            # data = self._post_color(data)
            f.apply(self._post_color)
        data = f.data

        return data

    def _sort_components(self, order):
        """Change the order of components in oca solution."""
        assert self.n_oscillations == len(order)
        # reorder components
        self._oscillators_._do_arange(order)
        order = self._oscillators_._expand_indices(order)
        self._mixing_matrix_inst_._do_arrange(order)
    
    def _get_global_coh_spectra(self, inst=None, start=None, stop=None,
                                reject_by_annotation=False):
        if inst is None:
            sigma2 = self._oscillators_.sigma2
            alpha = self._oscillators_.alpha
            source_powers = np.repeat(sigma2 / np.abs(1 - alpha**2), 2)
            mixing_mat_ = self.get_components()
            source_powers *= (mixing_mat_ ** 2).sum(axis=0)
            spectra = np.reshape(source_powers, (-1, 2)).sum(axis=-1)
        else:
            spectra = oca_explained_variance(self, inst, normalize=False)
        return spectra

    def copy(self):
        """Copy the OCA object.
        Returns
        -------
        oca : instance of OCA
            The copied object.
        """
        return deepcopy(self)

    def plot_components(self, picks=None, ch_type=None, res=64,
                        vmin=None, vmax=None, cmap='interactive', sensors=True,
                        colorbar=False, title=None, show=True, outlines='head',
                        contours=6, image_interp='cubic', plot_phase=False,
                        mapscale='2-norm', inst=None, plot_std=True,
                        topomap_args=None, image_args=None, psd_args=None,
                        reject='auto', sphere=None, verbose=None):
        return plot_oca_components(self, picks=picks, ch_type=ch_type,
                                   res=res, vmin=vmin,
                                   vmax=vmax, cmap=cmap, sensors=sensors,
                                   colorbar=colorbar, title=title, show=show,
                                   outlines=outlines, contours=contours,
                                   image_interp=image_interp, plot_phase=plot_phase,
                                   mapscale=mapscale, inst=inst, plot_std=plot_std,
                                   topomap_args=topomap_args,
                                   image_args=image_args, psd_args=psd_args,
                                   reject=reject, sphere=sphere,
                                   verbose=verbose)


    def plot_properties(self, inst, picks=None, axes=None, dB=True,
                        plot_std=True, log_scale=False, topomap_args=None,
                        image_args=None, psd_args=None, figsize=None,
                        show=True, *, verbose=None):
        return plot_oca_properties(self, inst, picks=picks, axes=axes, dB=dB,
                        plot_std=plot_std, log_scale=log_scale, topomap_args=topomap_args,
                        image_args=image_args, psd_args=psd_args, figsize=figsize,
                        show=show, verbose=None)

    def plot_sources(self, inst, picks=None, start=None,
                     stop=None, title=None, show=True, block=False,
                     show_first_samp=False, show_scrollbars=True,
                     time_format='float', precompute=None,
                     use_opengl=None, *, theme=None, overview_mode=None):
        return plot_oca_sources(self, inst, picks=picks, start=start,
                                stop=stop, title=title, show=show, block=block,
                                show_first_samp=show_first_samp,
                                show_scrollbars=show_scrollbars,
                                time_format=time_format, precompute=precompute,
                                use_opengl=use_opengl, theme=theme,
                                overview_mode=overview_mode)
    
    def plot_global_coh_spectra(oca, inst=None, start=None, stop=None,
                                reject_by_annotation=False, ax=None, bands=None):
        return plot_oca_coh_spectra(oca, inst=inst, start=start, stop=stop,
                                    reject_by_annotation=reject_by_annotation, ax=ax, bands=bands)

    def _check_n_pca_components(self, _n_pca_comp, verbose=None):
        """Aux function."""
        if isinstance(_n_pca_comp, float):
            n, ev = _exp_var_ncomp(
                self.pca_explained_variance_, _n_pca_comp)
            logger.info(f'    Selected {n} PCA components by explained '
                        f'variance ({100 * ev}≥{100 * _n_pca_comp}%)')
            _n_pca_comp = n
        elif _n_pca_comp is None:
            _n_pca_comp = self._max_pca_components
            if _n_pca_comp is None:
                _n_pca_comp = self.pca_components_.shape[0]
        elif _n_pca_comp < self.n_pca_components_:
            _n_pca_comp = self.n_pca_components_

        return _n_pca_comp
 

class OCA(BaseOCA):
    def _fit(self, data, fit_type):
        """Aux function."""
        with DataManager(data) as f:
            n_channels, n_samples = f.data.shape
            self._compute_pre_whitener(f.data)
            f.apply(self._pre_whiten)
            f.apply(self._compute_pca)
            # take care of OCA
            sel = slice(0, self.n_pca_components_)
            f.apply(lambda x: x[sel, :])
        data = f.data
        self.n_samples_ = n_samples

        _validate_type(self.fit_params['n_oscillations'], ('int-like', None))
        if self.fit_params['n_oscillations'] is None:
            self.fit_params['n_oscillations'] = n_channels * 2
        n_osc = self.fit_params['n_oscillations']
        ar_order = self.fit_params.setdefault('ar_order', max(7, 2 * n_osc // n_channels + 1)) 
        # 7 is the magic number (arbitrary)
        initial_guess = self.fit_params.setdefault('initial_guess', None)
        scalar_alpha = self.fit_params.setdefault('scalar_alpha', True)
        update_sigma2 = self.fit_params.setdefault('update_sigma2', True)
        tol = self.fit_params.setdefault('tol', 1e-5)
        
        # noise prior computation
        pca_components_ = self.pca_components_[:self.n_pca_components_]
        if self.pca_whiten_:
            norms = self.pca_explained_variance_[:self.n_pca_components_]
            norms = np.sqrt(norms)[:, np.newaxis]
            norms[norms == 0] = 1.
            pca_components_ = pca_components_ / norms
        noise_cov_prior = dict(
            data=pca_components_ @ np.eye(pca_components_.shape[-1]) @ pca_components_.T,
            df=self.noise_cov['nfree']
            )
        sfreq = self.info['sfreq']

        osc, mixing_mat_inst, noise_cov, mm_prior, lls, dkls = \
            _fit_oca(data, ar_order, n_osc, max_iter=self.max_iter,
                    initial_guess=initial_guess, sfreq=sfreq,
                    scalar_alpha=scalar_alpha,
                    update_sigma2=update_sigma2, tol=tol,
                    noise_cov_prior=noise_cov_prior,
                    proj=None, ncomp=0)
        free_energies = np.array(lls) - np.array(dkls)
        self._oscillators_ = osc
        self.n_oscillations = len(osc)
        self._update_oca_names()
        self._mixing_matrix_inst_ = mixing_mat_inst
        self._noise_var_fit_ = noise_cov
        self.free_energies_ = free_energies
        self._mm_prior = mm_prior
        self.current_fit = fit_type


class NewOCA(OCA):
    """For back compatibility"""
    def _fit(self, *args, **kwargs):
        warnings.warn("eprecated, use OCA instead.", DeprecationWarning)


class OCACV(BaseOCA):
    def _fit(self, data, fit_type):
        """Aux function."""
        with DataManager(data) as f:
            n_channels, n_samples = f.data.shape
            self._compute_pre_whitener(f.data)
            f.apply(self._pre_whiten)
            f.apply(self._compute_pca)
            # take care of OCA
            sel = slice(0, self.n_pca_components_)
            f.apply(lambda x: x[sel, :])
        data = f.data
        self.n_samples_ = n_samples

        try:
            n_oscs = np.unique(np.sort(self.fit_params['n_oscillations']))
            _validate_type(n_oscs[0], ('int-like', None))
        except np.AxisError:
            raise TypeError('n_oscillations needs to be a np.ndarry of integers or'
                            ' or object that can be transformed to np.ndarry of integers'
                            ' such as list, tuples.')
        ar_order = self.fit_params.setdefault('ar_order', max(7, 2 * n_oscs[-1] // n_channels + 1)) 
        # 7 is the magic number (arbitrary)
        initial_guess = self.fit_params.setdefault('initial_guess', None)
        scalar_alpha = self.fit_params.setdefault('scalar_alpha', True)
        update_sigma2 = self.fit_params.setdefault('update_sigma2', True)
        tol = self.fit_params.setdefault('tol', 1e-5)
        
        # noise prior computation
        pca_components_ = self.pca_components_[:self.n_pca_components_]
        if self.pca_whiten_:
            norms = self.pca_explained_variance_[:self.n_pca_components_]
            norms = np.sqrt(norms)[:, np.newaxis]
            norms[norms == 0] = 1.
            pca_components_ = pca_components_ / norms
        noise_cov_prior = dict(
            data=pca_components_ @ np.eye(pca_components_.shape[-1]) @ pca_components_.T,
            df=self.noise_cov['nfree']
            )
        sfreq = self.info['sfreq']

        osc, mixing_mat_inst, noise_cov, mm_prior, _, _ = \
            _fit_oca(data, ar_order, n_oscs[-1], max_iter=self.max_iter,
                    initial_guess=initial_guess, sfreq=sfreq,
                    scalar_alpha=scalar_alpha,
                    update_sigma2=update_sigma2, tol=tol,
                    noise_cov_prior=noise_cov_prior,
                    proj=None, ncomp=0)
        to_sensorspace = self._transform_to_sensorspace()[:, :self.n_pca_components_]
        order = _get_sorted_order(data, to_sensorspace, mixing_mat_inst, osc, noise_cov)
        osc._do_arange(order)
        order = osc._expand_indices(order)
        mixing_mat_inst._do_arrange(order)
        mm_prior = mm_prior if scalar_alpha else mm_prior[order]
        self.fit_params['initial_guess'] = (osc, mixing_mat_inst, noise_cov_prior, mm_prior)

        # This can be parallelized using parallel?
        fits = []
        if not osc.n in n_oscs:
            np.append(n_oscs, osc.n)
        n_oscs = n_oscs[n_oscs <= osc.n]

        for ii in n_oscs:
            sel = np.arange(ii)
            initial_guess = (osc[sel],) 
            sel = initial_guess[0]._expand_indices(sel)
            initial_guess = initial_guess + (mixing_mat_inst[sel], noise_cov.copy(), 
                                             mm_prior if scalar_alpha else mm_prior[sel])
            fits.append(_fit_oca(data, ar_order, ii, max_iter=self.max_iter,
                    initial_guess=initial_guess, sfreq=sfreq,
                    scalar_alpha=scalar_alpha,
                    update_sigma2=update_sigma2, tol=tol,
                    noise_cov_prior=noise_cov_prior,
                    proj=None, ncomp=0))
        self._cv_fits = fits
        self.crossvalidate()
        self.current_fit = fit_type

    def crossvalidate(self, use_diffbic=True, force_select=None):
        if use_diffbic:
            metrics, _ = self._crossvalidate_metrics(self._cv_fits)
            opt = metrics[-1].argmax()
        else:
            self._cv_fits = sorted(self._cv_fits, key=lambda fit: (fit[-2][-1] - fit[-1][-1]),
                            reverse=True)
            opt = 0
        if force_select is not None:
            n_oscs = np.fromiter(map(len, list(zip(*self._cv_fits))[0]), dtype=int)
            idx = np.where(n_oscs==force_select)[0]
            if idx.size != 1:
                warnings.warn(f"Found {idx}. Needed only one for selection", RuntimeWarning)
            else:
                opt = int(idx)

        osc, mixing_mat_inst, noise_cov, mm_prior, lls, dkls = self._cv_fits[opt]        
        free_energies = np.array(lls) - np.array(dkls)
        self._oscillators_ = osc
        self.n_oscillations = len(osc)
        self._update_oca_names()
        self._mixing_matrix_inst_ = mixing_mat_inst
        self._noise_var_fit_ = noise_cov
        self.free_energies_ = free_energies
        self._mm_prior = mm_prior
 

    def _crossvalidate_metrics(self, fits=None):
        if fits is None:
            fits = getattr(self, '_cv_fits')
        model_posteriors = [(len(fit[0]), fit[-2][-1]-fit[-1][-1]) for fit in fits]
        idx, model_posteriors = [np.array(ii) for ii in zip(*model_posteriors)]
        return DiffBic(idx, model_posteriors)

    def plot_cv(self):
        import matplotlib.pyplot as plt
        if getattr(self, '_cv_fits', None) is None:
            raise RuntimeError("No crossvalidation was performed.")
        metrics, trend = self._crossvalidate_metrics()
        opt = metrics[-1].argmax()
        opt_m, opt_free_energy = metrics[0, opt], metrics[-1, opt]
        temp = metrics.min()
        metrics = metrics[:, np.argsort(metrics[0])]
        idx = metrics[0]
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(idx, metrics[1], marker='o', color='b', label='C1')
        ax.plot(idx, metrics[2], marker='o', color='r', label='C2')
        ax.plot(idx, metrics[3], marker='o', color='k',
                label='(C1+C2)/2' if trend == 'increasing' else '|C1 - C2|/2')
        ax.legend()
        ax.annotate(f'Best Model (with trend={trend})', (opt_m, opt_free_energy),
                    (opt_m, 0.5*(temp+opt_free_energy)),
                    arrowprops={'arrowstyle': '->'})
        ax.set(xlabel="m, # of oscillations", ylabel="diff free energy metric",
                xlim=[idx[0]-0.4, idx[-1]-0.6])
        plt.xticks(idx, ["%i" % m for m in idx],
                    rotation='vertical')
        return fig


class DataManager():
    type = None
    """
    Usage:

    with DataManager(data) as f:
        f.apply(fun1)
        f.apply(fun2)
        ....
    data = f.data
    """
    def __init__(self, data):
        self.data = data
        self.lengths = None

    def __enter__(self):
        data = self.data
        if isinstance(data, collections.abc.Sequence):
            self.type = 'list'
            lengths = [data_.shape[-1] for data_ in data]
            self.data = np.concatenate(data, axis=-1)
        elif isinstance(data, np.ndarray):
            self.type = 'ndarray'
            if data.ndim == 3:
                lengths = [data_.shape[-1] for data_ in data]
                self.data = np.concatenate(list(data), axis=-1)
            elif data.ndim == 2:
                lengths = None
        self.lengths = lengths
        return self

    def apply(self, fun):
        data = fun(self.data)
        if data is not None:
            self.data = data

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.lengths is None:
            data_ = self.data
        else:
            data_ = []
            ii = 0
            for length in self.lengths:
                data_.append(self.data[:, ii:ii+length])
                ii = ii + length
            if self.type == 'ndarray':
                data_ = np.stack(data_, axis=0)
        self.data = data_


def oca_explained_variance(oca, inst, normalize=False):
    if not isinstance(oca, BaseOCA):
        raise TypeError('first argument must be an instance of OCA.')
    if not isinstance(inst, (BaseRaw, BaseEpochs)):
        raise TypeError('second argument must an instance of either Raw or '
                        'Epochs.')

    sources = oca.get_sources(inst)._data

    # if epochs - reshape to channels x timesamples
    if isinstance(inst, BaseEpochs):
        sources = np.hstack(sources)
    source_powers = np.sum(sources ** 2, axis=-1)
    mixing_mat_ = oca.get_components()
    source_powers *= (mixing_mat_ ** 2).sum(axis=0)
    var = np.reshape(source_powers, (-1, 2)).sum(axis=-1)
    if normalize:
        var /= var.sum()
    return var


def _check_start_stop(raw, start, stop):
    """Aux function."""
    out = list()
    for st, none_ in ((start, 0), (stop, raw.n_times)):
        if st is None:
            out.append(none_)
        else:
            try:
                out.append(st)
            except TypeError:  # not int-like
                out.append(raw.time_as_index(st)[0])
    return out


def _get_sorted_order(data, _to_sensorspace, _mixing_mat_inst, _oscillators, _noise_var_fit):
    """Aux function. operates on preprocessed (whitened->pca) data only"""
    sources, *rest = _oscillators.get_hidden_posterior(data, None, _noise_var_fit, None, 0.0, _mixing_mat_inst)
    sources = np.hstack([sources_.T for sources_ in sources])
    source_powers = np.sum(sources ** 2, axis=-1)
    mixing_matrix_, _ = _mixing_mat_inst._get_vals()
    mixing_matrix_ = _to_sensorspace @ mixing_matrix_
    source_powers *= (mixing_matrix_ ** 2).sum(axis=0)
    source_powers = np.reshape(source_powers, (-1, 2)).sum(axis=-1)
    return np.argsort(source_powers)[::-1]


def _rerange(source, target):
    slope = (target.max() - target.min()) / (source.max() - source.min())
    intercept = target.min()
    return slope * (source - source.min()) + intercept


def DiffBic(x, y):
    """Metric for detecting the knee point of the resulting model posterior curve.

    x: model orders
    y: model slecton metric (model posterior/ BIC)

    The original method is described in [1].
    [1] Knee Point Detection on Bayesian Information Criterion
    DOI 10.1109/ICTAI.2008.154
    """
    order = np.argsort(x)

    C1 = _rerange(y, x)
    Cm = C1 / x
    C2 = _rerange(Cm, x)

    trend = 'increasing' if y[order[-1]] > y[order[0]] else 'decreasing'
    C = (C1 + C2) / 2 if trend == 'increasing' else np.abs(C1 - C2) / 2
    return np.vstack((x, C1, C2, C)), trend
