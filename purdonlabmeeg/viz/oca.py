import numpy as np
from mne import EpochsArray, BaseEpochs
from mne.io import RawArray, BaseRaw
from mne.defaults import _handle_default
from mne.io.pick import pick_types
from mne.epochs import plot_epochs_image
from mne.viz.utils import plt_show

from .topomap import _plot_oca_topomap, _get_data_scales

_deafult_bands = [(0, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'),
        (12, 30, 'Beta'), (30, 45, 'Gamma')]


def set_constrained_layout(fig):
    import matplotlib
    mversion = list(map(lambda x: int(x), matplotlib._version.version.split('.')))
    if mversion[0] > 3 and mversion[1] > 6:
        fig.set_layout_engine('constrained')
    else:
        fig.set_constrained_layout(True)


def plot_oca_sources(oca, inst, picks=None, start=None,
                     stop=None, title=None, show=True, block=False,
                     show_first_samp=False,
                     scale_by_mixing_vec='extrema', show_scrollbars=True,
                     time_format='float', precompute=None,
                     use_opengl=None, *, theme=None, overview_mode=None):
    from mne.io.pick import _picks_to_idx

    picks = _picks_to_idx(oca.n_oscillations, picks, 'all')

    if isinstance(inst, (BaseRaw, BaseEpochs)):
        fig = _plot_sources(oca, inst, picks, start=start, stop=stop,
                            show=show, title=title, block=block,
                            show_first_samp=show_first_samp,
                            scale_by_mixing_vec=scale_by_mixing_vec,
                            show_scrollbars=show_scrollbars,
                            time_format=time_format, precompute=precompute,
                            use_opengl=use_opengl, theme=theme,
                            overview_mode=overview_mode)
        return fig
    else:
        raise ValueError('Data input must be of Raw or Epochs type')


def _create_properties_layout(figsize=None, fig=None, topomap_cbar_mode='each'):
    """Create main figure and axes layout used by plot_oca_properties."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid
    if fig is not None and figsize is not None:
        raise ValueError('Cannot specify both fig and figsize.')
    if figsize is None:
        figsize = [7., 6.]
    if fig is None:
        fig = plt.figure(figsize=figsize, facecolor=[0.95] * 3)

    axes_params = (('topomap', [0.08, 0.63, 0.40, 0.32], (1, 2), topomap_cbar_mode, 'bottom'),
                   ('spectrum', [0.57, 0.60, 0.39, 0.35], (2, 1), None, 'right'),
                   ('image', [0.08, 0.175, 0.88, 0.32], (1, 2), None, 'right'),
                   ('erp', [0.08, 0.075, 0.88, 0.10], (1, 2), None, 'right'))

    axes = [ImageGrid(fig, rect, nrows_ncols, axes_pad=(0.10, 0.15), aspect=False, cbar_mode=cbar_mode,
                      cbar_location=cbar_location)
       for name, rect, nrows_ncols, cbar_mode, cbar_location in axes_params]

    return fig, axes


def _convert_psds(psds, dB, estimate, scaling, unit):
    """Convert PSDs to dB (if necessary) and appropriate units.
    The following table summarizes the relationship between the value of
    parameters ``dB`` and ``estimate``, and the type of plot and corresponding
    units.
    | dB    | estimate    | plot | units             |
    |-------+-------------+------+-------------------|
    | True  | 'power'     | PSD  | amp**2/Hz (dB)    |
    | True  | 'amplitude' | ASD  | amp/sqrt(Hz) (dB) |
    | True  | 'auto'      | PSD  | amp**2/Hz (dB)    |
    | False | 'power'     | PSD  | amp**2/Hz         |
    | False | 'amplitude' | ASD  | amp/sqrt(Hz)      |
    | False | 'auto'      | ASD  | amp/sqrt(Hz)      |
    where amp are the units corresponding to the variable, as specified by
    ``unit``.
    """
    if estimate == 'auto':
        estimate = 'power' if dB else 'amplitude'

    if estimate == 'amplitude':
        np.sqrt(psds, out=psds)
        psds *= scaling
        ylabel = r'$\mathrm{%s/\sqrt{Hz}}$' % unit
    else:
        psds *= scaling * scaling
        if '/' in unit:
            unit = '(%s)' % unit
        ylabel = r'$\mathrm{%s²/Hz}$' % unit
    if dB:
        np.log10(np.maximum(psds, np.finfo(float).tiny), out=psds)
        psds *= 10
        ylabel += r'$\ \mathrm{(dB)}$'

    return ylabel


def _plot_oca_properties(pick, oca, inst, psds, freqs, n_trials,
                         plot_lowpass_edge, epochs_src,
                         set_title_and_labels, plot_std, dB,
                         num_std, log_scale, topomap_args, image_args,
                         fig, axes, kind,):
    """Plot OCA properties (helper)."""
    topo_axes, spec_axes, image_axes, erp_axes = axes
    # all of them actually needs to be a Imagegrid! :(

    # plotting
    # --------
    # component topomap
    if topomap_args is None:
        topomap_args = dict(mapscale='extrema')
    _plot_oca_topomap(oca, [pick], grids=[topo_axes], colorbar=True, **topomap_args)

    picks = oca._oscillators_._expand_indices([pick])
    # spectrum
    for spec_ax, idx in zip(spec_axes, picks):
        # calculate component-specific spectrum stuff
        psd_ylabel, psds_mean, spectrum_std = _get_psd_label_and_std(
            psds[:, idx, :].copy(), dB, num_std)
        spec_ax.plot(freqs, psds_mean, color='k')
        if plot_std:
            spec_ax.fill_between(freqs, psds_mean - spectrum_std[0],
                                psds_mean + spectrum_std[1],
                                color='k', alpha=.2)
        if plot_lowpass_edge:
            spec_ax.axvline(inst.info['lowpass'], lw=2, linestyle='--',
                            color='k', alpha=0.2)

    # image and erp
    # # we create a new epoch with dropped rows
    # epoch_data = epochs_src.get_data()
    # epoch_data = np.insert(arr=epoch_data,
    #                        obj=(dropped_indices -
    #                             np.arange(len(dropped_indices))).astype(int),
    #                        values=0.0,
    #                        axis=0)
    # epochs_src = EpochsArray(epoch_data, epochs_src.info, tmin=epochs_src.tmin,
    #                          verbose=0)

    [plot_epochs_image(epochs_src, picks=pick, axes=this_ax_pair,
                      combine=None, colorbar=False, show=False,
                      **image_args)
                      for pick, *this_ax_pair in zip(picks, image_axes, erp_axes)]

    # aesthetics
    # ----------
    set_title_and_labels(image_axes, kind + ' image and ERP/ERF', [], kind)

    # erp
    set_title_and_labels(erp_axes, [], 'Time (s)', 'AU')
    for erp_ax in erp_axes:
        erp_ax.spines["right"].set_color('k')
        erp_ax.set_xlim(epochs_src.times[[0, -1]])
        # remove half of yticks if more than 5
        yt = erp_ax.get_yticks()
        if len(yt) > 5:
            erp_ax.yaxis.set_ticks(yt[::2])

    # remove xticks - erp plot shows xticks for both image and erp plot
    for image_ax in image_axes:
        image_ax.xaxis.set_ticks([])
        yt = image_ax.get_yticks()
        image_ax.yaxis.set_ticks(yt[1:])
        image_ax.set_ylim([-0.5, n_trials + 0.5])
        image_ax.axhline(0, color='k', linewidth=.5)

    def _set_scale(axes, scale):
        """Set the scale of a matplotlib axis."""
        for ax in axes:
            ax.set_xscale(scale)
            ax.set_yscale(scale)
            ax.relim()
            ax.autoscale()

    # spectrum
    set_title_and_labels(spec_axes, 'Spectrum', 'Frequency (Hz)', psd_ylabel)
    for spec_ax in spec_axes:
        spec_ax.yaxis.labelpad = 0
        spec_ax.set_xlim(freqs[[0, -1]])
        ylim = spec_ax.get_ylim()
        air = np.diff(ylim)[0] * 0.1
        spec_ax.set_ylim(ylim[0] - air, ylim[1] + air)

    if log_scale:
        _set_scale(spec_axes, 'log')

    def _plot_oca_properties_on_press(event, oca, pick, topomap_args):
        """Handle keypress events for oca properties plot."""
        import matplotlib.pyplot as plt
        fig = event.canvas.figure
        if event.key == 'escape':
            plt.close(fig)
        if event.key in ('t', 'l'):
            raise NotImplementedError('Not implemented')
            ax_labels = [ax.get_label() for ax in fig.axes]
            if event.key == 't':
                ax = fig.axes[ax_labels.index('topomap')]
                ax.clear()
                ch_types = list(set(oca.get_channel_types()))
                ch_type = \
                    ch_types[(ch_types.index(ax._ch_type) + 1) % len(ch_types)]
                _plot_oca_topomap(oca, pick, ch_type=ch_type, show=False,
                                  axes=ax, **topomap_args)
                ax._ch_type = ch_type
            elif event.key == 'l':
                ax = fig.axes[ax_labels.index('spectrum')]
            del ax
            fig.canvas.draw()

    # add keypress event handler
    fig.canvas.mpl_connect(
        'key_press_event', lambda event: _plot_oca_properties_on_press(
            event, oca, pick, topomap_args))
    set_constrained_layout(fig)
    return fig


def _get_psd_label_and_std(this_psd, dB, num_std):
    """Handle setting up PSD for one component, for plot_oca_properties."""
    import warnings
    psd_ylabel = _convert_psds(this_psd, dB, estimate='auto', scaling=1.,
                               unit='AU')
    psds_mean = this_psd.mean(axis=0)
    diffs = this_psd - psds_mean
    # the distribution of power for each frequency bin is highly
    # skewed so we calculate std for values below and above average
    # separately - this is used for fill_between shade
    with warnings.catch_warnings():  # mean of empty slice
        warnings.simplefilter('ignore')
        spectrum_std = [
            [np.sqrt((d[d < 0] ** 2).mean(axis=0)) for d in diffs.T],
            [np.sqrt((d[d > 0] ** 2).mean(axis=0)) for d in diffs.T]]
    spectrum_std = np.array(spectrum_std) * num_std

    return psd_ylabel, psds_mean, spectrum_std


def plot_oca_properties(oca, inst, picks=None, axes=None, dB=True,
                        plot_std=True, log_scale=False, topomap_args=None,
                        image_args=None, psd_args=None, figsize=None,
                        show=True, *, verbose=None):
    """Display component properties.
    Properties include the topography, epochs image, ERP/ERF, power
    spectrum, and epoch variance.
    Parameters
    ----------
    oca : instance of OCA
        The OCA solution.
    inst : instance of Epochs or Raw
        The data to use in plotting properties.
        .. note::
           You can interactively cycle through topographic maps for different
           channel types by pressing :kbd:`T`.
    picks : str | list | slice | None
        Components to include. Slices and lists of integers will be interpreted
        as component indices. ``None`` (default) will use the first five
        components. Each component will be plotted in a separate figure.
    axes : list of Axes | None
        List of five matplotlib axes to use in plotting: [topomap_axis,
        image_axis, erp_axis, spectrum_axis, variance_axis]. If None a new
        figure with relevant axes is created. Defaults to None.
    dB : bool
        Whether to plot spectrum in dB. Defaults to True.
    plot_std : bool | float
        Whether to plot standard deviation/confidence intervals in ERP/ERF and
        spectrum plots.
        Defaults to True, which plots one standard deviation above/below for
        the spectrum. If set to float allows to control how many standard
        deviations are plotted for the spectrum. For example 2.5 will plot 2.5
        standard deviation above/below.
        For the ERP/ERF, by default, plot the 95 percent parametric confidence
        interval is calculated. To change this, use ``ci`` in ``ts_args`` in
        ``image_args`` (see below).
    log_scale : bool
        Whether to use a logarithmic frequency axis to plot the spectrum.
        Defaults to ``False``.
        .. note::
           You can interactively toggle this setting by pressing :kbd:`L`.
        .. versionadded:: 1.1
    topomap_args : dict | None
        Dictionary of arguments to ``plot_topomap``. If None, doesn't pass any
        additional arguments. Defaults to None.
    image_args : dict | None
        Dictionary of arguments to ``plot_epochs_image``. If None, doesn't pass
        any additional arguments. Defaults to None.
    psd_args : dict | None
        Dictionary of arguments to ``psd_multitaper``. If None, doesn't pass
        any additional arguments. Defaults to None.
    figsize : array-like, shape (2,) | None
        Allows to control size of the figure. If None, the figure size
        defaults to [7., 6.].
    show : bool
        Show figure if True.
    reject : 'auto' | dict | None
        Allows to specify rejection parameters used to drop epochs
        (or segments if continuous signal is passed as inst).
        If None, no rejection is applied. The default is 'auto',
        which applies the rejection parameters used when fitting
        the OCA object.
    %(reject_by_annotation_raw)s
        .. versionadded:: 0.21.0
    %(verbose)s
    Returns
    -------
    fig : list
        List of matplotlib figures.
    Notes
    -----
    .. versionadded:: 0.13
    """
    return _fast_plot_oca_properties(oca, inst, picks=picks, axes=axes, dB=dB,
                                     plot_std=plot_std, log_scale=log_scale,
                                     topomap_args=topomap_args,
                                     image_args=image_args, psd_args=psd_args,
                                     figsize=figsize, show=show,
                                     verbose=verbose, precomputed_data=None)


def _fast_plot_oca_properties(oca, inst, picks=None, axes=None, dB=True,
                              plot_std=True, log_scale=False,
                              topomap_args=None, image_args=None,
                              psd_args=None, figsize=None, show=True,
                              precomputed_data=None, *, verbose=None):
    """Display component properties."""
    from mne.io.pick import _picks_to_idx
    from ..oca import BaseOCA

    # input checks and defaults
    # -------------------------
    if not isinstance(oca, BaseOCA):
        raise TypeError(f"oca must be an instance of OCA, "
                        f"got {type(oca)} instead.")
    if not isinstance(plot_std, (bool, float)):
        raise TypeError(f"oca must be an instance of bool or fraction, "
                        f"got {type(plot_std)} instead.")
    if isinstance(plot_std, bool):
        num_std = 1. if plot_std else 0.
    else:
        num_std = float(plot_std)
        plot_std = True

    # if no picks given - plot the first 5 components
    limit = min(5, oca.n_oscillations) if picks is None else oca.n_oscillations
    picks = _picks_to_idx(oca.info, picks, 'all')[:limit]
    if not isinstance(topomap_args, dict) or not topomap_args.get('plot_phase', False):
        cbar_mode = 'single'
    else:
        cbar_mode = 'each'

    if axes is None:
        fig, axes = _create_properties_layout(figsize=figsize, topomap_cbar_mode=cbar_mode)
    else:
        if len(picks) > 1:
            raise ValueError('Only a single pick can be drawn '
                             'to a set of axes.')
        #_validate_if_list_of_axes(axes, obligatory_len=5)
        # TODO
        fig = axes[0].get_figure()

    psd_args = dict() if psd_args is None else psd_args
    topomap_args = dict() if topomap_args is None else topomap_args
    image_args = dict() if image_args is None else image_args
    image_args["ts_args"] = dict(truncate_xaxis=False, show_sensors=False)
    if plot_std:
        # from ..stats.parametric import _parametric_ci
        # image_args["ts_args"]["ci"] = _parametric_ci
        pass
    elif "ts_args" not in image_args or "ci" not in image_args["ts_args"]:
        image_args["ts_args"]["ci"] = False

    for item_name, item in (("psd_args", psd_args),
                            ("topomap_args", topomap_args),
                            ("image_args", image_args)):
        if not isinstance(item, dict):
            raise TypeError(f"{item_name} must be an instance of dictionary, "
                            f"got {type(item)} instead.")
    if not isinstance(dB, bool):
        raise TypeError(f"dB must be an instance of bool, "
                        f"got {type(dB)} instead.")
    if not isinstance(log_scale, (bool, None)):
        raise TypeError(f"log_scale must be an instance of bool or None, "
                        f"got {type(log_scale)} instead.")

    # calculations
    # ------------
    mapscale = topomap_args.setdefault('mapscale', 'extrema')
    if isinstance(precomputed_data, tuple):
        kind, dropped_indices, epochs_src, data = precomputed_data
    else:
        kind, dropped_indices, epochs_src, data = _prepare_data_oca_properties(
            inst, oca, mapscale)
    # oca_data = np.swapaxes(data[:, picks, :], 0, 1)
    # dropped_src = oca_data

    # spectrum
    Nyquist = inst.info['sfreq'] / 2.
    lp = inst.info['lowpass']
    if 'fmax' not in psd_args:
        psd_args['fmax'] = min(lp * 1.25, Nyquist)
    plot_lowpass_edge = lp < Nyquist and (psd_args['fmax'] > lp)
    # psds, freqs = psd_multitaper(epochs_src, picks='all', **psd_args)  # Deprecated in MNE > 1.2 
    spectrum = epochs_src.compute_psd(method='multitaper', picks='all', **psd_args)
    psds, freqs = spectrum.get_data(picks='all', return_freqs=True)

    def set_title_and_labels(axes, title, xlab, ylab):
        for ax, suffix in zip(axes, ('real', 'imag')):
            if title:
                ax.set_title(f"{title} {suffix}")
            if xlab:
                ax.set_xlabel(xlab)
            if ylab:
                ax.set_ylabel(ylab)
            ax.axis('auto')
            ax.tick_params('both', labelsize=8)
            ax.axis('tight')

    # plot
    # ----
    all_fig = list()
    for idx, pick in enumerate(picks):
        # if more than one component, spawn additional figures and axes
        if idx > 0:
            fig, axes = _create_properties_layout(figsize=figsize, topomap_cbar_mode=cbar_mode)

        # we reconstruct an epoch_variance with 0 where indexes where dropped
        # epoch_var = np.var(oca_data[idx], axis=1)
        # drop_var = np.var(dropped_src[idx], axis=1)
        # drop_indices_corrected = \
        #     (dropped_indices -
        #      np.arange(len(dropped_indices))).astype(int)
        # epoch_var = np.insert(arr=epoch_var,
        #                       obj=drop_indices_corrected,
        #                       values=drop_var[dropped_indices],
        #                       axis=0)

        # the actual plot
        fig = _plot_oca_properties(
            pick, oca, inst, psds, freqs, len(epochs_src),
            plot_lowpass_edge, epochs_src,
            set_title_and_labels, plot_std, dB, num_std,
            log_scale, topomap_args, image_args, fig, axes, kind)
        all_fig.append(fig)

    plt_show(show)
    return all_fig


def _prepare_data_oca_properties(inst, oca, mapscale):
    """Prepare Epochs sources to plot OCA properties.
    Parameters
    ----------
    oca : instance of OCA
        The OCA solution.
    inst : instance of Epochs or Raw
        The data to use in plotting properties.
    reject_by_annotation : bool, optional
        [description], by default True
    reject : str, optional
        [description], by default 'auto'
    Returns
    -------
    kind : str
        "Segment" for BaseRaw and "Epochs" for BaseEpochs
    dropped_indices : list
        Dropped epochs indexes.
    epochs_src : instance of Epochs
        Segmented data of OCA sources.
    data : array of shape (n_epochs, n_oca_sources, n_times)
        A view on epochs OCA sources data.
    """
    if not isinstance(inst, (BaseRaw, BaseEpochs)):
        raise TypeError(f"inst must be an instance of Raw or Epochs, "
                        f"got {type(inst)} instead.")
    scale = _get_data_scales(oca, np.arange(2*oca.n_oscillations), mapscale)
    if isinstance(inst, BaseRaw):
        # when auto, delegate reject to the oca
        from mne.epochs import make_fixed_length_epochs
        dropped_indices = []
        # break up continuous signal into segments
        epochs_src = make_fixed_length_epochs(
            oca.get_sources(inst),
            duration=2,
            preload=True,
            reject_by_annotation=False,
            proj=False,
            verbose=False)
        kind = "Segment"
    else:
        epochs_src = oca.get_sources(inst)
        dropped_indices = []
        kind = "Epochs"
    epochs_src._data[:, :scale.shape[0], :] *= scale[:, None]
    return kind, dropped_indices, epochs_src, epochs_src.get_data()


def _plot_sources(oca, inst, picks, start, stop, show, title, block,
                  show_scrollbars, show_first_samp, scale_by_mixing_vec,
                  time_format, precompute, use_opengl, *,
                  scalings=None, theme=None, overview_mode=None, ):
    """Plot the oscillation components as a RawArray or EpochsArray.
    scale_by_mixing_vec: str | None
    'extrema' , '1-norm' , '2-norm'
    """
    from mne.viz._figure import _get_browser
    from mne.viz.utils import (_make_event_color_dict,
                               _compute_scalings)
    from mne.io.meas_info import create_info

    # handle defaults / check arg validity
    is_raw = isinstance(inst, BaseRaw)
    is_epo = isinstance(inst, BaseEpochs)
    sfreq = inst.info['sfreq']
    color = _handle_default('color', (0., 0., 0.))
    units = _handle_default('units', None)
    scalings = (_compute_scalings(None, inst) if is_raw else
                _handle_default('scalings_plot_raw'))
    # default:
    if scalings is None:
        scalings = {}
    scalings.update(dict(misc=0.001, whitened=1.))
    unit_scalings = _handle_default('scalings', None)
    picks = oca._oscillators_._expand_indices(picks)

    # data
    if is_raw:
        data = oca._transform_raw(inst, 0, len(inst.times))[picks]
    else:
        data = oca._transform_epochs(inst)[:, picks]

    # scale the data to make mixing_vectors unit infty/1/2-norm.
    if scale_by_mixing_vec is not None:
        if not scale_by_mixing_vec in ('extrema', '1-norm', '2-norm'):
            raise ValueError(f"scale_by_mixing_vec needs to be 'extrema' | '1-norm' | '2-norm'| None, found {mapscale}")
        scale  = _get_data_scales(oca, picks, scale_by_mixing_vec)
        data *= scale[:, None]

    # events
    if is_epo:
        event_id_rev = {v: k for k, v in inst.event_id.items()}
        event_nums = inst.events[:, 2]
        event_color_dict = _make_event_color_dict(None, inst.events,
                                                  inst.event_id)

    # channel properties / trace order / picks
    ch_names = list(oca._osc_names)  # copy
    ch_types = ['misc' for _ in picks]
    # add EOG/ECG channels if present
    eog_chs = pick_types(inst.info, meg=False, eog=True, ref_meg=False)
    ecg_chs = pick_types(inst.info, meg=False, ecg=True, ref_meg=False)
    for eog_idx in eog_chs:
        ch_names.append(inst.ch_names[eog_idx])
        ch_types.append('eog')
    for ecg_idx in ecg_chs:
        ch_names.append(inst.ch_names[ecg_idx])
        ch_types.append('ecg')
    extra_picks = np.concatenate((eog_chs, ecg_chs)).astype(int)
    if len(extra_picks):
        if is_raw:
            eog_ecg_data, _ = inst[extra_picks, :]
        else:
            eog_ecg_data = inst.get_data(extra_picks)
        data = np.append(data, eog_ecg_data, axis=1)
    picks = np.concatenate(
        (picks, 2 * oca.n_oscillations + np.arange(len(extra_picks))))

    ch_order = np.arange(len(picks))
    n_channels = min([20, len(picks)])
    ch_names_picked = [ch_names[x] for x in picks]

    # create info
    info = create_info(ch_names_picked, sfreq, ch_types=ch_types)
    with info._unlock():
        info['meas_date'] = inst.info['meas_date']
    if is_raw:
        inst_array = RawArray(data, info, inst.first_samp)
        inst_array.set_annotations(inst.annotations)
    else:
        inst_array = EpochsArray(data, info)

    # handle time dimension
    start = 0 if start is None else start
    _last = inst.times[-1] if is_raw else len(inst.events)
    stop = min(start + 20, _last) if stop is None else stop
    first_time = getattr(inst, '_first_time', 0) if show_first_samp else 0
    if is_raw:
        duration = stop - start
        start += first_time
    else:
        n_epochs = int(stop - start)
        total_epochs = len(inst)
        epoch_n_times = len(inst.times)
        n_epochs = min(n_epochs, total_epochs)
        n_times = total_epochs * epoch_n_times
        duration = n_epochs * epoch_n_times / sfreq
        event_times = (np.arange(total_epochs) * epoch_n_times
                       + inst.time_as_index(0)) / sfreq
        # NB: this includes start and end of data:
        boundary_times = np.arange(total_epochs + 1) * epoch_n_times / sfreq
    if duration <= 0:
        raise RuntimeError('Stop must be larger than start.')

    # misc
    bad_color = (0.8, 0.8, 0.8)
    title = 'OCA components' if title is None else title
    precompute = _handle_precompute(precompute)
    from mne.preprocessing import ICA

    params = dict(inst=inst_array,
                  ica=ICA(),    # To spoof the browser that we are not dealing with epochs/raw data 
                  ica_inst=inst,
                  info=info,
                  # channels and channel order
                  ch_names=np.array(ch_names_picked),
                  ch_types=np.array(ch_types),
                  ch_order=ch_order,
                  picks=picks,
                  n_channels=n_channels,
                  picks_data=list(),
                  # time
                  t_start=start if is_raw else boundary_times[start],
                  duration=duration,
                  n_times=inst.n_times if is_raw else n_times,
                  first_time=first_time,
                  time_format=time_format,
                  decim=1,
                  # events
                  event_times=None if is_raw else event_times,
                  # preprocessing
                  projs=list(),
                  projs_on=np.array([], dtype=bool),
                  apply_proj=False,
                  remove_dc=True,  # for EOG/ECG
                  filter_coefs=None,
                  filter_bounds=None,
                  noise_cov=None,
                  # scalings
                  scalings=scalings,
                  units=units,
                  unit_scalings=unit_scalings,
                  # colors
                  ch_color_bad=bad_color,
                  ch_color_dict=color,
                  # display
                  butterfly=False,
                  clipping=None,
                  scrollbars_visible=show_scrollbars,
                  scalebars_visible=False,
                  window_title=title,
                  precompute=precompute,
                  use_opengl=use_opengl,
                  theme=theme,
                  overview_mode='empty' if overview_mode is None else overview_mode
                  )
    if is_epo:
        params.update(n_epochs=n_epochs,
                      boundary_times=boundary_times,
                      event_id_rev=event_id_rev,
                      event_color_dict=event_color_dict,
                      event_nums=event_nums,
                      epoch_color_bad=(1, 0, 0),
                      epoch_colors=None,
                      xlabel='Epoch number')
    fig = _get_browser(show=show, block=block, **params)

    return fig


def plot_oca_coh_spectra(oca, inst=None, start=None, stop=None,
                            reject_by_annotation=False, ax=None, bands=None):
    coh_spectra = oca._get_global_coh_spectra(inst, start, stop, reject_by_annotation)
    coh_spectra *= 1e12     # mv^2
    import matplotlib
    from matplotlib import collections
    import matplotlib.text as mtext
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    if ax is None:
        fig, ax = plt.subplots(figsize=(25, 3))
    else:
        fig = ax.get_figure()
    freq = oca.get_frequencies()
    seg = [((x, 0), (x, y)) for x, y in zip(freq, coh_spectra)]
    lc1 = collections.LineCollection(seg, colors=('k',))
    ax.add_collection(lc1)

    offsets = [(x, y) for x, y in zip(freq, coh_spectra)]
    lc2 = collections.RegularPolyCollection(50, sizes=(20,),
                                            offsets=offsets, transOffset=ax.transData,
                                            )
    ax.add_collection(lc2)

    lc3 = collections.LineCollection((((freq.min()-0.1, 0), (freq.max()+0.1, 0)),), colors='k')
    ax.add_collection(lc3)
    ax.autoscale_view()

    if bands is None:
        bands = _deafult_bands
    trans = ax.get_xaxis_transform(which="grid")
    cmap = matplotlib.cm.get_cmap('Spectral')
    colors = [cmap(x) for x in np.linspace(0, 1, len(bands))]
    for (x1, x2, name), c in zip(bands, colors):
        if x1 > freq.max()+0.1:  # skip this band
            continue
        x2 = min(x2, freq.max()+0.1)  # shorten any bands within the freq range
        rect = mpatches.Rectangle((x1, 0), width=x2-x1, height=1, color=c, alpha=.4,
                            ec='k', linewidth=2.0, linestyle='--')
        rect.get_path()._interpolation_steps = 100
        rect.set_transform(trans)
        ax.add_patch(rect)
        text = mtext.Text((x1+x2)/2, 0.8, name,
                            horizontalalignment="center",
                            family='sans-serif', size=8,
                            transform=trans)
        text.set(alpha=0.8)
        ax.add_artist(text)
    
    ax._request_autoscale_view(scaley=False)

    ax.set(xlabel="center frequency (in Hz)", ylabel="power (in $mV^2$)",
            title='Global Coh spectra',
            xlim=[-0.1, freq.max()+0.1])
    ax.set_xticks(freq, ["%0.2f" % m for m in freq], rotation='vertical', )
    ax.xaxis.set_tick_params(which='major', labelsize=6)
    return fig


def _handle_precompute(precompute):
    from mne.utils import check, config
    check._validate_type(precompute, (bool, str, None), 'precompute')
    if precompute is None:
        precompute = config.get_config('MNE_BROWSER_PRECOMPUTE', 'auto').lower()
        check._check_option('MNE_BROWSER_PRECOMPUTE',
                      precompute, ('true', 'false', 'auto'),
                      extra='when precompute=None is used')
        precompute = dict(true=True, false=False, auto='auto')[precompute]
    return precompute
