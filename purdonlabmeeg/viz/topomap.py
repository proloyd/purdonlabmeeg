from warnings import warn
import math
import numpy as np
import copy
from mne.io.pick import channel_type
from mne.channels.layout import _merge_ch_data
from mne.viz.utils import plt_show
from mne.viz.topomap import plot_topomap, _prepare_topomap_plot
from matplotlib.cm import register_cmap
from matplotlib.colors import LinearSegmentedColormap


def make_cmaps():
    """Create custom colormaps and register them with matplotlib"""
    # Bipolar
    # -----
    # bi-polar, blue-white-red based
    cmap = LinearSegmentedColormap.from_list(
        "bipolar", (
            (0.0, (0.0, 0.0, 0.3)),
            (0.25, (0.0, 0.0, 1.0)),
            (0.5, (1.0, 1.0, 1.0)),
            (0.5, (1.0, 1.0, 1.0)),
            (0.75, (1.0, 0.0, 0.0)),
            (1.0, (0.5, 0.0, 0.0)),
        ))
    cmap.set_bad('w', alpha=0.)
    register_cmap(cmap=cmap)

    # Bipolar-transparent middle
    # -----
    # bi-polar, blue-transparent-red based
    cmap = LinearSegmentedColormap.from_list(
        "bipolar-a", (
            (0.0, (0.0, 0.0, 0.3, 1.0)),
            (0.25, (0.0, 0.0, 1.0, 1.0)),
            (0.5, (0.0, 0.0, 1.0, 0.0)),
            (0.5, (1.0, 0.0, 0.0, 0.0)),
            (0.75, (1.0, 0.0, 0.0, 1.0)),
            (1.0, (0.5, 0.0, 0.0, 1.0)),
        ))
    cmap.set_bad('w', alpha=0.)
    register_cmap(cmap=cmap)

    # polar-alpha: middle is transparent instead of white
    cmap = LinearSegmentedColormap.from_list(
        "unipolar", (
            (0.0, (1.0, 1.0, .0)),
            (0.5, (1.0, 0.0, 0.0)),
            (1.0, (0.5, 0.0, 0.0)),
        ))
    cmap.set_bad('w', alpha=0.)
    register_cmap(cmap=cmap)

    # polar-alpha: middle is transparent instead of white
    cmap = LinearSegmentedColormap.from_list(
        "unipolar-a", (
            (0.0, (1.0, 0.0, 0.0, 0.0)),
            (0.5, (1.0, 0.0, 0.0, 1.0)),
            (1.0, (0.5, 0.0, 0.0, 1.0)),
        ))
    cmap.set_bad('w', alpha=0.)
    register_cmap(cmap=cmap)

    # extra-polar light: ends are light instead of dark
    cmap = LinearSegmentedColormap.from_list(
        "lunipolar", (
            (0.0, (0.0, 0.0, 0.0, 1.0)),
            (0.4, (1.0, 0.0, 0.0, 1.0)),
            (0.9, (1.0, 1.0, 0.0, 1.0)),
            (1.0, (1.0, 1.0, 0.5, 1.0)),
        ))
    cmap.set_bad('w', alpha=0.)
    register_cmap(cmap=cmap)

    # extra-polar light: ends are light instead of dark
    cmap = LinearSegmentedColormap.from_list(
        "lunipolar-a", (
            (0.0, (0.0, 0.0, 0.0, 0.5)),
            (0.4, (1.0, 0.0, 0.0, 1.0)),
            (0.9, (1.0, 1.0, 0.0, 1.0)),
            (1.0, (1.0, 1.0, 0.5, 1.0)),
        ))
    cmap.set_bad('w', alpha=0.)
    register_cmap(cmap=cmap)

    # phase
    # -----
    cmap = LinearSegmentedColormap.from_list(
        "phase-oca", (
            (0.0, (0.0, 0.0, 1.0)),
            (0.5, (1.0, 0.0, 0.0)),
            (1.0, (0.0, 0.0, 1.0)),
        ))
    cmap.set_bad('w', alpha=0.)
    register_cmap(cmap=cmap)


make_cmaps()


def _setup_cmap(cmap, n_axes=1, norm=False, phase=False):
    """Set color map interactivity."""
    if cmap == 'interactive':
        if phase:
            cmap = ('phase', True)
        else:
            cmap = ('lunipolar-a' if norm else 'bipolar-a', True)
    elif not isinstance(cmap, tuple):
        if cmap is None:
            if phase:
                cmap = 'phase'
            else:
                cmap = 'lunipolar-a' if norm else 'biploar-a'
        cmap = (cmap, False if n_axes > 2 else True)
    return cmap


def _setup_vmin_vmax(data, vmin, vmax, norm=False, phase=False):
    """Handle vmin and vmax parameters for visualizing topomaps.

    For the normal use-case (when `vmin` and `vmax` are None), the parameter
    `norm` drives the computation. When norm=False, data is supposed to come
    from a mag and the output tuple (vmin, vmax) is symmetric range
    (-x, x) where x is the max(abs(data)). When norm=True (a.k.a. data is the
    L2 norm of a gradiometer pair) the output tuple corresponds to (0, x).

    Otherwise, vmin and vmax are callables that drive the operation.
    """
    should_warn = False
    if phase:
        return (-np.pi, np.pi)
    if vmax is None and vmin is None:
        vmax = np.abs(data).max()
        vmin = 0. if norm else -vmax
        if vmin == 0 and np.min(data) < 0:
            should_warn = True

    else:
        if callable(vmin):
            vmin = vmin(data)
        elif vmin is None:
            vmin = 0. if norm else np.min(data)
            if vmin == 0 and np.min(data) < 0:
                should_warn = True

        if callable(vmax):
            vmax = vmax(data)
        elif vmax is None:
            vmax = np.max(data)

    if should_warn:
        warn_msg = ("_setup_vmin_vmax output a (min={vmin}, max={vmax})"
                    " range whereas the minimum of data is {data_min}")
        warn_val = {'vmin': vmin, 'vmax': vmax, 'data_min': np.min(data)}
        warn(warn_msg.format(**warn_val), UserWarning)

    return vmin, vmax


def _draw_colorbar(cax, im, cmap, orientation="horizontal", pad=.05,
                   title=None, format=None):
    """Draw colorbar on an caxis."""
    import matplotlib.pyplot as plt
    cbar = plt.colorbar(im, cax=cax, orientation=orientation,
                        format='%3.2f')
    cbar.ax.tick_params(labelsize=6, rotation=90)
    if title is not None:
        cax.set_title(title, y=-2.00, fontsize=6, pad=pad)
    return cbar, cax


def _add_image_grid(fig, loc, cbar_mode):
    """Add image grid at loc
    See https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.axes_grid1.axes_grid.ImageGrid.html
    for more options.
    """
    from mpl_toolkits.axes_grid1 import ImageGrid
    grid = ImageGrid(fig, int(loc),  # similar to subplot(int(loc))
                     nrows_ncols=(1, 2),
                     axes_pad=(0.10, 0.15),
                     label_mode="1",
                     share_all=False,
                     cbar_location="bottom",
                     cbar_mode=cbar_mode,
                     cbar_size="7%",
                     cbar_pad="2%",
                     )
    return grid


def _prepare_image_grids(n_cells, ncols, nrows='auto', title=False,
                         cbar_mode=False, size=3):
    if n_cells == 1:
        nrows = ncols = 1
    elif isinstance(ncols, int) and n_cells <= ncols:
        nrows, ncols = 1, n_cells
    else:
        if ncols == 'auto' and nrows == 'auto':
            ncols = 2 if n_cells == 4 else 3
            nrows = math.ceil(n_cells / ncols)
        elif ncols == 'auto':
            ncols = math.ceil(n_cells / nrows)
        elif nrows == 'auto':
            nrows = math.ceil(n_cells / ncols)
        else:
            naxes = ncols * nrows
            if naxes < n_cells:
                raise ValueError("Cannot plot {} axes in a {} by {} "
                                 "figure.".format(n_cells, nrows, ncols))
    width = size * ncols
    height = (size + max(0, 0.1 * (4 - size)) - 0.7) * nrows
    try:
        from mne.viz._figure import _figure
        fig = _figure(toolbar=False, figsize=(width, 0.25 + height))
    except Exception:
        from matplotlib.pyplot import figure
        fig = figure(figsize=(width, 0.25 + height))
    grids = [_add_image_grid(fig, f'{nrows}{ncols}{ii+1}', cbar_mode)
             for ii in range(n_cells)]
    return fig, grids


def plot_oca_components(oca, picks=None, ch_type=None, res=64,
                        vmin=None, vmax=None, cmap='interactive',
                        sensors=True, colorbar=False, title=None,
                        show=True, outlines='head', contours=6,
                        image_interp='cubic',
                        plot_phase=False, mapscale='extrema',
                        inst=None, plot_std=True, 
                        grids=None, topomap_args=None,
                        image_args=None, psd_args=None, reject='auto',
                        sphere=None, layout=None, verbose=None):
    """Project mixing matrix on interpolated sensor topography.
    Parameters
    ----------
    oca : instance of OCA
        The OCA solution.
    %(picks_all)s
        If None all are plotted in batches of 20.
    ch_type : 'mag' | 'grad' | 'planar1' | 'planar2' | 'eeg' | None
        The channel type to plot. For 'grad', the gradiometers are
        collected in pairs and the RMS for each pair is plotted.
        If None, then channels are chosen in the order given above.
    res : int
        The resolution of the topomap image (n pixels along each side).
    vmin : float | callable | None
        The value specifying the lower bound of the color range.
        If None, and vmax is None, -vmax is used. Else np.min(data).
        If callable, the output equals vmin(data). Defaults to None.
    vmax : float | callable | None
        The value specifying the upper bound of the color range.
        If None, the maximum absolute value is used. If callable, the output
        equals vmax(data). Defaults to None.
    cmap : matplotlib colormap | (colormap, bool) | 'interactive' | None
        Colormap to use. If tuple, the first value indicates the colormap to
        use and the second value is a boolean defining interactivity. In
        interactive mode the colors are adjustable by clicking and dragging the
        colorbar with left and right mouse button. Left mouse button moves the
        scale up and down and right mouse button adjusts the range. Hitting
        space bar resets the range. Up and down arrows can be used to change
        the colormap. If None, 'Reds' is used for all positive data,
        otherwise defaults to 'RdBu_r'. If 'interactive', translates to
        (None, True). Defaults to 'RdBu_r'.
        .. warning::  Interactive mode works smoothly only for a small amount
                      of topomaps.
    sensors : bool | str
        Add markers for sensor locations to the plot. Accepts matplotlib
        plot format string (e.g., 'r+' for red plusses). If True (default),
        circles  will be used.
    colorbar : bool
        Plot a colorbar.
    title : str | None
        Title to use.
    show : bool
        Show figure if True.
    plot_phase : bool
    mapscale : str
    'extrema' , '1-norm' , '2-norm'

    %(topomap_outlines)s
    contours : int | array of float
        The number of contour lines to draw. If 0, no contours will be drawn.
        When an integer, matplotlib ticker locator is used to find suitable
        values for the contour thresholds (may sometimes be inaccurate, use
        array for accuracy). If an array, the values represent the levels for
        the contours. Defaults to 6.
    image_interp : str
        The image interpolation to be used. All matplotlib options are
        accepted.
    inst : Raw | Epochs | None
        To be able to see component properties after clicking on component
        topomap you need to pass relevant data - instances of Raw or Epochs
        (for example the data that OCA was trained on). This takes effect
        only when running matplotlib in interactive mode.
    plot_std : bool | float
        Whether to plot standard deviation in ERP/ERF and spectrum plots.
        Defaults to True, which plots one standard deviation above/below.
        If set to float allows to control how many standard deviations are
        plotted. For example 2.5 will plot 2.5 standard deviation above/below.
    topomap_args : dict | None
        Dictionary of arguments to ``plot_topomap``. If None, doesn't pass any
        additional arguments. Defaults to None.
    image_args : dict | None
        Dictionary of arguments to ``plot_epochs_image``. If None, doesn't pass
        any additional arguments. Defaults to None.
    psd_args : dict | None
        Dictionary of arguments to ``psd_multitaper``. If None, doesn't pass
        any additional arguments. Defaults to None.
    reject : 'auto' | dict | None
        Allows to specify rejection parameters used to drop epochs
        (or segments if continuous signal is passed as inst).
        If None, no rejection is applied. The default is 'auto',
        which applies the rejection parameters used when fitting
        the OCA object.
    %(topomap_sphere_auto)s
    %(verbose)s
    Returns
    -------
    fig : instance of matplotlib.figure.Figure or list
        The figure object(s).
    Notes
    -----
    When run in interactive mode, ``plot_oca_components`` allows to reject
    components by clicking on their title label. The state of each component
    is indicated by its label color (gray: rejected; black: retained). It is
    also possible to open component properties by clicking on the component
    topomap (this option is only available when the ``inst`` argument is
    supplied).
    """
    from mne.io import BaseRaw
    from mne.epochs import BaseEpochs

    if oca.info is None:
        raise RuntimeError('The OCA\'s measurement info is missing. Please '
                           'fit the OCA or add the corresponding info object.')

    if picks is None:
        picks = np.arange(oca.n_oscillations)

    # picks = _picks_to_idx(oca.info, picks)
    if len(picks) > 9:  # plot components by sets of 9
        if grids is not None:
            assert len(grids) == len(picks)
        n_components = len(picks)
        p = 9
        figs = []
        for k in range(0, n_components, p):
            picks_ = picks[np.arange(k, min(k + p, n_components))]
            grids_ = grids if grids is None else grids[np.arange(k, min(k + p, n_components))]
            fig = plot_oca_components(
                oca, picks=picks_, ch_type=ch_type, res=res, vmax=vmax,
                cmap=cmap, sensors=sensors, colorbar=colorbar, title=title,
                show=show, outlines=outlines, contours=contours,
                image_interp=image_interp, plot_phase=plot_phase, inst=inst,
                plot_std=plot_std, topomap_args=topomap_args, grids=grids_,
                image_args=image_args, psd_args=psd_args, reject=reject,
                sphere=sphere, layout=layout,)
            figs.append(fig)
        return figs
    
    # prepare and/or check the axes/ grids
    cbar_mode = "single" if colorbar else None
    if colorbar and plot_phase:
        cbar_mode = "each"
    elif colorbar:
        cbar_mode = "single"
    else:
        cbar_mode = None
    if grids is None:
        fig, grids = _prepare_image_grids(len(picks), ncols='auto',
                                        cbar_mode=cbar_mode)
    else:
        from mpl_toolkits.axes_grid1 import ImageGrid
        from matplotlib.axes import Axes
        assert len(grids) == len(picks)
        if isinstance(grids[0], ImageGrid):
            fig = grids[0][0].get_figure()
        elif isinstance(grids[0], (tuple, list, np.ndarray)) \
            and isinstance(grids[0][0], Axes):
            fig = grids[0][0].get_figure()
        else:
            raise TypeError(f"grids must be an ndarray of ImageGrids, or tuple/list of Axes-pairs "
                            f"got {type(grids[0])} instead."
                            f"Try grids=None.")

    if title is None:
        title = 'OCA components'
    fig.suptitle(title)

    _plot_oca_topomap(oca, picks, ch_type=ch_type, res=res, vmin=vmin, vmax=vmax,
                      cmap=cmap, sensors=sensors, colorbar=colorbar, outlines=outlines,
                      contours=contours, image_interp=image_interp, plot_phase=plot_phase,
                      mapscale=mapscale, grids=grids, topomap_args=topomap_args,
                      sphere=sphere, layout=layout)
    # tight_layout(fig=fig)
    fig.subplots_adjust(top=1.0, bottom=0.0)
    fig.canvas.draw()

    plt_show(show)
    return fig


def _plot_oca_topomap(oca, picks, ch_type=None, res=64, vmin=None, vmax=None, cmap='interactive',
                      sensors=True, colorbar=False, outlines='head', contours=6,
                      image_interp='cubic', plot_phase=False, mapscale='extrema',
                      grids=None, topomap_args=None, sphere=None, layout=None):
    topomap_args = dict() if topomap_args is None else topomap_args
    topomap_args = copy.copy(topomap_args)
    if 'sphere' not in topomap_args:
        topomap_args['sphere'] = sphere

    picks = oca._oscillators_._expand_indices(picks)
    if ch_type is None:
        ch_type = np.unique([channel_type(oca.info, idx) for idx in range(len(oca.info['chs']))])


    data = oca.get_components()[:, picks].copy()
    scale = _get_data_scales(oca, picks, mapscale)
    data /= scale[None, :]
    data = data.T

    # freqs = np.abs(oca.freq) * oca.info['sfreq']
    freqs = oca.get_frequencies()

    data_picks, pos, merge_channels, names, ch_type, sphere, clip_origin = \
        _prepare_topomap_plot(oca, ch_type, sphere=sphere)
    if layout is not None:
        from mne.channels.layout import Layout, _find_topomap_coords
        if isinstance(layout, Layout):
            pos = _find_topomap_coords(oca.info, None, layout,
                                       False, True, sphere)
            pos[:, 0] += 0.5 * pos[:, 2]
            pos[:, 1] += 0.5 * pos[:, 3]
            pos = pos[:, :2]
            pos -= 0.5
        else:
            raise ValueError(f"Either mne.channels.Layout object or None is"
                             f"expected as layout, but got {type(layout)}")
    outlines = _make_head_outlines(sphere, pos, outlines, clip_origin)

    colorbar = colorbar and not getattr(grids[0], 'cbar_axes', None) is None

    data = np.atleast_2d(data)
    data = data[:, data_picks]

    titles = list()
    for ii, grid, data_x, data_y in zip(picks[::2], grids, data[::2],
                                        data[1::2]):
        # Set the title on the left of the grid.
        freq = freqs[ii // 2]
        titles.append(grid[0].set_title(oca._osc_names[ii][:-1] + f"({freq:0.3f} Hz)",
                      fontsize=8, loc='left'))
        if plot_phase:
            data_c = data_x + 1j*data_y
            data_p = [np.abs(data_c), np.angle(data_c)]
            units = ["AU", "rad"]
            norms = [True, False]
            phase = [False, True]
            for jj, (ax, data_, unit, norm, _phase) in enumerate(zip(grid, data_p,
                                                             units, norms, phase)):
                if merge_channels:
                    data_, names_ = _merge_ch_data(data_, ch_type,
                                                   names.copy())
                vlim = _setup_vmin_vmax(data_, vmin, vmax, norm=norm,  phase=_phase)
                this_cmap = _setup_cmap(cmap, n_axes=len(picks), norm=norm, phase=_phase)
                im = plot_topomap(
                    data_.flatten(), pos, vlim=vlim, res=res,
                    axes=ax, cmap=this_cmap[0], outlines=outlines,
                    contours=contours, image_interp=image_interp, show=False,
                    sensors=sensors, ch_type=ch_type, **topomap_args)[0]
                if colorbar:
                    cax = grid.cbar_axes[jj]
                    cbar, cax = _draw_colorbar(cax, im, this_cmap,
                                               orientation="horizontal",
                                               pad=.05,
                                               title=unit,
                                               format='%3.2f')
                    cbar.set_ticks(vlim)
                    _hide_frame(ax)
        else:
            vlim = _setup_vmin_vmax(np.stack([data_x, data_y]),
                                            vmin, vmax, norm=False)
            for ax, data_ in zip(grid, [data_x, data_y]):
                if merge_channels:
                    data_, names_ = _merge_ch_data(data_, ch_type,
                                                   names.copy())
                this_cmap = _setup_cmap(cmap, n_axes=len(picks), norm=False, phase=False)
                im = plot_topomap(
                    data_.flatten(), pos, vlim=vlim, res=res,
                    axes=ax, cmap=this_cmap[0], outlines=outlines,
                    contours=contours, image_interp=image_interp, show=False,
                    sensors=sensors, ch_type=ch_type, **topomap_args)[0]
            if colorbar:
                cax = grid.cbar_axes[0]
                cbar, cax = _draw_colorbar(cax, im, this_cmap,
                                           orientation="horizontal",
                                           pad=.05,
                                           title="AU",
                                           format='%3.2f')
                cbar.set_ticks(vlim)
                _hide_frame(ax)


def _get_data_scales(oca, picks, mapscale):
    # data = oca._mixing_mat_data[:, picks].copy()
    data = oca.get_components()[:, picks].copy()

    # scale the data to make mixing_vectors unit infty/1/2-norm.
    if mapscale == 'extrema':
        scale = np.sum(np.reshape(data, (data.shape[0], data.shape[1] // 2 , 2)) ** 2, axis=-1).max(axis=0)
        scale **= 0.5
    elif mapscale == '1-norm':
        scale = np.abs(data).sum(axis=0)
        scale = np.reshape(scale, (-1, 2)).sum(axis=1)
    elif mapscale == '2-norm':
        scale = (data * data).sum(axis=0)
        scale = np.reshape(scale, (-1, 2)).sum(axis=1)
        scale = np.sqrt(scale)
    else:
        raise ValueError(f"mapscale needs to be 'extrema' | '1-norm' | '2-norm', found {mapscale}")
    scale = np.repeat(scale, 2)
    return scale


def _make_head_outlines(sphere, pos, outlines, clip_origin):
    """Check or create outlines for topoplot."""
    assert isinstance(sphere, np.ndarray)
    x, y, _, radius = sphere
    del sphere

    if outlines in ('head', 'skirt', None):
        ll = np.linspace(0, 2 * np.pi, 101)
        head_x = np.cos(ll) * radius + x
        head_y = np.sin(ll) * radius + y
        dx = np.exp(np.arccos(np.deg2rad(12)) * 1j)
        dx, dy = dx.real, dx.imag
        nose_x = np.array([-dx, 0, dx]) * radius + x
        nose_y = np.array([dy, 1.15, dy]) * radius + y
        ear_x = np.array([.497, .510, .518, .5299, .5419, .54, .547,
                          .532, .510, .489]) * (radius * 2)
        ear_y = np.array([.0555, .0775, .0783, .0746, .0555, -.0055, -.0932,
                          -.1313, -.1384, -.1199]) * (radius * 2) + y

        if outlines is not None:
            # Define the outline of the head, ears and nose
            outlines_dict = dict(head=(head_x, head_y), nose=(nose_x, nose_y),
                                 ear_left=(ear_x + x, ear_y),
                                 ear_right=(-ear_x + x, ear_y))
        else:
            outlines_dict = dict()

        # Make the figure encompass slightly more than all points
        mask_scale = 1.25 if outlines == 'skirt' else 1.
        # We probably want to ensure it always contains our most
        # extremely positioned channels, so we do:
        mask_scale = max(
            mask_scale, np.linalg.norm(pos, axis=1).max() * 1.01 / radius)
        outlines_dict['mask_pos'] = (mask_scale * head_x, mask_scale * head_y)
        clip_radius = radius * mask_scale
        outlines_dict['clip_radius'] = (clip_radius,) * 2
        outlines_dict['clip_origin'] = clip_origin
        outlines = outlines_dict

    elif isinstance(outlines, dict):
        if 'mask_pos' not in outlines:
            raise ValueError('You must specify the coordinates of the image '
                             'mask.')
    else:
        raise ValueError('Invalid value for `outlines`.')

    return outlines


def _hide_frame(ax):
    """Hide axis frame for topomaps."""
    ax.get_yticks()
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.set_frame_on(False)
