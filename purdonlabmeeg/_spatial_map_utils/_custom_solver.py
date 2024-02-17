# -*- coding: utf-8 -*-
"""
.. _ex-custom-inverse:

================================================
Source localization with a custom inverse solver
================================================

The objective of this example is to show how to plug a custom inverse solver
in MNE in order to facilate empirical comparison with the methods MNE already
implements (wMNE, dSPM, sLORETA, eLORETA, LCMV, DICS, (TF-)MxNE etc.).

This script is educational and shall be used for methods
evaluations and new developments. It is not meant to be an example
of good practice to analyse your data.

The example makes use of 2 functions ``apply_solver`` and ``solver``
so changes can be limited to the ``solver`` function (which only takes three
parameters: the whitened data, the gain matrix and the number of orientations)
in order to try out another inverse algorithm.
"""
import itertools
import numpy as np
from scipy import linalg
import mne


# Auxiliary function to run the solver

def apply_solver(epochs, forward, noise_cov, loose=0.2, depth=0.8, extent=(11,), local_smoothness=0.9999, subjects_dir=None, subject=None,
                average_first=True):
    """Call a custom solver on evoked data.

    This function does all the necessary computation:

    - to select the channels in the forward given the available ones in
      the data
    - to take into account the noise covariance and do the spatial whitening
    - to apply loose orientation constraint as MNE solvers
    - to apply a weigthing of the columns of the forward operator as in the
      weighted Minimum Norm formulation in order to limit the problem
      of depth bias.

    Parameters
    ----------
    epochs : instance of mne.Epochs
        The epochs data
    forward : instance of Forward
        The forward solution.
    noise_cov : instance of Covariance
        The noise covariance.
    loose : float in [0, 1] | 'auto'
        Value that weights the source variances of the dipole components
        that are parallel (tangential) to the cortical surface. If loose
        is 0 then the solution is computed with fixed orientation.
        If loose is 1, it corresponds to free orientations.
        The default value ('auto') is set to 0.2 for surface-oriented source
        space and set to 1.0 for volumic or discrete source space.
    depth : None | float in [0, 1]
        Depth weighting coefficients. If None, no depth weighting is performed.

    Returns
    -------
    stc : instance of SourceEstimate
        The source estimates.
    """
    # Import the necessary private functions
    from mne.inverse_sparse.mxne_inverse import \
        (_prepare_gain, is_fixed_orient,
         _reapply_source_weighting, _make_sparse_stc)

    all_ch_names = epochs.ch_names

    # Handle depth weighting and whitening (here is no weights)
    forward, gain, gain_info, whitener, source_weighting, mask = _prepare_gain(
        forward, epochs.info, noise_cov, pca=False, depth=depth,
        loose=loose, weights=None, weights_min=None, rank=None)

    from ._parcel_utils import data_driven_parcellation
    ddp = data_driven_parcellation(epochs, forward, 0.05, 0.15, extent, subjects_dir, subject)

    Q2s = setup_local_smoothness_priors(forward, local_smoothness, ddp)
    Q1s = setup_sensor_smoothness_priors(gain_info)

    # Select channels of interest
    sel = [all_ch_names.index(name) for name in gain_info['ch_names']]
    if average_first:
        M = epochs.average().data[sel]
        # Whiten data
        M = np.dot(whitener, M)
    else:
        M = epochs.get_data(picks=sel)
        # Whiten data
        M = np.einsum('ij,...jk->...ik', whitener, M)

    out = hierarchial_bayes_solver(M, gain, Q1s, list(itertools.chain(*Q2s)),
                                   use_multiplicative_update=True,
                                   max_iter=500, rtol=1e-15)
    nu2 = out[1]
    labels = list(itertools.chain(*[ii['labels'] for ii in ddp.values()]))

    X = out[2]
    if X.ndim == 3:
        X *= source_weighting[None, :, None]
        X = X.mean(axis=0)
    else:
        X *= source_weighting[:, None]
    stc = mne.SourceEstimate(X, [src['vertno'] for src in forward['src']],
                             tmin=epochs.times[0],
                             tstep=1. / epochs.info['sfreq'],
                             subject=subject)
    return stc, labels, nu2


def setup_sensor_smoothness_priors(gain_info):
    meg_sel = []
    eeg_sel = []
    for i, name in enumerate(gain_info['ch_names']):
        if name.startswith('MEG'):
            meg_sel.append(i)
        elif name.startswith('EEG'):
            eeg_sel.append(i)
    Q1s = []
    for sel in [meg_sel, eeg_sel]:
        if len(sel) > 1:
            sel = np.array(sel)
            Q1s.append((np.ones(len(sel), dtype=np.float64), (sel, sel)))
    return Q1s


def setup_local_smoothness_priors(forward, local_smoothness, ddp):
    adjacency = mne.spatial_src_adjacency(forward['src'])
    smoothness_weights = _spatial_smoothness_prior(adjacency, local_smoothness)
    Q2s = _get_local_smoothness_model(smoothness_weights, ddp)
    return Q2s


# Define your solver


def hierarchial_bayes_solver(M, G, Q1s, Q2s, use_multiplicative_update=False,
                 max_iter=100, rtol=1e-15):
    from ._hierarchial import restricted_ml
    nu1, nu2, X, e1 = restricted_ml(M, G, Q1s, Q2s, None, None, use_multiplicative_update, max_iter, rtol)
    return nu1, nu2, X, e1


def solver(M, G, n_orient):
    """Run L2 penalized regression and keep 10 strongest locations.

    Parameters
    ----------
    M : array, shape (n_channels, n_times)
        The whitened data.
    G : array, shape (n_channels, n_dipoles)
        The gain matrix a.k.a. the forward operator. The number of locations
        is n_dipoles / n_orient. n_orient will be 1 for a fixed orientation
        constraint or 3 when using a free orientation model.
    n_orient : int
        Can be 1 or 3 depending if one works with fixed or free orientations.
        If n_orient is 3, then ``G[:, 2::3]`` corresponds to the dipoles that
        are normal to the cortex.

    Returns
    -------
    X : array, (n_active_dipoles, n_times)
        The time series of the dipoles in the active set.
    active_set : array (n_dipoles)
        Array of bool. Entry j is True if dipole j is in the active set.
        We have ``X_full[active_set] == X`` where X_full is the full X matrix
        such that ``M = G X_full``.
    """
    inner = np.dot(G, G.T)
    trace = np.trace(inner)
    K = linalg.solve(inner + 4e-6 * trace * np.eye(G.shape[0]), G).T
    K /= np.linalg.norm(K, axis=1)[:, None]
    X = np.dot(K, M)

    indices = np.argsort(np.sum(X ** 2, axis=1))[-10:]
    active_set = np.zeros(G.shape[1], dtype=bool)
    for idx in indices:
        idx -= idx % n_orient
        active_set[idx:idx + n_orient] = True
    X = X[active_set]
    return X, active_set


def _spatial_smoothness_prior(adjacency, sigma):
    import scipy.sparse as sparse
    degree = np.squeeze(np.array(adjacency.sum(axis=-1)))
    diags = sparse.diags(degree, shape=adjacency.shape)
    laplacian = -adjacency + diags
    scaled_laplacian = sigma * laplacian
    w = sparse.eye(*adjacency.shape)
    v = sparse.eye(*adjacency.shape)
    for i in range(8):
        v *= scaled_laplacian / (i+1)
        w += v
    w /= w.max()
    return w


def _get_local_smoothness_model(smoothness_weights, ddp):
    "nested list of covariance matrices that defines local smoothness model"
    from ._parcel_utils import format_component
    from functools import partial
    Q2s = []
    for this_ddp in ddp.values():
        parc, nparc = this_ddp['parc'], this_ddp['nparc']
        func = partial(format_component, parc=parc, smoothness_weights=smoothness_weights.tolil()) 
        Q2s.append([func(i) for i in range(nparc)])
    return Q2s


# %%
# Apply your custom solver
if __name__ == '__main__':
    from mne.datasets import sample
    data_path = sample.data_path()
    subject = 'sample'
    meg_path = data_path / 'MEG' / 'sample'
    raw_fname = meg_path / 'sample_audvis_filt-0-40_raw.fif'
    event_fname = meg_path / 'sample_audvis_filt-0-40_raw-eve.fif'
    fwd_fname = meg_path / 'sample_audvis-meg-eeg-oct-6-fwd.fif'
    trans = meg_path / 'sample_audvis_raw-trans.fif'
    ave_fname = meg_path / 'sample_audvis-ave.fif'
    cov_fname = meg_path / 'sample_audvis-cov.fif'
    subjects_dir = data_path / 'subjects'
    bem_sol_fname = subjects_dir / 'sample' / 'bem' / 'sample-5120-5120-5120-bem-sol.fif'
    condition = 'Left Auditory'

    # Read noise covariance matrix
    noise_cov = mne.read_cov(cov_fname)
    # # Handling average file
    # evoked = mne.read_evokeds(ave_fname, condition=condition, baseline=(None, 0))
    # evoked.plot()
    # evoked.crop(tmin=0.04, tmax=0.18)
    # Handling raw files 
    raw = mne.io.read_raw_fif(raw_fname)
    events = mne.read_events(event_fname)

    raw.info['bads'] += ['MEG 2443']
    tmin = -0.2
    tmax = 0.3  # Use a lower tmax to reduce multiple comparisons
    picks = mne.pick_types(raw.info, meg=True, eeg=True, eog=True, exclude='bads')
    event_id = [1,]  # L auditory
    reject = dict(grad=1000e-13, mag=4000e-15, eog=150e-6)
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                        baseline=(None, 0), reject=reject, preload=True)
    evoked = epochs.pick_types(eeg=True, meg=True)
    evoked = epochs.average(picks=['meg', 'eeg', 'eog'])

    # Handling forward solution
    forward = mne.read_forward_solution(fwd_fname)
    src = mne.setup_source_space('fsaverage', spacing='ico4', subjects_dir=subjects_dir,
                                 n_jobs=-1, add_dist=False)
    src = mne.morph_source_spaces(src, 'sample', subjects_dir=subjects_dir)
    bem = mne.read_bem_solution(bem_sol_fname)
    fwd = mne.make_forward_solution(raw.info, trans, src=src, bem=bem,
                                    meg=True, eeg=True, n_jobs=-1)
    # forward = forward.pick_channels(epochs.ch_names)

    # # Handle data driven parcellation first
    # from purdonlabmeeg._spatial_map_utils._parcel_utils import data_driven_parcellation
    # ddp = data_driven_parcellation(epochs, forward, 0.05, 0.15, [7, 5, 3], subjects_dir, subject)

    # adjacency = mne.spatial_src_adjacency(forward['src'])
    # smoothness_weights = _spatial_smoothness_prior(adjacency, 0.8)

    # Q2s = _get_local_smoothness_model(smoothness_weights, ddp)
    
    # # Checks
    # import matplotlib.pyplot as plt
    # import scipy.sparse as sparse
    # fig, ax = plt.subplots(figsize=(5.8, 7.2))
    # ax.imshow(sparse.coo_array(Q2s[0][0], shape=(7498, 7498)).toarray(), cmap='bwr', vmax=200, vmin=-200)
    # fig.show()

    # # noise cov
    # noise_cov_coo = sparse.coo_array(noise_cov.data)
    # Q1s = (noise_cov_coo.data, (noise_cov_coo.row, noise_cov_coo.col))

    stc, labels, nu = apply_solver(epochs, fwd, noise_cov, loose=0., depth=0.5,
                       extent=[9,], local_smoothness=0.9999, subject=subject, subjects_dir = subjects_dir,
                       average_first=True)

    # Brain = mne.viz.get_brain_class()
    # brain = Brain('sample', 'both', 'pial', subjects_dir=subjects_dir,
    #             cortex='low_contrast', alpha=0.1, background='white', size=(800, 800),)

    # loose, depth = 0.2, 0.8  # corresponds to loose orientation
    # loose, depth = 1., 0.  # corresponds to free orientation
    # stc = apply_solver(solver, evoked, forward, noise_cov, loose, depth)


    # # View in 2D and 3D ("glass" brain like 3D plot)
    # plot_sparse_source_estimates(forward['src'], stc, bgcolor=(1, 1, 1),
    #                             opacity=0.1)
