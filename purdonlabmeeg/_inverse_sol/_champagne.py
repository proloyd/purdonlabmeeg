# author: Proloy Das <proloyd94@gmail.com>
import numpy as np
from mne.utils import logger


def _champagne_opt(data, lead_field, dc, max_iter, noise_cov, rank=None,
                      initialization_stategy='infinity'):
    if rank is None:
        rank = np.linalg.matrix_rank(noise_cov)

    c_b = data.dot(data.T) / data.shape[-1]
    eig, eigvec = np.linalg.eigh(c_b)
    n_zeros = eig > 0
    n_zeros[:-rank] = 0.
    whitener = np.zeros_like(eig)
    whitener[n_zeros] = np.sqrt(eig[n_zeros])
    y_hat = whitener[None, :] * eigvec

    # # Initialization
    if initialization_stategy == 'infinity':
        X = lead_field.T.dot(np.linalg.pinv(lead_field.dot(lead_field.T))).dot(y_hat)
        gamma = (X * X).sum(axis=-1)
    elif initialization_stategy == 'balance':
        gamma = np.ones(lead_field.shape[1])
        multiplier = ((np.trace(c_b) - lead_field.shape[0]) /
                      np.trace(lead_field.dot(gamma[:, None] * lead_field.T)))
        gamma *= multiplier
    elif initialization_stategy == 'mne':
        # Works only when noise covariance is identity
        assert np.allclose(noise_cov, np.eye(noise_cov.shape[0]))
        gamma = np.ones(lead_field.shape[1])
        snr = 2.
        multiplier = (np.trace(noise_cov) /
                      np.trace(lead_field.dot(gamma[:, None] * lead_field.T)))
        gamma *= multiplier
        gamma *= (snr ** 2)

    loglikelihoods = []

    for jj in range(max_iter):
        # gammaLt.dot((noise_cov + L.dot(gammaLt))^-1) 
        gammaLt = gamma[:, None] * lead_field.T
        sigma_b = noise_cov + lead_field.dot(gammaLt)
        eig, eigvec = np.linalg.eigh(sigma_b)
        eigvec = eigvec.conj().T
        n_zeros = eig > 0
        n_zeros[:-rank] = 0
        whitener = np.zeros_like(eig)[:, None]
        whitener[n_zeros, 0] = 1 / np.sqrt(eig[n_zeros])
        whitener = whitener * eigvec
        loglikelihood = - eig[n_zeros].sum() - (whitener.dot(c_b) * whitener).sum()
        loglikelihoods.append(loglikelihood)
        X = gammaLt.dot(whitener.T).dot(whitener.dot(y_hat))
        wL = whitener.dot(lead_field)
        Z = wL.T.dot(wL)
        xs = []
        zs = []
        for ii in range(0, lead_field.shape[-1], dc):
            xs.append(X[ii:ii+dc].dot(X[ii:ii+dc].T))
            zs.append(Z[ii:ii+dc, ii:ii+dc])

        # Gamma update
        alphas = np.array([np.sqrt(np.trace(x) / np.trace(z)) for x, z in zip(xs, zs)])
        new_gamma = np.repeat(alphas, 3)
        gamma[:] = new_gamma
    gammaLt = gamma[:, None] * lead_field.T
    sigma_b = noise_cov + lead_field.dot(gammaLt)
    eig, eigvec = np.linalg.eigh(sigma_b)
    eigvec = eigvec.conj().T
    n_zeros = eig > 0
    n_zeros[:-rank] = 0
    whitener = np.zeros_like(eig)[:, None]
    whitener[n_zeros, 0] = 1 / np.sqrt(eig[n_zeros])
    whitener = whitener * eigvec
    return whitener, sigma_b, gamma, loglikelihoods


def champagne(data,
              info,
              forward,
              noise_cov,
              loose="auto",
              depth=0.8,
              xyz_same_gamma=True,
              maxit=10000,
              tol=1e-6,
              update_mode=1,
              gammas=None,
              pca=True,
              return_residual=False,
              return_as_dipoles=False,
              rank=None,
              pick_ori=None,
              percentile=95,):
    from mne.inverse_sparse.mxne_inverse import _prepare_gain, _check_ori, _make_sparse_stc
    from mne.forward import is_fixed_orient
    forward, gain, gain_info, whitener, source_weighting, mask = _prepare_gain(
        forward, info, noise_cov, pca, depth, loose, rank)
    _check_ori(pick_ori, forward)

    if not xyz_same_gamma:
        raise NotImplementedError
    group_size = 1 if (is_fixed_orient(forward) or not xyz_same_gamma) else 3

    logger.info("Whitening data matrix.")
    whitened_data = whitener.dot(data)
    noise_cov = np.eye(whitener.shape[0])
    whitener, sigma_b, gamma, lls = _champagne_opt(whitened_data, gain, group_size, maxit, noise_cov, rank=None, initialization_stategy='mne')
    gammaLt = gamma[:, None] * gain.T
    X = gammaLt.dot(whitener.T).dot(whitener.dot(whitened_data))
    X *= source_weighting[:, None]

    if return_residual:
        sel = [forward["sol"]["row_names"].index(c) for c in gain_info["ch_names"]]
        recon = np.dot(forward["sol"]["data"][sel, :], X)
        residual = data - recon
    import pdb; pdb.set_trace()
    power = np.sum(X[:, :2] ** 2, axis=-1)
    active_set = np.where(power > np.percentile(power, percentile))[0]
    X = X[active_set]
    idx, offset = divmod(active_set, 3)
    active_src = np.unique(idx)
    if len(X) < 3 * len(active_src):
        X_xyz = np.zeros((len(active_src), 3, X.shape[1]), dtype=X.dtype)
        idx = np.searchsorted(active_src, idx)
        X_xyz[idx, offset, :] = X
        X_xyz.shape = (len(active_src) * 3, X.shape[1])
        X = X_xyz
    active_set = (active_src[:, np.newaxis] * 3 + np.arange(3)).ravel()
    import pdb; pdb.set_trace()

    out = _make_sparse_stc(
        X,
        active_set,
        forward,
        0.0,
        1 / info['sfreq'],
        active_is_idx=True,
        pick_ori=pick_ori,
    )
    if return_residual:
        out = out, residual
    return out
