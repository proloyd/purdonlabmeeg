# Author: Proloy Das <pdas6@mgh.harvard.edu>
import warnings
import numpy as np
from scipy import linalg
from mne.utils import logger


def sskf(y, a, fw, q, r, c, delta, xs, initial_cond, return_pred=False,
         compute_logdet=False):
    """Computes steady-state smoothed distribution
    y_{i} = fx_{i} + n_{i}   n_{i} ~ N(0, r)
    x_{i} = ax_{i-1} + u_{i},  u_{i} ~ N(0, q)
    Parameters
    ----------
    y: ndarray of shape (n_samples, n_channels)
    a: ndarray of shape (n_factors*order, n_factors*order)
    f: ndarray of shape (n_channels, n_sources)
    q: ndarray of shape (n_factors*order, n_factors*order)
    r: ndarray of shape (n_channels, n_channels)
    ws: ndarray of shape (n_component, n_sources)
    sigma_ws: list of ndarry | ndarray of shape
              (n_component, n_source, n_source)
    xs: tuple of two ndarrays of shape (n_samples, n_sources*order)
        if provided needs to be F contiguous
    Returns
    -------
    x_ : ndarray of shape (n_samples, n_sources*order)
    s : ndarray of shape (n_sources*order, n_sources*order)
        smoothed error covariances
    b : ndarray of shape (n_sources*order, n_sources*order)
        smoothing gain
    s_hat : ndarray of shape (n_sources*order, n_sources*order)
        smoothed error covariances for sampling
    Notes:
    See README and/or _[1] for the difference between s and s_hat.
    _[1] Fruhwirth-Schnatter, Sylvia (1992) Data Augmentation and Dynamic
    Linear Models. URL: https://epub.wu.ac.at/id/eprint/392
    """
    warnings.warn("kalcvf and kalcfs is recommended", DeprecationWarning)
    assert y.shape[1] == fw.shape[0]

    t, dy = y.shape
    _, dx = fw.shape
    if xs is not None:
        _x, x_ = xs
        assert len(_x) == len(y)
    else:
        _x = np.empty((t, dx), dtype=np.float64)
        x_ = np.empty_like(_x)
    if c is None:
        c = fw.T
        r_ = r
    else:
        r_ = np.eye(c.shape[1])

    # try:
    #     res = _solve_discrete_are(a, fw, q, r, delta)
    # except Exception:
    #     res = None
    # if res is None: print(f'res is {res}')
    #
    try:
        try:
            _s = linalg.solve_discrete_are(a.T, c, q, r_,
                                           balanced=True)
        except np.linalg.LinAlgError as e:
            logger.info(e)
            _s = linalg.solve_discrete_are(a.T, c, q, r_,
                                           balanced=False)
        except ValueError as e:  # q is not symmetric
            logger.info(e)
            q = (q + q.T) / 2
            _s = linalg.solve_discrete_are(a.T, c, q, r_,
                                           balanced=True)
    except Exception:
        res = _solve_discrete_are(a, fw, q, r, delta)
        _s = res[0]

    try:
        temp = fw.dot(_s)
        temp2 = temp.dot(fw.T) + r
        (l, low) = linalg.cho_factor(temp2, check_finite=False)
        k = linalg.cho_solve((l, low), temp, check_finite=False)
    except Exception:
        logger.info(f"{r} and {temp.dot(fw.T)} are not balanced")
        # raise Exception
        e, v = linalg.eigh(_s)
        e[e < 0] = 0
        _s = (v * e[None, :]).dot(v.T)
        temp = fw.dot(_s)
        temp2 = temp.dot(fw.T) + r
        (l, low) = linalg.cho_factor(temp2, check_finite=False)
        k = linalg.cho_solve((l, low), temp, check_finite=False)
    inv_innov_cov = linalg.cho_solve((l, low), np.eye(dy), check_finite=False)
    logdet_inno_cov = np.log(np.diag(l)).sum()  # already multiplied by 1/2
    k = k.T  # Kalman Gain
    s_bar = _s.copy()
    s_bar -= k.dot(temp)
    if delta is None:
        temp = s_bar * 0.0
    elif delta.ndim == 1:
        temp = s_bar * delta[None, :]
    elif delta.ndim == 2:
        temp = s_bar.dot(delta)
    # Use Schur decomosition to compute the inverse
    _t, z = linalg.schur(temp, output='complex')
    _t.flat[::_t.shape[0]+1] += 1.0
    p = z.dot(linalg.solve_triangular(_t, z.conj().T)).real
    s = p.dot(s_bar)

    temp = a.dot(s)
    try:
        (l, low) = linalg.cho_factor(_s, lower=True, check_finite=False)
        b = linalg.cho_solve((l, low), temp, check_finite=False)
    except (np.linalg.LinAlgError, Exception):
        b, *rest = linalg.lstsq(_s, temp, check_finite=False)

    b = b.T  # Smoother Gain
    s_hat = s - b.dot(_s).dot(b.T)  # See README what this means!
    s_ = linalg.solve_discrete_lyapunov(b, s_hat)
    if (np.diag(s_) < 0).any():  # Use approximation from Elvira's paper
        logger.info("diag(s_) values are not non-negative! Using approximation"
              "from Elvira's paper")
        s_ = s + b.dot(s - _s).dot(b.T)
    if (np.diag(s_) < 0).any():
        res = _solve_discrete_are(a, fw, q, r, delta)
        _s, s, *rest = res
        temp = a.dot(s)
        b = linalg.solve(_s, temp).T
        s_hat = s - b.dot(_s).dot(b.T)  # See README what this means!
        s_ = linalg.solve_discrete_lyapunov(b, s_hat)
        # raise ValueError("diag(s_) values are not non-negative!"
        #                  "Both approximation methods fail.")

    fw, a, k, b = _align_cast((fw, a, k, b), use_lapack=False)

    temp = np.empty(dy, dtype=np.float64)
    temp1 = np.empty(dx, dtype=np.float64)
    temp2 = np.empty(dx, dtype=np.float64)
    ll = 0.0
    for i in range(t):
        if i == 0:
            _x[i] = initial_cond
        else:
            _x[i] = np.dot(a, x_[i - 1], out=_x[i])
        # x_[i] = p.dot(_x[i] + k.dot(y[i]-f.dot(_x[i])))
        temp = np.dot(fw, _x[i], out=temp)
        temp *= -1
        temp += y[i]
        temp1 = np.dot(k, temp, out=temp1)
        temp1 += _x[i]
        x_[i] = np.dot(p, temp1, out=x_[i])

        ll += 0.5 * np.sum(inv_innov_cov.dot(temp) * temp)

    ll += t * logdet_inno_cov
    if return_pred:
        return _x, _s, -ll
    # i = t-1 case is already taken care of.
    for i in reversed(range(t - 1)):
        # temp = x_[i+1] - _x[i+1]
        # x_[i] += b.dot(temp)
        temp1[:] = x_[i + 1]
        temp1 -= _x[i + 1]
        temp2 = np.dot(b, temp1, out=temp2)
        x_[i] += temp2

    # Update initial cond
    initial_cond = initial_cond + np.dot(b, x_[0] - _x[0])
    if compute_logdet:
        pass

    return x_, s, s_, b, s_hat, -ll, initial_cond


def _solve_discrete_are(a, fw, q, r, delta):
    # s = np.diag(q) / (1 - (a * a).sum(axis=1))
    # generalization of previous method when off-diagonals are non-zero
    s = linalg.solve_discrete_lyapunov(a, q)
    s = np.nan_to_num(s, copy=True, nan=0.0, posinf=q.max(), neginf=-1e-6)
    logdet_inno_covs = []
    for j in range(100):
        e, v = linalg.eigh(s)
        if np.any(e < 0):
            e[e < 0] = 0
            s = (v * e[:, None]).dot(v.T)
        _s = a.dot(s).dot(a.T) + q
        try:
            temp = fw.dot(_s)
            temp2 = temp.dot(fw.T) + r
            (l, low) = linalg.cho_factor(temp2, check_finite=False)
            k = linalg.cho_solve((l, low), temp, check_finite=False)
        except Exception:
            # import ipdb; ipdb.set_trace()
            logger.info(f"{r} and {temp.dot(fw.T)} are not balanced")
            # raise Exception
            e, v = linalg.eigh(_s)
            e[e < 0] = 0
            _s = (v * e[:, None]).dot(v.T)
            temp = fw.dot(_s)
            temp2 = temp.dot(fw.T) + r
            (l, low) = linalg.cho_factor(temp2, check_finite=False)
            k = linalg.cho_solve((l, low), temp, check_finite=False)
        logdet_inno_covs.append(np.log(np.diag(l)).sum())  # multiplied by 1/2
        k = k.T  # Kalman Gain
        s_bar = _s.copy()
        s_bar -= k.dot(temp)
        if delta is None:
            temp = s_bar * 0.0
        elif delta.ndim == 1:
            temp = s_bar * delta[None, :]
        elif delta.ndim == 2:
            temp = s_bar.dot(delta)
        # Use Schur decomosition to compute the inverse
        _t, z = linalg.schur(temp, output='complex')
        _t.flat[::_t.shape[0]+1] += 1.0
        p = z.dot(linalg.solve_triangular(_t, z.conj().T)).real
        s = p.dot(s_bar)
    return _s, s, k, logdet_inno_covs


def stableblockfiltering(y, a, fw, q, r, fw_vars, xs, initial_cond,
                         mixing_mat_inst=None, rinv=None,
                         return_pred=False, compute_logdet=False, proj=None, ncomp=0):
    """Utilizes the block triangluar structure to solve for the
    posterior distribution of the states, instead of Kalamn filtering.
    
    params
    ------
    proj: ndarray | None (deafult None)
        SSP operator created via MNE (make_projector).
    ncomp: int (default 0)
        number of SSP operator used.
    """
    from ._block_tri_diag_opt import inverse_block_factors, Ainvx
    if mixing_mat_inst is None and fw is None:
        raise ValueError(f"Both mixing_mat_inst and fw cannot be None.")
    if mixing_mat_inst is not None:
        fw, fw_vars = mixing_mat_inst._get_vals()
    if proj is None:
        assert ncomp == 0
    assert y.shape[1] == fw.shape[0]
    n, dy = y.shape
    m, dx = fw.shape
    if xs is not None:
        _x, x_ = xs
        assert len(_x) == len(y)
    else:
        _x = np.empty((n, dx), dtype=np.float64)
        x_ = np.empty_like(_x)

    # rinv = linalg.inv(r)
    if rinv is None:
        rinv, _ = get_noise_cov_inverse(r, proj, ncomp,)
    fwT_rinv = fw.T.dot(rinv)
    b = fwT_rinv.dot(y.T).T
    
    qinv = linalg.inv(q)
    if mixing_mat_inst is not None:
        ctrc = mixing_mat_inst.ctrc(rinv)
    else:
        ctrc = fwT_rinv.dot(fw)
        if fw_vars is None:
            pass
        else:
            delta = (np.asanyarray(fw_vars) *
                     np.diag(rinv)[:, None, None]).sum(axis=0)
            ctrc += delta        
    
    M_1_inv = ctrc + qinv
    _temp = qinv.dot(a)  # B.T
    temp = a.T.dot(_temp)
    Ai = [M_1_inv + temp] * (n-1)
    Ai.append(M_1_inv)
    Bi = [_temp.T] * (n - 1)
    Di, Si, logdet = inverse_block_factors(Ai, Bi)
    x_ = Ainvx(Di, Si, b)
    X_minus_1 = [di.dot(si).T for di, si in zip(Di, Si)]  # A{i+1,i}, lower block
                                                          # Sigma_{t, t-1}

    e = linalg.eigvalsh(r)
    e = e[e > 0.0]
    # ll = - (np.sum(y * rinv.dot(y.T - fw.dot(x_.T)).T)
    #         + n * np.log(e).sum()
    #         + n * np.log(np.diag(q)).sum()
    #         + logdet) / 2
    ll = - (
                # - (x_ * b).sum()
                # + (y * rinv.dot(y.T).T).sum()
                # + n * np.log(e).sum()
                # - m * n
                + n * np.log(np.diag(q)).sum()
                + logdet
            ) / 2

    return x_, np.stack(Di), np.stack(X_minus_1), ll


def _align_cast(args, use_lapack):
    """internal function to typecast and/or memory-align ndarrays

    type cast is done to np.float64, and memory is aligned with 'F'.
    Parameters
    ----------
    args: tuple of ndarrays of arbitrary shape
    use_lapack: bool
        whether to make F_contiguous or not.
    Returns
    -------
    args: tuple
        after alignment and typecasting
    """
    args = tuple([arg if arg.dtype == np.float64 else arg.astype(np.float64)
                  for arg in args])
    if use_lapack:
        args = tuple([arg if arg.flags['F_CONTIGUOUS'] else arg.copy(order='F')
                      for arg in args])
    return args


def get_noise_cov_inverse(noise_cov, proj, ncomp,):
    "Author: Proloy Das <pd640@mgh.harvard.edu>"
    from mne.utils import logger
    eig, eigvec, _ = _smart_eigh(noise_cov, proj, ncomp)
    n_chan = len(eig)
    nzero = (eig > 0)  # Reason behind not using mask is to eliminate additional negative values.
    eig[~nzero] = 0.
    if eigvec.dtype.kind == 'c':
        dtype = np.complex128
    else:
        dtype = np.float64
    INV = np.zeros((n_chan, 1), dtype)
    INV[nzero, 0] = 1.0 / eig[nzero]
    logdet_by_2 = np.log(INV[nzero]).sum() / 2
    #   Rows of eigvec are the eigenvectors
    INV = INV * eigvec  # C ** -0.5
    n_nzero = nzero.sum()
    logger.info('    Created the whitener using a noise covariance matrix '
                'with rank %d (%d small eigenvalues omitted)'
                % (n_nzero, n_chan - n_nzero))

    INV = np.dot(eigvec.conj().T, INV)

    return INV, logdet_by_2


def _smart_eigh(C, proj, ncomp):
    """Compute eigh of C taking into account rank.
    Copied from MNE (See https://github.com/mne-tools/mne-python/blob/9e4a0b492299d3638203e2e6d2264ea445b13ac0/mne/cov.py#L1478)
    Author: Proloy Das <pd640@mgh.harvard.edu>
    """
    rank = C.shape[0]   # by deafult full rank
    if ncomp > 0:
        C = np.dot(proj, np.dot(C, proj.T))
        rank -= ncomp   # decrease rank based on the projection
    eig, eigvec, mask =  _get_ch_whitener(C, rank)
    return eig, eigvec, mask


def _get_ch_whitener(A, rank):
    """Get whitener params for a set of channels.
    
    Author: Proloy Das <pd640@mgh.harvard.edu>"""
    # whitening operator
    from mne.utils.linalg import eigh
    eig, eigvec = eigh(A, overwrite_a=True)
    eigvec = eigvec.conj().T
    mask = np.ones(len(eig), bool)
    eig[:-rank] = 0.0
    mask[:-rank] = False
    # logger.info('    Setting small %s eigenvalues to zero (%s)'
    #             % (ch_type, 'using PCA' if pca else 'without PCA'))
    return eig, eigvec, mask