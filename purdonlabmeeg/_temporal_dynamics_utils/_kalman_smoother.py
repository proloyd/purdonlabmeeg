# author: Proloy Das <pdas6@mgh.harvard.edu>
import numpy as np
from scipy import linalg
from warnings import warn

from .numba_opt import (kalcvf_numba, kalcvs_numba, vcvg_kalcvf_numba,
                        ss_kalcvf_numba, ss_kalcvs_numba)


def kalman_smoother(F, Q, mu0, Q0, G, R, y, a=None, b=None, ss=True, use_numba=True):
    """Expose Alex API"""
    Nz = F.shape[0]
    Ny = y.shape[0]
    if a is None:
        a = np.zeros(Nz)
    if b is None:
        b = np.zeros(Ny)
    var = np.zeros((Nz+Ny, Nz+Ny))
    var[:Nz, :Nz] = Q
    var[Nz:, Nz:] = R
    z = mu0
    P = Q0
    filter, smoother = (ss_kalcvf, ss_kalcvs) if ss else (kalcvf, kalcvs)
    out = filter(y, 1, a, F, b, G, var, z, P, return_pred=True,
                 return_filt=False, return_L=True, return_Dinve=True,
                 return_Dinv=True, use_numba=use_numba)
    out_sm = smoother(y, a, F, b, G, var, out['pred'], out['vpred'],
                      Dinvs=out['Dinv'], Dinves=out['Dinve'], Ls=out['L'],
                      use_numba=use_numba)
    out_sm.update(out)
    out_sm['ss'] = ss
    return out_sm


def cho_inv(a):
    "helper function of matrix inversion using cholesky"
    ci, low = linalg.cho_factor(a)
    logdet = 2 * np.log(np.diagonal(ci)).sum()
    a_inv = linalg.cho_solve((ci, low), np.eye(a.shape[1]))
    return a_inv, logdet


def qr_inv(a):
    "helper function of matrix inversion using QR"
    q, r = linalg.qr(a)
    logdet = np.log(np.diagonal(r)).sum()
    a_inv = linalg.solve_triangular(r, q.T)
    return a_inv, logdet


def kalcvf(data, lead, a, F, b, H, var, z=None, P=None, return_pred=False,
           return_filt=False, return_L=False, return_Dinve=False,
           return_Dinv=False, return_K=False, use_numba=True):
    """The (original/vanilla) Kalman filter

    State space model is defined as follows:
        z(t+1) = a+F*z(t)+eta(t)     (state or transition equation)
        y(t) = b+H*z(t)+eps(t)     (observation or measurement equation)

    [logl, <pred, vpred, <filt, vfilt>]=kalcvf(data, lead, a, F, b, H, var, <z0, vz0>)
    computes the one-step prediction and the filtered estimate, as well as
    their covariance matrices. The function uses forward recursions, and you
    can also use it to obtain k-step estimates.

    Input:
    data: a Ny x T matrix containing data (y(1), ... , y(T)).
        NaNs are treated as missing values.
    lead: int
     the number of steps to forecast after the end of the data.
    a: Nz x 1 vector or Nz x T matrix
        input vector in the transition equation.
    F: Nz x Nz matrix
        a time-invariant transition matrix in the transition equation.
    b: Ny x T vector
        a input vector in the measurement equation.
    H: Ny x Nz matrix
        a time-invariant measurement matrix in the measurement equation.
    var: (Ny+Nz) x (Ny+Nz) matrix
        time-invariant variance matrix for the error in the transition equation
        and the error in the measurement equation, i.e., [eta(t)', eps(t)']'.
    z0: (optional) Nz x 1 initial state vector.
    vz0: (optional) Nz x Nz covariance matrix of an initial state vector.

    Output:
    logl: float
        value of the average log likelihood function of the SSM
        under assumption that observation noise eps(t) is normally distributed
    pred: (optional) Nz x (T+lead) matrix
        one-step predicted state vectors.
    vpred: (optional) Nz x Nz x (T+lead) matrix
        mean square errors of predicted state vectors.
    filt: (Optional) Nz x T matrix
        filtered state vectors.
    vfilt: (optional) Nz x Nz x T matrix
        mean square errors of filtered state vectors.

    The initial state vector and its covariance matrix of the time invariant
    Kalman filters are computed under the stationarity condition:
    z0 = (I-F)\a
    vz0 = (I-kron(F,F)) \ V(:)
    where F and V are the time invariant transition matrix and the covariance
    matrix of transition equation noise, and vec(V) is an Nz^2 x 1 column
    vector that is constructed by the stacking Nz columns of matrix V.
    Note that all eigenvalues of the matrix F are inside the unit circle
    when the SSM is stationary. When the preceding formula cannot be applied,
    is given by 1E6I. Optionally, you can specify initial values.
    INFO: http://karibzhanov.com/src/kalcvf.m
    NOTE: use_numba could be ~5 times faster than vanilla numpy
    """
    T = data.shape[-1]
    Nz = a.shape[0]
    Ny = b.shape[0]

    # check input matrix dimensions
    if data.shape[0] != Ny:
        raise ValueError('data and b must have the same number of rows')
    if a.ndim == 1:
        a = [a] * T
        a = np.asanyarray(a)
        a = np.swapaxes(a, 0, -1)
    elif a.shape[-1] != T:
        raise ValueError('a, if matrix, must have same mumber of columns as'
                         'data')
    if F.shape != (Nz, Nz):
        raise ValueError('F must be square')
    if b.ndim == 1:
        b = [b] * T
        b = np.asanyarray(b)
        b = np.swapaxes(b, 0, -1)
    elif b.shape[-1] != T:
        raise ValueError('b, if matrix, must have same mumber of columns as'
                         'data')
    if H.shape != (Ny, Nz):
        raise ValueError('H must be Ny by Nz matrix')
    if var.shape != ((Ny+Nz), (Ny+Nz)):
        raise ValueError('var must be (Ny+Nz) by (Ny+Nz) matrix')

    # V(t) and R(t) are variances of eta(t) and eps(t), respectively,
    # and G(t) is a covariance of eta(t) and eps(t)
    V = var[:Nz, :Nz]
    R = var[Nz:, Nz:]
    G = var[:Nz, Nz:]
    if z is None or P is None:
        e = linalg.eigvals(F)
        if np.all(e != 1.0):
            z = linalg.solve(np.eye(Nz) - F, a[..., 0])
            try:
                P = linalg.solve_discrete_are(F.T, H.T, V, R, balanced=False)
                print('Method 1')
            except np.linalg.LinAlgError:
                try:
                    P = linalg.solve_discrete_lyapunov(F, V)
                    print('Method 2')
                except np.linalg.LinAlgError:
                    # P = np.eye(Nz) * 1e2
                    P = V.copy
                    print('Method 3')
                    # P = linalg.solve(np.eye(Nz ** 2) - np.kron(F, F), V.ravel())
                    # P = P.reshape((Nz, Nz))
        else:
            z = a[..., 0]
            P = np.eye(Nz) * 1e2
            print('Method 4')
    P = F @ P @ F.T + V  # to match Alex's implementation

    if use_numba:
        data, a, F, V, b, H, R, G, var, z, P = map(
                        lambda x: x.astype(np.float64),
                        (data, a, F, V, b, H, R, G, var, z, P)
                )
        # data, lead, a, F, V, b, H, R, G, var, z, P
        ll = np.zeros(1, np.float64)
        pred = np.zeros((Nz, T+lead+1), np.float64)
        vpred = np.zeros((Nz, Nz, T+lead+1), np.float64)
        filt = np.zeros((Nz, T), np.float64)
        vfilt = np.zeros((Nz, Nz, T), np.float64)
        Ls = np.zeros((Nz, Nz, T), np.float64)
        Dinvs = np.zeros((Ny, Ny, T), np.float64)
        Dinves = np.zeros((Ny, T), np.float64)
        Ks = np.zeros((Nz, Ny, T), np.float64)
        kalcvf_numba(T, Nz, Ny, data, lead, a, F, V, b, H, R, G, var, z, P,
                     ll, pred, vpred, filt, vfilt, Ls, Dinvs, Dinves, Ks)
    else:
        if return_pred:
            pred = np.zeros((Nz, T+lead+1))
            vpred = np.zeros((Nz, Nz, T+lead+1))
        if return_filt:
            filt = np.zeros((Nz, T))
            vfilt = np.zeros((Nz, Nz, T))
        if return_L:
            Ls = np.zeros((Nz, Nz, T))
        if return_Dinv:
            Dinvs = np.zeros((Ny, Ny, T))
        if return_Dinve:
            Dinves = np.zeros((Ny, T))
        if return_K:
            Ks = np.zeros((Nz, Ny, T))
        ll = 0

        if return_pred:
            pred[..., 0] = z
            vpred[..., 0] = P
        for t in range(T):
            i = np.logical_not(np.isnan(data[..., t])).nonzero()[0]
            h = H[i]
            Pht = P.dot(h.T)
            e = data[i, t] - b[i, t] - h.dot(z)  # prediction error
            D = h.dot(Pht) + R[np.ix_(i, i)]  # prediction error variance
            try:
                Dinv, logdet = cho_inv(D)
            except linalg.LinAlgError:
                Dinv, logdet = qr_inv(D)
            Dinve = Dinv.dot(e)
            if return_Dinv:
                Dinvs[i, i, t] = Dinv
            if return_Dinve:
                Dinves[i, t] = Dinve
            if return_filt:
                filt[..., t] = z + Pht.dot(Dinve)
                vfilt[..., t] = P - Pht.dot(np.dot(Dinv, Pht.T))
                # vfilt[..., t] = P - Pht.dot(np.linalg.solve(D, Pht.T))
            K = (F.dot(Pht) + G[..., i]).dot(Dinv)
            if return_K:
                Ks[..., t] = K
            L = F - K.dot(h)
            if return_L:
                Ls[..., t] = L
            z = a[..., t] + F.dot(z) + K.dot(e)
            P = F.dot(P).dot(L.T) + V
            P = (P+P.T) / 2
            if return_pred:
                pred[..., t+1] = z
                vpred[..., t+1] = P

            # L = L - (np.log(np.linalg.det(D)) + (e*Dinve).sum() + i.sum()*np.log(2*np.pi))/2
            # sign, logdet = np.linalg.slogdet(D)
            if np.isnan(logdet): raise ValueError
            ll = ll - (logdet + (e*Dinve).sum() + len(i)*np.log(2*np.pi)) / 2

        if lead >= 1 and return_pred:
            for t in range(T, T+lead):
                z = F.dot(z)       # Does not take care of a
                P = F.dot(P).dot(F.T) + V
                pred[..., t+1] = z
                vpred[..., t+1] = P

    out = dict(ll=ll, pred=None, vpred=None, filt=None, vfilt=None)
    if return_pred:
        out['pred'] = pred
        out['vpred'] = vpred
    if return_filt:
        out['filt'] = filt
        out['vfilt'] = vfilt
    if return_L:
        out['L'] = Ls
    if return_Dinv:
        out['Dinv'] = Dinvs
    if return_Dinve:
        out['Dinve'] = Dinves
    if return_K:
        out['K'] = Ks
    return out


def kalcvs(data, a, F, b, H, var, pred, vpred, Dinvs=None, Dinves=None, Ls=None,
           u=None, vu=None, use_numba=True):
    """KALCVS The Kalman smoothing

    State space model is defined as follows:
     z(t+1) = a+F*z(t)+eta(t)     (state or transition equation)
       y(t) = b+H*z(t)+eps(t)     (observation or measurement equation)

    [sm, vsm] = kalcvs(data, a, F, b, H, var, pred, vpred, <un, vun>)
    uses backward recursions to compute the smoothed estimate z(t|T) and
    its covariance matrix, P(t|T),
    where T is the number of observations in the complete data set.

    Input:
    data: a Ny x T matrix containing data (y(1), ... , y(T)).
        NaNs are treated as missing values.
    lead: int
     the number of steps to forecast after the end of the data.
    a: Nz x 1 vector or Nz x T matrix
        input vector in the transition equation.
    F: Nz x Nz matrix
        a time-invariant transition matrix in the transition equation.
    b: Ny x T vector
        a input vector in the measurement equation.
    H: Ny x Nz matrix
        a time-invariant measurement matrix in the measurement equation.
    var: (Ny+Nz) x (Ny+Nz) matrix
        time-invariant variance matrix for the error in the transition equation
        and the error in the measurement equation, i.e., [eta(t)', eps(t)']'.
    pred: Nz x T matrix
        one-step forecasts (z(1|0), ... , z(T|T-1))'.
    vpred: Nz x Nz x T matrix
        mean square error matrices of predicted state vectors, i.e.,
        (P(1|0), ... , P(T|T-1))'.
    Dinvs: (optional) Nz x Nz x T matrix
        output from the kalcvf
    Dinves: (optional) Nz x T matrix
        output from the kalcvf
    Ls: (optional) Nz x Nz x T matrix
        output from the kalcvf
    un: (optional) Nz x 1 vector containing u(T).
    vun: (optional) Nz x Nz covariance matrix containing U(T).

    Output:
    sm: Nz x T matrix
        smoothed state vectors (z(1|T), ... , z(T|T))'.
    vsm: Nz x Nz x T matrix
        covariance matrices of smoothed state vectors (P(1|T), ... , P(T|T))'.

    INFO: http://karibzhanov.com/src/kalcvs.m
    """
    T = data.shape[-1]
    Nz = a.shape[0]
    Ny = b.shape[0]

    if u is None:
        u = np.zeros((Nz, 1), np.float64)
    if vu is None:
        vu = np.zeros((Nz, Nz), np.float64)

    # check input matrix dimensions
    if data.shape[0] != Ny:
        raise ValueError('data and b must have the same number of rows')
    if a.ndim == 1:
        a = [a] * T
        a = np.asanyarray(a)
        a = np.swapaxes(a, 0, -1)
    elif a.shape[-1] != T:
        raise ValueError('a, if matrix, must have same mumber of columns as'
                         'data')
    if F.shape != (Nz, Nz):
        raise ValueError('F must be square')
    if b.ndim == 1:
        b = [b] * T
        b = np.asanyarray(b)
        b = np.swapaxes(b, 0, -1)
    elif b.shape[-1] != T:
        raise ValueError('b, if matrix, must have same mumber of columns as'
                         'data')
    if H.shape != (Ny, Nz):
        raise ValueError('H must be Ny by Nz matrix')
    if var.shape != ((Ny+Nz), (Ny+Nz)):
        raise ValueError('var must be (Ny+Nz) by (Ny+Nz) matrix')
    if pred.shape != (Nz, T):
        ValueError('pred must be Nz by T matrix')
    if vpred.shape != (Nz, Nz, T):
        ValueError('vpred must be Nz by Nz by T matrix')
    if u.shape != (Nz, 1):
        raise ValueError('un must be column vector of length Nz')
    if vu.shape != (Nz, Nz):
        raise ValueError('vun must be Nz by Nz matrix')

    if use_numba:
        if Dinvs is None or Dinves is None or Ls is None:
            use_numba = False
            warn('Cannot use numba_opt: falling back to numpy'
                 'Dinvs, Dinves, Ls must be supplied for numba to work.')

    R = var[Nz:, Nz:]  # variance of eps(t)
    G = var[:Nz, Nz:]  # covariance of eta(t) and eps(t)
    u = np.squeeze(u)

    sm = np.zeros((Nz, T), np.float64)
    vsm = np.zeros((Nz, Nz, T), np.float64)
    cvsm = np.zeros((Nz, Nz, T), np.float64)

    if use_numba:
        H, u, vu = map(lambda x: x.astype(np.float64), (H, u, vu))
        kalcvs_numba(T, Nz, H, u, vu, pred, vpred, Ls, Dinvs, Dinves,
                     sm, vsm, cvsm)
    else:
        P_ = np.zeros((Nz, Nz))
        vu_ = np.zeros((Nz, Nz))
        for t in range(T-1, -1, -1):
            z = pred[..., t]
            P = vpred[..., t]
            i = np.logical_not(np.isnan(data[..., t])).nonzero()[0]
            h = H[i]
            Pht = P.dot(h.T)
            if Dinvs is None:
                D = h.dot(Pht) + R[np.ix_(i, i)]  # prediction error variance
                try:
                    Dinv, logdet = cho_inv(D)
                except linalg.LinAlgError:
                    Dinv, logdet = qr_inv(D)
            else:
                Dinv = Dinvs[..., t]
            if Dinves is None:
                e = data[i, t] - b[i, t] - h.dot(z)  # prediction error
                Dinve = Dinv.dot(e)
            else:
                Dinve = Dinves[..., t]
            if Ls is None:
                K = (F.dot(Pht) + G[..., i]).dot(Dinv)
                L = F - K.dot(h)
            else:
                L = Ls[..., t]
            # L, Dinve, Dinv
            u = h.T.dot(Dinve) + L.T.dot(u)  # t-2
            vu = h.T.dot(Dinv.dot(h)) + L.T.dot(vu).dot(L)  # t-2
            sm[..., t] = z + P.dot(u)
            vsm[..., t] = P - P.dot(vu).dot(P.T)
            # P_t_tmin1_n[..., t] = Ls[..., t-1]*vpred[..., t-1].T \
            # - P.T*vu.T*Ls[..., t-1]*vpred[..., t-1].T
            if t < T-1: # P t-1, t-2 = Lt-2 Pt-2 - 
                cvsm[..., t] = (L - P_.dot(vu_).dot(L)).dot(P.T)
            P_[:] = P
            vu_[:] = vu
        # derived using Theorem 1 and Lemma 1, m = t, s = t+1
    return dict(sm=sm, vsm=vsm, cvsm=cvsm)


def ss_kalcvf(data, lead, a, F, b, H, var, z=None, P=None, return_pred=False,
              return_filt=False, return_L=False, return_Dinve=False,
              return_Dinv=False, return_K=False,  use_numba=True):
    """The (original/vanilla) Kalman filter

    State space model is defined as follows:
        z(t+1) = a+F*z(t)+eta(t)     (state or transition equation)
        y(t) = b+H*z(t)+eps(t)     (observation or measurement equation)

    [logl, <pred, vpred, <filt, vfilt>]=kalcvf(data, lead, a, F, b, H, var, <z0, vz0>)
    computes the one-step prediction and the filtered estimate, as well as
    their covariance matrices. The function uses forward recursions, and you
    can also use it to obtain k-step estimates.

    Input:
    data: a Ny x T matrix containing data (y(1), ... , y(T)).
        NaNs are treated as missing values.
    lead: int
     the number of steps to forecast after the end of the data.
    a: Nz x 1 vector or Nz x T matrix
        input vector in the transition equation.
    F: Nz x Nz matrix
        a time-invariant transition matrix in the transition equation.
    b: Ny x T vector
        a input vector in the measurement equation.
    H: Ny x Nz matrix
        a time-invariant measurement matrix in the measurement equation.
    var: (Ny+Nz) x (Ny+Nz) matrix
        time-invariant variance matrix for the error in the transition equation
        and the error in the measurement equation, i.e., [eta(t)', eps(t)']'.
    z0: (optional) Nz x 1 initial state vector.
    vz0: (optional) Nz x Nz covariance matrix of an initial state vector.

    Output:
    logl: float
        value of the average log likelihood function of the SSM
        under assumption that observation noise eps(t) is normally distributed
    pred: (optional) Nz x (T+lead) matrix
        one-step predicted state vectors.
    vpred: (optional) Nz x Nz x (T+lead) matrix
        mean square errors of predicted state vectors.
    filt: (Optional) Nz x T matrix
        filtered state vectors.
    vfilt: (optional) Nz x Nz x T matrix
        mean square errors of filtered state vectors.

    The initial state vector and its covariance matrix of the time invariant
    Kalman filters are computed under the stationarity condition:
    z0 = (I-F)\a
    vz0 = (I-kron(F,F)) \ V(:)
    where F and V are the time invariant transition matrix and the covariance
    matrix of transition equation noise, and vec(V) is an Nz^2 x 1 column
    vector that is constructed by the stacking Nz columns of matrix V.
    Note that all eigenvalues of the matrix F are inside the unit circle
    when the SSM is stationary. When the preceding formula cannot be applied,
    is given by 1E6I. Optionally, you can specify initial values.
    INFO: http://karibzhanov.com/src/kalcvf.m
    NOTE: use_numba could be ~5 times faster than vanilla numpy
    """
    T = data.shape[-1]
    Nz = a.shape[0]
    Ny = b.shape[0]

    # check input matrix dimensions
    if data.shape[0] != Ny:
        raise ValueError('data and b must have the same number of rows')
    if a.ndim == 1:
        a = [a] * T
        a = np.asanyarray(a)
        a = np.swapaxes(a, 0, -1)
    elif a.shape[-1] != T:
        raise ValueError('a, if matrix, must have same mumber of columns as'
                         'data')
    if F.shape != (Nz, Nz):
        raise ValueError('F must be square')
    if b.ndim == 1:
        b = [b] * T
        b = np.asanyarray(b)
        b = np.swapaxes(b, 0, -1)
    elif b.shape[-1] != T:
        raise ValueError('b, if matrix, must have same mumber of columns as'
                         'data')
    if H.shape != (Ny, Nz):
        raise ValueError('H must be Ny by Nz matrix')
    if var.shape != ((Ny+Nz), (Ny+Nz)):
        raise ValueError('var must be (Ny+Nz) by (Ny+Nz) matrix')

    # V(t) and R(t) are variances of eta(t) and eps(t), respectively,
    # and G(t) is a covariance of eta(t) and eps(t)
    V = var[:Nz, :Nz]
    R = var[Nz:, Nz:]
    G = var[:Nz, Nz:]

    if z is None:
        e = linalg.eigvals(F)
        if np.all(e != 1.0):
            z = linalg.solve(np.eye(Nz) - F, a[..., 0])
            # P = linalg.solve(np.eye(Nz ** 2) - np.kron(F, F), V.ravel())
            # P = P.reshape((Nz, Nz))
        else:
            z = a[..., 0]
            # P = np.eye(Nz) * 1e6
        P = linalg.solve_discrete_lyapunov(F, V)

    vpred = np.zeros((Nz, Nz), np.float64)
    Ks = np.zeros((Nz, Ny), np.float64)
    Ls = np.zeros((Nz, Nz), np.float64)
    Dinvs = np.zeros((Ny, Ny), np.float64)
    Tmax = 25
    F, V, H, R, G, P = map(lambda x: x.astype(np.float64), (F, V, H, R, G, P))
    logdet = vcvg_kalcvf_numba(Tmax, Nz, Ny, F, V, H, R, G, P,
                               vpred, Ks, Ls, Dinvs)
    if use_numba:
        data, a, F, V, b, H, R, G, var, z, P = map(
                        lambda x: x.astype(np.float64),
                        (data, a, F, V, b, H, R, G, var, z, P)
                )
        # data, lead, a, F, V, b, H, R, G, var, z, P
        ll = np.zeros(1, np.float64)
        pred = np.zeros((Nz, T+lead+1), np.float64)
        filt = np.zeros((Nz, T), np.float64)
        vfilt = np.zeros((Nz, Nz), np.float64)
        Dinves = np.zeros((Ny, T), np.float64)
        ss_kalcvf_numba(T, Nz, Ny, data, lead, a, F, V, b, H, R, G, var, z, P,
                        ll, pred, vpred, filt, vfilt, Ls, Dinvs, Dinves,
                        Ks, logdet)
    else:
        Pht = vpred.dot(H.T)
        if return_pred:
            pred = np.zeros((Nz, T+lead+1))
        if return_Dinve:
            Dinves = np.zeros((Ny, T))
        if return_filt:
            filt = np.zeros((Nz, T))
            vfilt = P - Pht.dot(Dinvs.dot(Pht.T))
        ll = 0

        if return_pred:
            pred[..., 0] = z
        for t in range(T):
            i = np.logical_not(np.isnan(data[..., t])).nonzero()[0]
            h = H[i]
            e = data[i, t] - b[i, t] - h.dot(z)  # prediction error
            Dinve = Dinvs.dot(e)
            if return_Dinve:
                Dinves[i, t] = Dinve
            if return_filt:
                filt[..., t] = z + Pht.dot(Dinve)
            z = a[..., t] + F.dot(z) + Ks.dot(e)
            P = F.dot(P).dot(Ls.T) + V
            P = (P+P.T) / 2
            if return_pred:
                pred[..., t+1] = z
            ll = ll - (logdet + (e*Dinve).sum() + len(i)*np.log(2*np.pi)) / 2

        if lead >= 1 and return_pred:
            for t in range(T, T+lead):
                z = F.dot(z)       # Does not take care of a
                pred[..., t+1] = z

    out = dict(ll=ll, pred=None, vpred=None, filt=None, vfilt=None)
    if return_pred:
        out['pred'] = pred
        out['vpred'] = vpred[:, :, None]
    if return_filt:
        out['filt'] = filt
        out['vfilt'] = vfilt[:, :, None]
    if return_L:
        out['L'] = Ls[:, :, None]
    if return_Dinv:
        out['Dinv'] = Dinvs[:, :, None]
    if return_Dinve:
        out['Dinve'] = Dinves
    if return_K:
        out['K'] = Ks[:, :, None]
    return out


def ss_kalcvs(data, a, F, b, H, var, pred, vpred, Dinvs=None, Dinves=None, Ls=None,
              u=None, vu=None, use_numba=True):
    """KALCVS The Kalman smoothing

    State space model is defined as follows:
     z(t+1) = a+F*z(t)+eta(t)     (state or transition equation)
       y(t) = b+H*z(t)+eps(t)     (observation or measurement equation)

    [sm, vsm] = kalcvs(data, a, F, b, H, var, pred, vpred, <un, vun>)
    uses backward recursions to compute the smoothed estimate z(t|T) and
    its covariance matrix, P(t|T),
    where T is the number of observations in the complete data set.

    Input:
    data: a Ny x T matrix containing data (y(1), ... , y(T)).
        NaNs are treated as missing values.
    lead: int
     the number of steps to forecast after the end of the data.
    a: Nz x 1 vector or Nz x T matrix
        input vector in the transition equation.
    F: Nz x Nz matrix
        a time-invariant transition matrix in the transition equation.
    b: Ny x T vector
        a input vector in the measurement equation.
    H: Ny x Nz matrix
        a time-invariant measurement matrix in the measurement equation.
    var: (Ny+Nz) x (Ny+Nz) matrix
        time-invariant variance matrix for the error in the transition equation
        and the error in the measurement equation, i.e., [eta(t)', eps(t)']'.
    pred: Nz x T matrix
        one-step forecasts (z(1|0), ... , z(T|T-1))'.
    vpred: Nz x Nz x T matrix
        mean square error matrices of predicted state vectors, i.e.,
        (P(1|0), ... , P(T|T-1))'.
    Dinvs: (optional) Nz x Nz x T matrix
        output from the kalcvf
    Dinves: (optional) Nz x T matrix
        output from the kalcvf
    Ls: (optional) Nz x Nz x T matrix
        output from the kalcvf
    un: (optional) Nz x 1 vector containing u(T).
    vun: (optional) Nz x Nz covariance matrix containing U(T).

    Output:
    sm: Nz x T matrix
        smoothed state vectors (z(1|T), ... , z(T|T))'.
    vsm: Nz x Nz x T matrix
        covariance matrices of smoothed state vectors (P(1|T), ... , P(T|T))'.

    INFO: http://karibzhanov.com/src/kalcvs.m
    """
    T = data.shape[-1]
    Nz = a.shape[0]
    Ny = b.shape[0]

    if u is None:
        u = np.zeros((Nz, 1), np.float64)
    if vu is None:
        vu = np.zeros((Nz, Nz), np.float64)

    # check input matrix dimensions
    if data.shape[0] != Ny:
        raise ValueError('data and b must have the same number of rows')
    if a.ndim == 1:
        a = [a] * T
        a = np.asanyarray(a)
        a = np.swapaxes(a, 0, -1)
    elif a.shape[-1] != T:
        raise ValueError('a, if matrix, must have same mumber of columns as'
                         'data')
    if F.shape != (Nz, Nz):
        raise ValueError('F must be square')
    if b.ndim == 1:
        b = [b] * T
        b = np.asanyarray(b)
        b = np.swapaxes(b, 0, -1)
    elif b.shape[-1] != T:
        raise ValueError('b, if matrix, must have same mumber of columns as'
                         'data')
    if H.shape != (Ny, Nz):
        raise ValueError('H must be Ny by Nz matrix')
    if var.shape != ((Ny+Nz), (Ny+Nz)):
        raise ValueError('var must be (Ny+Nz) by (Ny+Nz) matrix')
    if pred.shape != (Nz, T):
        ValueError('pred must be Nz by T matrix')
    if vpred.shape != (Nz, Nz, T):
        ValueError('vpred must be Nz by Nz by T matrix')
    if u.shape != (Nz, 1):
        raise ValueError('un must be column vector of length Nz')
    if vu.shape != (Nz, Nz):
        raise ValueError('vun must be Nz by Nz matrix')

    if use_numba:
        if Dinvs is None or Dinves is None or Ls is None:
            use_numba = False
            warn('Cannot use numba_opt: falling back to numpy'
                 'Dinvs, Dinves, Ls must be supplied for numba to work.')

    R = var[Nz:, Nz:]  # variance of eps(t)
    G = var[:Nz, Nz:]  # covariance of eta(t) and eps(t)
    u = np.squeeze(u)

    sm = np.zeros((Nz, T), np.float64)
    vsm = np.zeros((Nz, Nz), np.float64)
    cvsm = np.zeros((Nz, Nz), np.float64)

    Dinv = Dinvs[..., 0]
    L = Ls[..., 0]
    P = vpred[..., 0]
    vu = linalg.solve_discrete_lyapunov(L.T, H.T.dot(Dinv.dot(H)))
    vsm = P - P.dot(vu).dot(P.T)
    cvsm = (L - P.T.dot(vu.T).dot(L)).dot(P.T)

    if use_numba:
        H, u, vu = map(lambda x: x.astype(np.float64), (H, u, vu))
        ss_kalcvs_numba(T, Nz, H, u, vu, pred, P, L, Dinv, Dinves,
                        sm, vsm, cvsm)
    else:
        for t in range(T-1, -1, -1):
            z = pred[..., t]
            i = np.logical_not(np.isnan(data[..., t])).nonzero()[0]
            h = H[i]
            Dinve = Dinves[..., t]
            # L, Dinve, Dinv
            u = h.T.dot(Dinve) + L.T.dot(u)
            sm[..., t] = z + P.dot(u)
        # derived using Theorem 1 and Lemma 1, m = t, s = t+1
    return dict(sm=sm, vsm=vsm[:, :, None], cvsm=cvsm[:, :, None])
