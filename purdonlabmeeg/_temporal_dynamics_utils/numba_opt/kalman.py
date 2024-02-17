import numpy as np
from numba import njit
from numba.pycc import CC


cc = CC('kalman_opt')
cc.verbose = True


@njit
def cho_inv_numba(a):
    "helper function of matrix inversion using cholesky"
    ci, low = np.linalg.cholesky(a)
    logdet = 2 * np.log(np.diag(ci)).sum()
    ci_inv = np.linalg.solve(ci, np.eye(a.shape[1]))
    a_inv = np.dot(ci_inv, ci_inv)
    return a_inv, logdet


@njit
@cc.export('qr_inv_numba', 'float64[:,:](float64[:,:], float64[:])')
def qr_inv_numba(a, logdet):
    "helper function of matrix inversion using QR"
    q, r = np.linalg.qr(a)
    logdet += np.log(np.diag(r)).sum()
    a_inv = np.linalg.solve(r, q.T)
    return a_inv


@njit('void(int64, int64, int64, float64[:,:], int64, float64[:,:],'
      ' float64[:,:], float64[:,:], float64[:,:], float64[:,:],'
      ' float64[:,:], float64[:,:], float64[:,:], float64[:], float64[:,:],'
      ' float64[:], float64[:,:], float64[:,:,:], float64[:,:],'
      ' float64[:,:,:], float64[:,:,:], float64[:,:,:], float64[:,:], float64[:,:,:])',
      cache=True, nogil=True)
@cc.export('kalcvf_numba',
           'void(int64, int64, int64, float64[:,:], int64, float64[:,:],'
           ' float64[:,:], float64[:,:], float64[:,:], float64[:,:],'
           ' float64[:,:], float64[:,:], float64[:,:], float64[:],'
           ' float64[:,:], float64[:], float64[:,:], float64[:,:,:],'
           ' float64[:,:], float64[:,:,:], float64[:,:,:], float64[:,:,:],'
           ' float64[:,:], float64[:,:,:])')
def kalcvf_numba(T, Nz, Ny, data, lead, a, F, V, b, H, R, G, var, z, P,
                 ll, pred, vpred, filt, vfilt, Ls, Dinvs, Dinves, Ks):
    pred[..., 0] = z
    vpred[..., 0] = P
    for t in range(T):
        # i = np.logical_not(np.isnan(data[..., t])).nonzero()[0]
        h = H
        Pht = P.dot(h.T)
        e = data[:, t] - b[:, t] - h.dot(z)  # prediction error
        D = h.dot(Pht) + R  # prediction error variance
        # Dinv = qr_inv_numba(D, logdet)
        q, r = np.linalg.qr(D)
        logdet = np.log(np.abs(np.diag(r))).sum()
        Dinv = np.linalg.solve(r, q.T)
        Dinve = Dinv.dot(e)
        Dinvs[..., t] = Dinv
        Dinves[..., t] = Dinve
        filt[..., t] = z + Pht.dot(Dinve)
        vfilt[..., t] = P - Pht.dot(Dinv.dot(Pht.T))
        K = (F.dot(Pht) + G).dot(Dinv)
        Ks[..., t] = K
        L = F - K.dot(h)
        Ls[..., t] = L
        z = a[..., t] + F.dot(z) + K.dot(e)
        P = F.dot(P).dot(L.T) + V
        P = (P+P.T) / 2
        pred[..., t+1] = z
        vpred[..., t+1] = P

        ll -= (logdet + (e*Dinve).sum() + Ny*np.log(2*np.pi)) / 2
    if lead >= 1:
        for t in range(T, T+lead):
            z = F.dot(z)       # Does not take care of a
            P = F.dot(P).dot(F.T) + V
            pred[..., t+1] = z
            vpred[..., t+1] = P
    return


@njit('void(int64, int64, float64[:,:], float64[:], float64[:,:],'
      ' float64[:,:], float64[:,:,:], float64[:,:,:], float64[:,:,:],'
      ' float64[:,:], float64[:,:], float64[:,:,:], float64[:,:,:])',
      cache=True, nogil=True)
@cc.export('kalcvs_numba',
           'void(int64, int64, float64[:,:], float64[:], float64[:,:],'
           ' float64[:,:], float64[:,:,:], float64[:,:,:], float64[:,:,:],'
           ' float64[:,:], float64[:,:], float64[:,:,:], float64[:,:,:])')
def kalcvs_numba(T, Nz, H, u, vu, pred, vpred, Ls, Dinvs, Dinves,
                 sm, vsm, cvsm):
    P_ = np.zeros((Nz, Nz))
    vu_ = np.zeros((Nz, Nz))

    for t in range(T-1, -1, -1):
        z = pred[..., t]
        P = vpred[..., t]
        # i = np.logical_not(np.isnan(data[..., t])).nonzero()[0]
        h = H
        Dinv = Dinvs[..., t]
        Dinve = Dinves[..., t]
        L = Ls[..., t]
        # L, Dinve, Dinv
        u = h.T.dot(Dinve) + L.T.dot(u)
        vu = h.T.dot(Dinv.dot(h)) + L.T.dot(vu).dot(L)
        sm[..., t] = z + P.dot(u)
        vsm[..., t] = P - P.dot(vu).dot(P.T)
        # P_t_tmin1_n[..., t] = Ls[..., t-1]*vpred[..., t-1].T \
        # - P.T*vu.T*Ls[..., t-1]*vpred[..., t-1].T
        if t < T-1:
            cvsm[..., t] = (L - P_.dot(vu_).dot(L)).dot(P.T)
        P_[:] = P
        vu_[:] = vu
    return


@njit('void(int64, int64, int64, float64[:,:], int64, float64[:,:],'
      ' float64[:,:], float64[:,:], float64[:,:], float64[:,:],'
      ' float64[:,:], float64[:,:], float64[:,:], float64[:],'
      ' float64[:,:], float64[:], float64[:,:], float64[:,:],'
      ' float64[:,:], float64[:,: ], float64[:,:], float64[:,:],'
      ' float64[:,:], float64[:,:], float64)',
      cache=True, nogil=True)
@cc.export('ss_kalcvf_numba',
           'void(int64, int64, int64, float64[:,:], int64, float64[:,:],'
           ' float64[:,:], float64[:,:], float64[:,:], float64[:,:],'
           ' float64[:,:], float64[:,:], float64[:,:], float64[:],'
           ' float64[:,:], float64[:], float64[:,:], float64[:,:],'
           ' float64[:,:], float64[:,:], float64[:,:], float64[:,:],'
           ' float64[:,:], float64[:,:], float64)')
def ss_kalcvf_numba(T, Nz, Ny, data, lead, a, F, V, b, H, R, G, var, z, P,
                    ll, pred, vpred, filt, vfilt, L, Dinv, Dinves, K, logdet):
    pred[..., 0] = z
    Pht = vpred.dot(H.T)
    vfilt[:] = vpred - Pht.dot(Dinv.dot(Pht.T))
    for t in range(T):
        e = data[:, t] - b[:, t] - H.dot(z)  # prediction error
        Dinve = Dinv.dot(e)
        Dinves[..., t] = Dinve
        filt[..., t] = z + Pht.dot(Dinve)
        z = a[..., t] + F.dot(z) + K.dot(e)
        pred[..., t+1] = z
        ll -= (logdet + (e*Dinve).sum() + Ny*np.log(2*np.pi)) / 2
    if lead >= 1:
        for t in range(T, T+lead):
            z = F.dot(z)       # Does not take care of a
            pred[..., t+1] = z
    return


@njit('void(int64, int64, float64[:,:], float64[:], float64[:,:],'
      ' float64[:,:], float64[:,:], float64[:,:], float64[:,:],'
      ' float64[:,:], float64[:,:], float64[:,:], float64[:,:])',
      cache=True, nogil=True)
@cc.export('ss_kalcvs_numba',
           'void(int64, int64, float64[:,:], float64[:], float64[:,:],'
           ' float64[:,:], float64[:,:], float64[:,:], float64[:,:],'
           ' float64[:,:], float64[:,:], float64[:,:], float64[:,:])')
def ss_kalcvs_numba(T, Nz, H, u, vu, pred, vpred, L, Dinv, Dinves,
                    sm, vsm, cvsm):
    P = vpred

    for t in range(T-1, -1, -1):
        z = pred[..., t]
        # i = np.logical_not(np.isnan(data[..., t])).nonzero()[0]
        h = H
        Dinve = Dinves[..., t]
        # L, Dinve, Dinv
        u = h.T.dot(Dinve) + L.T.dot(u)
        sm[..., t] = z + P.dot(u)
        # P_t_tmin1_n[..., t] = Ls[..., t-1]*vpred[..., t-1].T \
        # - P.T*vu.T*Ls[..., t-1]*vpred[..., t-1].T
    return


@njit('float64(int64, int64, int64, float64[:,:], float64[:,:],'
      ' float64[:,:], float64[:,:], float64[:,:], float64[:,:],'
      ' float64[:,:], float64[:,:], float64[:,:], float64[:,:])',
      cache=True, nogil=True)
@cc.export('vcvg_kalcvf_numba',
           'float64(int64, int64, int64, float64[:,:], float64[:,:],'
           ' float64[:,:], float64[:,:], float64[:,:], float64[:,:],'
           ' float64[:,:], float64[:,:], float64[:,:], float64[:,:])')
def vcvg_kalcvf_numba(T, Nz, Ny, F, V, H, R, G, P,
                      vpred, Ks, Ls, Dinvs):
    # P_init = linalg.solve_discrete_lyapunov(F, V)
    wt = 1
    h = H
    Pht = P.dot(h.T)
    D = h.dot(Pht) + R  # prediction error variance
    gk = np.zeros_like(P)
    dk = np.zeros_like(P)
    for t in range(T):
        # Dinv = qr_inv_numba(D, logdet)
        q, r = np.linalg.qr(D)
        logdet = np.log(np.diag(r)).sum()
        Dinv = np.linalg.solve(r, q.T)
        K = (F.dot(Pht) + G).dot(Dinv)
        L = F - K.dot(h)
        gkold = gk
        gk = P - (F.dot(P).dot(L.T) + V)
        # beta = ((gk-gkold) * gk).sum() / (gkold * gkold).sum() if t > 0 else 0
        beta = 0
        dk = - gk + beta * dk
        # print(f"{t}: {np.linalg.norm(P - (F.dot(P).dot(L.T) + V), 'fro')}")
        P = P + wt * dk
        # wt takes care of the acceleartion
        P = (P+P.T) / 2
        # i = np.logical_not(np.isnan(data[..., t])).nonzero()[0]
        h = H
        Pht = P.dot(h.T)
        D_new = h.dot(Pht) + R  # prediction error variance
        rerr = (np.linalg.norm((D-D_new).ravel(), ord=2)
                / np.linalg.norm(D.ravel(), ord=2))
        if rerr < D.shape[0] * 1e-8:
            break
        D = D_new
        # wt = 1 / (t+1) if t > 5 else 1
        wt = 1

    vpred[:] = P
    Dinvs[:] = Dinv
    Ks[:] = K
    Ls[:] = L
    return logdet


"""
@njit('float64(int64, int64, int64, float64[:,:], float64[:,:],'
      ' float64[:,:], float64[:,:], float64[:,:], float64[:,:],'
      ' float64[:, :], float64[:,:], float64[:,:], float64[:,:])',
      cache=True, nogil=True)
@cc.export('vss_kalcvf_numba',
           'float64(int64, int64, int64, float64[:,:], float64[:,:],'
           ' float64[:,:], float64[:,:], float64[:,:], float64[:,:],'
           ' float64[:, :], float64[:,:], float64[:,:], float64[:,:])')"""


def vss_kalcvf_numba(T, Nz, Ny, F, V, H, R, G, P,
                     vpred, Ks, Ls, Dinvs):
    """
    Note that if F is singular or ill-conditioned, or S has multiple or
    nearly multiple eigenvalues, then other methods should be used to obtain
    the solution to the DARE.
    REF:
    [1] W. Q. Malik, W. Truccolo, E. N. Brown and L. R. Hochberg, "Efficient
    Decoding With Steady-State Kalman Filter in Neural Interface Systems,"
    in IEEE Transactions on Neural Systems and Rehabilitation Engineering,
    vol. 19, no. 1, pp. 25-34, Feb. 2011, doi: 10.1109/TNSRE.2010.2092443.
    """
    # form Hamiltonian Matrix (18)
    S = np.zeros((2*Nz, 2*Nz))
    q, r = np.linalg.qr(F)
    Finv = np.linalg.solve(r, q.T)
    S[:Nz, :Nz] = Finv.T
    q, r = np.linalg.qr(R)
    RinvH = np.linalg.solve(r, q.T.dot(H))
    S[:Nz, Nz:] = Finv.T.dot(H.T.dot(RinvH))
    S[Nz:, :Nz] = V.dot(Finv.T)
    S[Nz:, Nz:] = F + V.dot(S[:Nz, Nz:])
    # Jordan form (19)
    e, v = np.linalg.eig(S)
    idx = np.argsort(np.abs(e))
    V2 = np.real(v[:, idx[Nz:]])  # Choose the eigenvectors with e > 1
    V12, V22 = V2[:Nz], V2[Nz:]
    # Matrix fraction (20)
    q, r = np.linalg.qr(V12)
    P = np.real(V22.dot(np.linalg.solve(r, q.T)))
    P = (P+P.T) / 2
    h = H
    Pht = P.dot(h.T)
    D = h.dot(Pht) + R  # prediction error variance
    q, r = np.linalg.qr(D)
    logdet = np.log(np.diag(r)).sum()
    Dinv = np.linalg.solve(r, q.T)
    K = (F.dot(Pht) + G).dot(Dinv)
    L = F - K.dot(h)
    vpred[:] = P
    Dinvs[:] = Dinv
    Ks[:] = K
    Ls[:] = L
    return logdet


if __name__ == "__main__":
    cc.distutils_extension()
    cc.compile()
