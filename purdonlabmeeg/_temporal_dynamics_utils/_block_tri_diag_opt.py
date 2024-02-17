# Author: Proloy Das <pdas6@mgh.harvard.edu>
"""
Implements stable inversion of block tridiagonal and banded matrices:
    [ A1  -B1          O     O]
    |-B1' A2  -B2      O     O|
M = |                         |
    |  O    -Bn-2'  An-1 -Bn-1|
    [  O   O    O  -Bn-1'   An]

Reference: Jain, Jitesh et. al., "Numerically Stable Algorithms for
Inversion of Block Tridiagonal and Banded Matrices" (2007). ECE Technical
Reports. Paper 357. http://docs.lib.purdue.edu/ecetr/357
"""
import numpy as np
from scipy import linalg
from numba import jit


def inverse_block_factors(Ai, Bi, use_numba=True):
    "returns Di, Si, and logdeterminant"
    assert len(Ai) == len(Bi) + 1, f"length of Ai = {len(Ai)} !=" \
                                   f"{len(Bi) + 1} = length of Bi + 1"
    assert Ai[0].shape == Bi[0].shape, f"Shapes of Ai {Ai[0].shape}," \
                                       f" Bi {Bi[0].shape} does not match"
    Si = np.empty((len(Bi),) + Bi[0].shape, Ai[0].dtype)
    Di = np.empty((len(Ai),) + Ai[0].shape, Ai[0].dtype)
    eye = np.eye(Ai[0].shape[0])

    if use_numba:
        logdet = _inverse_block_factors_opt(np.stack(Ai), np.stack(Bi), Si, Di, eye)
    else:
        logdet = 0
        temp = np.zeros_like(Ai[0])
        for i, (ai, bi) in enumerate(zip(Ai[1:][::-1], Bi[::-1])):
            temp = ai - temp  # A_i - S_i B_i.T
            ci, low = linalg.cho_factor(temp)
            logdet = logdet + 2 * np.log(np.diagonal(ci)).sum()
            ci_inv = linalg.cho_solve((ci, low), eye)
            Di[-i-1] = ci_inv.T
            Si[-i-1] = bi.dot(ci_inv)
            temp = np.dot(Si[-i-1], bi.T, out=temp)
        temp = Ai[0] - temp
        ci, low = linalg.cho_factor(temp)
        logdet = logdet + 2 * np.log(np.diagonal(ci)).sum()
        ci_inv = linalg.cho_solve((ci, low), eye)
        Di[0] = ci_inv.T

        for i, (bi, si) in enumerate(zip(Bi, Si)):
            temp = eye + bi.T.dot(Di[i].dot(si))
            Di[i+1] = np.dot(Di[i+1], temp)
    return Di, Si, logdet


@jit(nopython=True, cache=True, nogil=True)
def _inverse_block_factors_opt(Ai, Bi, Si, Di, eye):
    logdet = 0
    temp = np.zeros_like(Ai[0])
    for i, (ai, bi) in enumerate(zip(Ai[1:][::-1], Bi[::-1])):
        temp = ai - temp  # A_i - S_i B_i.T
        ci= np.linalg.cholesky(temp)
        logdet = logdet + 2 * np.log(np.diag(ci)).sum()
        ci_inv = np.linalg.solve(ci, eye)
        ci_inv = ci_inv.T.dot(ci_inv)
        Di[-i-1] = ci_inv.T
        Si[-i-1] = bi.dot(ci_inv)
        temp = np.dot(Si[-i-1], bi.T, out=temp)
    temp = Ai[0] - temp
    ci= np.linalg.cholesky(temp)
    logdet = logdet + 2 * np.log(np.diag(ci)).sum()
    ci_inv = np.linalg.solve(ci, eye)
    ci_inv = ci_inv.T.dot(ci_inv)
    Di[0] = ci_inv.T

    for i, (bi, si) in enumerate(zip(Bi, Si)):
        temp = eye + bi.T.dot(Di[i].dot(si))
        Di[i+1] = np.dot(Di[i+1], temp)
    return logdet


def Ainvx(Di, Si, x, use_numba=True):
    "returns A^{-1}x"
    assert len(Di) == len(Si) + 1, f"length of Di = {len(Di)} !=" \
                                   f"{len(Si) + 1} = length of Si + 1"
    assert Di[0].shape == Si[0].shape, f"Shapes of Di {Di[0].shape}," \
                                       f" Si {Si[0].shape} does not match"
    assert len(Di) == len(x), f"length of Di = {len(Di)} !=" \
                              f"{len(x)} = length of x"
    assert Di[0].shape[0] == x[0].shape[0], f"Dims of Di {Di[0].shape[0]}," \
                                            f" x {x[0].shape[0]} doesn't match"
    p = np.zeros_like(x)
    q = np.zeros_like(x)
    y = np.zeros_like(x)
    if use_numba:
        _Ainvx_opt(p, q, y, Di, Si, x)
    else:
        p[-1] = x[-1]
        for i, (si, xi) in enumerate(zip(Si[::-1], x[:-1][::-1])):
            p[-i-2] = xi + si.dot(p[-i-1])
        y[0] = Di[0].dot(p[0])
        q[0] = Si[0].T.dot(Di[0].dot(x[0]))
        for i, (di, si, pi, xi) in enumerate(zip(Di[1:-1], Si[1:],
                                                p[1:-1], x[1:-1])):
            q[i+1] = si.T.dot(q[i] + di.dot(xi))
            y[i+1] = di.dot(pi) + q[i]
        y[-1] = Di[-1].dot(p[-1]) + q[-2]
    return y


@jit(nopython=True, cache=True, nogil=True)
def _Ainvx_opt(p, q, y, Di, Si, x):
    p[-1] = x[-1]
    for i, (si, xi) in enumerate(zip(Si[::-1], x[:-1][::-1])):
        p[-i-2] = xi + si.dot(p[-i-1])
    y[0] = Di[0].dot(p[0])
    q[0] = Si[0].T.dot(Di[0].dot(x[0]))
    for i, (di, si, pi, xi) in enumerate(zip(Di[1:-1], Si[1:],
                                             p[1:-1], x[1:-1])):
        q[i+1] = si.T.dot(q[i] + di.dot(xi))
        y[i+1] = di.dot(pi) + q[i]
    y[-1] = Di[-1].dot(p[-1]) + q[-2]


def test_block_tri_diag_opt():
    from scipy import sparse
    from scipy.sparse import coo_matrix
    from codetiming import Timer
    Ai = [np.eye(2), np.eye(2), 0.8*np.eye(2)]
    Bi = [np.eye(2)*0.2, np.array([[0.1, 0.05], [0.0, 0.05]])]
    A = sparse.bmat([[coo_matrix(Ai[0]), coo_matrix(-Bi[0]), None],
                     [coo_matrix(-Bi[0]).T, coo_matrix(Ai[1]),
                      coo_matrix(-Bi[1])],
                     [None, coo_matrix(-Bi[1]).T, coo_matrix(Ai[2])]]
                    ).toarray()
    Ainv = linalg.inv(A)
    Di, Si, logdet = inverse_block_factors(Ai, Bi)
    E = Di[0].dot(Si[0])
    F = Di[0].dot(Si[0]).dot(Si[1])
    G = Di[1].dot(Si[1])

    Ainv_ = sparse.bmat([[coo_matrix(Di[0]), coo_matrix(E), coo_matrix(F)],
                         [coo_matrix(E).T, coo_matrix(Di[1]), coo_matrix(G)],
                         [coo_matrix(F).T, coo_matrix(G).T, coo_matrix(Di[2])]]
                        ).toarray()

    assert np.allclose(Ainv, Ainv_)
    assert np.allclose(np.log(np.linalg.det(A)), logdet)

    b = np.ones((3, 2))
    y = Ainv.dot(b.ravel())
    y_ = Ainvx(Di, Si, b)
    assert np.allclose(y_.ravel(), y)

    with Timer():
        for i in range(100):
            Di, Si, logdet = inverse_block_factors(Ai, Bi)
            y = Ainvx(Di, Si, b)


    with Timer():
        for i in range(100):
            Di_, Si_, logdet_ = inverse_block_factors(Ai, Bi, False)
            y_ = Ainvx(Di_, Si_, b, False)
    
    assert np.allclose(y_, y)
    assert logdet_ == logdet


if __name__ == '__main__':
    test_block_tri_diag_opt()
