import numpy as np
from scipy import linalg


def polyeig(*A):
    """
    Solve the polynomial eigenvalue problem:
        (A0 + e A1 +...+  e**p Ap)x=0 

    Return the eigenvectors [x_i] and eigenvalues [e_i] that are solutions.

    Usage:
        X, e = polyeig(A0, A1, ..., Ap)

    Most common usage, to solve a second order system: (K + C e + M e**2) x =0
        X, e = polyeig(K, C, M)

    """
    if len(A) <= 0:
        raise Exception('Provide at least one matrix')
    for Ai in A:
        if Ai.shape[0] != Ai.shape[1]:
            raise Exception('Matrices must be square')
        if Ai.shape != A[0].shape:
            raise Exception('All matrices must have the same shapes')

    n = A[0].shape[0]
    l = len(A) - 1
    # Assemble matrices for generalized problem
    C = np.block([
        [np.zeros((n*(l-1), n)), np.eye(n*(l-1))],
        [-np.column_stack(A[0:-1])]
        ])
    D = np.block([[np.eye(n*(l-1)), np.zeros((n*(l-1), n))],
                  [np.zeros((n, n*(l-1))), A[-1]]])
    # Solve generalized eigenvalue problem
    e, X = linalg.eig(C, D)
    if np.all(np.isreal(e)):
        e = np.real(e)
    X = X[:n, :]

    # # Sort eigenvalues/vectors
    # I = np.argsort(e)
    # X = X[:,I]
    # e = e[I]

    # Scaling each mode by max
    X /= np.tile(np.max(np.abs(X), axis=0), (n, 1))

    return X, e


if __name__ == '__main__':
    M = np.diag([3, 1, 3, 1])
    C = np.array([[0.4, 0, -0.3, 0],
                  [0, 0, 0, 0],
                  [-0.3, 0, 0.5, -0.2],
                  [0, 0, -0.2, 0.2]])
    K = np.array([[-7, 2, 4, 0],
                  [2, -4, 2, 0],
                  [4, 2, -9, 3],
                  [0, 0, 3, -3]])
    X, e = polyeig(K, C, M)
    print('X:\n', X)
    print('e:\n', e)
    # Test that first eigenvector and value satisfy eigenvalue problem:
    s = e[0]
    x = X[:, 0]
    res = (M * s**2 + C * s + K).dot(x)  # residuals
    assert(np.all(np.abs(res) < 1e-12))
