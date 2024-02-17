import numpy as np

from numpy.random import SeedSequence, default_rng
from joblib import Parallel, delayed
from multiprocessing import cpu_count
try:
    from scipy import linalg
except ImportError:
    from numpy import linalg


def _pa_eigs(p, n, stream,):
    vals = stream.standard_normal((p, n))
    vals_norm = (vals * vals).sum(axis=1) ** 0.5
    vals = vals / vals_norm[:, None]
    corr = np.dot(vals, vals.T) / vals.shape[1]
    eigs = linalg.eigvalsh(corr)[::-1]
    return eigs


def parallel_analysis(p, n, repeat=100, seed=12345, n_jobs=-1):
    """returns the eigenvalues from parallel analysis

    Reference:
    [1] Franklin SB, Gibson DJ, Robertson PA, Pohlmann JT, Fralish JS. Parallel
        Analysis: a method for determining significant principal components.
        J Veg Sci. 1995;6(1):99–106.
    """
    ss = SeedSequence(seed)
    child_seeds = ss.spawn(repeat)
    streams = [default_rng(s) for s in child_seeds]
    # serial implementation
    # eigvals = [_pa_eigs(p, n, streams[i]) for i in range(repeat)]
    # Parallel implementation
    n_jobs = cpu_count() // 2 if n_jobs == -1 else min(cpu_count() // 2, n_jobs)
    eigvals = Parallel(n_jobs=n_jobs)(delayed(_pa_eigs)(p, n, streams[i])
                                      for i in range(repeat))
    eigvals = np.stack(eigvals)
    return eigvals
