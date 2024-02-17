from mne.utils import logger
try:
    from .kalman_opt import (kalcvf_numba, kalcvs_numba,
                             ss_kalcvf_numba, ss_kalcvs_numba)
    logger.debug('using the cc one')
except Exception:
    from .kalman import (kalcvf_numba, kalcvs_numba,
                         ss_kalcvf_numba, ss_kalcvs_numba)
    logger.debug('using the njit one')

from .kalman import vcvg_kalcvf_numba
