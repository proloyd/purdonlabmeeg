from ._scale_update import (costf, gradf, _solve_for_qw,
                            update_spatial_scales, spatial_filter,
                            cg_solver)
from ._sparse_map import (update_sparse_spatial_hyperparameters,
                          sparse_spatial_filter)
from ._bb_utils import stab_bb
from ._kernels import (SquaredExponentailGPKernel, MaternGPKernel,
                       MaternGPKernel2, GammaExpGPKernel)
from . import _sensor_maps as sensor_map
