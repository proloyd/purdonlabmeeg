# Author: Proloy Das <pdas6@mgh.harvard.edu>
import tqdm
import numpy as np
from scipy.sparse import coo_array
from scipy import linalg


def restricted_ml(y, X1, Q1s, Q2s, nu1, nu2, use_multiplicative_update=False,
                 max_iter=100, rtol=1e-15):
    """Empirical Bayesian solution to source-reconstruction 
    in MEG/EEG neuro-imaging.

    Bayesian Inference on the following model:
        :math: y       = X1 theta1 + e1,  e1 ~ N(0, C1)  
        :math: theta1 =  0  theta2 + e2,  e2 ~ N(0, C2)
    with the following constraints:
        :math: C1 = lambda1 Q11 + lambda2 Q12 +  ... + lambdaj Q1j
        :math: C2 = lambda1 Q21 + lambda2 Q22 +  ... + lambdai Q2i
    
        
    References: 
    ----------
    [1] Friston, K.J., et. al. 2002. “Classical and Bayesian Inference
    in Neuroimaging: Theory.” NeuroImage 16 (2): 465–83.
    https://doi.org/10.1006/nimg.2002.1090.
    [2] Phillips, Christophe et. al. 2005. “An Empirical Bayesian
    Solution to the Source Reconstruction Problem in EEG.” NeuroImage
    24 (4): 997–1011.
    https://doi.org/10.1016/j.neuroimage.2004.10.030.
    [3] Laporte F, Charcosset A, Mary-Huard T (2022) Efficient ReML 
    inference in variance component mixed models using a Min-Max
    algorithm. PLOS Computational Biology 18(1): e1009659.
    https://doi.org/10.1371/journal.pcbi.1009659


    Parameters:
    ----------
    y: 2d ndarray
        spatial map at sensor space
    X1: ndarray
        forward solution matrix a.k.a. lead-filed matrix 
    Q1s: list
        atoms that construct C1
    Q2s: list
        atoms that construct C2
    nu1: ndarray
        initial weights for the atoms in C1
    nu2: ndarray
        initial weights for the atoms in C2
    max_iter: int | 100 (default)
        maximum number of EM iterations
    
    NOTE: 
    ----
    Components, Q1k or Q2k are represented as `(data, (i, j))`,
    as in coo_array so that they can be readily converted to sparse
    matrices:
                `Qij = coo_array(Qij, shape=(m, m))`
    with (i, j) indices are indices wrt to y for Q1, X1 theta1 for Q2.

    `use_multiplicative_update=True` uses update rules based on a
    min-max algorithm described in [3].

    """
    n, m = X1.shape
    
    _C1s = _process_level1_atoms(Q1s, n)
    _C2s = _process_level2_atoms(X1, Q2s)
    _C1s = np.stack(_C1s)
    _C2s = np.stack(_C2s)
    _Cs = np.concatenate((_C1s, _C2s), axis=0)
    # TODO: Debug the following normalization
    # This is forcing the initialization of nu to be 0 for some reason.
    traces = np.einsum('...ii', _Cs)
    _Cs /= traces[:, None, None]

    if y.ndim == 3:
        y = np.swapaxes(y, 0, 1)
        orig_shape = y.shape[1:]
        y = np.reshape(y, (y.shape[0], -1))
    else:
        orig_shape = None
    yyt = y.dot(y.T) / y.shape[-1]

    if nu1 is None or nu2 is None:
        nu = initialize_nu(_Cs, yyt)
    else:
        assert nu1 == len(_C1s)
        assert nu2 == len(_C2s)
        nu = np.concatenate((nu1, nu2), axis=0)
    nu[nu<=0] = nu[nu>0].min() / 100
    pivot = len(_C1s)
    active = np.ones(len(nu), np.bool)

    free_energies = np.zeros(max_iter, np.float64)
    for i in tqdm.tqdm(range(max_iter)):
        active_nu = nu[active]
        active_Cs = _Cs[active]
        # Compute the recent data covariance under the model, its
        # inverse and log-determinant
        Cy = _calculate_Cy(active_nu, active_Cs)
        try:
            Cy_inv = linalg.inv(Cy)
        except linalg.LinAlgError:
            import ipdb; ipdb.set_trace()
        _, logdet = np.linalg.slogdet(Cy)
        
        # Finally compute the recent free energy, which is identical 
        # to the data log-likelihood under the model
        free_energies[i] = -0.5 * (logdet + (Cy_inv * yyt).sum())
        # print(free_energies[i], len(active_nu))

        # Check if free energy is changing beyond the relative
        # tolerence 
        if i > 2 and np.abs((free_energies[i] - free_energies[i-1])) < rtol:
            break
        
        if use_multiplicative_update:
            active_nu = multiplicative_update_rule(yyt, active_nu, active_Cs, Cy_inv)
        else:
            active_nu = grad_descent_update_rule(yyt, active_nu, active_Cs, Cy_inv)

        nu[active] = active_nu
        if (nu < rtol * nu.max()).any():
            # print('inactive source')
            pass
        active[nu < np.sqrt(rtol) * nu.max()] = False
        nu[~active] = 0

    # Finally Compute the Inverse mapping of the data
    Cy = _calculate_Cy(nu, _Cs)
    nu1, nu2 =  np.split(nu, [pivot,])
    C1 = _calculate_Cy(nu1, _C1s)
    inv_map = _constuct_inv_map(nu2, Q2s, X1)
    whitened_y = linalg.solve(Cy, y)
    theta1 = inv_map.dot(whitened_y)
    e1 = C1.dot(whitened_y)
    if orig_shape:
        theta1, e1 = [np.reshape(elem, (-1,) + orig_shape) for elem in (theta1, e1)]
        theta1, e1 = [np.swapaxes(elem, 0, 1) for elem in (theta1, e1)]
    print(nu)
    print(free_energies[i])
    return nu1, nu2, theta1, e1


def initialize_nu(_Cs, yyt):
    "initializes nu"
    # Uses the whole matrix
    Cs = np.reshape(_Cs, (_Cs.shape[0], -1))
    y = yyt.ravel()

    # Uses only the diag
    Cs = np.einsum('...ii->...i', _Cs)
    y = np.einsum('ii->i', yyt)

    Sigma = Cs.dot(Cs.T)
    b = Cs.dot(y)
    try:
        nu = linalg.solve(Sigma, b)
        use_nnls = False 
    except linalg.LinAlgError:
        from scipy.optimize import nnls
        nu, _ = nnls(Sigma, b)
        use_nnls = True 

    if (nu < 0).any() or ((nu > 1e5).any() and use_nnls is False):
        from scipy.optimize import nnls
        nu, _ = nnls(Sigma, b)
    return nu


def grad_descent_update_rule(yyt, nu, Cs, Cy_inv):
    # Compute the new descent direction and take a full step
    d = get_ascent_direction(Cs, yyt, Cy_inv)
    next_nu = nu + d
    while (next_nu < 0).any():
        d /= 2
        next_nu = nu + d
    return next_nu


def multiplicative_update_rule(yyt, nu, Cs, Cy_inv):
    "uses update rules based on a min-max algorithm described in [3] above."
    temp1 = Cy_inv
    temp2 = Cy_inv.dot(yyt).dot(Cy_inv)
    term1, term2 = [np.sum(temp[None, :, :] * Cs, axis=(-2, -1))
                    for temp in (temp1, temp2)]
    factor = np.sqrt(term2 / term1)
    factor[np.isnan(factor)] = 0.
    if not np.count_nonzero(factor):
        import ipdb; ipdb.set_trace()
    return nu * factor


def get_ascent_direction(_Cs, yyt, Cy_inv):
    g = _calculate_gi(yyt, Cy_inv, _Cs)
    H = _calculate_Hij(Cy_inv, _Cs)
    d = linalg.solve(H, g)
    return d
    
       
def _calculate_Cy(nu, Cs):
    return np.sum(nu[:, None, None] * Cs, axis=0)


def _calculate_gi(yyt, Cy_inv, Cs):
    temp1 = Cy_inv
    temp2 = Cy_inv.dot(yyt).dot(Cy_inv)
    term1, term2 = [np.sum(temp[None, :, :] * Cs, axis=(-2, -1))
                    for temp in (temp1, temp2)]
    return 0.5 * (-term1 + term2)
    

def _calculate_Hij(Cy_inv, Cs):
    PQP11 = np.swapaxes(Cy_inv.dot(Cs.dot(Cy_inv)), 0, 1)
    # PQP22 =  not needed
    Hijs = [0.5 * np.sum(PQP11 * this_C, axis=(-2, -1)) for this_C in Cs]
    H = np.vstack(Hijs)
    return H


def _process_level2_atoms(X1, Q2s):
    _C2s = [_compute_inner_prod(X1, this_Q2) for this_Q2 in Q2s]
    return _C2s


def _process_level1_atoms(Q1s, n):
    _C1s = [coo_array(this_Q1, shape=(n, n)).toarray() for this_Q1 in Q1s]
    return _C1s


def _compute_inner_prod(X1, Q2):
    """returns X1 prod Q2 prod X1^T"""
    m = X1.shape[1]
    Q2 = coo_array(Q2, shape=(m, m)).tocsr()
    temp = Q2.dot(X1.T)
    return X1.dot(temp)     # TODO: try optimizing this. 


def _get_initial_coeffs(X1, Q2):
        m, n = X1.shape
        Q2 = coo_array(Q2, shape=(m, m)).tocsr()
        temp = Q2.dot(X1)
        return np.sum(X1.T * temp.T) / n / Q2.trace()


def _constuct_inv_map(nu2, Q2s, X1):
    m = X1.shape[1]
    IM = np.zeros_like(X1.T)
    for this_nu, this_Q2 in zip(nu2, Q2s):
        Q2 = coo_array(this_Q2, shape=(m, m))
        IM += this_nu * Q2.tocsr().dot(X1.T)
    return IM


def plot_comparison(n2, theta1, name=None):
    from matplotlib import pyplot as plt
    from matplotlib import colors
    
    fig, axes = plt.subplots(2)
    images = []
    for to_plot, title, ax in zip((n2, theta1), ('ground truth', 'Baeysian Inference'), axes):
        images.append(ax.matshow(to_plot))
        ax.set_title(title)
    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for im in images:
        im.set_norm(norm)
    
    fig.colorbar(images[0], ax=axes, orientation='vertical', fraction=.1)
    
    def update(changed_image):
        for im in images:
            if (changed_image.get_cmap() != im.get_cmap()
                    or changed_image.get_clim() != im.get_clim()):
                im.set_cmap(changed_image.get_cmap())
                im.set_clim(changed_image.get_clim())

    for im in images:
        im.callbacks.connect('changed', update)

    if name is None:
        fig.show()
    else:
        fig.savefig(name)



def test_restricted_ml(seed=123456):
    from numpy.random import default_rng
    rng = default_rng(seed)
    tmax = 100
    ny = 5
    nx = 10

    X1 = rng.standard_normal(size=(ny, nx)) / np.sqrt(nx)
    # X1 = np.eye(nx)
    e1 = rng.standard_normal(size=(ny, tmax)) 
    e2 = rng.standard_normal(size=(nx, tmax))

    Q1s = [
        (np.array([0.1,]), (np.array([0,]),np.array([0,]))),
        (np.array([0.1, -0.005, 0.1, -0.005]), (np.array([1, 1 , 2, 2]), np.array([1, 2, 2, 1]))),
        (np.ones(ny-3), (np.arange(3, ny), np.arange(3, ny)))
    ]

    Q2s = [
        (np.array([0.5, -0.25, 0.5, -0.25]), (np.array([1, 1 , 2, 2]), np.array([1, 2, 2, 1]))),
        
        (np.array([    1,  -0.5, 0.25,  0.05,
                    -0.5,     1, -0.5, -0.25,
                    0.25,  -0.5,   1,    0.5,
                    0.05, -0.25,  0.5,     1]), 
         (np.array([6, 6, 6, 6,
                    7, 7, 7, 7,
                    8, 8, 8, 8,
                    9, 9, 9, 9]), 
        np.array([6, 7, 8, 9,
                  6, 7, 8, 9,
                  6, 7, 8, 9,
                  6, 7, 8, 9,]))),
        
        (np.array([    1,  -0.5, 0.25,  0.05,
                    -0.5,     1, -0.5, -0.25,
                    0.25,  -0.5,   1,    0.5,
                    0.05, -0.25,  0.5,     1]), 
        (np.array([0, 0, 0, 0,
                   3, 3, 3, 3,
                   4, 4, 4, 4,
                   5, 5, 5, 5]), 
        np.array([0, 3, 4, 5,
                  0, 3, 4, 5,
                  0, 3, 4, 5,
                  0, 3, 4, 5])))
    ]

    nu1 = np.array([1., 1., 1.])
    nu2 = np.array([0.00, 1., 0.00])
    C1 = np.sum(np.stack([coo_array(Q1, shape=(ny, ny)).toarray() for Q1 in Q1s], axis=0) * nu1[:, None, None], axis=0)
    C2 = np.sum(np.stack([coo_array(Q2, shape=(nx, nx)).toarray() for Q2 in Q2s], axis=0) * nu2[:, None, None], axis=0)

    ns = []
    for C, e in zip([C1, C2], [e1, e2]):
        u, s, v = linalg.svd(C)
        ns.append(u.dot(np.sqrt(s)[:, None] * e))
    n1, n2 = ns

    # Check individual functions
    _process_level1_atoms(Q1s, ny)
    _compute_inner_prod(X1, Q2s[0])
    _process_level2_atoms(X1, Q2s)

    y = X1.dot(n2) + n1

    # nu_guess = rng.uniform(size=len(Q1s) + len(Q2s))
    nu_guess = np.ones(len(Q1s) + len(Q2s), np.float64)
    nu1_guess, nu2_guess = np.split(nu_guess, [nu1.size])
    nu1, nu2, theta1, e1 = restricted_ml(y, X1, Q1s, Q2s, nu1_guess, nu2_guess,
                                        rtol=1e-4)

    plot_comparison(n2, theta1, 'compn2-1')
    plot_comparison(n1, e1, 'compn1-1')

    nu1, nu2, theta1, e1 = restricted_ml(y, X1, Q1s, Q2s, nu1_guess, nu2_guess,
                                         use_multiplicative_update=True, rtol=1e-4)

    plot_comparison(n2, theta1, 'compn2-2')
    plot_comparison(n1, e1, 'compn1-2')

