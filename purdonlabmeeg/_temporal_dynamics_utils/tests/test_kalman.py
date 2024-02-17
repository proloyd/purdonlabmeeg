import numpy as np
from codetiming import Timer

from purdonlabmeeg._temporal_dynamics_utils._kalman_smoother import (
                                                kalcvf, kalcvs,
                                                kalman_smoother,
                                                ss_kalcvf,
                                                ss_kalcvs)

from purdonlabmeeg._temporal_dynamics_utils._temporal_filtering import (sskf,
                                                                        stableblockfiltering)


def test_kf():
    from numpy.random import default_rng
    rng = default_rng(seed=0)
    w = rng.standard_normal(1000)
    v = rng.standard_normal((1, 1000))
    u = np.empty_like(w)
    z = np.empty_like(w)
    u[0] = rng.standard_normal()
    z[0] = rng.standard_normal()
    for k in range(1, 1000):
        u[k] = 0.7 * u[k-1] + w[k-1]
        z[k] = 0.8 * z[k-1] + u[k-1]
    x = np.vstack((u, z))
    # x_noisy = x + x.std(axis=1)[:, None] * 0.1 * v
    f = np.asanyarray([[1, 1]]) / np.sqrt(2)
    y = f.dot(x)
    y_noisy = y + y.std(axis=-1)[:, None] * 0.1 * v
    # x_noisy = linalg.solve(f, y_noisy)

    a = np.asanyarray([[0.7, 0], [1, 0.8]])
    q = np.asanyarray([[1, 0], [0, 0]])
    r = np.diag(1 * y.var(axis=1))
    # f = np.eye(2)

    F = a.copy()
    H = f.copy()
    a = np.zeros(2)
    b = np.zeros(1)
    var = np.zeros((3, 3))
    var[:2, :2] = q
    var[2:, 2:] = r
    timer1 = Timer(name='numba', text='numba Elapsed time: {:0.4f} seconds')
    timer2 = Timer(name='no-numba', text='no-numba Elapsed time: {:0.4f} seconds')

    mu0, Q0 = np.zeros(2), q
    for i in range(10):
        print('Vanilla:')
        print('--------')
        with timer1:
            out_numba = kalcvf(y_noisy, 1, a, F, b, H, var, return_pred=True,
                         return_filt=False, return_L=True, return_Dinve=True,
                         return_Dinv=True, return_K=True)
            # out_sm = kalcvs(y_noisy, a, F, b, H, var, out['pred'], out['vpred'],
            #                 Dinvs=out['Dinv'],
            #                 Dinves=out['Dinve'],
            #                 Ls=out['L'], use_numba=True)
            # out = kalman_smoother(F, q, mu0, Q0, H, r, y, ss=False,
            #                       use_numba=False)
            
        # print('Steady State:')
        # print('-------------')
        # with timer1:
        #     out_sm = kalman_smoother(F, q, mu0, Q0, H, r, y, ss=True,
        #                              use_numba=False)

        # print('Vanilla:')
        # print('--------')
        # with timer2:
        #     out_ = kalman_smoother(F, q, mu0, Q0, H, r, y, ss=False,
        #                            use_numba=True)

        print('Steady State:')
        print('-------------')
        with timer2:
            # out_ss_ = kalman_smoother(F, q, mu0, Q0, H, r, y, ss=True,
            #                           use_numba=True)

            out = kalcvf(y_noisy, 1, a, F, b, H, var, return_pred=True,
                         return_filt=False, return_L=True, return_Dinve=True,
                         return_Dinv=True, use_numba=False, return_K=True)
            # out_sm = kalcvs(y_noisy, a, F, b, H, var, out['pred'], out['vpred'],
            #                 Dinvs=out['Dinv'],
            #                 Dinves=out['Dinve'],
            #                 Ls=out['L'], use_numba=False)

        print(np.allclose(out_numba['Ks'], out['Ks']))
    # import ipdb; ipdb.set_trace()
    # print(out['ll'])
    # print('Convergent\n')
    # with timer1:
    #     out_ss = ss_kalcvf(y_noisy, 1, a, F, b, H, var, return_pred=True,
    #                        return_filt=False, return_L=True, return_Dinve=True,
    #                        return_Dinv=True, use_numba=True)
    #     out_ss_sm = ss_kalcvs(y_noisy, a, F, b, H, var, out_ss['pred'],
    #                           out_ss['vpred'], Dinvs=out_ss['Dinv'],
    #                           Dinves=out_ss['Dinve'],
    #                           Ls=out_ss['L'], use_numba=True)
    #
    # with timer2:
    #     out_ss = ss_kalcvf(y_noisy, 1, a, F, b, H, var, return_pred=True,
    #                        return_filt=False, return_L=True, return_Dinve=True,
    #                        return_Dinv=True, use_numba=False)
    #     out_ss_sm = ss_kalcvs(y_noisy, a, F, b, H, var, out_ss['pred'],
    #                           out_ss['vpred'], Dinvs=out_ss['Dinv'],
    #                           Dinves=out_ss['Dinve'],
    #                           Ls=out_ss['L'], use_numba=False)
    # print(out_ss['ll'])
    # print(out['Dinv'][..., 5:10])
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2)
    for i, ax in enumerate(axes):
        # ax.plot(out['filt'][i], label='filt')
        ax.plot(out['pred'][i][:-1], label='pred')
        ax.plot(out['sm'][i][:-1], label='sm')
        ax.plot(out_ss_['pred'][i][:-1], label='ss-pred')
        ax.plot(out_ss_['sm'][i][:-1], label='ss-sm')
        ax.plot(x[i], label='True')
        # ax.plot(x_noisy[i], label='noisy')
        ax.legend()
    fig.show()
    fig, ax = plt.subplots()
    ax.plot(out_ss_['sm'][i][:-1] - out['sm'][i][:-1], label='pred')
    ax.plot(out_ss_['pred'][i][:-1] - out['pred'][i][:-1], label='pred')
    ax.legend()
    fig.show()
    """

def time_kalman():
    import os
    dir_path = os.path.dirname(os.path.realpath(__file__))
    import scipy.io
    # results = scipy.io.loadmat(os.path.join(dir_path, "runtime_inputs.mat"))
    results = scipy.io.loadmat("/autofs/cluster/purdonlab/users/alexhe/djkalman/runtime_testing/runtime_inputs.mat")
    for elem in ('__header__', '__version__', '__globals__'):
        print(f'{elem}: {results.pop(elem)}')
    results['mu0'] = np.squeeze(results['mu0'])
    for elem in results.keys():
        results[elem] = results[elem].copy()
    with Timer():
        out = kalman_smoother(**results, ss=True, use_numba=True)
    return out



def test_sskf():
    from numpy.random import default_rng
    from scipy import linalg
    rng = default_rng(seed=0)
    w = rng.standard_normal(1000)
    v = rng.standard_normal((2, 1000))
    u = np.empty_like(w)
    z = np.empty_like(w)
    u[0] = rng.standard_normal()
    z[0] = rng.standard_normal()
    for k in range(1, 1000):
        u[k] = 0.7 * u[k-1] + w[k-1]
        z[k] = 0.8 * z[k-1] + u[k-1]
    x = np.vstack((u, z))
    # x_noisy = x + x.std(axis=1)[:, None] * 0.1 * v
    f = np.asanyarray([[1, 1], [1, -1]]) / np.sqrt(2)
    y = f.dot(x)
    y_noisy = y + y.std(axis=1)[:, None] * 0.1 * v
    x_noisy = linalg.solve(f, y_noisy)

    xs = (np.zeros_like(x).T, np.zeros_like(x).T)

    a = np.asanyarray([[0.7, 0], [1, 0.8]])
    q = np.asanyarray([[1, 0], [0, 0]])
    r = np.diag(0.01 * y.var(axis=1))
    # f = np.eye(2)
    delta = np.zeros(2)
    c = (10 * np.diag(1/y.std(axis=1))).dot(f)

    out = sskf(y_noisy.T, a, f, q, r, c, delta, None, initial_cond=0.0)

    assert np.all((out[0] - x.T).var(axis=0) < (x_noisy.T - x.T).var(axis=0))

    q = np.asanyarray([[1, 0], [0, 0.00001]])
    out1 = stableblockfiltering(y.T, a, f, q, r, delta, xs, 0.0,
                               mixing_mat_inst=None,
                               return_pred=False, compute_logdet=False)
    assert np.all((out1[0] - x.T).var(axis=0) < (x_noisy.T - x.T).var(axis=0))

    mu0, Q0 = np.zeros(2), q
    out2 = kalman_smoother(a, q, mu0, Q0, f, r, y, ss=False,
                           use_numba=True)
    assert np.all((out2['sm'].T - x.T).var(axis=0) < (x_noisy.T - x.T).var(axis=0))

    out3 = kalman_smoother(a, q, mu0, Q0, f, r, y, ss=True,
                           use_numba=True)
    assert np.all((out3['sm'].T - x.T).var(axis=0) < (x_noisy.T - x.T).var(axis=0))

    # assert np.allclose(out[0], out1[0])

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1)
    # ax.plot(out[0][:,1:], label='sskf')
    ax.plot(out2['sm'].T[:,1:], linestyle='--', label='numba')
    ax.plot(out3['sm'].T[:,1:], linestyle='-.', label='ss-kf')
    ax.plot(out1[0][:,1:], alpha=0.5, linewidth=1.2, label='stable-block-filter')
    ax.plot(x.T[:,1:], alpha=0.35, linewidth=2, label='True')
    ax.plot(x_noisy.T[:,1:], linestyle=':', label='noisy')
    ax.legend()
    ax.set_xlim([400, 450])
    fig.savefig('test.svg')
