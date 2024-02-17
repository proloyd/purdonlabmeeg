import numpy as np
from scipy.signal import fftconvolve
from purdonlabmeeg.sserp._ss_erp import _fit_state_erps, _fit_erp


def erpdata(times, seed=0):
    """Function to generate random source time courses

    copied from:
    https://mne.tools/dev/auto_examples/simulation/simulate_evoked_data.html"""
    return (7 * np.sin(30 * times) *
            np.exp(- (times - 0.15 + 0.05 * seed) ** 2 / 0.01))


def stateerpdata(times, seed=0):
    """Function to generate random source time courses

    copied from:
    https://mne.tools/dev/auto_examples/simulation/simulate_evoked_data.html"""
    return (7 * np.sin(3 * times) *
            np.exp(- (times - 0.15 + 0.05 * seed) ** 2 / 0.01))


def test_sserp(seed=42):
    from numpy.random import default_rng
    from math import sqrt
    from matplotlib import pyplot as plt
    rng = np.random.default_rng(seed)

    sfreq = 80 # Hz
    ntimes = int(np.round(sfreq * 60.))
    times = np.arange(0.5*sfreq, dtype=np.float64) / sfreq - 0.1

    # State-params
    phi = np.array(
        [[0.9649,   -0.0870,         0,         0],
         [0.0870,    0.9649,         0,         0],
         [     0,         0,    0.7294,   -0.6243],
         [     0,         0,    0.6243,    0.7294]]
    )
    q = np.diag(np.array([15.2061, 15.2061, 11.5350, 11.5350]))
    a = np.vstack([stateerpdata(times, rng.normal(size=1)) for _ in range(4)])
    a *= np.array([0, 0, 0, 0])[:, None]

    # Observation-params
    M = np.array([1, 0, 1, 0])
    b = 2 * erpdata(times, rng.normal(1))
    rng = default_rng(seed=0)
    w = rng.normal(size=(1, ntimes))
    v = np.sqrt(q).dot(rng.normal(size=(4, ntimes)))
    
    events1 = np.zeros(ntimes)
    events1[np.arange(100, ntimes, 200) + rng.integers(-25, 25)] = 1
    offset = int(0.1*sfreq)
    state_erp_data = fftconvolve(np.vstack([events1]*4), a, 'full', axes=-1)[:,offset:events1.shape[-1]+offset]
    obsrv_erp_data = fftconvolve(events1, b, 'full')[offset:events1.shape[-1]+offset]

    x = np.empty((4, ntimes+1),)
    x[:, 0] = rng.normal(4)
    for k in range(1, ntimes+1):
        x[:, k] = phi.dot(x[:, k-1]) + state_erp_data[:, k-1] + v[:, k-1]
    y = M.dot(x[:, 1:]) + obsrv_erp_data[None, :]
    data = y + 4 * w 
    print(f'noise_variance: {w.std() ** 2}')
    fig, [ax, ax1, ax2] = plt.subplots(nrows=3, ncols=1, figsize=[12.5, 7.5])
    lines = ax.plot(np.arange(ntimes) / sfreq, x[::2, 1:].T)
    ax.legend(lines, ['osc 1', 'osc 2'])
    ax.set_frame_on(False)
    ax.grid(True, linestyle='-.', linewidth=0.2)
    ax.tick_params(left=False, bottom=False)
    ax.set_xlabel('time (in s)')
    ax.set_ylabel('amplitude (in a.u.)')

    lines = ax1.plot(np.arange(ntimes) / sfreq, state_erp_data.T)
    lines.extend(ax1.plot(np.arange(ntimes) / sfreq, obsrv_erp_data))
    ax1.legend(lines, ['osc 1x', 'osc 1y', 'osc 2x', 'osc 2y', 'obsrv'])
    # ax1.plot(np.arange(ntimes) / sfreq, events, 'r|')
    ax1.set_frame_on(False)
    ax1.grid(True, linestyle='-.', linewidth=0.2)
    ax1.tick_params(left=False, bottom=False)
    ax1.set_xlabel('time (in s)')
    ax1.set_ylabel('amplitude (in a.u.)')
    ax1.set_title('simulated ERP')

    lines = ax2.plot(np.arange(ntimes) / sfreq, y.T, label='y')
    lines = ax2.plot(np.arange(ntimes) / sfreq, y.T, label='y + noise')
    ax2.legend()
    ax2.set_frame_on(False)
    ax2.grid(True, linestyle='-.', linewidth=0.2)
    ax2.tick_params(left=False, bottom=False)
    ax2.set_xlabel('time (in s)')
    ax2.set_ylabel('amplitude (in a.u.)')
    plt.show(block=False)
    fig.tight_layout()
    fig.savefig('sim.svg')

    out = _fit_state_erps(data, sfreq, [events1], 32, a=1, n_osc=2, ar_order=17,
             initial_guess=None, debug=True)

    fig, axes = plt.subplots(5)
    out[1][0].plot(confidence=.9, sfreq=out[-1], ax=axes[0], alpha=0.5)
    axes[0].plot(times, b, 'g')
    axes[0].plot(times[8:], sum(out[5][0]).T/ len(out[5][0]), 'k')
    out[1][0].plot(confidence=.9, sfreq=out[-1], ax=axes[0], alpha=0.5)
    [fir.plot(confidence=.9, sfreq=out[-1], ax=ax, alpha=0.5) for fir, ax in zip(out[1][1], axes[1:])]
    [ax.plot(times, this_a, 'g') for this_a, ax in zip(a, axes[1:])]
    fig.savefig('sim-(1).svg')
    return out

def test_sserp_2(seed=42):
    from numpy.random import default_rng
    from math import sqrt
    from matplotlib import pyplot as plt
    rng = np.random.default_rng(seed)

    sfreq = 80 # Hz
    ntimes = int(np.round(sfreq * 20.))
    times = np.arange(0.5*sfreq, dtype=np.float64) / sfreq - 0.1

    # State-params
    phi = np.array(
        [[0.9649,   -0.0870,         0,         0],
         [0.0870,    0.9649,         0,         0],
         [     0,         0,    0.7294,   -0.6243],
         [     0,         0,    0.6243,    0.7294]]
    )
    q = np.diag(np.array([15.2061, 15.2061, 11.5350, 11.5350]))
    a = np.vstack([stateerpdata(times, rng.normal(size=1)) for _ in range(4)])
    a *= np.array([0, 0, 0, 0])[:, None]

    # Observation-params
    M = np.array([1, 0, 1, 0])
    b = 2. * erpdata(times, rng.normal(1))
    rng = default_rng(seed=0)
    w = rng.normal(size=(1, ntimes))
    v = np.sqrt(q).dot(rng.normal(size=(4, ntimes)))
    
    events1 = np.zeros(ntimes)
    events1[np.arange(100, ntimes, 180)] = 1
    offset = int(0.1*sfreq)
    state_erp_data = fftconvolve(np.vstack([events1]*4), a, 'full', axes=-1)[:,offset:events1.shape[-1]+offset]
    obsrv_erp_data = fftconvolve(events1, b, 'full')[offset:events1.shape[-1]+offset]

    x = np.empty((4, ntimes+1),)
    x[:, 0] = rng.normal(4)
    for k in range(1, ntimes+1):
        x[:, k] = phi.dot(x[:, k-1]) + state_erp_data[:, k-1] + v[:, k-1]
    y = M.dot(x[:, 1:]) + obsrv_erp_data[None, :]
    data = y + 4 * w 
    print(f'noise_variance: {w.std() ** 2}')
    fig, [ax, ax1, ax2] = plt.subplots(nrows=3, ncols=1, figsize=[12.5, 7.5])
    lines = ax.plot(np.arange(ntimes) / sfreq, x[::2, 1:].T)
    ax.legend(lines, ['osc 1', 'osc 2'])
    ax.set_frame_on(False)
    ax.grid(True, linestyle='-.', linewidth=0.2)
    ax.tick_params(left=False, bottom=False)
    ax.set_xlabel('time (in s)')
    ax.set_ylabel('amplitude (in a.u.)')

    lines = ax1.plot(np.arange(ntimes) / sfreq, state_erp_data.T)
    lines.extend(ax1.plot(np.arange(ntimes) / sfreq, obsrv_erp_data))
    ax1.legend(lines, ['osc 1x', 'osc 1y', 'osc 2x', 'osc 2y', 'obsrv'])
    # ax1.plot(np.arange(ntimes) / sfreq, events, 'r|')
    ax1.set_frame_on(False)
    ax1.grid(True, linestyle='-.', linewidth=0.2)
    ax1.tick_params(left=False, bottom=False)
    ax1.set_xlabel('time (in s)')
    ax1.set_ylabel('amplitude (in a.u.)')
    ax1.set_title('simulated ERP')

    lines = ax2.plot(np.arange(ntimes) / sfreq, y.T, label='y')
    lines = ax2.plot(np.arange(ntimes) / sfreq, y.T, label='y + noise')
    ax2.legend()
    ax2.set_frame_on(False)
    ax2.grid(True, linestyle='-.', linewidth=0.2)
    ax2.tick_params(left=False, bottom=False)
    ax2.set_xlabel('time (in s)')
    ax2.set_ylabel('amplitude (in a.u.)')
    plt.show(block=False)
    fig.tight_layout()
    fig.savefig('sim.svg')

    out = _fit_erp(data, sfreq, [events1], 32, a=1, n_osc=2, ar_order=17,
             initial_guess=None, debug=True)

    fig, axes = plt.subplots(5)
    out[1].plot(confidence=.9, sfreq=out[-1], ax=axes[0], alpha=0.5)
    axes[0].plot(times, b, 'g')
    axes[0].plot(times[8:], sum(out[5][0]).T/ len(out[5][0]), 'k')
    out[1].plot(confidence=.9, sfreq=out[-1], ax=axes[0], alpha=0.5)
    # [fir.plot(confidence=.9, sfreq=out[-1], ax=ax, alpha=0.5) for fir, ax in zip(out[1][1], axes[1:])]
    [ax.plot(times, this_a, 'g') for this_a, ax in zip(a, axes[1:])]
    fig.savefig('sim-(3).svg')
    return out