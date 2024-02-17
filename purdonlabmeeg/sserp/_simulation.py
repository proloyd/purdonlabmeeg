import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import signal
from mne import time_frequency
from purdonlabmeeg.sserp import fit_erp, confidence_interval
from purdonlabmeeg._temporal_dynamics_utils.tests._generate_data import (
                                                                   ARData
                                                                   )


def erpdata(times, seed=0):
    """Function to generate random source time courses

    copied from:
    https://mne.tools/dev/auto_examples/simulation/simulate_evoked_data.html"""
    return (50e-9 * np.sin((25 + 0.1*np.random.randn(len(times))) * times) *
            np.exp(- (times - 0.15 + 0.05 * seed) ** 2 / 0.01))


debug = False
rng = np.random.RandomState(84)
sfreq = 100     # in Hz
ntimes = int(np.round(sfreq * 50.))
slow_data = ARData(ntimes + 10, noise_var=3.8995,
                   coeffs=[2*0.9688*np.cos(2*np.pi*0.1449/sfreq),
                           -0.9385],
                   num_prev=2)
fast_data = ARData(ntimes, noise_var=3.3963,
                   coeffs=[2*0.9601*np.cos(2*np.pi*9.0134/sfreq),
                           -0.9218],
                   num_prev=2)

times = np.arange(0.5*sfreq, dtype=np.float64) / sfreq - 0.1
erp = 1 * erpdata(times, rng.randn(1)) / 5e-8
events1 = np.zeros(ntimes)
events1[np.arange(100, ntimes, 200) + rng.randint(-25, 25)] = 1
erp_data1 = signal.convolve(events1, erp, 'full')[10:events1.shape[-1]+10]
events2 = np.zeros(ntimes)
events2[np.arange(20, ntimes, 200) + rng.randint(-25, 25)] = 1
erp_data2 = signal.convolve(events2, erp, 'full')[10:events1.shape[-1]+10]

source_time_series1 = 15e-9 * slow_data.y[10:] / slow_data.y.std()
source_time_series2 = 2 * np.sqrt(5/4) * slow_data.y[:-10] / slow_data.y.std()
source_time_series3 = 2 * np.sqrt(3/4) * fast_data.y / fast_data.y.std()
source_time_serieses = (source_time_series1, source_time_series2,
                        source_time_series3)

source_time_series = np.stack(source_time_serieses)
# source_time_series -= source_time_series.mean(axis=-1)[:, None]
weights = np.array([0, 1, 1])
data = weights.dot(source_time_series) + 1 * erp_data1 - 0 * erp_data2
noisy_data = data + 0.01 * rng.randn(*data.shape)

if debug:
        fig, [ax, ax1, ax2] = plt.subplots(nrows=3, ncols=1, figsize=[12.5, 7.5])
        lines = ax.plot(np.arange(ntimes) / sfreq, np.vstack((source_time_series2,
                                                        source_time_series3)
                                                        ).T)
        ax.legend(lines, ['osc 1', 'osc 2'])
        ax.set_frame_on(False)
        ax.grid(True, linestyle='-.', linewidth=0.2)
        ax.tick_params(left=False, bottom=False)
        ax.set_xlabel('time (in s)')
        ax.set_ylabel('amplitude (in a.u.)')

        # fig, ax1 = plt.subplots(figsize=[12.5, 2.5])
        lines = ax1.plot(np.arange(ntimes) / sfreq, erp_data1)
        lines = ax1.plot(np.arange(ntimes) / sfreq, erp_data2)
        # ax1.plot(np.arange(ntimes) / sfreq, events, 'r|')
        ax1.set_frame_on(False)
        ax1.grid(True, linestyle='-.', linewidth=0.2)
        ax1.tick_params(left=False, bottom=False)
        ax1.set_xlabel('time (in s)')
        ax1.set_ylabel('amplitude (in a.u.)')
        ax1.set_title('simulated ERP')
        axins = inset_axes(ax1, width="20%", height="15%", loc=1)
        lineins = axins.plot(times, erp)
        axins.grid(True, linestyle='-.', linewidth=0.2)
        axins.tick_params(labelleft=False, left=False, bottom=False,
                                labelbottom=True)

        lines = ax2.plot(np.arange(ntimes) / sfreq, np.vstack((data,
                                                        noisy_data)).T)
        ax2.legend(lines, ['osc 1 + osc 2 + erp',
                                'osc 1 + osc 2 + erp + noise'])
        ax2.set_frame_on(False)
        ax2.grid(True, linestyle='-.', linewidth=0.2)
        ax2.tick_params(left=False, bottom=False)
        ax2.set_xlabel('time (in s)')
        ax2.set_ylabel('amplitude (in a.u.)')
        plt.show(block=False)
        fig.tight_layout()

# estimation
a = .9
lag_max = 50
lag_min = -10
n_osc, ar_order = 2, 13
y = noisy_data[None, :].copy()
stims = [events1]
# initial_guess = ([0.96, 0.96], [0.00184, 0.09044242], [1, 0.3])
# initial_guess = ([0.99999, 0.99999], [2 / sfreq, 9. / sfreq], [0.1, 0.1])
initial_guess = None
# osc, fir, osc_tc, y2, tau, all_epochs, lls, sfreq = _fit_erp(y, sfreq, stims,
#                                                              lag_max, lag_min, a,
#                                                              n_osc, ar_order,
#                                                              initial_guess)
sserp = fit_erp(y, sfreq, stims,
                0.5, -0.1, a,
                n_osc, ar_order, initial_guess)

osc_tc = sserp._osc_timecourse
all_epochs = sserp._epochs

if debug:
        fig = plt.figure(figsize=[12.5, 7.5], tight_layout=True)
        axes = []

        axes.append(fig.add_subplot(311))
        line4 = axes[0].plot(np.arange(ntimes) / sfreq, y.T, 'k', alpha=0.2, label = 'data')
        line1 = axes[0].plot(np.arange(ntimes) / sfreq, osc_tc[::2].sum(axis=0), 'b', label='osc')
        # line2 = axes[0].plot(np.arange(ntimes) / sfreq, y2.T, 'r', label = 'evoked')
        line3 = axes[0].plot(np.arange(ntimes) / sfreq, y.T - osc_tc[::2].sum(axis=0)[:, None], 'g', label='data -osc')
        axes[0].legend()
        axes[0].set_xlabel('time (in s)')
        axes[0].set_ylabel('amplitude (in a.u.)')
        axes[0].set_title('Osc + ERP decomposition')

        axes.append(fig.add_subplot(323))
        # line1 = axes[1].plot(np.arange(n_lags) / sfreq, fir.h().T, color='b')
        # ci = 2.576 * fir.h_ci()
        # [axes[1].fill_between(np.arange(n_lags) / sfreq,
        #                       lb,
        #                       ub,
        #                       color='b',
        #                       alpha=.5)
        #  for lb, ub in zip(fir.h() - ci, fir.h() + ci)]
        line1 = sserp.fir.plot(confidence=.9, sfreq=sfreq, ax=axes[1], alpha=0.5)
        classical_erp, lci, uci = confidence_interval(all_epochs[0], axis=0,
                                                confidence=.9)
        line2 = axes[1].plot(np.arange(lag_min, lag_max) / sfreq, classical_erp.T,
                        color='r')
        axes[1].fill_between(np.arange(lag_min, lag_max) / sfreq,
                        lci[0],
                        uci[0],
                        color=line2[0].get_color(),
                        alpha=.5)
        line3 = axes[1].plot(times, erp, 'g')
        # line3 = axes[1].plot(times, np.vstack((erp, -2*erp)).T, 'k')
        lines = [line1[0], line2[0], line3[0]]
        axes[1].legend(lines, ['proposed ERP', 'classical ERP', 'True ERP'])
        axes[1].set_xlabel('time (in s)')
        axes[1].set_ylabel('amplitude (in a.u.)')
        axes[1].set_title('ERP comparison')
        axes[-1].set_xlim([-0.1, 0.5])

        axes.append(fig.add_subplot(324))
        sserp.oscs.plot_osc_spectra(N=200, ax=axes[-1], fs=sfreq)
        x = noisy_data - noisy_data.mean(axis=0)
        psd, freq = time_frequency.psd_array_multitaper(x,
                                                        sfreq,
                                                        bandwidth=2.,
                                                        adaptive=False,
                                                        low_bias=True,
                                                        normalization='length',)
        axes[-1].semilogy(freq, psd, '-.k')
        axes[-1].set_xlabel('Frequency (in Hz)')
        axes[-1].set_ylabel('PSD')
        axes[-1].set_title('Recovered oscillators')

        axes.append(fig.add_subplot(325))
        line1 = sserp.fir.plot(confidence=.9, sfreq=sfreq, ax=axes[-1], alpha=0.5)
        classical_erp, lci, uci = confidence_interval(all_epochs[0], axis=0,
                                                confidence=.9)
        line2 = axes[-1].plot(np.arange(lag_min, lag_max) / sfreq, classical_erp.T,
                        color='r')
        axes[-1].fill_between(np.arange(lag_min, lag_max) / sfreq,
                        lci[0],
                        uci[0],
                        color=line2[0].get_color(),
                        alpha=.5)
        line3 = axes[-1].plot(times, erp, 'g')
        # line3 = axes[1].plot(times, np.vstack((erp, -2*erp)).T, 'k')
        lines = [line1[0], line2[0], line3[0]]
        axes[-1].legend(lines, ['proposed ERP', 'classical ERP', 'True ERP'])
        axes[-1].set_xlabel('time (in s)')
        axes[-1].set_ylabel('amplitude (in a.u.)')
        axes[-1].set_title('ERP comparison')
        axes[-1].set_ylim([-0.5, 0.5])
        axes[-1].set_xlim([-0.1, 0.5])

        axes.append(fig.add_subplot(326))
        ll_plot = axes[-1].plot(range(3, len(sserp._lls)), sserp._lls[3:])
        axes[-1].set_xlabel('Iterations')
        axes[-1].set_ylabel('ll')
        axes[-1].set_title('Convergence')
        for ax in axes:
                ax.set_frame_on(False)
                ax.grid(True, linestyle='-.', linewidth=0.2)
                ax.tick_params(labelleft=True, left=False, bottom=False,
                                labelbottom=True)
                plt.show(block=False)
                fig.tight_layout()
