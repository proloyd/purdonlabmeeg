import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import signal
from mne import time_frequency
from purdonlabmeeg.sserp import _fit_erp, confidence_interval
from purdonlabmeeg._temporal_dynamics_utils.tests._generate_data import (
                                                                   ARData
                                                                   )
import scipy.io as spio


def erpdata(times, seed=0):
    """Function to generate random source time courses

    copied from:
    https://mne.tools/dev/auto_examples/simulation/simulate_evoked_data.html"""
    return (1 * np.sin(30 * times) *
            np.exp(- (times - 0.15 + 0.05 * seed) ** 2 / 0.01))


data = spio.loadmat('/homes/1/pd640/Workspace/purdonlabmeeg/purdonlabmeeg/sserp/example data/orig_eeg.mat')
source_time_series = np.squeeze(data['y']) / 20
sfreq = data['Fs'][0][0]

rng = np.random.RandomState(42)
ntimes = int(np.round(sfreq * 20.))

times = np.arange(1.*sfreq, dtype=np.float64) / sfreq - 0.1
erp = erpdata(times, rng.randn(1))
events1 = np.zeros(ntimes)
tt = np.arange(100, ntimes, 200)
# events1[tt + rng.randint(-25, 25, tt.shape)] = 1
events1[tt] = 1
shift = int(sfreq * 0.1)
erp_data1 = signal.convolve(events1, erp, 'full')[shift:events1.shape[-1]+shift]

start = rng.randint(0, high=source_time_series.shape[-1]-ntimes)
data = source_time_series[start:start + ntimes]
data -= data.mean()
data -= erp_data1.mean()
noisy_data = data + erp_data1 + 0.8 * rng.randn(*data.shape)

fig, [ax, ax1, ax2] = plt.subplots(nrows=3, ncols=1, figsize=[12.5, 7.5])
lines = ax.plot(np.arange(ntimes) / sfreq, noisy_data)
ax.set_frame_on(False)
ax.grid(True, linestyle='-.', linewidth=0.2)
ax.tick_params(left=False, bottom=False)
ax.set_xlabel('time (in s)')
ax.set_ylabel('amplitude (in microVolts)')

# fig, ax1 = plt.subplots(figsize=[12.5, 2.5])
lines = ax1.plot(np.arange(ntimes) / sfreq, erp_data1.T)
# lines = ax1.plot(np.arange(ntimes) / sfreq, erp_data2)
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

lines = ax2.plot(np.arange(ntimes)[:, None] / sfreq, np.vstack((data,
                                                                noisy_data)).T)
ax2.legend(lines, ['eeg data',
                   'eeg data + erp'])
ax2.set_frame_on(False)
ax2.grid(True, linestyle='-.', linewidth=0.2)
ax2.tick_params(left=False, bottom=False)
ax2.set_xlabel('time (in s)')
ax2.set_ylabel('amplitude (in a.u.)')
plt.show(block=False)
fig.tight_layout()


a = 1.
n_lags = 40
n_osc, ar_order = 2, 8
y = noisy_data[None, :].copy()
stims = [events1]
# initial_guess = ([0.99999, 0.99999], [1.0 / sfreq, 10. / sfreq], [1e-3, 1e-3])
initial_guess = None
osc, fir, osc_tc, y2, tau, all_epochs, lls, sfreq = _fit_erp(y, sfreq, stims,
                                                             n_lags, a,
                                                             n_osc, ar_order,
                                                             initial_guess,
                                                             use_sparsifying_prior=False)


fig = plt.figure(figsize=[12.5, 7.5], tight_layout=True)
axes = []

axes.append(fig.add_subplot(311))
line1 = axes[0].plot(np.arange(ntimes) / sfreq, osc_tc[::2].sum(axis=0), 'b')
line2 = axes[0].plot(np.arange(ntimes) / sfreq, y2.T, 'r')
line3 = axes[0].plot(np.arange(ntimes) / sfreq, noisy_data, 'k', alpha=0.2)
lines = [line1[0], line2[0], line3[0]]
axes[0].legend(lines, ['osc', 'evoked', 'data'])
axes[0].set_xlabel('time (in s)')
axes[0].set_ylabel('amplitude (in a.u.)')
axes[0].set_title('Osc + ERP decomposition')

axes.append(fig.add_subplot(323))
line1 = fir.plot(confidence=.9, sfreq=sfreq, ax=axes[1], alpha=0.5)
classical_erp, lci, uci = confidence_interval(all_epochs[0], axis=0,
                                              confidence=.9)
line2 = axes[1].plot(np.arange(n_lags) / sfreq, classical_erp.T,
                     color='r')
axes[1].fill_between(np.arange(n_lags) / sfreq,
                     lci[0],
                     uci[0],
                     color=line2[0].get_color(),
                     alpha=.5)
line3 = axes[1].plot(times.T, erp.T, 'g')
# line3 = axes[1].plot(times, np.vstack((erp, -2*erp)).T, 'k')
lines = [line1[0], line2[0], line3[0]]
axes[1].legend(lines, ['proposed ERP', 'classical ERP', 'True ERP'])
axes[1].set_xlabel('time (in s)')
axes[1].set_ylabel('amplitude (in a.u.)')
axes[1].set_title('ERP comparison')
# axes[-1].set_xlim([0, 0.5])

axes.append(fig.add_subplot(324))
osc.plot_osc_spectra(N=200, ax=axes[-1], fs=sfreq)
x = noisy_data-noisy_data.mean(axis=-1)
psd, freq = time_frequency.psd_array_multitaper(x,
                                                sfreq,
                                                bandwidth=2.,
                                                adaptive=False,
                                                low_bias=True,
                                                normalization='length',)
axes[-1].semilogy(freq[:, None], psd.T, '-.k')
axes[-1].set_xlabel('Frequency (in Hz)')
axes[-1].set_ylabel('PSD')
axes[-1].set_title('Recovered oscillators')

axes.append(fig.add_subplot(325))
line1 = fir.plot(confidence=.9, sfreq=sfreq, ax=axes[-1], alpha=0.5)
classical_erp, lci, uci = confidence_interval(all_epochs[0], axis=0,
                                              confidence=.9)
line2 = axes[-1].plot(np.arange(n_lags) / sfreq, classical_erp.T,
                      color='r')
axes[-1].fill_between(np.arange(n_lags) / sfreq,
                      lci[0],
                      uci[0],
                      color=line2[0].get_color(),
                      alpha=.5)
line3 = axes[-1].plot(times.T, erp.T, 'g')
# line3 = axes[1].plot(times, np.vstack((erp, -2*erp)).T, 'k')
lines = [line1[0], line2[0], line3[0]]
axes[-1].legend(lines, ['proposed ERP', 'classical ERP', 'True ERP'])
axes[-1].set_xlabel('time (in s)')
axes[-1].set_ylabel('amplitude (in a.u.)')
axes[-1].set_title('ERP comparison')
axes[-1].set_ylim([-25, 25])
axes[-1].set_xlim([0, 0.5])

axes.append(fig.add_subplot(326))
ll_plot = axes[-1].plot(range(3, len(lls[0])), lls[0][3:])
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
