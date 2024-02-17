import os
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.lines as mlines
from scipy import signal
from mne import time_frequency
from purdonlabmeeg.sserp import fit_erp, confidence_interval
from purdonlabmeeg._temporal_dynamics_utils.tests._generate_data import (
                                                                   ARData
                                                                   )
from purdonlabmeeg.sserp._erp import ImpulseResponse, KernelImpulseResponse


plt.rc('font', family='Helvetica')
plt.rc('text', usetex=False)
plt.rc('mathtext', fontset='cm', rm='serif')
plt.rc('savefig', dpi=300, format='svg')
plt.rc('ytick', labelsize='small')
plt.rc('xtick', labelsize='small')
plt.rc('axes', labelsize='small')
plt.rc('axes', titlesize='medium')
plt.rc('grid', color='0.75', linestyle=':')

outdir = 'results_for_publication'
if os.path.exists(outdir):
    pass
else:
    os.mkdir(outdir)


def erpdata(times, seed=0):
    """Function to generate random source time courses

    copied from:
    https://mne.tools/dev/auto_examples/simulation/simulate_evoked_data.html"""
    return (50e-9 * np.sin((25 + 0.1*np.random.randn(len(times))) * times) *
            np.exp(- (times - 0.15 + 0.05 * seed) ** 2 / 0.01))


debug = True
rng = np.random.RandomState(84)
sfreq = 100     # in Hz
ntimes = int(np.round(sfreq * 200))
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
    figure, [ax, ax1, ax2] = plt.subplots(nrows=3, ncols=1, figsize=[12.5, 7.5])
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
    figure.tight_layout()

sserps = []
for tw in (50., 100., 200.):
    # estimation
    a = 1.
    lag_max = 50
    lag_min = -10
    n_osc, ar_order = 2, 13
    y = noisy_data[None, :int(np.round(sfreq * tw))].copy()
    stims = [events1[:int(np.round(sfreq * tw))]]
    initial_guess = None
    sserp = fit_erp(y, sfreq, stims,
                    0.5, -0.1, a,
                    n_osc, ar_order, initial_guess,
                    window="gaussian", window_params=0.070)
    sserps.append(sserp)

if debug:            
    osc_tc = sserp._osc_timecourse
    all_epochs = sserp._epochs
    figure = plt.figure(figsize=[12.5, 7.5], tight_layout=True)
    axes = []

    axes.append(figure.add_subplot(311))
    line4 = axes[0].plot(np.arange(ntimes) / sfreq, y.T, 'k', alpha=0.2, label = 'data')
    line1 = axes[0].plot(np.arange(ntimes) / sfreq, osc_tc[::2].sum(axis=0), 'b', label='osc')
    # line2 = axes[0].plot(np.arange(ntimes) / sfreq, y2.T, 'r', label = 'evoked')
    line3 = axes[0].plot(np.arange(ntimes) / sfreq, y.T - osc_tc[::2].sum(axis=0)[:, None], 'g', label='data -osc')
    axes[0].legend()
    axes[0].set_xlabel('time (in s)')
    axes[0].set_ylabel('amplitude (in a.u.)')
    axes[0].set_title('Osc + ERP decomposition')

    axes.append(figure.add_subplot(323))
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

    axes.append(figure.add_subplot(324))
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

    axes.append(figure.add_subplot(325))
    line1 = sserp.fir.plot(confidence=.9, sfreq=sfreq, ax=axes[-1], alpha=0.5)
    classical_erp, lci, uci = confidence_interval(all_epochs[0], axis=0,
                                            confidence=.9)
    line2 = axes[-1].plot(np.arange(lag_min, lag_max) / sfreq, classical_erp.T,
                          color='r')
    axes[-1].fill_between(np.arange(lag_min, lag_max) / sfreq,
                        lci[0],
                        uci[0],
                        color=line2[0].get_color(),
                        alpha=.2)
    line3 = axes[-1].plot(times, erp, 'g')
    # line3 = axes[1].plot(times, np.vstack((erp, -2*erp)).T, 'k')
    lines = [line1[0], line2[0], line3[0]]
    axes[-1].legend(lines, ['proposed ERP', 'classical ERP', 'True ERP'])
    axes[-1].set_xlabel('time (in s)')
    axes[-1].set_ylabel('amplitude (in a.u.)')
    axes[-1].set_title('ERP comparison')
    axes[-1].set_ylim([-0.5, 0.5])
    axes[-1].set_xlim([-0.1, 0.5])

    axes.append(figure.add_subplot(326))
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
        figure.tight_layout()


figure = plt.figure(figsize=(7.5, 3.7),) #layout='constrained')
gs = figure.add_gridspec(2, 3, left=0.08, right=0.95, top=0.92, hspace=0.8, wspace=0.5)

tstart = 600
tend = 1000
tidx = np.arange(tstart, tend) / sfreq
raw_data = y[:, tstart:tend].T
osc_tc = sserp._osc_timecourse[::2, tstart:tend].sum(axis=0)[:, None]
fir = KernelImpulseResponse(50, -10)
stim_lagged, _ = fir._setup_rcorr(stims, ('gaussian', 7))
evoked_res = sserps[1].fir._mean.dot(stim_lagged)[tstart:tend]

axes = []
axes.append(figure.add_subplot(gs[0, 0]))
line1 = axes[-1].plot(tidx, raw_data, 'k', alpha=0.5, linewidth=1)
axes[-1].set_xlabel('raw data')
axes[0].set_ylabel('amplitude (in a.u.)')
axes.append(figure.add_subplot(gs[0, 1]))
line1 = axes[-1].plot(tidx, osc_tc, 'r', linewidth=1)
axes[-1].set_xlabel('background oscillations')
axes.append(figure.add_subplot(gs[0, 2]))
line3 = axes[-1].plot(tidx, raw_data - osc_tc, 'b', alpha=0.2, linewidth=1, )
line4 = axes[-1].plot(tidx, evoked_res, 'k', linewidth=1, )
axes[-1].set_xlabel('stimulus evoked response')
axes[-1].legend()

for ax in axes:
    # ax.set_xlabel('time (in s)')
    ax.grid(True)
    ax.set_frame_on(False)
    ax.set_xticklabels([])

lower, _ = axes[0].get_ylim()
axes[0].hlines(lower, 6, 8, lw=2)
axes[0].text(7.0, lower + 0.4, '2 s', horizontalalignment='center', fontsize='xx-small')

axes = []
for kk, sserp in enumerate(sserps):
    axes.append(figure.add_subplot(gs[1, kk]))
    classical_erp, lci, uci = confidence_interval(sserp._epochs[0], axis=0, confidence=.9)
    line2 = axes[-1].plot(np.arange(lag_min, lag_max) / sfreq, classical_erp.T,
                          color='r')
    axes[-1].fill_between(np.arange(lag_min, lag_max) / sfreq,
                          lci[0], uci[0], color=line2[0].get_color(),
                          alpha=.2)
    line1 = sserp.fir.plot(confidence=.9, sfreq=sfreq, ax=axes[-1], alpha=0.5)
    line3 = axes[-1].plot(times, erp, 'g')
    lines = [line1[0], line2[0], line3[0]]
    axes[-1].set_title(f'n = {sserp._epochs[0].shape[0]}')

axes[0].set_ylabel('amplitude (in a.u.)')
axes[0].set_xlabel('time (s)')
axes[0].grid(True)
axes[0].set_frame_on(False)
for ax in axes[1:]:
    ax.sharex(axes[0])
    ax.sharey(axes[0])
    ax.set_xlabel('time (s)')
    ax.grid(True)
    ax.set_frame_on(False)

axes[0].set_xlim([-0.1, 0.5])
axes[0].set_ylim([-1.5, 1.5])

figure.text(.01, .95, 'A) SS-ERP effectively removes the background oscillations', size=10)
figure.text(.01, .52, 'B) ERPs as more trials are added', size=10)

# axes[1].set_title('ERP comparison')
aux_figure = plt.figure(figsize=[3.75, 2.0])
gs = aux_figure.add_gridspec(2, 1)
axes = [aux_figure.add_subplot(gs[kk, 0]) for kk in range(2)] 

blue_line = mlines.Line2D([], [], color='b', label='oscillation removed')
orange_line = mlines.Line2D([], [], color='k', label='estimated evoked response')
axes[0].legend(handles=[blue_line, orange_line], ncol=1,
               fontsize='x-small', frameon=False)

labels = ['proposed ERP', 'classical ERP', 'True ERP']
handles = [mlines.Line2D([], [], color=line.get_c(), label=label) 
           for line, label in zip(lines, labels)]
axes[1].legend(handles=handles, ncol=3, fontsize='x-small', frameon=False)
for ax in axes:
    ax.set_frame_on(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

figure.savefig(os.path.join(outdir, 'sim-Example.svg'))
aux_figure.savefig(os.path.join(outdir, 'sim-Example-aux.svg'))
figure.savefig(os.path.join(outdir, 'sim-Example.eps'))
aux_figure.savefig(os.path.join(outdir, 'sim-Example-aux.eps'))