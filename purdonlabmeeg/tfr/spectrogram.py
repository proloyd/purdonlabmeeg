import numpy as np
import mne

def _compute_mtm_spectrogram(raw, window_len, overlap, fmin=0.0, fmax=np.inf,
                            halfbandwidth=4, adaptive=False, low_bias=True,
                            normalization='length', output='power',
                            max_iter=50, verbose=None):
    events = mne.make_fixed_length_events(raw, duration=window_len, overlap=overlap)
    epochs = mne.Epochs(raw, events, tmin=0, tmax=window_len,
                        baseline=None, detrend=0, reject_by_annotation=False)
    psds = epochs.compute_psd(method='multitaper', fmin=fmin, fmax=fmax, 
                              bandwidth=halfbandwidth, adaptive=adaptive,
                              low_bias=low_bias, normalization=normalization,
                              output=output, verbose=verbose)
    data, freqs = psds.get_data(return_freqs=True)
    sfreq = raw.info['sfreq']
    tstart = int(window_len * sfreq / 2)
    tstep = int((window_len - overlap) * sfreq)
    times = raw.times[tstart::tstep]
    times = times[:data.shape[0]]
    if output == 'complex':
        weights = psds._mt_weights
        return data, freqs, times, weights
    return data, freqs, times


def compute_mtm_spectrogram(raw, window_len, overlap, fmin=0.0, fmax=np.inf,
                            halfbandwidth=4, adaptive=False, low_bias=True,
                            normalization='length', output='power',
                            max_iter=50, verbose=None):
    out = _compute_mtm_spectrogram(raw, window_len, overlap, fmin, fmax,
                                   halfbandwidth, adaptive, low_bias,
                                   normalization, output, max_iter, verbose)
    if output == 'complex':
        data, freqs, times, weights = out
    else:
        data, freqs, times = out
        weights = None
    return MTSpectrogram(raw.info, data, freqs, times, halfbandwidth,
                           normalization, output, weights)



class MTSpectrogram:
    def __init__(self, info, data, freqs, times, halfbandwidth,
                 normalization, output, weights):
        self.info = info
        self.data = data
        n_times, n_freqs = data.shape[0], data.shape[-1]
        assert n_times == len(times)
        assert n_freqs == len(freqs)
        self.freqs = freqs
        self.times = times
        self.halfbandwidth = halfbandwidth
        self.normalization = normalization
        self.output = output
        if self.output == 'complex':
            self._mt_weights = weights

    def plot(self, picks, dB=True):
        from mne.io.pick import _picks_to_idx
        from matplotlib import pyplot as plt
        if isinstance(picks, str):
            picks = [picks]
        idx = _picks_to_idx(self.info, picks, none='all', exclude=(), allow_empty=True)
        Y = np.repeat(self.times[None,:], len(self.freqs), axis=0)
        X = np.repeat(self.freqs[:, None], len(self.times), axis=1)
        C = self.data.copy()[:, idx]
        if self.output == 'complex':
            C = np.mean(np.abs(C) ** 2 * self._mt_weights,
                       axis=-2).real 
        unit = 'V^2/Hz'
        if dB == True:
            C = 10 * np.log10(C)
            C[np.isnan(C)] = -100
            unit += ' (dB)'
        figs = []
        for ii, pick in enumerate(picks):
            fig, ax = plt.subplots()
            fig.suptitle(f'{pick}')
            im = ax.pcolormesh(Y, X, C[:, ii,].T)
            ax.set_xlabel('Time (s)')
            ax.set_label('Freq (Hz)')
            cax = fig.colorbar(im, label=f'Power {unit}')
            figs.append(fig)
        return figs
        
