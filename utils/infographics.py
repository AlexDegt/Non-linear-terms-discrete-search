import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal


def get_figure(name=None, x_name=None, y_name=None, with_fig=False):
    plt.rcParams.update({'font.size': 14})
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    if name is not None:
        ax.set_title(name)
    if x_name is not None:
        ax.set_xlabel(x_name)
    if y_name is not None:
        ax.set_ylabel(y_name)

    if with_fig:
        return fig, ax
    else:
        return ax


def spectrum(ax, x, label, is_complex=True):
    if is_complex:
        ax.psd(x.detach().cpu().view(-1), NFFT=2048, label=label)
    else:
        ax.psd(x.detach().cpu()[0, 0] + 1j *
               x.detach().cpu()[0, 1], NFFT=2048, label=label)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power spectral density (dB/Hz)")

def plot_psd(*signals, Fs=1.0, nfft=2048//1, filename='', legend=(), is_save=False,
             window='blackman', nfig=None, ax=None, bottom_text='', top_text='', title='',#'Power spectral density',
             figsize_x=7, figsize_y=5, ylim = [-60, 10], xshift=0, clf=True, nperseg=None, noverlap=None, y_shift=0):
    """ Plotting power spectral density """
    if nfig is None:
        nfig = plt.figure(figsize=(figsize_x, figsize_y))
        ax = plt.subplot(111)
    else:
        if ax is None:
            ax = plt.subplot(111)
    
    if clf:
        ax.cla()
      
    ax.set_xlabel('frequency')
    xlim = np.array([-Fs/2, Fs/2])
    xlim += xshift
    ax.set_xlim(xlim)
    ax.set_ylabel('Magnitude [dB]')
    ax.set_ylim(ylim)
    ax.set_title(title, fontsize=13)
    ax.grid(True)

    for iisignal in signals:

        win = signal.get_window(window, nfft, True)
        freqs, psd = signal.welch(iisignal, 1, win,
                                  return_onesided=False, detrend=False, nperseg = nperseg, noverlap = noverlap)
        freqs = np.fft.fftshift(freqs)*Fs
        freqs += xshift
        psd = 10.0*np.log10(np.fft.fftshift(psd)) + y_shift
        ax_ptr, = ax.plot(freqs, psd)

    if len(bottom_text):
        plt.figtext(0.5,-0.1, bottom_text, fontsize=13, ha='center', va='bottom')
    
    if len(top_text):
        plt.figtext(0.5,1, top_text, fontsize=13, ha='center', va='top')
    
    if len(legend) > 1:
        ax.legend(legend, fontsize=13)
    if is_save:
        nfig.savefig(filename)
    plt.show()
    return ax_ptr