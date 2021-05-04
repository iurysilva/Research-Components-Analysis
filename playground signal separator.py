import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import FastICA
from scipy.signal import lfilter
from scipy import linalg


def return_mask(half_life, half_lives_per_mask, max_mask_lenght):
    mask_lenght = min(half_lives_per_mask*half_life, max_mask_lenght)
    print('Returning mask for Complexity Pursuit with size: ', mask_lenght)
    mask = (2**(-1/half_life))**np.arange(0, mask_lenght).T
    mask[0] = 0
    mask = mask/(np.sum(np.abs(mask)))
    mask[0] = -1
    return mask


def apply_BSS(components):
    print('Applying BSS')
    short_mask = return_mask(1.0, 10, 50)
    long_mask = return_mask(900000.0, 10, 50)
    print('calculating filters')
    short_filter = lfilter(short_mask, 1, components, axis=0)
    long_filter = lfilter(long_mask, 1, components, axis=0)
    print('Calculating covariance matrix')
    short_cov = np.cov(short_filter, bias=1, rowvar=False)
    long_cov = np.cov(long_filter, bias=1, rowvar=False)
    print('Calculating eigenvectors and eigenvalues')
    eigen_values, mixture_matrix = linalg.eig(long_cov, short_cov)
    print('mixing matrix shape: ', mixture_matrix.shape, '\n')
    mixture_matrix = np.real(mixture_matrix)
    unmixed = -np.matmul(components, mixture_matrix)
    unmixed = -np.flip(unmixed, axis=1)
    return mixture_matrix, unmixed


n_samples = 2000
time = np.linspace(0, 8, n_samples)

signal_1 = np.sin(2 * time)
signal_2 = np.sign(np.sin(3 * time))
signal_3 = signal.sawtooth(2 * np.pi * time)
signals = np.c_[signal_1, signal_2, signal_3]
signals += 0.2 * np.random.normal(size=signals.shape)
signals /= signals.std(axis=0)
A = np.array([[1, 1, 1], [0.5, 2, 1], [1.5, 1, 2]])
X = np.dot(signals, A.T)
print(X.shape)
autovalores, result = apply_BSS(X)
result = result
fig, ax = plt.subplots(7, 1)
fig.tight_layout()

ax[0].plot(signal_1, color='red')
ax[1].plot(signal_2, color='blue')
ax[2].plot(signal_3, color='green')
ax[3].plot(X[:, 1])
ax[4].plot(result[:, 0], color='green')
ax[5].plot(result[:, 1], color='blue')
ax[6].plot(result[:, 2], color='red')
ax[6].set_xlabel('Tempo')
ax[3].set_ylabel('Amplitude')
for plot in ax:
    plot.spines["top"].set_alpha(0.0)
    plot.spines["bottom"].set_alpha(0.3)
    plot.spines["right"].set_alpha(0.0)
    plot.spines["left"].set_alpha(0.3)
