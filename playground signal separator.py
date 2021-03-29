import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import FastICA
from scipy.signal import lfilter
from scipy import linalg


def return_mask(half_life, half_lives_per_mask, max_mask_lenght):
    mask_lenght = min(half_lives_per_mask*half_life, max_mask_lenght)
    print('Retornando máscara para Complexity Pursuit com tamanho: ', mask_lenght)
    mask = (2**(1/half_life))**(np.arange(mask_lenght - 1))
    mask = mask/(np.sum(np.abs(mask[1:])))
    mask[0] = -1
    return mask


def apply_BSS(principal_components):
    print('Applying BSS')
    short_mask = return_mask(1.0, 8, 500)
    long_mask = return_mask(900000.0, 8, 500)
    print('Calculando filtros')
    short_filter = lfilter(short_mask, 1, principal_components)
    long_filter = lfilter(long_mask, 1, principal_components)
    print('Calculando matrizes de covariância')
    short_cov = np.cov(short_filter)
    long_cov = np.cov(long_filter)
    print('Calculando Auto Valores e Auto Vetores')
    eigen_values, eigen_vectors = linalg.eig(short_cov, long_cov)
    return eigen_values, eigen_vectors


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
autovalores, result = apply_BSS(X.T)
print(result)
result = np.dot(X, result)
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
