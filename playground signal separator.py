import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import FastICA

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
ica = FastICA(n_components=3)
result = ica.fit_transform(X)
fig, ax = plt.subplots(7, 1)
fig.tight_layout()

ax[0].plot(signal_1, color='red')
ax[1].plot(signal_2, color='blue')
ax[2].plot(signal_3, color='green')
ax[3].plot(X)
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
