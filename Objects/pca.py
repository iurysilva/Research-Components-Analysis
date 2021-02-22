import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, data):
        self.mean = np.mean(data, axis=0)
        data = data-self.mean
        cov = np.cov(data.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        eigenvectors = eigenvectors.T
        id = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[id]
        eigenvectors = eigenvectors[id]
        self.components = eigenvectors[0:self.n_components]

    def transform(self, data):
        data = data-self.mean
        return np.dot(data, self.components.T)
