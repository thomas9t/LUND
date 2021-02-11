import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_circles

def main():
    C1 = np.eye(2)
    C2 = np.random.rand(2,2)
    C2 = C2.dot(C2.T)
    C3 = 3*np.random.rand(2,2)
    C3 = C3.dot(C3.T)

    X1 = np.random.multivariate_normal([3,3],   C1, size=100)
    X2 = np.random.multivariate_normal([-3,-3], C2, size=100)
    X3 = np.random.multivariate_normal([-3,3],  C3, size=100)

    X = np.concatenate((X1, X2, X3), axis=0)
    y = np.zeros(X.shape[0])
    y[100:200] = 1
    y[200:] = 2

    GMM = GaussianMixture(n_components=3, covariance_type="full")
    GMM.fit(X)

    XX, YY = np.meshgrid(np.linspace(X[:,0].min()-1, X[:,0].max()+1),
                         np.linspace(X[:,1].min()-1, X[:,1].max()+1))
    ZZ = -GMM.score_samples(
            np.array([XX.ravel(), YY.ravel()]).T).reshape(XX.shape)
    plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.Paired)
    CS = plt.contour(XX, YY, ZZ, norm=LogNorm(vmin=1.0, vmax=1000.0),
                 levels=np.logspace(0, 3, 20))
    plt.show()

    X, y = make_circles(n_samples=300, noise=0.03)
    GMM = GaussianMixture(n_components=30, covariance_type="full")
    GMM.fit(X)

    XX, YY = np.meshgrid(np.linspace(X[:,0].min()-0.3, X[:,0].max()+0.3),
                         np.linspace(X[:,1].min()-0.3, X[:,1].max()+0.3))
    ZZ = -GMM.score_samples(
            np.array([XX.ravel(), YY.ravel()]).T).reshape(XX.shape)
    plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.Paired)
    CS = plt.contour(XX, YY, ZZ, norm=LogNorm(vmin=1.0, vmax=1000.0),
                 levels=np.logspace(0, 3, 20))
    plt.show()