import numpy as np
import matplotlib.pyplot as plt

from mmd_clustering import kernel
from sklearn.datasets import make_blobs
from sklearn.neighbors import KernelDensity
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.kernel_approximation import RBFSampler


def main():
    N = 5000
    X = sample_mvn(N)

    gamma = 0.175
    S = RBFSampler(gamma=gamma, n_components=300)
    
    G = S.fit_transform(X)

    f, axs = plt.subplots(1,2)
    Q = np.tanh((1/N)*G.sum(axis=0))
    p_hat = G.dot(Q)
    s = axs[0].scatter(X[:,0], X[:,1], c=p_hat, cmap=plt.cm.jet)
    f.colorbar(s, ax=axs[0])

    V = np.sign(G.sum(axis=0))
    p_hat2 = G.dot(V)
    s = axs[1].scatter(X[:,0], X[:,1], c=p_hat2, cmap=plt.cm.jet)
    f.colorbar(s, ax=axs[1])

    z = np.zeros(G.shape[1])
    for ix in range(10000):
        if (ix % 1000) == 0:
            print(ix)
        
        S = RBFSampler(gamma=gamma, n_components=G.shape[1])
        S.fit(X)

        s = G.sum(axis=0)
        V = np.sign(s)
        Q = np.tanh((1/N)*s)

        z += V - Q

    h = z / 10000


def sample_mvn(N):
    pi = np.random.binomial(1, 0.5, size=(N,1))
    x1 = np.random.multivariate_normal(mean=[4,4], cov=np.eye(2), size=(N,))
    x2 = np.random.multivariate_normal(mean=[-4,-4], cov=np.eye(2), size=(N,))
    return pi*x1 + (1-pi)*x2

