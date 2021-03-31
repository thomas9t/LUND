import os

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from matplotlib import animation
from diffusion_distances import kernel
from sklearn.datasets import make_circles
from sklearn.linear_model import LogisticRegressionCV


class Encoder:
    def __init__(self, d):
        self.d = d

    def fit(self, X):
        n = X.shape[1]
        self.Phi = np.random.normal(size=(self.d,n))
        self.Phi /= np.linalg.norm(self.Phi, axis=1).reshape(-1,1)
        self.b = np.random.uniform(X.min(), X.max(), size=(1,self.d))
    
    def encode(self, X):
        return np.sign(self.Phi.dot(X.T).T + self.b)


def main():
    X, y = make_circles(n_samples=300, noise=0.03, factor=0.5)
    plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.Paired)
    plt.show()

    n = X.shape[0]
    K = kernel(X, 1.0)
    alpha = cp.Variable((n,1))
    
    mu1 = cp.sum(cp.multiply(alpha, X), axis=0)
    mu2 = cp.sum(cp.multiply(1-alpha, X), axis=0)
    x1 = cp.square(cp.norm(mu1 - mu2))
    x2 = cp.square(alpha)
    prob = cp.Problem(cp.Maximize(x1 + x2), [0 <= alpha, alpha <= 1])

def mmd_cluster(centers, max_centers, x):
    if len(centers) < max_centers:
        centers.append(x.copy())
        return centers
    
    scores = np.array([cosine(x,c) for c in centers])
    weights = scores / float(scores.sum())

    for c, w in zip(centers, weights):
        c += x*w
    
    return centers


def assign_clusters(X, centers):
    C = np.vstack(centers)
    C /= np.linalg.norm(C, axis=1).reshape(-1,1)

    X_ = X / np.linalg.norm(X, axis=1).reshape(-1,1)
    v = C.dot(X_.T)
    return np.argmax(v, axis=0)


def cosine(x, y):
    return x.dot(y) / (np.linalg.norm(x)*np.linalg.norm(y))


def animation_from_imgs(paths, save_path=None):
    fig = plt.figure()
    imgs = []
    for p in paths:
        img = plt.imread(p)
        imgs.append([plt.imshow(img)])
        os.unlink(p)

    ani = animation.ArtistAnimation(fig, imgs)
    plt.axis("off")

    if save_path is not None:
        writer = animation.PillowWriter(fps=2)
        ani.save(save_path, writer=writer)
        plt.close()