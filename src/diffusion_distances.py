import os
import re

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation
from sklearn.neighbors import KernelDensity
from sklearn.datasets import make_circles, make_blobs
from sklearn.metrics.pairwise import euclidean_distances


def main():
    X, y = make_circles(n_samples=300, noise=0.03, factor=0.5)
    # X, y = make_blobs(n_samples=[300,300,300])
    # X, y = make_bottleneck(100)

    sigma = 0.175
    W = kernel(X, sigma)
    D = np.diag(W.sum(axis=0))
    P = np.linalg.inv(D).dot(W)

    plt.scatter(X[:,0], X[:,1], c=W[:,1], cmap=plt.cm.jet)
    plt.scatter(X[1,0], X[1,1], marker="*", s=100, c="red")
    plt.axis("off")
    plt.savefig("../output/kernel.png", bbox_inches="tight")

    (U,V) = eigs_ordered( P )

    # Hacky way to create an animated plot
    t_list = [2**x for x in np.arange(np.ceil(np.log2(1e6)))]
    for t in t_list:
        D = diffusion_distances(U, V, t)
        plt.scatter(X[:,0], X[:,1], c=D[:,1], cmap=plt.cm.jet)
        plt.scatter(X[1,0], X[1,1], marker="*", s=100, c="red")
        plt.title("$t = 2^{{{}}}$".format(int(np.log2(t))))
        plt.axis("off")
        plt.savefig("../temp/dists{}.png".format(t), bbox_inches="tight")
        plt.close()

    fig = plt.figure()
    imgs = []
    for t in t_list:
        img = plt.imread("../temp/dists{}.png".format(t))
        imgs.append([plt.imshow(img)])
        os.unlink("../temp/dists{}.png".format(t))

    ani = animation.ArtistAnimation(fig, imgs)
    plt.axis("off")

    writer = animation.PillowWriter(fps=10)
    ani.save("../output/diffusion_circles.gif", writer=writer)


def make_bottleneck(N):
    X1 = np.random.multivariate_normal([-1,0.8], np.diag([0.01,0.01]), size=N)
    X2 = np.random.multivariate_normal([-1,-0.8], np.diag([0.01,0.01]), size=N)
    X3 = np.zeros((int(N/2),2))
    X3[:,0] = np.random.uniform(-1.2, -0.8, size=(int(N/2),))
    X3[:,1] = np.random.uniform(-0.6, 0.6, size=((int(N/2),)))

    XL = np.concatenate((X1, X2, X3), axis=0)
    XR = XL + np.array([2,0])
    XU = np.random.multivariate_normal([0.25,1.0], np.diag([0.01,0.01]), size=N)
    X = np.concatenate((XL, XR, XU), axis=0)
    y = np.concatenate(
        (np.zeros(XL.shape[0]), np.ones(XR.shape[0]), 2*np.ones(XU.shape[0])))

    return X, y


def eigs_ordered( P, truncate=-1, min_eigs=4 ):
    (U,V) = np.linalg.eig( P )

    assert np.linalg.norm(np.imag(U)) < 1e-9, "Complex Eigenvalues Encountered"

    perm = np.argsort(-1*U)
    U = np.real(U[perm])
    V = np.real(V[:,perm])

    if truncate > 0:
        power = np.cumsum(np.square(U))
        power /= power[-1]
        mask = power <= truncate
        mask[:min_eigs] = True

        U = U[mask]
        V = V[:,mask]

    return (U, V)


def diffusion_distances(U, V, t):
    return euclidean_distances(np.power(U, t)*V)


def kernel(X, sigma):
    return np.exp(-euclidean_distances(X) / np.square(sigma))


def lund(X, sigma, t, k, truncate=0.95, plot_stub="", bw=-1, animate_clustering=False):
    W = kernel(X, sigma)

    if plot_stub != "":
        plt.scatter(X[:,0], X[:,1], c=W[:,1], cmap=plt.cm.jet)
        plt.scatter(X[1,0], X[1,1], marker="*", s=100, c="red")
        plt.colorbar()
        path = "../output/kernel{}.png".format(plot_stub)
        plt.savefig(path, bbox_inches="tight")

    D = np.diag(W.sum(axis=0))
    P = np.linalg.inv(D).dot(W)

    if bw < 0:
        bw = np.power(X.shape[0], -1.0/(X.shape[1]+4))

    KDE = KernelDensity(bandwidth=bw)
    KDE.fit(X)
    px  = np.exp(KDE.score_samples(X))

    if plot_stub != "":
        plt.scatter(X[:,0], X[:,1], c=px, cmap=plt.cm.jet)
        plt.colorbar()
        path = "../output/kde{}.png".format(plot_stub)
        plt.savefig(path, bbox_inches="tight")

    (U, V) = eigs_ordered(P, truncate=truncate)
    D = diffusion_distances(U, V, t)

    dw = np.zeros(X.shape[0])
    ix_max = np.argmax(px)
    for ix in range(X.shape[0]):
        if ix != ix_max:
            mask = px >= px[ix]
            mask[ix] = False
            dw[ix] = D[ix,mask].min()*px[ix]
        else:
            dw[ix] = px[ix]*D[ix,:].max()
    
    ixs_dw = np.argsort(-1*dw)
    ixs_px = np.argsort(-1*px)
    if plot_stub != "":
        plt.bar(np.arange(X.shape[0]), dw, color="darkgreen")
        path = "../output/weighted_dist_scatter{}.png".format(plot_stub)
        plt.savefig(path, bbox_inches="tight")

        plt.scatter(X[:,0], X[:,1], c=dw, cmap=plt.cm.jet)
        plt.scatter(X[ixs_dw[:k],0], X[ixs_dw[:k],1], marker="*", s=100, c="red")
        path = "../output/weighted_dists{}.png".format(plot_stub)
        plt.savefig(path, bbox_inches="tight")

    counter = 0
    y_hat = np.zeros(X.shape[0])
    y_hat[ixs_dw[:k]] = np.arange(1,k+1)
    for ix in ixs_px:
        if y_hat[ix] == 0:
            mask = px >= px[ix]
            mask[ix] = False
            ixl = np.argmin(D[ix,mask])
            ixy = np.where(mask)[0][ixl]
            y_hat[ix] = y_hat[ixy]
            if animate_clustering & (counter % 5 == 0):
                plt.scatter(X[:,0], X[:,1], c=y_hat, cmap=plt.cm.Paired)
                plt.scatter(X[ixs_dw[:k],0], X[ixs_dw[:k],1], 
                    marker="*", s=100, c="red")
                plt.plot(X[[ix, ixy],0], X[[ix, ixy],1], c="cyan", lw=2)
                plt.savefig("../temp/animate_cluster{}.png".format(counter))
                plt.close()
            counter += 1

    if animate_clustering:
        fig = plt.figure()
        imgs = []
        paths = filter(lambda x: "animate" in x, os.listdir("../temp"))
        paths = sorted(paths, key=lambda x: int("".join(re.findall("\d", x))))
        for p in paths:
            img = plt.imread("../temp/{}".format(p))
            imgs.append([plt.imshow(img)])
            os.unlink("../temp/{}".format(p))
        
        ani = animation.ArtistAnimation(fig, imgs)
        plt.axis("off")
        
        writer = animation.PillowWriter(fps=10)
        path = "../output/animated_clustering{}.gif".format(plot_stub)
        ani.save(path, writer=writer)

    if plot_stub != "":
        plt.scatter(X[:,0], X[:,1], c=y_hat, cmap=plt.cm.Paired)
        path = "../output/imputed_clusters{}.png".format(plot_stub)
        plt.savefig(path, bbox_inches="tight")

    return y_hat