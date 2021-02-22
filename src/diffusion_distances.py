import os
import re

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.keras.datasets import mnist
from scipy.spatial.transform import Rotation
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.datasets import make_circles, make_blobs, make_swiss_roll


def main():
    sigma = 0.175

    X, y = make_circles(n_samples=300, noise=0.03, factor=0.5)
    animate_diffusion(
        X, sigma, 1e8, "_circles", plot_heatmap=True, animate=True)
    lund(X, sigma, t=1e8, k=2, t_max=1e6,
        plot_stub="_circles", animate_clustering=True, amimate_time_search=True)

    X, y = make_blobs(n_samples=[100]*16, cluster_std=0.5)
    
    X, y = make_bottleneck(100)
    animate_diffusion(X, sigma, 1e14, "_bottleneck")


def plot_eigenvalue_decay(X, sigma_max, stub):
    sigma_grid = np.linspace(1,sigma_max)
    paths = []
    for s in sigma_grid:
        W = kernel(X, s)
        D = np.diag(W.sum(axis=0))
        P = np.linalg.inv(D).dot(W)

        (U,V) = eigs_ordered( P, truncate=0.99 )
        U = np.clip(U, -1.0, 1.0)
        plt.scatter(np.arange(U.size), U)
        plt.title("Sigma: {} - gap: {}".format(s, U[0] / U[1]))
        path = "../temp/decay{}.png".format(s)
        plt.savefig(path)
        paths.append(path)
        plt.close()

    animation_from_imgs(paths, "../output/eigen_decay{}.gif".format(stub))


def animate_diffusion(X, sigma, t_max, stub, plot_heatmap=False, animate=False):
    W = kernel(X, sigma)
    D = np.diag(W.sum(axis=0))
    P = np.linalg.inv(D).dot(W)

    (U,V) = eigs_ordered( P, truncate=0.99 )
    U = np.clip(U, -1.0, 1.0)

    # Hacky way to create an animated plot
    t = 1.0
    scatter_paths = []
    heatmap_paths = []
    while t < t_max:
        print(t)
        D = diffusion_distances(U, V, t)
        if X.shape[1] == 2:
            plt.scatter(X[:,0], X[:,1], c=D[:,1], cmap=plt.cm.jet)
            plt.scatter(X[1,0], X[1,1], marker="*", s=100, c="red")
        if X.shape[1] == 3:
            ix = np.argmax(X[:,2])
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(X[:,0], X[:,1], X[:,2], c=D[:,ix], cmap=plt.cm.jet)
            ax.scatter(X[ix,0], X[ix,1], X[ix,2], marker="*", s=100, c="red")
            ax.view_init(elev=18, azim=73)

        plt.title("$t = 10^{{{:.3f}}}$".format(np.log10(t)))
        plt.axis("off")
        p = "../temp/dists{:.3f}.png".format(t)
        scatter_paths.append(p)
        plt.savefig(p, bbox_inches="tight")
        plt.close()

        if plot_heatmap:
            plt.imshow(D, cmap="viridis")
            plt.title("$t = 10^{{{:.3f}}}$".format(np.log10(t)))
            p = "../temp/heatmap{:.3f}.png".format(t)
            plt.savefig(p, bbox_inches="tight")
            heatmap_paths.append(p)
            plt.close()
        
        t *= 1.5

    if animate:
        animation_from_imgs(
            scatter_paths, "../output/diffusion{}.gif".format(stub))

        if plot_heatmap:
            animation_from_imgs(
                heatmap_paths, "../output/heatmap{}.gif".format(stub))


def animation_from_imgs(paths, save_path):
    fig = plt.figure()
    imgs = []
    for p in paths:
        img = plt.imread(p)
        imgs.append([plt.imshow(img)])
        os.unlink(p)

    ani = animation.ArtistAnimation(fig, imgs)
    plt.axis("off")

    writer = animation.PillowWriter(fps=2)
    ani.save(save_path, writer=writer)
    plt.close()


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


def make_taurus(N):
    X1 = np.zeros((N,3))
    t = np.linspace(-2*np.pi, 2*np.pi, N)
    X1[:,0] = np.sin(t)
    X1[:,1] = np.cos(t)
    X1 = X1 + np.random.uniform(-0.1, 0.1, size=X1.shape)

    R = Rotation.from_euler("xyz", [45, 30, 20], degrees=True)
    t = np.array([[0.5, 0.5, 0]])
    X2 = X1.dot(R.as_matrix())
    X3 = X2.dot(R.as_matrix())

    X = np.concatenate((X1, X2 + t, X3 + 2*t))
    y = np.zeros(X.shape[0])
    y[N:2*N]   = 1
    y[2*N:3*N] = 2

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


def lund(X, sigma, t=-1, k=-1, 
        truncate=0.95, plot_stub="", t_max=-1,
        bw=-1, animate_clustering=False, animate_time_search=True):
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
        plt.close()

    (U, V) = eigs_ordered(P, truncate=truncate)

    t_actual = t

    N = X.shape[0]
    ratios = []
    t = 1
    while t < t_max:
        dw, D = compute_weighted_dists(U, V, N, t, px)
        dws = -1*np.sort(-1*dw)
        if np.all(dws == 0):
            break 
        dws = np.clip(dws, dws[dws != 0].min(), dws.max())
        
        ratio = dws / np.roll(dws, -1)
        ratios.append((t, np.max(ratio), np.argmax(ratio), dws))
        t *= 2

    if animate_time_search:
        paths = []
        for t, r, k, dws in ratios:
            plt.plot(np.arange(32), dws[:32], color="darkgreen", lw=3)
            plt.title("t: {} - r: {} - k: {}".format(int(t), r, k+1))
            p = "../temp/weighted_dists{}.png".format(int(t))
            plt.savefig(p, bbox_inches="tight")
            paths.append(p)
            plt.close()

        animation_from_imgs(
            paths, "../output/weighted_dists{}.gif".format(plot_stub))

    if t_actual < 0:
        maxv = np.argmax([x[1] for x in ratios])
        t_actual = ratios[maxv][0]
        k = ratios[maxv][2]+1

    t = t_actual
    dw, D = compute_weighted_dists(U, V, N, t, px)

    ixs_dw = np.argsort(-1*dw)
    ixs_px = np.argsort(-1*px)
    if plot_stub != "":
        plt.bar(np.arange(10*k), dw[ixs_dw][:10*k], color="darkgreen")
        plt.yscale("log")
        plt.ylabel("$D_{{t}}(x)$")
        path = "../output/weighted_dist_scatter{}.png".format(plot_stub)
        plt.savefig(path, bbox_inches="tight")
        plt.close()

        plt.scatter(X[:,0], X[:,1])
        plt.scatter(X[ixs_dw[:k],0], X[ixs_dw[:k],1], marker="*", s=100, c="red")
        path = "../output/weighted_dists{}.png".format(plot_stub)
        plt.savefig(path, bbox_inches="tight")
        plt.close()

    counter = 0
    y_hat = np.zeros(X.shape[0])
    y_hat[ixs_dw[:k]] = np.arange(1,k+1)
    paths = []
    for ix in ixs_px:
        if y_hat[ix] == 0:
            mask = np.logical_and(px >= px[ix], y_hat > 0)
            mask[ix] = False
            ixl = np.argmin(D[ix,mask])
            ixy = np.where(mask)[0][ixl]
            y_hat[ix] = y_hat[ixy]
            if animate_clustering & (counter % 5 == 0):
                plt.scatter(X[:,0], X[:,1], c=y_hat, cmap=plt.cm.Paired)
                plt.scatter(X[ixs_dw[:k],0], X[ixs_dw[:k],1], 
                    marker="*", s=100, c="red")
                plt.plot(X[[ix, ixy],0], X[[ix, ixy],1], c="cyan", lw=2)
                p = "../temp/animate_cluster{}.png".format(counter)
                paths.append(p)
                plt.savefig(p)
                plt.close()
            counter += 1

    if animate_clustering:
        animation_from_imgs(
            paths, "../output/animated_clustering{}.gif".format(plot_stub))

    if plot_stub != "":
        plt.scatter(X[:,0], X[:,1], c=y_hat, cmap=plt.cm.Paired)
        path = "../output/imputed_clusters{}.png".format(plot_stub)
        plt.savefig(path, bbox_inches="tight")

    return y_hat


def compute_weighted_dists(U, V, n, t, px):
    D = diffusion_distances(U, V, t)

    dw = np.zeros(n)
    ix_max = np.argmax(px)
    for ix in range(n):
        if ix != ix_max:
            mask = px >= px[ix]
            mask[ix] = False
            dw[ix] = D[ix,mask].min()*px[ix]
        else:
            dw[ix] = px[ix]*D[ix,:].max()

    return dw, D


if __name__=="__main__":
    main()
