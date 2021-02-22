import os

import numpy as np
import matplotlib.pyplot as plt

from scipy import sparse
from matplotlib import animation
from sklearn.cluster import KMeans
from skimage.measure import block_reduce
from sklearn.neighbors import KernelDensity
from sklearn.metrics.pairwise import euclidean_distances
from diffusion_distances import lund, diffusion_distances, kernel, eigs_ordered


def main():
    img = plt.imread("../input/obelix.png")[:,:,:3]
    img = block_reduce(img, (10,10,1), np.mean)


def basic_segmentation(X, sigma, k):
    W = kernel(X.reshape(-1,3), sigma)

    U_ = U[1:k+1]
    V_ = V[:,1:k+1]

    S = V_*U_.reshape(1,-1)
    K = KMeans(n_clusters=2)
    K.fit(S)

    y_hat = K.predict(S).reshape(X.shape[0], X.shape[1])



def spectral_embedding(W):    
    ixs = np.arange(W.shape[0])
    diag = np.asarray(W.sum(axis=0)).ravel()
    D_inv = sparse.csr_matrix((1./diag, (ixs, ixs)), shape=W.shape)
    P = D_inv.dot(W)

    (U, V) = sparse_eigs_ordered(P, min_eigs=100)
    U = np.clip(U, -1.0, 1.0)

    return U, V, W


def sparse_kernel(X, sigma):
    rows = []; cols = []; data = []

    combs = [(r,c) for r in range(8) for c in range(8)]
    for my_r in range(X.shape[0]):
        for my_c in range(X.shape[1]):
            ix = sub2ind(X.shape, my_r, my_c)
            for r,c in combs:
                if my_r + r < X.shape[0] and my_c + c < X.shape[1]:
                    ix_n = sub2ind(X.shape, my_r+r, my_c+c)
                    rows.append(ix)
                    cols.append(ix_n)
                    data.append(gaussian_kernel(X[my_r,my_c,:], X[r,c,:], sigma))
    
    size = (X.shape[0])*(X.shape[1])
    return sparse.csr_matrix((data, (rows, cols)), shape=(size, size))


def neighbor_graph(X, )


def image_kde(X, bw, wsize=4):
    Z = np.zeros((X.shape[0], X.shape[1]))
    K = KernelDensity(bandwidth=bw)
    for r in range(X.shape[0]):
        for c in range(X.shape[1]):
            minr, maxr = max(r-wsize, 0), min(r+wsize, X.shape[0])
            minc, maxc = max(c-wsize, 0), min(c+wsize, X.shape[1])
            W = X[minr:maxr,minc:maxc,:].reshape(-1,3)
            K.fit(W)
            Z[r,c] = K.score_samples(X[r,c,:].reshape(1,3))
    return Z

def sub2ind(array_shape, rows, cols):
    return rows*array_shape[1] + cols


def ind2sub(array_shape, ind):
    rows = (int(ind) // array_shape[1])
    cols = (int(ind) % array_shape[1])
    return (rows, cols)


def gaussian_kernel(x, y, sigma):
    return np.exp(-np.linalg.norm(x - y) / np.square(sigma))


def sparse_eigs_ordered( P, truncate=-1, min_eigs=4 ):
    (U,V) = sparse.linalg.eigs( P, k=min_eigs )

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


def lund_segmentation(X, sigma, t, k, truncate=0.95, plot_stub="", bw=-1):
    W = sparse_kernel(X, 0.175)
    
    ixs = np.arange(W.shape[0])
    diag = np.asarray(W.sum(axis=0)).ravel()
    D_inv = sparse.csr_matrix((1./diag, (ixs, ixs)), shape=W.shape)
    P = D_inv.dot(W)

    if plot_stub != "":
        plt.imshow(W[0,:].toarray().reshape(X.shape[0], X.shape[1]))
        path = "../output/kernel{}.png".format(plot_stub)
        plt.savefig(path, bbox_inches="tight")
        plt.close()

    # wsize = 4
    # if bw < 0:
    #     bw = np.power(wsize**2, -1.0/(3+4))

    # px = np.exp(image_kde(X, bw))
    px = np.asarray(W.sum(axis=0)).ravel() / (W != 0).sum(axis=0).ravel()

    if plot_stub != "":
        plt.imshow(px.reshape(X.shape[0], X.shape[1]))
        plt.colorbar()
        path = "../output/kde{}.png".format(plot_stub)
        plt.savefig(path, bbox_inches="tight")
        plt.close()

    N = X.shape[0]*X.shape[1]
    (U, V) = sparse_eigs_ordered(P, min_eigs=100)
    U = np.clip(U, -1.0, 1.0)
    ratios = []
    t = 1
    while t < 1e6:
        print(t)
        dw, D = compute_weighted_dists(U, V, N, t, px)
        dws = -1*np.sort(-1*dw)
        dws = np.clip(dws, dws[dws != 0].min(), dws.max())
        ratio = dws / np.roll(dws, -1)
        ratios.append((t, np.max(ratio), np.argmax(ratio)))
        t *= 1.5

    maxv = np.argmax([x[1] for x in ratios])
    t = ratios[maxv][0]
    k = ratios[maxv][2]+1
    dw, D = compute_weighted_dists(U, V, X_.shape[0], t, px)

    ixs_dw = np.argsort(-1*dw)
    ixs_px = np.argsort(-1*px)
    if plot_stub != "":
        plt.bar(np.arange(10*k), dw[ixs_dw][:10*k], color="darkgreen")
        plt.yscale("log")
        plt.ylabel("$D_{{t}}(x)$")
        path = "../output/weighted_dist_scatter{}.png".format(plot_stub)
        plt.savefig(path, bbox_inches="tight")
        plt.close()

    plt.imshow(X)
    for ix in range(k):
        r,c = ind2sub(X.shape, ixs_dw[ix])
        plt.scatter(c,r, marker="*", s=10, c="red")

    counter = 0
    y_hat = np.zeros(X_.shape[0])
    y_hat[ixs_dw[:k]] = np.arange(1,k+1)
    for ix in ixs_px:
        if y_hat[ix] == 0:
            mask = np.logical_and(px >= px[ix], y_hat != 0)
            mask[ix] = False
            ixl = np.argmin(D[ix,mask])
            ixy = np.where(mask)[0][ixl]
            y_hat[ix] = y_hat[ixy]
            counter += 1

    if plot_stub != "":
        plt.scatter(X[:,0], X[:,1], c=y_hat, cmap=plt.cm.Paired)
        path = "../output/imputed_clusters{}.png".format(plot_stub)
        plt.savefig(path, bbox_inches="tight")

    return y_hat


def animate_diffusion(X, sigma, t_max, stub):
    X_ = X.reshape(-1,3)
    W = kernel(X_, sigma)
    D = np.diag(W.sum(axis=0))
    P = np.linalg.inv(D).dot(W)

    (U,V) = eigs_ordered( P, truncate=0.99 )
    U = np.clip(U, 0.0, 1.0)

    # Hacky way to create an animated plot
    t_list = [2**x for x in np.arange(np.ceil(np.log2(t_max)))]
    for t in t_list:
        D = diffusion_distances(U, V, t)
        plt.imshow(D[0,:].reshape(X.shape[0], X.shape[1]))
        plt.title("$t = 2^{{{}}}$".format(int(np.log2(t))))
        plt.axis("off")
        plt.savefig("../temp/dists{}.png".format(t), bbox_inches="tight")
        plt.close()

    paths = ["../temp/dists{}.png".format(t) for t in t_list]
    animation_from_imgs(paths, "../output/diffusion{}.gif".format(stub))


def animation_from_imgs(paths, save_path):
    fig = plt.figure()
    imgs = []
    for p in paths:
        img = plt.imread(p)
        imgs.append([plt.imshow(img)])
        os.unlink(p)

    ani = animation.ArtistAnimation(fig, imgs)
    plt.axis("off")

    writer = animation.PillowWriter(fps=10)
    ani.save(save_path, writer=writer)
    plt.close()