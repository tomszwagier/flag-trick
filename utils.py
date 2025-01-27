import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np


def subspace_distance(U_true, U_found):  # cf. [ye_schubert_2016]
    error = np.linalg.norm(np.arccos(np.clip(np.linalg.svd(U_true.T @ U_found, full_matrices=False)[1], -1, 1)))
    return float(error)


def plot_subspace_distances(U_Gr_list, U_Fl, signature):
    plt.figure()
    plt.plot([subspace_distance(U_Gr_list[k], U_Gr_list[k+1]) for k in range(len(signature) - 1)], label="Gr")
    plt.plot([subspace_distance(U_Fl[:, :signature[k]], U_Fl[:, :signature[k+1]]) for k in range(len(signature) - 1)], label="Fl")
    plt.legend()
    plt.show(block=False)


def plot_nestedness_scatter(X, U_Gr_1, U_Gr_2, U_Fl, y=None):
    n = X.shape[1]
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    X_subspace_1 = U_Gr_1.T @ X
    axes[0, 0].scatter(X_subspace_1, np.zeros(n,), c='tab:red' if y is None else y, alpha=.5)
    axes[0, 0].set_title('Subspace - 1D')
    X_subspace_2 = U_Gr_2.T @ X
    axes[1, 0].scatter(*X_subspace_2, c='tab:red' if y is None else y, alpha=.5)
    axes[1, 0].set_title('Subspace - 2D')
    X_flag = U_Fl.T @ X
    axes[0, 1].scatter(X_flag[0], np.zeros(n,), c='tab:green' if y is None else y, alpha=.5)
    axes[0, 1].set_title('Flag - 1D')
    axes[1, 1].scatter(X_flag[0], X_flag[1], c='tab:green' if y is None else y, alpha=.5)
    axes[1, 1].set_title('Flag - 2D')
    plt.show(block=False)


def plot_nestedness_images(image, center, U_Gr_list, U_Fl, signature):
    fig, axes = plt.subplots(nrows=2, ncols=len(signature) + 2)
    plt.set_cmap("gray")
    data = [U_Gr @ U_Gr.T @ image.flatten() for U_Gr in U_Gr_list] + [U_Fl[:, :q_k] @ U_Fl[:, :q_k].T @ image.flatten() for q_k in signature] + [center.flatten(), (center + image).flatten()]
    vmin, vmax = np.min(data), np.max(data)
    for k, ax in enumerate(axes[0, 1:-1]):
        ax.imshow(center + (U_Gr_list[k] @ U_Gr_list[k].T @ image.flatten()).reshape(image.shape), vmin=vmin, vmax=vmax)
    for k, ax in enumerate(axes[1, 1:-1]):
        ax.imshow(center + (U_Fl[:, :signature[k]] @ U_Fl[:, :signature[k]].T @ image.flatten()).reshape(image.shape), vmin=vmin, vmax=vmax)
    axes[0, -1].imshow(center + image, vmin=vmin, vmax=vmax)
    axes[1, -1].imshow(center + image, vmin=vmin, vmax=vmax)
    axes[0, 0].imshow(center, vmin=vmin, vmax=vmax)
    axes[1, 0].imshow(center, vmin=vmin, vmax=vmax)
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show(block=False)


def plot_explained_variance(X, U_Gr_list, U_Fl, signature):
    p, n = X.shape
    U_Fl_list = [np.zeros((p, p))] + [U_Fl[:, :q_k] for q_k in signature] + [np.eye(p)]
    U_Gr_list = [np.zeros((p, p))] + U_Gr_list + [np.eye(p)]
    plt.figure()
    plt.plot((0,)+signature+(p,), [np.trace(U.T @ X @ X.T @ U) / np.trace(X @ X.T) for U in U_Gr_list], label='Gr')
    plt.plot((0,)+signature+(p,), [np.trace(U.T @ X @ X.T @ U) / np.trace(X @ X.T) for U in U_Fl_list], label='Fl')
    plt.legend()
    plt.show(block=False)


def plot_reconstruction_errors(X, n_in, U_Gr, U_Fl, signature):
    p, n = X.shape
    reconstruction_errors_gr = np.linalg.norm(X - U_Gr @ U_Gr.T @ X, axis=0)
    argsrt = np.argsort(reconstruction_errors_gr)
    reconstruction_errors_gr = reconstruction_errors_gr[argsrt]
    labels_gr = np.array([0] * n_in + [1] * (n - n_in))[argsrt]
    flag_trick = np.zeros((p, p))
    for q_k in signature:
        flag_trick += 1 / len(signature) * U_Fl[:, :q_k] @ U_Fl[:, :q_k].T
    reconstruction_errors_fl = np.linalg.norm(X - flag_trick @ X, axis=0)
    argsrt = np.argsort(reconstruction_errors_fl)
    reconstruction_errors_fl = reconstruction_errors_fl[argsrt]
    labels_fl = np.array([0] * n_in + [1] * (n - n_in))[argsrt]
    plt.figure()
    plt.scatter(np.arange(n)[labels_gr==0], reconstruction_errors_gr[labels_gr==0], label='inliers', color="tab:blue", alpha=.8)
    plt.scatter(np.arange(n)[labels_gr==1], reconstruction_errors_gr[labels_gr==1], label='outliers', color="tab:red", alpha=.8)
    plt.plot(np.arange(n), reconstruction_errors_gr, color='k', label='Reconstruction Errors - Grassmann')
    plt.legend()
    plt.show(block=False)
    plt.figure()
    plt.scatter(np.arange(n)[labels_fl==0], reconstruction_errors_fl[labels_fl==0], label='inliers', color="tab:blue", alpha=.8)
    plt.scatter(np.arange(n)[labels_fl==1], reconstruction_errors_fl[labels_fl==1], label='outliers', color="tab:red", alpha=.8)
    plt.plot(np.arange(n), reconstruction_errors_fl, color='k', label='Reconstruction Errors - Flag')
    plt.legend()
    plt.show()
    plt.show(block=False)


def plot_nestedness_scatter_sc(U_Gr_1, U_Gr_2, U_Fl, y=None):
    n = U_Gr_1.shape[0]
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    axes[0, 0].scatter(U_Gr_1[:, 0], np.zeros(n,), c='tab:red' if y is None else y, alpha=.8)
    axes[0, 0].set_title('Subspace - 1D')
    axes[1, 0].scatter(U_Gr_2[:, 0], U_Gr_2[:, 1], c='tab:red' if y is None else y, alpha=.8)
    axes[1, 0].set_title('Subspace - 2D')
    axes[0, 1].scatter(U_Fl[:, 0], np.zeros(n,), c='tab:green' if y is None else y, alpha=.8)
    axes[0, 1].set_title('Flag - 1D')
    axes[1, 1].scatter(U_Fl[:, 0], U_Fl[:, 1], c='tab:green' if y is None else y, alpha=.8)
    axes[1, 1].set_title('Flag - 2D')
    plt.show(block=False)
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    plt.set_cmap("Accent")
    axes[0, 0].scatter(U_Gr_1/np.linalg.norm(U_Gr_1, axis=1)[:, np.newaxis], np.zeros(n,), c='tab:red' if y is None else y, alpha=.8)
    axes[0, 0].set_title('Subspace - 1D')
    axes[1, 0].scatter(*(U_Gr_2/np.linalg.norm(U_Gr_2, axis=1)[:, np.newaxis]).T, c='tab:red' if y is None else y, alpha=.8)
    axes[1, 0].set_title('Subspace - 2D')
    axes[0, 1].scatter(U_Fl[:, :1]/np.linalg.norm(U_Fl[:, :1], axis=1)[:, np.newaxis], np.zeros(n,), c='tab:green' if y is None else y, alpha=.8)
    axes[0, 1].set_title('Flag - 1D')
    axes[1, 1].scatter(*(U_Fl[:, :2]/np.linalg.norm(U_Fl[:, :2], axis=1)[:, np.newaxis]).T, c='tab:green' if y is None else y, alpha=.8)
    axes[1, 1].set_title('Flag - 2D')
    plt.show(block=False)


def plot_scatter_3D(X, y, U_Gr_list=None, U_Fl=None):
    fig = plt.figure(figsize=(7, 7))
    cmap = plt.get_cmap('tab20c')
    colors = cmap(np.array([0, 4, 8, 12, 16, 1, 5, 9, 13, 17, 2, 6, 10, 14, 18, 3, 7, 11, 15, 19]))  # issue if more than 20 classes
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.scatter(*X, alpha=.5, c=colors[y])
    if U_Fl is not None:
        mu = np.mean(X, axis=1)
        ax1.quiver(*mu[:3], *U_Gr_list[0][:, 0], color='k', length=5, linewidth=3, alpha=.8, label='1D subspace')
        ax1.quiver(*mu[:3], *U_Gr_list[1][:, 0], color='k', length=5, linewidth=3, alpha=.8, linestyle="dashed", label='2D subspace')
        ax1.quiver(*mu[:3], *U_Gr_list[1][:, 1], color='k', length=5, linewidth=3, alpha=.8, linestyle="dashed")
        ax1.legend()
    plt.axis('equal')
    plt.axis('off')
    plt.show(block=False)
