import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np


def subspace_error(U_true, U_found, type="angle"):
    if type == "angle":  # Square root of sum of squared principal angles. Cf. review.
        error = np.linalg.norm(np.arccos(np.clip(np.linalg.svd(U_true.T @ U_found, full_matrices=False)[1], -1, 1)))  # False retains only the number of angles of the smallest subspace (True also actually...)
    elif type == "projection":  # Cf. GMS paper.
        error = np.linalg.norm(U_true @ U_true.T - U_found @ U_found.T)
    else:
        raise NotImplementedError
    return float(error)


def plot_nestedness_scatter(X, U_Gr_1, U_Gr_2, U_Fl):
    n = X.shape[1]
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)  # plt.set_cmap("Accent")
    X_subspace_1 = U_Gr_1.T @ X
    axes[0, 0].scatter(X_subspace_1, np.zeros(n,), c='tab:red', alpha=.5)
    axes[0, 0].set_title('Subspace - 1D')
    X_subspace_2 = U_Gr_2.T @ X
    axes[1, 0].scatter(*X_subspace_2, c='tab:red', alpha=.5)
    axes[1, 0].set_title('Subspace - 2D')
    X_flag = U_Fl.T @ X
    axes[0, 1].scatter(X_flag[0], np.zeros(n,), c='tab:green', alpha=.5)
    axes[0, 1].set_title('Flag - 1D')
    axes[1, 1].scatter(X_flag[0], X_flag[1], c='tab:green', alpha=.5)
    axes[1, 1].set_title('Flag - 2D')
    plt.interactive(False)
    plt.show(block=True)


def plot_nestedness_images(image, U_Gr_list, U_Fl, signature):  # TODO should we add PCA before as commonly done? Should we add the mean, and so an axis for the mean in the plots too?
    fig, axes = plt.subplots(nrows=2, ncols=len(signature) + 1)
    plt.set_cmap("gray")
    for k, ax in enumerate(axes[0, :-1]):
        ax.imshow((U_Gr_list[k] @ U_Gr_list[k].T @ image.flatten()).reshape(image.shape))
    for k, ax in enumerate(axes[1, :-1]):
        ax.imshow((U_Fl[:, :signature[k]] @ U_Fl[:, :signature[k]].T @ image.flatten()).reshape(image.shape))
    axes[0, -1].imshow(image)
    axes[1, -1].imshow(image)
    plt.interactive(False)
    plt.show(block=True)


if __name__ == "__main__":
    U_true = np.eye(5)[:, :2]
    U_found = np.eye(5)[:, 1:2]
    # U_found = np.eye(5)[:, 1:3]
    # U_found = np.eye(5)[:, 2:4]
    # U_found = np.eye(5)[:, 3:5]
    print(subspace_error(U_true, U_found, type="angle"))
