import matplotlib.pyplot as plt
import autograd.numpy as anp
import pymanopt
from sklearn.datasets import load_digits
from time import time

from data import synthetic_tr_rings
from flag import Flag
from utils import plot_explained_variance, plot_nestedness_scatter, subspace_distance, plot_subspace_distances, plot_scatter_3D, color


def lda_scatter_matrices(X, y):
    X_list = [X[:, y==c] for c in list(anp.unique(y))]
    mu_list = [anp.mean(X_c, axis=1)[:, anp.newaxis] for X_c in X_list]
    n_list = [X_c.shape[1] for X_c in X_list]
    mu = anp.mean(X, axis=1)[:, anp.newaxis]
    Sw = anp.sum([(X_c - mu_c) @ (X_c - mu_c).T for (X_c, mu_c) in zip(X_list, mu_list)], axis=0)
    Sb = anp.sum([n_c * (mu_c - mu) @ (mu_c - mu).T for (n_c, mu_c) in zip(n_list, mu_list)], axis=0)
    return Sw, Sb


def generate_lda_data(X, y):
    center = anp.mean(X, axis=1)[:, anp.newaxis]
    X = X - center
    p, n = X.shape
    C = len(anp.unique(y))
    U_pca = anp.linalg.svd(X, full_matrices=True, compute_uv=True)[0][:, :min(n - C, p)]
    if n - C < p:
        print("CAUTION: PCA done")
    X_pca = U_pca.T @ X
    Sw, Sb = lda_scatter_matrices(X_pca, y)
    Sw = Sw + 10 ** (-5) * anp.trace(Sw) * anp.eye(X_pca.shape[0])
    Sb = Sb + 10 ** (-5) * anp.trace(Sb) * anp.eye(X_pca.shape[0])
    Sw /= anp.trace(Sw)
    Sb /= anp.trace(Sb)
    return U_pca, X_pca, Sw, Sb, center


def learn(p, signature, Sb, Sw, init="random"):
    flag = Flag(p, signature)
    init_point = {"random": flag.random_point(), "svd": anp.linalg.svd(Sb, full_matrices=True, compute_uv=True)[0][:, :signature[-1]]}

    @pymanopt.function.autograd(flag)
    def cost_fl(point):
        flag_trick = anp.zeros((p, p))
        for q_k in signature:
            flag_trick += 1 / len(signature) * point[:, :q_k] @ point[:, :q_k].T
        return - anp.trace(flag_trick @ Sb) / anp.trace(flag_trick @ Sw)

    problem_fl = pymanopt.Problem(flag, cost_fl)
    optimizer_fl = pymanopt.optimizers.SteepestDescent(verbosity=0)
    result_fl = optimizer_fl.run(problem_fl, initial_point=init_point[init])
    return result_fl.point


if __name__ == "__main__":
    anp.random.seed(42)

    X, y = synthetic_tr_rings(n=200)
    (p, n), C = X.shape, len(anp.unique(y))
    signature = (1, 2)
    q = signature[-1]

    # dataset = load_digits()
    # X, y = dataset.data.T, dataset.target
    # (p, n), C = X.shape, len(anp.unique(y))
    # signature = tuple(anp.arange(1, p))
    # q = signature[-1]

    U_pca, X_pca, Sw, Sb, center = generate_lda_data(X, y)

    start_fl = time()
    U_Fl = learn(p, signature, Sb, Sw, init="random")
    time_fl = time() - start_fl
    start_gr = time()
    U_Gr_list = []
    for dim in signature:
        U_Gr = learn(p, (dim,), Sb, Sw, init="random")
        U_Gr_list.append(U_Gr)
    time_gr = time() - start_gr

    print(f"Gr: nestedness_errors = {[subspace_distance(U_Gr_list[k], U_Gr_list[k+1]) for k in range(len(signature) - 1)]}, time = {time_gr}")
    print(f"Fl: nestedness_errors = {[subspace_distance(U_Fl[:, :signature[k]], U_Fl[:, :signature[k+1]]) for k in range(len(signature) - 1)]}, time = {time_fl}")
    plot_nestedness_scatter(X, U_Gr_list[0], U_Gr_list[1], U_Fl, colors=[color(label/4) for label in y])
    # plot_explained_variance(X, U_Gr_list, U_Fl, signature)
    # plot_subspace_distances(U_Gr_list, U_Fl, signature)
    plot_scatter_3D(X, U_Gr_list=U_Gr_list, U_Fl=U_Fl, length=3, colors=[color(label/4) for label in y])
