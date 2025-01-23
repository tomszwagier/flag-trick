import matplotlib.pyplot as plt
import autograd.numpy as anp
import pymanopt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics.pairwise import euclidean_distances

from time import time

from data import synthetic_sc_moon
from flag import Flag
from utils import subspace_distance, plot_subspace_distances, plot_nestedness_scatter_sc


def normalized_graph_Laplacian(X):
    pairwise_dist = euclidean_distances(X.T, X.T, squared=False)
    W = anp.exp(- pairwise_dist**2 / (2 * anp.median(pairwise_dist)**2))
    D_12 = anp.diag(anp.sum(W, axis=1)**(-1/2))
    L = anp.eye(n) - D_12 @ W @ D_12
    return L


def learn(n, signature, L, beta, init="random"):
    flag = Flag(n, signature)
    init_point = {"random": flag.random_point(), "svd": anp.linalg.svd(L, full_matrices=True, compute_uv=True)[0][:, -signature[-1]:]}

    @pymanopt.function.autograd(flag)
    def cost_fl(point):
        flag_trick = anp.zeros((n, n))
        for q_k in signature:
            flag_trick += 1 / len(signature) * point[:, :q_k] @ point[:, :q_k].T
        return anp.trace(flag_trick @ L) + beta * anp.sum(anp.abs(flag_trick))

    problem_fl = pymanopt.Problem(flag, cost_fl)
    optimizer_fl = pymanopt.optimizers.SteepestDescent(verbosity=0)
    result_fl = optimizer_fl.run(problem_fl, initial_point=init_point[init])
    return result_fl.point


if __name__ == "__main__":
    anp.random.seed(42)

    dataset = load_breast_cancer()
    X, y = dataset.data.T, dataset.target
    X = anp.concatenate([X[:, y==c][:, :100//len(anp.unique(y))] for c in anp.unique(y)], axis=1)  # 100 samples equally distributed between classes
    y = anp.concatenate([y[y==c][:100//len(anp.unique(y))] for c in anp.unique(y)])
    (p, n), C = X.shape, len(anp.unique(y))
    signature = tuple(anp.arange(1, n))
    q = signature[-1]

    X, y = synthetic_sc_moon(n=100)
    (p, n), C = X.shape, len(anp.unique(y))
    signature = (1, 2)
    q = signature[-1]

    L = normalized_graph_Laplacian(X)
    beta = 0.01

    start_fl = time()
    U_Fl = learn(n, signature, L, beta, init="random")
    time_fl = time() - start_fl
    start_gr = time()
    U_Gr_list = []
    for dim in signature:
        U_Gr = learn(n, (dim,), L, beta, init="random")
        U_Gr_list.append(U_Gr)
    time_gr = time() - start_gr

    print(f"Gr: nestedness_errors = {[subspace_distance(U_Gr_list[k], U_Gr_list[k+1]) for k in range(len(signature) - 1)]}, time = {time_gr}")
    print(f"Fl: nestedness_errors = {[subspace_distance(U_Fl[:, :signature[k]], U_Fl[:, :signature[k+1]]) for k in range(len(signature) - 1)]}, time = {time_fl}")
    cmap = plt.get_cmap('tab20c')
    colors = cmap(anp.unique(y)/5)
    plot_nestedness_scatter_sc(U_Gr_list[0], U_Gr_list[1], U_Fl, y=colors[y])
    plot_subspace_distances(U_Gr_list, U_Fl, signature)
