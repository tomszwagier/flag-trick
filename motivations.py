import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import autograd.numpy as anp
import pymanopt
from time import time

from flag import Flag
from utils import plot_nestedness_scatter


def synthetic(n_in, n_out):
    center = anp.zeros((3,))
    X_in = anp.random.multivariate_normal(center, anp.diag([1, .1, .1]), size=n_in).T
    X_out = anp.random.multivariate_normal(center, anp.diag([.1, 5, .1]), size=n_out).T
    # X_out = 2 * X_out / anp.linalg.norm(X_out, axis=0)
    X, y = anp.concatenate([X_in, X_out], axisa=1), anp.array([0] * n_in + [1] * n_out)
    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.scatter(*X[:, :n_in], alpha=0.5, color='tab:blue', label='inlier')
    ax1.scatter(*X[:, n_in:], alpha=0.3, color='black', marker="x", label='outlier')
    ax1.legend()
    plt.axis('off')
    plt.show(block=False)
    return X, y, center


if __name__ == "__main__":
    anp.random.seed(42)

    n_in, n_out = 950, 50
    X, y, center = synthetic(n_in, n_out)
    p, n = X.shape

    signature = (1, 2)
    q = signature[-1]

    start_fl = time()
    flag = Flag(p, signature)

    @pymanopt.function.autograd(flag)
    def cost_fl(point):
        flag_trick = anp.zeros((flag._p, flag._p))
        for q_k in flag._signature:
            flag_trick += 1 / len(flag._signature) * point[:, :q_k] @ point[:, :q_k].T
        return anp.sum(anp.linalg.norm(X - flag_trick @ X, ord=2, axis=0))

    problem_fl = pymanopt.Problem(flag, cost_fl)
    optimizer_fl = pymanopt.optimizers.SteepestDescent()
    result_fl = optimizer_fl.run(problem_fl, initial_point=pymanopt.manifolds.grassmann.Grassmann(p, q).random_point())
    time_fl = time() - start_fl

    results_gr = []
    start_gr = time()
    for dim in signature:
        grassmann = Flag(p, (dim,))

        @pymanopt.function.autograd(grassmann)
        def cost_gr(point):
            return anp.sum(anp.linalg.norm(X - point @ point.T @ X, ord=2, axis=0))

        problem_gr = pymanopt.Problem(grassmann, cost_gr)
        optimizer_gr = pymanopt.optimizers.SteepestDescent()
        result_gr = optimizer_gr.run(problem_gr, initial_point=pymanopt.manifolds.grassmann.Grassmann(p, dim).random_point())
        results_gr.append(result_gr)
    time_gr = time() - start_gr

    plot_nestedness_scatter(X, results_gr[0].point, results_gr[1].point, result_fl.point, y=y)
