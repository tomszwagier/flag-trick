import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import autograd.numpy as anp
import pymanopt
from time import time

from flag import Flag
from utils import plot_nestedness_scatter, subspace_error


# TODO: use the Grassmann class for Grassmann optimization parts (now that they are the same)?


def synthetic(n_in, n_out):
    center = anp.zeros((3,))
    X_in = anp.random.multivariate_normal(center, anp.diag([1, .1, .1]), size=n_in).T
    X_out = anp.random.multivariate_normal(center, anp.diag([.1, .1, 5]), size=n_out).T
    # X_out = 2 * X_out / anp.linalg.norm(X_out, axis=0)
    X, y = anp.concatenate([X_in, X_out], axis=1), anp.array([0] * n_in + [1] * n_out)
    fig = plt.figure(figsize=(7, 7))
    cmap = matplotlib.cm.get_cmap('tab20c')
    colors = cmap(anp.unique(y)/5)
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.scatter(*X, alpha=.5, c=colors[y])
    ax1.legend()
    plt.axis('equal')
    plt.axis('off')
    plt.show(block=False)
    return X, y


if __name__ == "__main__":
    anp.random.seed(42)

    n_in, n_out = 450, 50
    X, y = synthetic(n_in, n_out)
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
    optimizer_fl = pymanopt.optimizers.SteepestDescent(verbosity=0)
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
        optimizer_gr = pymanopt.optimizers.SteepestDescent(verbosity=0)
        result_gr = optimizer_gr.run(problem_gr, initial_point=pymanopt.manifolds.grassmann.Grassmann(p, dim).random_point())
        results_gr.append(result_gr)
    time_gr = time() - start_gr

    cmap = matplotlib.cm.get_cmap('tab20c')
    colors = cmap(anp.unique(y)/5)
    plot_nestedness_scatter(X, results_gr[0].point, results_gr[1].point, result_fl.point, y=colors[y])
    print(f"Gr: nestedness_errors = {[subspace_error(results_gr[k].point, results_gr[k+1].point, type='angle') for k in range(len(signature) - 1)]}, time = {time_gr}")
    print(f"Fl: nestedness_errors = {[subspace_error(result_fl.point[:, :signature[k]], result_fl.point[:, :signature[k+1]], type='angle') for k in range(len(signature) - 1)]}, time = {time_fl}")
